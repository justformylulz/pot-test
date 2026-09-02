#################################################################
# rdf.py
#
# Radial distribution functions (total and per species-pair) for any
# periodic ASE.Atoms structure or MD trajectory, built on top of ASE's own
# ase.geometry.analysis.Analysis (which already handles periodic boundary
# conditions and multi-frame averaging correctly) rather than hand-rolled
# pair-distance code.
#
# ---------------------------------------------------------------
# NOTES ON THE LAMMPS MD RUN THIS IS MEANT TO ANALYZE
# ---------------------------------------------------------------
#   dump 1 all custom <every> <file> id type xu yu zu
#
# - Same dump style as frenkel.py: "id type xu yu zu", unwrapped
#   coordinates. RDFs don't actually care about wrapping (they use the
#   minimum-image convention internally either way), so this isn't as
#   strict a requirement here as it was for frenkel.py's hop detection --
#   but using the same dump command for every analysis keeps one dump file
#   reusable everywhere, so it's still what's recommended.
# - A SINGLE relaxed structure gives you a "static" RDF, which is fine for
#   a quick sanity check but not a proper thermal RDF -- real broadening of
#   the peaks only shows up once you average over many DECORRELATED MD
#   snapshots. Run a production NVT/NVE trajectory and dump every ~10-50 fs
#   over tens of ps; analyze_rdf() below averages over every frame handed
#   to it automatically.
# - Supercell size matters directly here: ASE's rdf routine requires the
#   cell to be big enough that a sphere of radius rmax fits inside the
#   periodic image convention (rmax < half the shortest cell width,
#   roughly). analyze_rdf() picks a safe default automatically, but if you
#   want to resolve further-out coordination shells, use a bigger
#   supercell and pass a larger rmax explicitly.
# - If you're comparing RDFs across temperatures (e.g. across Li2O's
#   superionic transition), run one trajectory per T and call analyze_rdf()
#   once per file -- exactly the same traj-per-temperature pattern
#   frenkel.py uses.
#
# NEW PACKAGES used here that are NOT already imported in pot_test.ipynb:
#   - scipy.signal (for automatic first-coordination-shell detection).
#     Same scipy package already flagged for frenkel.py/point-defects.py.
#################################################################

# General imports
import sys
import itertools

# Mathematical imports
import numpy as np
import matplotlib.pyplot as plt
# scipy renamed cumtrapz -> cumulative_trapezoid in 1.6 and removed the old
# name entirely in 1.14, so import whichever this scipy version actually has
# rather than assuming one or the other.
try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid
from scipy.signal import find_peaks

# Import from ase
from ase import Atoms
from ase.io import read
from ase.data import atomic_numbers
from ase.geometry.analysis import Analysis

from pot_functions import Logger


########################################################
################# input loading helper ###################
########################################################

def _load_structures(source, type_map=None, index=':'):
    """
    Accepts an ASE.Atoms object, a list of ASE.Atoms, or a path to a
    trajectory file (LAMMPS dump: .lammpstrj/.dump/.lmp, or anything else
    ASE can read by extension, e.g. .xyz) and returns a list of Atoms
    frames either way. A single Atoms object gives a "static" RDF; a
    trajectory gives a proper frame-averaged one.
    """
    if isinstance(source, Atoms):
        return [source]
    if isinstance(source, (list, tuple)) and all(isinstance(a, Atoms) for a in source):
        return list(source)

    path = str(source)
    if path.endswith(('.lammpstrj', '.dump', '.lmp')):
        if type_map is None:
            raise ValueError(
                "_load_structures: type_map is required when reading a "
                "LAMMPS dump file, e.g. {1: 'Li', 2: 'O'}."
            )
        specorder = [type_map[t] for t in sorted(type_map)]
        frames = read(path, index=index, format='lammps-dump-text', specorder=specorder)
    else:
        frames = read(path, index=index)

    return [frames] if isinstance(frames, Atoms) else frames


########################################################
##################### RDF computation #####################
########################################################

##########################################
########## START OF compute_rdf() ########
##########################################

#### INPUTS:
# source   : ASE.Atoms, list of Atoms, or trajectory file path (see
#            _load_structures())
# type_map, index : passed straight to _load_structures()
# rmax     : maximum distance (Ang). If None (default), a safe value is
#            picked automatically from the (smallest) supercell in `source`
#            -- 95% of the strict minimum-image limit ASE's rdf routine
#            itself enforces.
# nbins    : number of histogram bins, default 200.
# pairs    : list of (element_a, element_b) tuples to compute PARTIAL RDFs
#            for. If None (default), every unique unordered pair of
#            elements present in the structure is used automatically (plus
#            same-element pairs, e.g. Li-Li), so nothing needs to be
#            specified by hand for the common case.
# logfile  : Logger() target, default sys.stdout.
#### RETURNS:
# dict with:
#   'r'       : (nbins,) bin-center distances (Ang)
#   'total'   : (nbins,) frame-averaged total g(r)
#   'partial' : {(element_a, element_b): (nbins,) frame-averaged partial g(r)}
#   'rmax', 'nbins', 'n_frames', 'n_atoms', 'volume'

def compute_rdf(source, type_map=None, index=':', rmax=None, nbins=200,
                 pairs=None, logfile=sys.stdout):
    log = Logger(logfile)

    frames = _load_structures(source, type_map=type_map, index=index)

    if rmax is None:
        # the same minimum-image safety check ASE's own get_rdf() enforces
        # (cell perpendicular width > 2*rmax in every periodic direction),
        # applied here with a 5% safety margin, using the SMALLEST such
        # width across all frames (cheap insurance against a cell that
        # fluctuates slightly frame to frame).
        h_min = np.inf
        for atoms in frames:
            cell = atoms.get_cell()
            vol = atoms.get_volume()
            for i in range(3):
                axb = np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])
                h = vol / np.linalg.norm(axb)
                h_min = min(h_min, h)
        rmax = 0.5 * h_min * 0.95
        log(f"No rmax given, using automatic rmax = {rmax:.3f} Ang "
            f"(95% of the minimum-image limit for this cell).")

    elements = sorted(set(itertools.chain.from_iterable(
        a.get_chemical_symbols() for a in frames)))
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(elements) for b in elements[i:]]
        log(f"No pairs given, computing every partial RDF automatically: {pairs}")

    ana = Analysis(frames)

    dr = rmax / nbins
    r = (np.arange(1, nbins + 1) - 0.5) * dr

    # ase.geometry.analysis.Analysis.get_rdf() (per image, no_dists=True)
    # already returns an array of exactly `nbins` values -- ase's own
    # internal underflow bin is sliced off before it gets here, so no
    # further trimming is needed (or correct: an already-length-nbins
    # array cannot be trimmed by one more without shifting every bin).
    total_per_frame = ana.get_rdf(rmax, nbins, elements=None)
    total = np.mean(np.array(total_per_frame), axis=0)

    partial = {}
    for el_a, el_b in pairs:
        Z_a, Z_b = atomic_numbers[el_a], atomic_numbers[el_b]
        per_frame = ana.get_rdf(rmax, nbins, elements=[Z_a, Z_b])
        partial[(el_a, el_b)] = np.mean(np.array(per_frame), axis=0)

    n_atoms = len(frames[0])
    volume = frames[0].get_volume()
    log(f"Averaged RDF over {len(frames)} frame(s), {n_atoms} atoms, "
        f"rmax = {rmax:.3f} Ang, {nbins} bins.")

    return {
        'r': r,
        'total': total,
        'partial': partial,
        'rmax': rmax,
        'nbins': nbins,
        'n_frames': len(frames),
        'n_atoms': n_atoms,
        'volume': volume,
    }

##########################################
########### END OF compute_rdf() #########
##########################################




########################################################
################## coordination numbers ###################
########################################################

##########################################
###### START OF coordination_number() ####
##########################################

#### INPUTS:
# r, gr        : (nbins,) distance and g(r) arrays, e.g. rdf_result['r'] and
#                rdf_result['total'] or rdf_result['partial'][(a, b)]
# n_density_B  : number density (atoms / Ang^3) of the "B" species being
#                counted around each "A" atom -- N_B / V for a partial A-B
#                RDF, or N_total / V for the total RDF.
#### RETURNS:
# (r, N_r) -- the running coordination number N(r) = integral_0^r
# 4*pi*r'^2 * n_density_B * g(r') dr', via cumulative trapezoidal
# integration.

def coordination_number(r, gr, n_density_B):
    integrand = 4 * np.pi * r**2 * n_density_B * gr
    N_r = cumulative_trapezoid(integrand, r, initial=0.0)
    return r, N_r

##########################################
####### END OF coordination_number() #####
##########################################


##########################################
##### START OF first_shell_cutoff() ######
##########################################

#### INPUTS:
# r, gr            : (nbins,) distance and g(r) arrays
# prominence_frac : how much a peak/valley must stand out from its
#                   surroundings to count, as a fraction of max(g(r)),
#                   default 0.1 (10%). Raw g(r) -- especially from a short
#                   trajectory, or a single static structure with
#                   near-delta-function peaks -- is noisy enough bin-to-bin
#                   that plain scipy.signal.argrelextrema (a strict
#                   "smaller than both neighbours" test) readily mistakes a
#                   tiny statistical wiggle on the shoulder of the real
#                   peak for "the first minimum", silently truncating the
#                   coordination-number integration far too early.
#                   scipy.signal.find_peaks's `prominence` argument is the
#                   right tool for this: it only counts a peak/valley that
#                   actually stands out by a meaningful amount, so isolated
#                   noise bins are ignored automatically without needing a
#                   separate (and, as it turns out, ringing-prone near
#                   sharp features) smoothing pass first.
#### RETURNS:
# the distance (Ang) of the first genuine local MINIMUM of g(r) after its
# first genuine local MAXIMUM -- the conventional definition of "the first
# coordination shell", used to report a single coordination-number value
# automatically rather than just the full N(r) curve. Returns None if no
# clear peak-then-minimum pattern is found (e.g. too few frames / too
# noisy, or a perfectly static single structure whose peaks are so sharp
# there's no smooth valley between them at all).

def first_shell_cutoff(r, gr, prominence_frac=0.1):
    if gr.max() <= 0:
        return None
    prominence = prominence_frac * gr.max()

    peaks, _ = find_peaks(gr, prominence=prominence)
    if len(peaks) == 0:
        return None
    first_peak = peaks[0]

    valleys, _ = find_peaks(-gr, prominence=prominence)
    valleys_after_peak = valleys[valleys > first_peak]
    if len(valleys_after_peak) == 0:
        return None

    return r[valleys_after_peak[0]]

##########################################
###### END OF first_shell_cutoff() #######
##########################################




########################################################
######################## plotting #########################
########################################################

##########################################
########### START OF plot_rdf() ##########
##########################################

#### INPUTS:
# rdf_result   : dict from compute_rdf()
# coordination : if True (default), also plot the running coordination
#                number N(r) for the total RDF on a second panel.
# savefig      : optional path to save the figure to.
#### RETURNS:
# matplotlib Figure

def plot_rdf(rdf_result, coordination=True, savefig=None):
    r = rdf_result['r']
    n_density = rdf_result['n_atoms'] / rdf_result['volume']

    fig, axes = plt.subplots(2 if coordination else 1, 1, sharex=True,
                              figsize=(6, 7 if coordination else 4))
    ax1 = axes[0] if coordination else axes

    ax1.plot(r, rdf_result['total'], color='black', linewidth=1.8, label='total')
    for (el_a, el_b), gr in rdf_result['partial'].items():
        ax1.plot(r, gr, label=f"{el_a}-{el_b}", linewidth=1.2)
    ax1.axhline(1.0, color='gray', linewidth=0.6, linestyle=':')
    ax1.set_ylabel("g(r)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_title(f"Radial distribution function ({rdf_result['n_frames']} frame(s))")

    if coordination:
        ax2 = axes[1]
        r_N, N_r = coordination_number(r, rdf_result['total'], n_density)
        ax2.plot(r_N, N_r, color='black', linewidth=1.5)
        cutoff = first_shell_cutoff(r, rdf_result['total'])
        if cutoff is not None:
            ax2.axvline(cutoff, color='tab:red', linestyle='--', linewidth=1,
                        label=f"1st shell cutoff = {cutoff:.2f} Ang")
            ax2.legend(fontsize=8)
        ax2.set_ylabel("N(r) (running coordination number)")
        ax2.set_xlabel("r (Ang)")
    else:
        ax1.set_xlabel("r (Ang)")

    fig.tight_layout()
    if savefig is not None:
        fig.savefig(savefig)

    return fig

##########################################
############ END OF plot_rdf() ###########
##########################################




########################################################
################ top-level convenience ###################
########################################################

##########################################
######### START OF analyze_rdf() #########
##########################################

#### INPUTS: same as compute_rdf(), plus:
# plot : if True (default), also call plot_rdf() on the result.
#### RETURNS:
# rdf_result (== compute_rdf()'s return dict) with two extra keys added:
#   'coordination' : {pair_or_'total': (r, N_r)} running coordination number
#                    for the total RDF and every partial RDF
#   'fig'          : matplotlib Figure if plot=True, else None
#
# This is the one function you actually need in pot_test.ipynb for the
# common case: hand it either an ASE.Atoms object or a trajectory file
# path, get back distances, total + partial g(r), coordination numbers,
# and a plot, with every species pair detected automatically.

def analyze_rdf(source, type_map=None, index=':', rmax=None, nbins=200,
                 pairs=None, plot=True, logfile=sys.stdout):
    result = compute_rdf(source, type_map=type_map, index=index, rmax=rmax,
                          nbins=nbins, pairs=pairs, logfile=logfile)

    n_density_total = result['n_atoms'] / result['volume']
    coordination = {'total': coordination_number(result['r'], result['total'], n_density_total)}

    # count how many atoms of the "B" species are actually present, for the
    # correct per-species number density in each partial coordination number
    frames = _load_structures(source, type_map=type_map, index=index)
    symbol_counts = {el: frames[0].get_chemical_symbols().count(el)
                      for el in set(frames[0].get_chemical_symbols())}

    for (el_a, el_b), gr in result['partial'].items():
        n_density_B = symbol_counts.get(el_b, 0) / result['volume']
        coordination[(el_a, el_b)] = coordination_number(result['r'], gr, n_density_B)

    result['coordination'] = coordination
    result['fig'] = plot_rdf(result, coordination=True) if plot else None

    return result

##########################################
########## END OF analyze_rdf() ##########
##########################################



#################################
######## END OF rdf.py ##########
#################################
