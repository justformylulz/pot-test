#
#Created frenkel.py with one main entry point for your notebook:
#
#
#results = analyze_frenkel_and_migration(
#    {300: "dump_300K.lammpstrj", 600: "dump_600K.lammpstrj", ...},
#    type_map={1: 'Li', 2: 'O'},
#)
#What your LAMMPS dump needs (put this at the top of the file too):
#
#
#dump 1 all custom <every> <file> id type xu yu zu
#id/type for identification, and critically xu yu zu (unwrapped coordinates) — using wrapped x y z would make an atom 
#crossing a periodic boundary look like a fake giant hop. 
#You don't need dump_modify sort id — ASE's reader re-sorts by id internally 
#(I verified this). No velocity/temperature dumping needed either; 
#you already know each run's target T from your own fix nvt/npt command, '
#'so that's just the dict key you pass in.
#
#How the two analyses work:
#
#Frenkel concentration: derives the ideal tetrahedral (normal Li) and 
#octahedral (defect) site grids straight from geometry — octahedral sites are just
#the O sublattice shifted by half the cubic cell diagonal — then classifies 
#every Li atom every frame via a periodic KD-tree nearest-site search. 
#Defaults to deriving this grid fresh per trajectory (from its own first frame) 
#rather than one fixed 0 K reference, so it tracks each temperature's own
# thermal expansion. Produces concentration-vs-T and an Arrhenius (ln c vs 1/T) 
#plot with a fitted apparent Frenkel formation energy.
#Migration mechanism: tracks each Li's nearest-site index frame-to-frame;'
#' whenever it changes, that's a hop, characterized by a clean site-to-site vector.
#Hops are clustered by (length, crystallographic direction family) 
#empirically — it doesn't assume "two paths," it just clusters what actually '
#'shows up, so you can directly check whether your ACE potential's dominant 
#jump geometries match DFT/AIMD literature. Outputs both a length histogram 
#and a 3D plot of the actual hop vectors as the "picture" of the mechanism.
#Testing note: since phonopy/pymatgen/etc. aren't installed in this sandbox, '
#'I couldn't run your real files, so I built a synthetic Li₂O antifluorite 
#supercell, hand-wrote real LAMMPS dump text files, and ran the entire 
#pipeline end-to-end against it — including a deliberately injected
#single-atom Frenkel hop. It correctly reproduced the known geometric 
#tetrahedral–octahedral distance (a√3/4 ≈ 1.996 Å for Li₂O) exactly. 
#That test also caught a real edge case — noise-driven site-reassignment
# flicker producing spurious near-zero-length "hops" — which I fixed with a 
# min_hop_length cutoff (default 0.5 Å) in detect_hops().


#################################################################
# frenkel.py
#
# Analyzes LAMMPS MD trajectories of antifluorite Li2X (Li2O, Li2S, Li2Se,
# ...) for:
#   (1) cationic Frenkel-defect concentration vs. temperature, and
#   (2) the geometry of the Li migration mechanism (nearest-neighbour
#       interstitial hops), by clustering the actual jump vectors observed
#       in the trajectory rather than assuming a fixed number of paths.
#
# This module does NOT run any MD itself -- you run the LAMMPS simulations
# yourself, one per temperature, and hand the resulting dump file paths to
# the functions here.
#
# ---------------------------------------------------------------
# WHAT YOUR LAMMPS DUMP NEEDS TO CONTAIN
# ---------------------------------------------------------------
#   dump 1 all custom <every> <file> id type xu yu zu
#
# - "id" and "type" are required so atoms can be identified and mapped to
#   elements. ASE's dump reader (used below) already re-sorts every frame
#   by "id" internally, so you do NOT need `dump_modify sort id`.
# - "xu yu zu" (UNWRAPPED coordinates) are required, not "x y z". Using
#   wrapped coordinates would make an atom crossing a periodic boundary
#   look like a huge, fake "hop" -- exactly the thing this analysis is
#   trying to detect, so getting this wrong would corrupt both the defect
#   count and the migration-mechanism analysis.
# - Run ONE trajectory per temperature of interest (fixed volume / NVT+NVE,
#   same style as most superionic-conductor MD studies of these materials).
#   You do not need to dump velocities or temperature -- you already know
#   the target T of each run from your own `fix nvt/npt` command, and that
#   is what you pass in as the dict key below.
# - The simulation cell must be ORTHOGONAL (cubic supercell of the
#   antifluorite conventional cell) -- this is what the periodic
#   nearest-site search below assumes, and it's how Li2O/Li2S/Li2Se are
#   normally set up anyway.
#
# NEW PACKAGES used here that are NOT already imported in pot_test.ipynb:
#   - scipy.spatial.cKDTree -- very likely already installed as a
#     dependency of ase/pymatgen even though it's not explicitly imported
#     in your notebook; `pip install scipy` if it's missing.
#################################################################

# General imports
import sys
from collections import Counter

# Mathematical imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

# Import from ase
from ase import Atoms
from ase.io import read
from ase.units import kB

# Import from scipy
from scipy.spatial import cKDTree

# Logger is a plain class (never reassigned at runtime), so it's safe to
# import directly by name -- unlike pot_functions.calc (a mutable global),
# there's no staleness concern here. See phonon_functions.py's import
# comment if you want the full story on that distinction.
from pot_functions import Logger


########################################################
############## ideal-lattice-site helpers ###############
########################################################

##########################################
######## START OF get_ideal_sites() ######
##########################################

#### INPUTS:
# reference_atoms : ASE.Atoms -- a near-ideal (undefected) snapshot of the
#                   SAME supercell used for the MD run, e.g. the very first
#                   frame of the trajectory itself (default usage below),
#                   or a separately supplied, freshly built perfect
#                   antifluorite supercell.
# cation, anion   : chemical symbols, e.g. 'Li' and 'O'.
#### RETURNS:
# dict with:
#   'tetra'  : (N_Li, 3) ideal tetrahedral (normal cation) site positions
#   'oct'    : (N_Li, 3) ideal octahedral (defect/interstitial) site positions
#   'all'    : (2*N_Li, 3) the two stacked together, tetra first then oct
#   'n_tetra': N_Li (== number of cation atoms == number of tetrahedral sites)
#   'cell_lengths' : (3,) the reference cell's orthogonal box lengths
#
#### The geometry, in one sentence:
# in antifluorite Fm-3m, the octahedral interstitial sites sit at exactly
# the anion positions shifted by half the conventional cubic cell diagonal,
# i.e. (a/2, a/2, a/2) -- so instead of hardcoding Wyckoff-position algebra,
# this just takes the reference anion sublattice, shifts it by half of its
# own nearest-neighbour spacing (a = sqrt(2) * nearest O-O distance, the
# standard FCC relation) along all three cubic axes, and wraps the result
# back into the cell. The tetrahedral (normal Li) sites are simply the
# reference cation positions themselves, since by definition every cation
# starts out sitting exactly on one in the ideal structure.

def get_ideal_sites(reference_atoms, cation='Li', anion='O'):

    if not reference_atoms.cell.orthorhombic:
        raise ValueError(
            "get_ideal_sites: reference cell is not orthogonal. This "
            "analysis assumes a cubic/orthogonal antifluorite supercell "
            "(same assumption the periodic nearest-site search downstream "
            "relies on)."
        )

    symbols = np.array(reference_atoms.get_chemical_symbols())
    cation_pos = reference_atoms.get_positions()[symbols == cation]
    anion_pos = reference_atoms.get_positions()[symbols == anion]

    if len(cation_pos) == 0 or len(anion_pos) == 0:
        raise ValueError(
            f"get_ideal_sites: found 0 '{cation}' or 0 '{anion}' atoms in "
            f"reference_atoms -- check the cation/anion symbols and your "
            f"type_map."
        )

    # a = sqrt(2) * (nearest anion-anion distance), the standard FCC
    # nearest-neighbour relation -- avoids ever needing the user to supply
    # the conventional cell parameter separately.
    anion_only = reference_atoms[symbols == anion]
    d_nn = anion_only.get_all_distances(mic=True)
    d_nn = d_nn[d_nn > 1e-6].min()
    a = d_nn * np.sqrt(2)

    shift = np.array([a / 2, a / 2, a / 2])
    oct_atoms = Atoms(
        symbols=[anion] * len(anion_pos),
        positions=anion_pos + shift,
        cell=reference_atoms.cell,
        pbc=True,
    )
    oct_atoms.wrap()
    oct_pos = oct_atoms.get_positions()

    return {
        'tetra': cation_pos,
        'oct': oct_pos,
        'all': np.vstack([cation_pos, oct_pos]),
        'n_tetra': len(cation_pos),
        'cell_lengths': reference_atoms.cell.lengths(),
    }

##########################################
######### END OF get_ideal_sites() #######
##########################################




########################################################
###############  site tracking ######
########################################################



##########################################
######### START OF classify_sites() ######
##########################################

#### INPUTS:
# atoms_list   : list of ASE.Atoms (one per frame), e.g. from
#                read_lammps_trajectory()
# ideal_sites  : dict from get_ideal_sites()
# cation       : chemical symbol of the mobile ion, e.g. 'Li'
#### RETURNS:
# site_type_traj  : (n_frames, n_cations) int array, 0 = tetrahedral
#                   (normal site), 1 = octahedral (defect/interstitial site)
# site_index_traj : (n_frames, n_cations) int array, index of the nearest
#                   ideal site (into ideal_sites['all']) for each cation at
#                   each frame -- used by detect_hops() below to see which
#                   specific site-to-site jump happened, not just its type.
#
#### How it works:
# every cation atom is assigned to whichever ideal site (tetrahedral or
# octahedral) it is currently closest to, using a periodic nearest-neighbour
# search built once on the fixed ideal-site grid (scipy's cKDTree with its
# `boxsize` option, which handles the periodic images natively for an
# orthogonal cell). Both the ideal sites and the queried cation positions
# have to be wrapped into [0, box_length) first -- scipy's periodic KDTree
# requires that, and the "xu yu zu" unwrapped coordinates from the dump can
# otherwise be arbitrarily far outside that range after a long trajectory.

def classify_sites(atoms_list, ideal_sites, cation='Li'):
    cell_lengths = ideal_sites['cell_lengths']
    n_tetra = ideal_sites['n_tetra']

    all_sites_wrapped = ideal_sites['all'] % cell_lengths
    tree = cKDTree(all_sites_wrapped, boxsize=cell_lengths)

    site_type_traj = []
    site_index_traj = []

    for atoms in atoms_list:
        symbols = np.array(atoms.get_chemical_symbols())
        cation_pos = atoms.get_positions()[symbols == cation]
        cation_pos_wrapped = cation_pos % cell_lengths

        _, nearest_idx = tree.query(cation_pos_wrapped)
        site_index_traj.append(nearest_idx)
        site_type_traj.append((nearest_idx >= n_tetra).astype(int))

    return np.array(site_type_traj), np.array(site_index_traj)

##########################################
########## END OF classify_sites() #######
##########################################




########################################################
############# Frenkel defect concentration ##############
########################################################

##########################################
#### START OF frenkel_concentration_series() ####
##########################################

#### INPUTS:
# site_type_traj : (n_frames, n_cations) array from classify_sites()
#### RETURNS:
# (n_frames,) array: fraction of cations sitting on octahedral (defect)
# sites at each frame -- the instantaneous Frenkel defect concentration.
# Every octahedral occupation implies exactly one vacant tetrahedral site
# elsewhere (fixed total cation count), so this single number is both the
# interstitial fraction and the vacancy fraction.

def frenkel_concentration_series(site_type_traj):
    return site_type_traj.mean(axis=1)

##########################################
##### END OF frenkel_concentration_series() #####
##########################################


##########################################
##### START OF analyze_defect_concentration() ####
##########################################

#### INPUTS:
# traj_dict     : dict {temperature_in_K: dump_file_path}, one trajectory
#                 per temperature you simulated.
# type_map      : {lammps_type_int: element_symbol}, see read_lammps_trajectory()
# cation, anion : chemical symbols, e.g. 'Li', 'O'
# reference_atoms : optional ASE.Atoms giving a single, fixed reference
#                 lattice (e.g. a 0 K optimized supercell) used for EVERY
#                 temperature. If None (default), each trajectory instead
#                 builds its own ideal-site grid from ITS OWN first frame --
#                 this is the better default here, since it automatically
#                 accounts for each temperature's own thermal expansion of
#                 the cell rather than comparing every T against one fixed
#                 0 K lattice.
# equil_fraction : fraction of each trajectory's frames to discard as
#                 equilibration before averaging (default 0.2 = first 20%).
# index         : ASE frame-selection string passed to the dump reader.
# plot          : if True, plot concentration vs. T and an Arrhenius-style
#                 ln(concentration) vs. 1/T fit.
# logfile       : Logger() target, default sys.stdout.
#### RETURNS:
# dict with:
#   'T'            : sorted array of temperatures
#   'concentration': mean defect fraction at each T
#   'concentration_std' : std. dev. across the equilibrated frames at each T
#   'series'       : {T: per-frame concentration array} for each trajectory
#   'fit'          : {'E_F_eV': ..., 'slope': ..., 'intercept': ...} if a fit
#                    was possible (needs >= 2 temperatures), else None
#   'fig'          : matplotlib Figure if plot=True, else None
#
# Also returns, per temperature, the site classification arrays -- reused
# by analyze_frenkel_and_migration() below so trajectories are only read
# and classified once even when you want both analyses.

def analyze_defect_concentration(traj_dict, type_map, cation='Li', anion='O',
                                  reference_atoms=None, equil_fraction=0.2,
                                  index=':', plot=True, logfile=sys.stdout):

    log = Logger(logfile)

    temperatures = sorted(traj_dict)
    conc_mean, conc_std, series = [], [], {}
    per_T_data = {}  # reused by analyze_frenkel_and_migration()

    for T in temperatures:
        log(f"=== T = {T} K: reading {traj_dict[T]} ===")
        atoms_list = read_lammps_trajectory(traj_dict[T], type_map, index=index)

        ref = reference_atoms if reference_atoms is not None else atoms_list[0]
        ideal_sites = get_ideal_sites(ref, cation=cation, anion=anion)

        site_type_traj, site_index_traj = classify_sites(atoms_list, ideal_sites, cation=cation)
        conc_series = frenkel_concentration_series(site_type_traj)

        n_equil = int(len(conc_series) * equil_fraction)
        equilibrated = conc_series[n_equil:]

        conc_mean.append(equilibrated.mean())
        conc_std.append(equilibrated.std())
        series[T] = conc_series

        per_T_data[T] = {
            'ideal_sites': ideal_sites,
            'site_index_traj': site_index_traj,
            'n_frames': len(atoms_list),
        }

        log(f"  {len(atoms_list)} frames, {n_equil} discarded as equilibration, "
            f"defect fraction = {equilibrated.mean():.3e} +/- {equilibrated.std():.3e}")

    temperatures = np.array(temperatures, dtype=float)
    conc_mean = np.array(conc_mean)
    conc_std = np.array(conc_std)

    # ---- Arrhenius-style fit: ln(c) = ln(c0) - E_F / (2 kB T) ----
    # the factor of 2 is the standard dilute-limit Frenkel-pair result
    # (formation of a pair costs one interstitial + one vacancy together).
    # Only meaningful away from the superionic transition, where the
    # dilute-defect approximation itself breaks down.
    fit = None
    if len(temperatures) >= 2 and np.all(conc_mean > 0):
        inv_T = 1.0 / temperatures
        ln_c = np.log(conc_mean)
        slope, intercept = np.polyfit(inv_T, ln_c, 1)
        E_F = -2 * kB * slope
        fit = {'E_F_eV': E_F, 'slope': slope, 'intercept': intercept}
        log(f"Arrhenius fit: apparent Frenkel pair formation energy = {E_F:.3f} eV")
    else:
        log("Skipping Arrhenius fit (need >= 2 temperatures with nonzero defect concentration).")

    fig = None
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.errorbar(temperatures, conc_mean, yerr=conc_std, marker='o', capsize=3)
        ax1.set_xlabel("Temperature (K)")
        ax1.set_ylabel("Frenkel defect fraction")
        ax1.set_title("Defect concentration vs. T")

        ax2.errorbar(1.0 / temperatures, np.log(conc_mean),
                     yerr=conc_std / conc_mean, marker='o', capsize=3, linestyle='none')
        if fit is not None:
            x_fit = np.array([min(1.0 / temperatures), max(1.0 / temperatures)])
            ax2.plot(x_fit, fit['slope'] * x_fit + fit['intercept'], '--',
                     label=f"E_F = {fit['E_F_eV']:.3f} eV")
            ax2.legend()
        ax2.set_xlabel("1 / T (1/K)")
        ax2.set_ylabel("ln(defect fraction)")
        ax2.set_title("Arrhenius plot")

        fig.tight_layout()

    return {
        'T': temperatures,
        'concentration': conc_mean,
        'concentration_std': conc_std,
        'series': series,
        'fit': fit,
        'fig': fig,
        '_per_T_data': per_T_data,
    }

##########################################
###### END OF analyze_defect_concentration() #####
##########################################




########################################################
################ migration mechanism ####################
########################################################

##########################################
######### START OF detect_hops() #########
##########################################

#### INPUTS:
# site_index_traj : (n_frames, n_cations) array from classify_sites()
# ideal_sites     : dict from get_ideal_sites()
# min_hop_length  : hops shorter than this (Angstrom) are discarded, default
#                   0.5. This guards against spurious "hops" from an atom
#                   whose classification flickers between two competing
#                   sites purely from thermal noise near a classification
#                   boundary -- real Li-hop distances in these materials are
#                   on the order of ~2 Ang (tetrahedral-octahedral) or more,
#                   so 0.5 Ang comfortably separates noise-driven
#                   reassignment from genuine jumps without touching them.
#### RETURNS:
# (n_hops, 3) array of hop displacement vectors: for every cation whose
# nearest ideal site changes between two consecutive analyzed frames, the
# minimum-image vector from its OLD ideal site to its NEW ideal site. Using
# the clean ideal-site-to-ideal-site vector (rather than the atom's own
# noisy raw displacement) is what makes the vectors cluster tightly by
# length/direction below instead of being smeared out by thermal vibration.

def detect_hops(site_index_traj, ideal_sites, min_hop_length=0.5):
    all_sites = ideal_sites['all']
    cell_lengths = ideal_sites['cell_lengths']

    changed = site_index_traj[1:] != site_index_traj[:-1]
    frame_idx, atom_idx = np.where(changed)

    old_sites = all_sites[site_index_traj[frame_idx, atom_idx]]
    new_sites = all_sites[site_index_traj[frame_idx + 1, atom_idx]]

    vec = new_sites - old_sites
    # minimum-image convention for an orthogonal cell: wrap each component
    # back into [-L/2, L/2)
    vec = (vec + cell_lengths / 2) % cell_lengths - cell_lengths / 2

    lengths = np.linalg.norm(vec, axis=1)
    return vec[lengths >= min_hop_length]

##########################################
########## END OF detect_hops() ##########
##########################################


# A handful of low-index cubic direction families used to label hop
# vectors below. Classification is done via each vector's sorted, absolute
# component pattern, which is invariant under all 48 symmetry operations of
# the cubic point group -- so comparing against ONE representative per
# family correctly covers every symmetry-equivalent version of that
# direction (e.g. [1,1,0], [-1,1,0], [1,0,1], ... all map to '<110>').
_DIRECTION_FAMILIES = {
    '<100>': (1, 0, 0),
    '<110>': (1, 1, 0),
    '<111>': (1, 1, 1),
    '<210>': (2, 1, 0),
    '<211>': (2, 1, 1),
}


def _direction_family(vec, tol_deg=15):
    v = np.sort(np.abs(vec))[::-1]
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return 'other'
    v = v / norm

    best_label, best_cos = 'other', -1.0
    for label, fam in _DIRECTION_FAMILIES.items():
        f = np.sort(np.abs(np.array(fam, dtype=float)))[::-1]
        f = f / np.linalg.norm(f)
        cos = np.dot(v, f)
        if cos > best_cos:
            best_cos, best_label = cos, label

    angle = np.degrees(np.arccos(np.clip(best_cos, -1, 1)))
    return best_label if angle <= tol_deg else 'other'


##########################################
#### START OF describe_and_plot_migration() ####
##########################################

#### INPUTS:
# hop_vectors : (n_hops, 3) array from detect_hops() -- can be pooled from
#               multiple temperatures/trajectories, since the AVAILABLE
#               jump paths are a property of the lattice geometry, not of
#               temperature (temperature changes how often each path is
#               used, not which paths exist).
# length_tol  : hops are grouped into the same "type" if their lengths are
#               within this many Angstrom of each other (default 0.15 Ang).
# plot        : if True, produce the length-histogram + 3D hop-vector figure.
# logfile     : Logger() target.
#### RETURNS:
# (summary, fig)
#   summary : list of dicts, most common jump type first, each with
#             {'length': ..., 'direction': '<hkl>', 'count': ..., 'fraction': ...}
#   fig     : matplotlib Figure (length histogram + 3D quiver of hop
#             vectors) if plot=True, else None
#
#### What it does:
# groups the observed hop vectors into distinct "types" by (rounded length,
# direction family) -- deliberately NOT assuming a fixed number of paths up
# front, so the data itself shows whether one, two, or more distinct jump
# geometries actually dominate.

def describe_and_plot_migration(hop_vectors, length_tol=0.15, plot=True, logfile=sys.stdout):
    log = Logger(logfile)

    if len(hop_vectors) == 0:
        log("No hops detected -- nothing to describe.")
        return [], None

    lengths = np.linalg.norm(hop_vectors, axis=1)
    families = np.array([_direction_family(v) for v in hop_vectors])

    # cluster by (rounded length, direction family)
    rounded_len = np.round(lengths / length_tol) * length_tol
    keys = list(zip(rounded_len, families))
    counts = Counter(keys)

    summary = []
    for (rlen, fam), count in counts.most_common():
        mask = (rounded_len == rlen) & (families == fam)
        summary.append({
            'length': lengths[mask].mean(),
            'direction': fam,
            'count': int(count),
            'fraction': count / len(hop_vectors),
        })

    log(f"Detected {len(hop_vectors)} hop(s), grouped into {len(summary)} distinct jump type(s):")
    for i, s in enumerate(summary):
        log(f"  Type {i + 1}: length {s['length']:.3f} Ang, direction family {s['direction']}, "
            f"{s['count']} hops ({100 * s['fraction']:.1f}%)")

    fig = None
    if plot:
        fig = plt.figure(figsize=(11, 4.5))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.hist(lengths, bins=30, color='tab:blue', alpha=0.8)
        ax1.set_xlabel("Hop length (Ang)")
        ax1.set_ylabel("Count")
        ax1.set_title("Hop length distribution")

        # a picture of the migration mechanism: every hop vector drawn as
        # an arrow from a common origin, so dominant paths show up as
        # tight bundles of arrows pointing in the same direction(s).
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        origin = np.zeros(len(hop_vectors))
        ax2.quiver(origin, origin, origin,
                   hop_vectors[:, 0], hop_vectors[:, 1], hop_vectors[:, 2],
                   length=1.0, normalize=False, color='tab:red', alpha=0.35, arrow_length_ratio=0.15)
        lim = lengths.max() * 1.05
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        ax2.set_zlim(-lim, lim)
        ax2.set_xlabel("x (Ang)")
        ax2.set_ylabel("y (Ang)")
        ax2.set_zlabel("z (Ang)")
        ax2.set_title("Hop vectors (migration mechanism)")

        fig.tight_layout()

    return summary, fig

##########################################
##### END OF describe_and_plot_migration() #####
##########################################




########################################################
################ top-level convenience ###################
########################################################

##########################################
#### START OF analyze_frenkel_and_migration() ####
##########################################

#### INPUTS: same as analyze_defect_concentration(), plus:
# length_tol     : passed through to describe_and_plot_migration()
# min_hop_length : passed through to detect_hops() -- discards spurious
#                  sub-threshold "hops" caused by noise-driven reassignment
#                  flicker rather than genuine ionic jumps.
#### RETURNS:
# dict with keys 'concentration' (== analyze_defect_concentration()'s
# return dict) and 'migration' (== (summary, fig) from
# describe_and_plot_migration(), pooling hops from every trajectory in
# traj_dict).
#
# This is the one function you actually need in pot_test.ipynb for the
# common case -- it reads and classifies every trajectory exactly once and
# reuses that work for both analyses, rather than reading each dump file
# twice.

def analyze_frenkel_and_migration(traj_dict, type_map, cation='Li', anion='O',
                                   reference_atoms=None, equil_fraction=0.2,
                                   length_tol=0.15, min_hop_length=0.5,
                                   index=':', plot=True, logfile=sys.stdout):

    log = Logger(logfile)

    conc_results = analyze_defect_concentration(
        traj_dict, type_map, cation=cation, anion=anion,
        reference_atoms=reference_atoms, equil_fraction=equil_fraction,
        index=index, plot=plot, logfile=logfile,
    )

    log("=== pooling hops from all temperatures for the migration-mechanism analysis ===")
    all_hops = []
    for T, data in conc_results['_per_T_data'].items():
        hops = detect_hops(data['site_index_traj'], data['ideal_sites'], min_hop_length=min_hop_length)
        log(f"  T = {T} K: {len(hops)} hop(s)")
        all_hops.append(hops)
    all_hops = np.vstack(all_hops) if all_hops else np.empty((0, 3))

    migration_summary, migration_fig = describe_and_plot_migration(
        all_hops, length_tol=length_tol, plot=plot, logfile=logfile,
    )

    return {
        'concentration': conc_results,
        'migration': {'summary': migration_summary, 'fig': migration_fig, 'hop_vectors': all_hops},
    }

##########################################
##### END OF analyze_frenkel_and_migration() #####
##########################################



#################################
##### END OF frenkel.py #########
#################################
