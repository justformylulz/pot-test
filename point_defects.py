#################################################################
# point-defects.py
#
# Point-defect formation energies (vacancies, interstitials, antisites) for
# any ASE.Atoms bulk structure, using the standard supercell method:
# generate the defect, relax it with the ASE calculator you've already set
# via pot_functions.set_calculator(), and compare its energy against a
# perfect reference supercell plus elemental chemical-potential references.
#
# ---------------------------------------------------------------
# THIS IS A STATIC (0 K) CALCULATION, NOT AN MD ONE
# ---------------------------------------------------------------
# Point-defect formation energies are conventionally computed on a
# *relaxed, static* structure (geometry optimization, done here via ASE's
# own BFGS), not from a LAMMPS MD trajectory -- there is no "MD advice" to
# give for the core method itself. That said:
#   - If you'd rather relax a defect by annealing-then-quenching it in
#     LAMMPS instead of letting BFGS do it here, that's fine: just point
#     defect_formation_energy() at the final structure (an ASE.Atoms, or a
#     dump/xyz trajectory file -- the last frame is used by default) and
#     set relax=False so this module doesn't re-relax it.
#   - If what you actually want is a defect's finite-temperature
#     CONCENTRATION or its diffusion/migration behaviour (not just its 0 K
#     formation energy), that needs an MD trajectory and is a different
#     question -- see frenkel.py, which was written for exactly that in
#     antifluorite Li2X.
#   - The chemical-potential references this module needs (ref_bulk_dict)
#     are already produced by pot_functions.get_bulk_ref_e() from your
#     existing bulk_strucs_dict -- nothing new to build there.
#
# NEW PACKAGES used here that are NOT already imported in pot_test.ipynb:
#   - scipy.spatial (Voronoi, cKDTree) -- for automatic interstitial-site
#     detection. Very likely already installed as an ase/pymatgen
#     dependency even though it's not explicitly imported in your
#     notebook; `pip install scipy` if it's missing. (Also used by
#     frenkel.py, if you've already set that up.)
#################################################################

# General imports
import sys
from collections import Counter

# Mathematical imports
import numpy as np
import matplotlib.pyplot as plt

# Import from ase
from ase import Atoms
from ase.io import read
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter

# Import from scipy
from scipy.spatial import Voronoi, cKDTree

# Logger is a plain class (never reassigned at runtime), safe to import
# directly. pot_functions itself is imported as a module (not
# `from pot_functions import calc`) so pot_functions.calc is always read at
# call time -- see phonon_functions.py's import comment for the full story
# on why that distinction matters.
import pot_functions
from pot_functions import Logger


########################################################
################# input loading helper ###################
########################################################

def _load_structures(source, type_map=None, index=':'):
    """
    Accepts an ASE.Atoms object, a list of ASE.Atoms, or a path to a
    trajectory file (LAMMPS dump: .lammpstrj/.dump/.lmp, or anything else
    ASE can read by extension, e.g. .xyz) and returns a list of Atoms
    frames either way, so the functions below can handle both input styles
    uniformly.
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
################ defect construction ####################
########################################################

##########################################
######### START OF make_vacancy() ########
##########################################

#### INPUTS:
# atoms : ASE.Atoms, a perfect (defect-free) supercell
# index : integer atom index to remove
#### RETURNS:
# a NEW Atoms object (atoms is left untouched) with that atom removed

def make_vacancy(atoms, index):
    defect = atoms.copy()
    del defect[index]
    return defect

##########################################
########## END OF make_vacancy() #########
##########################################


##########################################
####### START OF make_interstitial() #####
##########################################

#### INPUTS:
# atoms    : ASE.Atoms, a perfect (defect-free) supercell
# species  : chemical symbol of the atom to insert, e.g. 'Li'
# position : Cartesian (3,) position to insert it at -- use
#            find_interstitial_sites() below to get automatic candidates
#            for an arbitrary structure, or (for antifluorite Li2X
#            specifically) frenkel.get_ideal_sites(atoms)['oct'] gives the
#            exact octahedral interstitial sites directly, if you already
#            have frenkel.py set up.
#### RETURNS:
# a NEW Atoms object with one extra atom of `species` appended at `position`

def make_interstitial(atoms, species, position):
    defect = atoms.copy()
    defect += Atoms(species, positions=[position], cell=defect.cell, pbc=defect.pbc)
    return defect

##########################################
######## END OF make_interstitial() ######
##########################################


##########################################
######### START OF make_antisite() #######
##########################################

#### INPUTS:
# atoms            : ASE.Atoms, a perfect (defect-free) supercell
# index_a, index_b : atom indices whose chemical symbols should be swapped
#### RETURNS:
# a NEW Atoms object with the two atoms' species exchanged (e.g. a Li atom
# sitting on an O site and vice versa)

def make_antisite(atoms, index_a, index_b):
    defect = atoms.copy()
    symbols = defect.get_chemical_symbols()
    symbols[index_a], symbols[index_b] = symbols[index_b], symbols[index_a]
    defect.set_chemical_symbols(symbols)
    return defect

##########################################
########## END OF make_antisite() ########
##########################################


##########################################
##### START OF find_interstitial_sites() #####
##########################################

#### INPUTS:
# atoms       : ASE.Atoms, a perfect (defect-free) bulk structure
# min_dist    : candidate sites closer than this (Ang) to any real atom are
#               discarded -- these are numerical Voronoi artifacts right
#               next to an atom, not actual open space. Default 1.0 Ang.
# cluster_tol : candidate sites within this distance (Ang) of each other are
#               merged into one representative site (Voronoi tessellation
#               of a periodic structure produces many near-duplicate
#               vertices for symmetric sites). Default 0.75 Ang.
#### RETURNS:
# (N, 3) array of candidate interstitial Cartesian positions, sorted with
# the most "open" (largest distance to nearest real atom) site first.
#
#### How it works, generally (works for ANY crystal structure, not just
#### antifluorite -- unlike frenkel.py's octahedral-site geometry, which is
#### specific to that one structure type):
# scipy.spatial.Voronoi doesn't understand periodic boundary conditions, so
# the standard trick is used: replicate the structure through its 26
# neighbouring periodic images (3x3x3 block), run a normal (non-periodic)
# Voronoi tessellation on that, and keep only the Voronoi VERTICES that
# land inside the original central cell. Voronoi vertices are exactly the
# points of maximum distance from the surrounding atoms -- i.e. candidate
# interstitial "holes" -- so this generalizes the idea used in frenkel.py
# for one specific structure to work automatically for any of them.

def find_interstitial_sites(atoms, min_dist=1.0, cluster_tol=0.75):
    if not atoms.cell.orthorhombic:
        raise ValueError(
            "find_interstitial_sites: this implementation assumes an "
            "orthogonal cell (for the simple fractional-coordinate cell "
            "membership test below)."
        )

    cell_lengths = atoms.cell.lengths()
    positions = atoms.get_positions()

    # replicate through the 26 neighbouring periodic images so the Voronoi
    # tessellation sees the correct local environment even for atoms near
    # the cell boundary
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    tiled = np.vstack([positions + shift * cell_lengths for shift in shifts])

    vor = Voronoi(tiled)
    vertices = vor.vertices

    # keep only vertices that fall inside the ORIGINAL (central) cell
    inside = np.all((vertices >= 0) & (vertices < cell_lengths), axis=1)
    candidates = vertices[inside]

    if len(candidates) == 0:
        return np.empty((0, 3))

    # discard candidates too close to any real atom (not genuine voids)
    atom_tree = cKDTree(positions, boxsize=cell_lengths)
    dist_to_atom, _ = atom_tree.query(candidates % cell_lengths)
    candidates = candidates[dist_to_atom > min_dist]
    dist_to_atom = dist_to_atom[dist_to_atom > min_dist]

    if len(candidates) == 0:
        return np.empty((0, 3))

    # sort by "openness" (largest distance to nearest atom first), then
    # greedily merge candidates within cluster_tol of an already-kept one
    order = np.argsort(-dist_to_atom)
    candidates, dist_to_atom = candidates[order], dist_to_atom[order]

    kept = []
    for cand in candidates:
        if not kept or np.min(np.linalg.norm(np.array(kept) - cand, axis=1)) > cluster_tol:
            kept.append(cand)

    return np.array(kept)

##########################################
##### END OF find_interstitial_sites() #####
##########################################




########################################################
############### relaxation & formation energy ############
########################################################

##########################################
########## START OF relax_defect() #######
##########################################

#### INPUTS:
# atoms      : ASE.Atoms to relax
# calc       : ASE calculator. If None (default), falls back to
#              pot_functions.calc (read at call time, see the import
#              comment at the top of this file).
# fmax       : force convergence threshold in eV/Ang, default 0.01
# relax_cell : if True, also relax the cell (via FrechetCellFilter). Default
#              False -- standard practice for DILUTE point defects is to
#              relax atomic positions only, at the fixed volume of the host
#              perfect crystal, so the defect's formation energy is
#              referenced to the same volume/pressure state as the bulk.
# logfile    : passed straight to ASE's BFGS logfile (NOT the Logger()
#              convention used elsewhere in this file -- this one comes
#              straight from BFGS itself, so pass None to silence it).
#### RETURNS:
# the relaxed Atoms object (same object that was passed in, modified in place,
# also returned for convenience)

def relax_defect(atoms, calc=None, fmax=0.01, relax_cell=False, logfile=None):
    if calc is None:
        calc = pot_functions.calc
    if calc is None:
        raise ValueError(
            "relax_defect: no calculator given, and pot_functions.calc is "
            "None. Either pass calc=... explicitly, or call "
            "pot_functions.set_calculator(...) first."
        )

    atoms.calc = calc
    target = FrechetCellFilter(atoms) if relax_cell else atoms
    BFGS(target, logfile=logfile).run(fmax=fmax)
    return atoms

##########################################
########### END OF relax_defect() ########
##########################################


##########################################
##### START OF defect_formation_energy() #####
##########################################

#### INPUTS:
# defect_atoms    : ASE.Atoms (or a trajectory -- see _load_structures())
#                   for the defected supercell. If a trajectory is given,
#                   the LAST frame is used by default (index=-1).
# perfect_atoms   : ASE.Atoms for the perfect reference supercell -- SAME
#                   size/shape as defect_atoms, just without the defect.
#                   Typically already relaxed/optimized in your own
#                   workflow (e.g. via pot_functions.opt_cell()).
# ref_bulk_dict   : {element: energy_per_atom} chemical potential references
#                   -- exactly the dict pot_functions.get_bulk_ref_e()
#                   already produces from your bulk_strucs_dict, reused
#                   directly here.
# relax           : if True (default), relax defect_atoms before computing
#                   its energy (see relax_defect() above). perfect_atoms is
#                   NOT relaxed here regardless -- pass in an already
#                   optimized reference.
# calc, fmax, relax_cell : passed straight to relax_defect()
# type_map, index : passed straight to _load_structures() if a trajectory
#                   path is given instead of an Atoms object
# logfile         : Logger() target, default sys.stdout
#### RETURNS:
# dict with:
#   'E_formation'   : formation energy in eV
#   'E_defect'      : (relaxed) total energy of the defect supercell
#   'E_perfect'     : total energy of the perfect supercell
#   'delta_n'       : {element: change in atom count}, positive = atoms
#                     added (interstitial-like), negative = removed
#                     (vacancy-like)
#   'relaxed_atoms' : the relaxed defect Atoms object
#
#### The formula (standard supercell method):
# E_f = E[defect] - E[perfect] - sum_i( delta_n_i * mu_i )
# where mu_i = ref_bulk_dict[i] is the chemical potential (energy per atom)
# of element i, and delta_n_i is how many atoms of element i were
# added/removed to build the defect from the perfect supercell. This is
# exactly the same convention pot_functions.get_formation_energy() already
# uses for compound formation energies, just generalized to a defect
# (which changes atom COUNTS, not just structure).

def defect_formation_energy(defect_atoms, perfect_atoms, ref_bulk_dict,
                             relax=True, calc=None, fmax=0.01, relax_cell=False,
                             type_map=None, index=-1, logfile=sys.stdout):
    log = Logger(logfile)

    # _load_structures() returns a single-element list whenever `index` is
    # a plain integer (or defect_atoms is already a bare Atoms object), and
    # a multi-element list when `index` is a slice/':' -- taking [-1]
    # always gives the right frame either way ("last frame" is also the
    # only frame in the single-frame case).
    defect_frames = _load_structures(defect_atoms, type_map=type_map, index=index)
    defect = defect_frames[-1].copy()

    if relax:
        relax_defect(defect, calc=calc, fmax=fmax, relax_cell=relax_cell)
    else:
        if calc is None:
            calc = pot_functions.calc
        defect.calc = calc

    E_defect = defect.get_potential_energy()

    perfect = perfect_atoms.copy()
    if calc is None:
        calc = pot_functions.calc
    perfect.calc = calc
    E_perfect = perfect.get_potential_energy()

    counts_defect = Counter(defect.get_chemical_symbols())
    counts_perfect = Counter(perfect.get_chemical_symbols())
    elements = set(counts_defect) | set(counts_perfect)
    delta_n = {el: counts_defect.get(el, 0) - counts_perfect.get(el, 0) for el in elements}

    missing_refs = [el for el, dn in delta_n.items() if dn != 0 and el not in ref_bulk_dict]
    if missing_refs:
        raise ValueError(
            f"defect_formation_energy: no chemical potential given for "
            f"{missing_refs} in ref_bulk_dict, but the defect changes their "
            f"atom count by {[delta_n[el] for el in missing_refs]}."
        )

    mu_correction = sum(dn * ref_bulk_dict[el] for el, dn in delta_n.items() if dn != 0)
    E_formation = E_defect - E_perfect - mu_correction

    log(f"E_defect = {E_defect:.4f} eV, E_perfect = {E_perfect:.4f} eV")
    log(f"delta_n = {delta_n}")
    log(f"E_formation = {E_formation:.4f} eV")

    return {
        'E_formation': E_formation,
        'E_defect': E_defect,
        'E_perfect': E_perfect,
        'delta_n': delta_n,
        'relaxed_atoms': defect,
    }

##########################################
###### END OF defect_formation_energy() ######
##########################################




########################################################
############# automated multi-defect survey ##############
########################################################

##########################################
#### START OF survey_point_defects() #####
##########################################

#### INPUTS:
# perfect_atoms : ASE.Atoms, the perfect (already optimized) supercell to
#                 generate defects from
# ref_bulk_dict : {element: energy_per_atom}, see defect_formation_energy()
# vacancies     : list of elements to create one vacancy for each of, e.g.
#                 ['Li', 'O'] -- for each element, one representative atom
#                 (the first one found) is removed. Default: every element
#                 present in perfect_atoms.
# interstitials : list of elements to try as an interstitial at the most
#                 "open" site found by find_interstitial_sites(). Default:
#                 every element present in perfect_atoms (i.e. also try
#                 self-interstitials).
# antisites     : list of (element_a, element_b) pairs to create one
#                 antisite defect for (first atom of each found swapped).
#                 Default: every unordered pair of elements present.
# relax, calc, fmax, relax_cell : passed straight to defect_formation_energy()
# plot          : if True, produce a bar chart comparing all computed
#                 formation energies.
# logfile       : Logger() target
#### RETURNS:
# dict {label: result_dict} for every defect computed (result_dict is
# defect_formation_energy()'s return value), plus a 'fig' entry with the
# bar-chart Figure if plot=True.
#
# This is the "as automated as possible" entry point: hand it one perfect,
# already-optimized structure and it surveys the common point-defect types
# for every element present, with no further input required (though every
# knob above can be set explicitly for a targeted study instead).

def survey_point_defects(perfect_atoms, ref_bulk_dict, vacancies=None,
                          interstitials=None, antisites=None,
                          relax=True, calc=None, fmax=0.01, relax_cell=False,
                          plot=True, logfile=sys.stdout):
    log = Logger(logfile)

    elements = sorted(set(perfect_atoms.get_chemical_symbols()))
    if vacancies is None:
        vacancies = elements
    if interstitials is None:
        interstitials = elements
    if antisites is None:
        antisites = [(a, b) for i, a in enumerate(elements) for b in elements[i + 1:]]

    results = {}

    for el in vacancies:
        idx = perfect_atoms.get_chemical_symbols().index(el)
        log(f"=== vacancy: V_{el} (removing atom {idx}) ===")
        defect = make_vacancy(perfect_atoms, idx)
        results[f"V_{el}"] = defect_formation_energy(
            defect, perfect_atoms, ref_bulk_dict, relax=relax, calc=calc,
            fmax=fmax, relax_cell=relax_cell, logfile=logfile,
        )

    if interstitials:
        sites = find_interstitial_sites(perfect_atoms)
        if len(sites) == 0:
            log("No candidate interstitial sites found -- skipping interstitials.")
        else:
            best_site = sites[0]
            for el in interstitials:
                log(f"=== interstitial: {el}_i (at {np.round(best_site, 3)}) ===")
                defect = make_interstitial(perfect_atoms, el, best_site)
                results[f"{el}_i"] = defect_formation_energy(
                    defect, perfect_atoms, ref_bulk_dict, relax=relax, calc=calc,
                    fmax=fmax, relax_cell=relax_cell, logfile=logfile,
                )

    for el_a, el_b in antisites:
        symbols = perfect_atoms.get_chemical_symbols()
        try:
            idx_a = symbols.index(el_a)
            idx_b = symbols.index(el_b)
        except ValueError:
            continue
        log(f"=== antisite: {el_a}_{el_b} (atoms {idx_a}<->{idx_b}) ===")
        defect = make_antisite(perfect_atoms, idx_a, idx_b)
        results[f"{el_a}_{el_b}_antisite"] = defect_formation_energy(
            defect, perfect_atoms, ref_bulk_dict, relax=relax, calc=calc,
            fmax=fmax, relax_cell=relax_cell, logfile=logfile,
        )

    fig = None
    if plot and results:
        labels = list(results.keys())
        energies = [results[label]['E_formation'] for label in labels]

        fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(labels)), 4))
        ax.bar(labels, energies, color='tab:blue')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel("Formation energy (eV)")
        ax.set_title("Point-defect formation energies")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        fig.tight_layout()

    results['fig'] = fig
    return results

##########################################
##### END OF survey_point_defects() ######
##########################################



#################################
##### END OF point-defects.py #####
#################################
