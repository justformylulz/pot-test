#################################################################
# stacking-fault.py
#
# Generalized stacking fault (GSF) energy curves and grain-boundary
# energies for any ASE.Atoms bulk structure, using the standard periodic
# supercell method: rigid in-plane shift (stacking fault) or an
# already-built bicrystal (grain boundary), relaxed with the ASE calculator
# you've set via pot_functions.set_calculator(), compared against a
# perfect-crystal reference.
#
# ---------------------------------------------------------------
# THIS IS A STATIC (0 K) CALCULATION, NOT AN MD ONE
# ---------------------------------------------------------------
# GSF curves and grain-boundary energies are, like point-defects.py,
# conventionally 0 K quantities computed on relaxed structures (BFGS here),
# not sampled via MD -- there's no core-method LAMMPS advice to give. That
# said:
#   - If you'd rather relax a faulted/bicrystal structure by
#     annealing-then-quenching it in LAMMPS instead of ASE's BFGS, both
#     functions below accept a trajectory file (dump/xyz) as well as a bare
#     Atoms object -- the last frame is used, and relax=False skips the
#     re-relaxation.
#   - This module assumes your structure already has the fault-plane (or
#     grain-boundary-plane) NORMAL along z -- the same z-is-special
#     convention pot_functions.py already uses elsewhere ("c has to be in
#     z-direction"). Build/orient the supercell that way first (e.g. with
#     ase.build.surface(), a manual rotation, or pymatgen's slab tools).
#   - Grain-boundary CONSTRUCTION (finding the coincidence-site-lattice,
#     matching the two grains, building the actual bicrystal) is its own
#     deep crystallographic problem and isn't reimplemented here -- use
#     pymatgen's GrainBoundaryGenerator (see build_bicrystal_gb() below,
#     which wraps it) or any other tool to build the bicrystal, then hand
#     the result to grain_boundary_energy() to get the actual energy.
#
# NEW PACKAGES used here that are NOT already imported in pot_test.ipynb:
#   - scipy.signal (automatic detection of the stable/unstable stacking
#     fault energies from the GSF curve). Same scipy package already
#     flagged for frenkel.py/point-defects.py/rdf.py.
#################################################################

# General imports
import sys

# Mathematical imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Import from ase
from ase import Atoms
from ase.io import read
from ase.optimize import BFGS
from ase.constraints import FixCartesian

# eV/Ang^2 -> J/m^2, the unit interface energies are conventionally quoted
# in in the literature (1 eV = 1.602176634e-19 J, 1 Ang^2 = 1e-20 m^2).
EV_PER_ANG2_TO_J_PER_M2 = 16.021766341

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
    frames either way.
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


def _last_frame(source, type_map=None, index=':'):
    return _load_structures(source, type_map=type_map, index=index)[-1].copy()


########################################################
########## generalized stacking fault (GSF) ##############
########################################################

##########################################
#### START OF generalized_stacking_fault() ####
##########################################

#### INPUTS:
# atoms          : ASE.Atoms (or trajectory, see _load_structures()) -- a
#                  perfect, already-optimized bulk supercell, oriented so
#                  the intended fault plane's NORMAL is along z (see module
#                  note above). Needs to be tall enough along z that the
#                  fault doesn't interact with its own periodic image
#                  (a handful of atomic layers on each side of the fault
#                  plane, same idea as a phonon supercell needing to be
#                  big enough -- see phonon_functions.py).
# slip_vector    : (2,) or (3,) Cartesian displacement vector (Ang) -- the
#                  FULL in-plane shift corresponding to fraction=1.0 (e.g.
#                  the Shockley partial or full Burgers vector of the slip
#                  system you're studying). Must be an in-plane vector (any
#                  z-component is ignored/zeroed). You supply this, since
#                  it depends on your specific crystal structure and slip
#                  system.
# fraction_range : (min, max) fraction of slip_vector to scan, default
#                  (0, 1) -- a full GSF curve from the perfect stacking
#                  back to the next perfect stacking (if slip_vector is a
#                  true lattice-translation vector, E(fraction=1) should
#                  come back close to E(fraction=0) -- see the sanity-check
#                  log message below).
# n_points       : number of fractions to sample, default 11.
# relax_normal   : if True (default), atoms are allowed to relax ALONG z
#                  (perpendicular to the fault plane) at each fixed
#                  in-plane shift, via ASE's FixCartesian constraint on
#                  every atom (mask=(1,1,0): x,y fixed, z free) -- this is
#                  the standard GSF prescription (the shift itself is
#                  fixed, but the interface is allowed to relax normal to
#                  itself).
# n_interfaces   : how many equivalent stacking faults exist in the
#                  periodic cell -- default 2. Shifting "everything above
#                  z_mid" in an otherwise fully periodic cell necessarily
#                  creates a SECOND, oppositely-shifted fault where the
#                  cell wraps around back to the bottom (the standard
#                  "axial" GSF construction) -- so the excess energy is
#                  conventionally divided by 2x the area. Set this to 1
#                  only if your structure has a free surface/vacuum on one
#                  side (so there's genuinely only one interface).
# calc, fmax     : passed to the relaxation.
# logfile        : Logger() target.
# plot           : if True (default), plot gamma(fraction) and mark the
#                  detected (meta)stable stacking fault energy and the
#                  unstable stacking fault energy.
#### RETURNS:
# dict with:
#   'fraction'        : (n_points,) array of shift fractions sampled
#   'energy'          : (n_points,) total energy at each fraction (eV)
#   'gamma'           : (n_points,) stacking fault energy at each fraction
#                        (eV/Ang^2)
#   'gamma_J_per_m2'  : same, in J/m^2
#   'gamma_stable'    : gamma (eV/Ang^2) at the first local MINIMUM after
#                       the first peak (the metastable/stable stacking
#                       fault energy), or None if no clear minimum was found
#   'gamma_stable_J_per_m2'   : same, in J/m^2 (None if gamma_stable is None)
#   'gamma_unstable'  : gamma (eV/Ang^2) at the global MAXIMUM of the curve
#                       (the unstable stacking fault energy, relevant for
#                       dislocation nucleation)
#   'gamma_unstable_J_per_m2' : same, in J/m^2
#   'area'            : in-plane cell area (Ang^2)
#   'fig'             : matplotlib Figure if plot=True, else None

def generalized_stacking_fault(atoms, slip_vector, fraction_range=(0.0, 1.0),
                                n_points=11, relax_normal=True, n_interfaces=2,
                                calc=None, fmax=0.01, type_map=None, index=-1,
                                plot=True, logfile=sys.stdout):
    log = Logger(logfile)

    if calc is None:
        calc = pot_functions.calc
    if calc is None:
        raise ValueError(
            "generalized_stacking_fault: no calculator given, and "
            "pot_functions.calc is None. Either pass calc=... explicitly, "
            "or call pot_functions.set_calculator(...) first."
        )

    reference = _last_frame(atoms, type_map=type_map, index=index)
    # z (the stacking/fault-normal direction) must be decoupled from the
    # in-plane a/b vectors -- i.e. cellpar's alpha (b-c angle) and beta
    # (a-c angle) must be 90 degrees -- but the IN-PLANE angle gamma (a-b)
    # is NOT required to be 90. This matters in practice: fcc{111}-type and
    # hcp basal-plane slabs (the most common real stacking-fault studies)
    # have a hexagonal in-plane cell with gamma = 60 or 120 degrees, not a
    # rectangular one, and are perfectly valid inputs here.
    alpha, beta = reference.cell.cellpar()[3:5]
    if not (np.isclose(alpha, 90, atol=1e-3) and np.isclose(beta, 90, atol=1e-3)):
        raise ValueError(
            "generalized_stacking_fault: the cell's third (z) lattice "
            "vector must be perpendicular to the first two (cellpar alpha "
            f"= {alpha:.3f}, beta = {beta:.3f}, both need to be ~90 "
            "degrees) -- reorient the structure so the fault-plane normal "
            "is exactly along z. The in-plane angle between a and b "
            "(gamma) does not need to be 90 -- a hexagonal in-plane cell "
            "(e.g. fcc{111} or hcp basal-plane slabs) is fine."
        )

    slip_vector = np.array(slip_vector, dtype=float)
    if slip_vector.shape == (2,):
        slip_vector = np.append(slip_vector, 0.0)
    slip_vector[2] = 0.0  # in-plane only, by construction

    area = np.linalg.norm(np.cross(reference.cell[0], reference.cell[1]))
    z_mid = reference.cell.lengths()[2] / 2.0
    upper_mask = reference.get_positions()[:, 2] > z_mid

    fractions = np.linspace(fraction_range[0], fraction_range[1], n_points)
    energies = []

    for frac in fractions:
        trial = reference.copy()
        trial.positions[upper_mask, :2] += frac * slip_vector[:2]
        trial.calc = calc

        if relax_normal:
            trial.set_constraint([FixCartesian(i, mask=(1, 1, 0)) for i in range(len(trial))])
            BFGS(trial, logfile=None).run(fmax=fmax)

        e = trial.get_potential_energy()
        energies.append(e)
        log(f"  fraction = {frac:.3f}, E = {e:.5f} eV")

    energies = np.array(energies)
    gamma = (energies - energies[0]) / (n_interfaces * area)
    gamma_SI = gamma * EV_PER_ANG2_TO_J_PER_M2

    if abs(fraction_range[1] - 1.0) < 1e-8:
        end_mismatch = abs(gamma[-1])
        if end_mismatch > 0.02 * max(abs(gamma.min()), abs(gamma.max()), 1e-6):
            log(f"WARNING: E(fraction=1) does not return close to E(fraction=0) "
                f"(gamma mismatch = {end_mismatch:.4f} eV/Ang^2). If slip_vector "
                f"is meant to be a full lattice-translation vector, double-check "
                f"it -- otherwise fraction=1 just isn't a repeat of the perfect "
                f"stacking, which is fine if that's what you intended.")

    # unlike rdf.py's first_shell_cutoff(), plain argrelextrema is fine
    # here: each point on this curve is a separate deterministic
    # relaxation, not a statistically noisy MD average, so there's no
    # bin-to-bin thermal noise for it to be fooled by.
    maxima = argrelextrema(gamma, np.greater)[0]
    minima = argrelextrema(gamma, np.less)[0]
    gamma_stable = gamma[minima[0]] if len(minima) > 0 else None
    gamma_unstable = gamma.max() if len(gamma) > 0 else None

    gamma_stable_SI = gamma_stable * EV_PER_ANG2_TO_J_PER_M2 if gamma_stable is not None else None
    gamma_unstable_SI = gamma_unstable * EV_PER_ANG2_TO_J_PER_M2 if gamma_unstable is not None else None

    log(f"gamma_stable = {gamma_stable} eV/Ang^2 ({gamma_stable_SI} J/m^2), "
        f"gamma_unstable (USF) = {gamma_unstable} eV/Ang^2 ({gamma_unstable_SI} J/m^2)")

    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(fractions, gamma_SI, marker='o', color='tab:blue')
        if gamma_stable_SI is not None:
            ax.axhline(gamma_stable_SI, color='tab:green',
                       linestyle='--', linewidth=1, label='stable SFE')
        if gamma_unstable_SI is not None:
            ax.axhline(gamma_unstable_SI, color='tab:red',
                       linestyle='--', linewidth=1, label='unstable SFE (USF)')
        ax.set_xlabel("Shift fraction of slip vector")
        ax.set_ylabel(r"$\gamma$ (mJ/m$^2$)" if gamma_SI.max() < 1 else r"$\gamma$ (J/m$^2$)")
        ax.set_title("Generalized stacking fault energy curve")
        ax.legend()
        fig.tight_layout()

    return {
        'fraction': fractions,
        'energy': energies,
        'gamma': gamma,
        'gamma_J_per_m2': gamma_SI,
        'gamma_stable': gamma_stable,
        'gamma_stable_J_per_m2': gamma_stable_SI,
        'gamma_unstable': gamma_unstable,
        'gamma_unstable_J_per_m2': gamma_unstable_SI,
        'area': area,
        'fig': fig,
    }

##########################################
##### END OF generalized_stacking_fault() ####
##########################################




########################################################
##################### grain boundaries ######################
########################################################

##########################################
###### START OF grain_boundary_energy() #####
##########################################

#### INPUTS:
# bicrystal        : ASE.Atoms (or trajectory) -- an already-built bicrystal
#                    supercell containing the grain boundary(-ies), oriented
#                    with the GB-plane normal along z (same convention as
#                    generalized_stacking_fault()). Build this with
#                    build_bicrystal_gb() below, pymatgen's
#                    GrainBoundaryGenerator directly, or any other method.
# bulk_energy_per_atom : reference energy per atom (eV) of the same
#                    material as a defect-free single crystal -- e.g.
#                    perfect_atoms.get_potential_energy() / len(perfect_atoms)
#                    for an already-optimized bulk reference.
# n_interfaces     : how many equivalent grain boundaries exist in the
#                    periodic cell -- default 2, same axial-construction
#                    reasoning as generalized_stacking_fault(). A bicrystal
#                    built by joining two grains back-to-back in a periodic
#                    cell has a GB where they meet AND a second one where
#                    the cell wraps around, unless you've deliberately
#                    built a slab with vacuum on one side (then use 1).
# gb_axes          : which two cell vectors span the GB plane, default
#                    (0, 1) i.e. the GB normal is along z -- override if
#                    your bicrystal is oriented differently.
# relax, calc, fmax : passed to the relaxation (position-only BFGS, fixed
#                    cell -- the standard choice for GB energies, same
#                    reasoning as point-defects.py's relax_defect()).
# type_map, index  : passed to _load_structures() if bicrystal is a
#                    trajectory path
# logfile          : Logger() target
#### RETURNS:
# dict with:
#   'E_total'            : relaxed total energy of the bicrystal (eV)
#   'gamma_gb'            : GB energy in eV/Ang^2
#   'gamma_gb_J_per_m2'    : GB energy in J/m^2 (the unit almost always
#                            quoted in the literature)
#   'area'                : GB-plane area (Ang^2)
#   'relaxed_atoms'        : the relaxed bicrystal Atoms object
#
#### The formula (standard supercell method, same idea as
#### pot_functions.surface_energy() -- this is that same excess-energy
#### construction, generalized from a free surface to a solid-solid
#### interface):
# gamma_GB = (E[bicrystal] - N_atoms * E_bulk_per_atom) / (n_interfaces * area)

def grain_boundary_energy(bicrystal, bulk_energy_per_atom, n_interfaces=2,
                           gb_axes=(0, 1), relax=True, calc=None, fmax=0.01,
                           type_map=None, index=-1, logfile=sys.stdout):
    log = Logger(logfile)

    if calc is None:
        calc = pot_functions.calc
    if calc is None:
        raise ValueError(
            "grain_boundary_energy: no calculator given, and "
            "pot_functions.calc is None. Either pass calc=... explicitly, "
            "or call pot_functions.set_calculator(...) first."
        )

    atoms = _last_frame(bicrystal, type_map=type_map, index=index)
    atoms.calc = calc

    if relax:
        BFGS(atoms, logfile=None).run(fmax=fmax)

    E_total = atoms.get_potential_energy()
    area = np.linalg.norm(np.cross(atoms.cell[gb_axes[0]], atoms.cell[gb_axes[1]]))

    gamma_gb = (E_total - len(atoms) * bulk_energy_per_atom) / (n_interfaces * area)
    gamma_gb_SI = gamma_gb * EV_PER_ANG2_TO_J_PER_M2

    log(f"E_total = {E_total:.4f} eV, area = {area:.3f} Ang^2, "
        f"n_interfaces = {n_interfaces}")
    log(f"gamma_GB = {gamma_gb:.5f} eV/Ang^2 = {gamma_gb_SI:.2f} mJ/m^2" if gamma_gb_SI < 1000
        else f"gamma_GB = {gamma_gb:.5f} eV/Ang^2 = {gamma_gb_SI:.2f} J/m^2")

    return {
        'E_total': E_total,
        'gamma_gb': gamma_gb,
        'gamma_gb_J_per_m2': gamma_gb_SI,
        'area': area,
        'relaxed_atoms': atoms,
    }

##########################################
####### END OF grain_boundary_energy() #####
##########################################


##########################################
####### START OF build_bicrystal_gb() ####
##########################################

#### INPUTS:
# bulk_atoms     : ASE.Atoms, a conventional bulk cell (NOT yet a
#                  supercell/bicrystal) -- the single-crystal structure to
#                  build a coincidence-site-lattice (CSL) grain boundary
#                  from.
# rotation_axis  : e.g. [0, 0, 1] -- the CSL rotation axis (Miller indices).
# rotation_angle : misorientation angle in degrees between the two grains.
# **kwargs       : forwarded to pymatgen's
#                  GrainBoundaryGenerator.gb_from_parameters() (e.g.
#                  expand_times, vacuum_thickness, plane, rm_ratio, ...) --
#                  see pymatgen's own documentation for the full parameter
#                  list, since this is a thin wrapper around that.
#### RETURNS:
# an ASE.Atoms bicrystal, ready to hand to grain_boundary_energy() above.
#
#### IMPORTANT CAVEAT:
# unlike everything else in this file, this specific convenience function
# has NOT been tested here (pymatgen isn't available in the sandbox this
# was written in). The CSL/bicrystal-construction machinery it wraps is
# pymatgen's own, well-documented GrainBoundaryGenerator -- if this errors
# out on an API mismatch, either check pymatgen's current
# pymatgen.analysis.gb.grain module directly, or just build the bicrystal
# with whatever tool you prefer and hand the resulting ASE.Atoms straight
# to grain_boundary_energy() instead -- that function has no dependency on
# how the bicrystal was made.

def build_bicrystal_gb(bulk_atoms, rotation_axis, rotation_angle, **kwargs):
    from pymatgen.analysis.gb.grain import GrainBoundaryGenerator
    from pymatgen.io.ase import AseAtomsAdaptor

    structure = AseAtomsAdaptor.get_structure(bulk_atoms)
    generator = GrainBoundaryGenerator(structure)
    gb_structure = generator.gb_from_parameters(rotation_axis, rotation_angle, **kwargs)

    return AseAtomsAdaptor.get_atoms(gb_structure)

##########################################
######## END OF build_bicrystal_gb() #####
##########################################



#################################
##### END OF stacking-fault.py #####
#################################
