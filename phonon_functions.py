#################################################################
# phonon_functions.py
#
# Universal phonon-spectrum calculation for any periodic (bulk) ASE.Atoms
# structure, using the finite-displacement method via phonopy, with forces
# computed by whatever ASE calculator you've set via
# pot_functions.set_calculator() (e.g. your PyACECalculator).
#
# NEW PACKAGES used here that are NOT already imported in pot_test.ipynb:
#   - phonopy   (pip install phonopy)
#   - seekpath  (pulled in automatically as a phonopy dependency; only used
#                internally by phonopy's auto_band_structure() to find the
#                correct high-symmetry q-point path for whatever space group
#                the structure has -- you never call it directly)
#################################################################

# General imports
import sys
import warnings

# Mathematical imports
import numpy as np

# Import from ase
from ase import Atoms

# Import from phonopy
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

# We need the *shared* calculator that set_calculator() in pot_functions.py
# sets, and the Logger() class already used throughout that file. We import
# the pot_functions MODULE (not `from pot_functions import calc`!) and reach
# into pot_functions.calc at call time instead -- `from pot_functions import
# calc` would copy today's value of calc (usually None) once at import time
# and never see later pot_functions.set_calculator(...) calls, since that
# only rebinds the name inside pot_functions's own namespace, not any copy
# already imported elsewhere.
import pot_functions


########################################################
############# supercell size helper ####################
########################################################

##########################################
### START OF _get_default_supercell_matrix() ###
##########################################

#### INPUTS:
# atoms      : ASE.Atoms, the primitive/unit cell to build a phonon
#              supercell for
# min_length : minimum length (in Angstrom) every supercell lattice vector
#              should have
#### RETURNS:
# a 3x3 diagonal numpy array suitable for phonopy's supercell_matrix

# Builds a simple diagonal supercell matrix that repeats each lattice
# vector just enough times to reach at least `min_length`. This keeps
# periodic images of a displaced atom far enough apart that the
# finite-displacement force constants aren't contaminated by spurious
# self-interaction across the supercell boundary. Same reasoning as
# get_kpoint_mesh()'s k_thresh in pot_functions.py, just inverted: a small
# unit cell needs MORE repeats, not fewer k-points.

def _get_default_supercell_matrix(atoms, min_length=15.0):
    cell_lengths = atoms.cell.lengths()
    repeats = np.ceil(min_length / cell_lengths).astype(int)
    repeats = np.maximum(repeats, 1)  # never shrink below the unit cell itself
    return np.diag(repeats)

##########################################
#### END OF _get_default_supercell_matrix() ####
##########################################




########################################################
################ phonon functions ######################
########################################################

##########################################
###### START OF get_phonon_spectrum() ####
##########################################

#### INPUTS:
# atoms  : ASE.Atoms -- the periodic (bulk) structure to compute a phonon
#          spectrum for. IMPORTANT: this should already be geometry- AND
#          cell-optimized (e.g. via pot_functions.opt_cell()) before being
#          passed in here -- any residual forces/stress at the input
#          geometry will show up as spurious soft/imaginary phonon modes,
#          since phonon theory assumes you're expanding the energy around
#          an actual local minimum.
# calc   : ASE calculator used to compute forces on the displaced
#          supercells. If None (default), falls back to whatever
#          calculator was last set via pot_functions.set_calculator() --
#          see the import comment above for why this is read from
#          pot_functions.calc at call time rather than imported directly.
# supercell_matrix : 3x3 (or plain diagonal) supercell matrix for the
#          finite-displacement method. If None (default), one is built
#          automatically via _get_default_supercell_matrix() so every
#          supercell lattice vector is at least min_supercell_length long.
# min_supercell_length : only used when supercell_matrix is None, see above.
# displacement : displacement distance in Angstrom used to generate the
#          finite-displacement supercells (phonopy default is 0.01 Ang).
# mesh   : q-point mesh, e.g. [20, 20, 20], for the phonon DOS. If None
#          (default), no DOS is computed and only the band structure is
#          returned/plotted.
# plot   : if True (default), produce a matplotlib figure of the band
#          structure (or band structure + DOS side-by-side if `mesh` was
#          given).
# savefig : optional file path to save the figure to (only used if
#          plot=True).
# logfile : where progress is logged, using the same Logger() convention
#          as the rest of pot_functions.py. Default = sys.stdout, pass
#          None to silence all output.
#
#### RETURNS:
# (phonon, fig)
#   phonon : the fully-populated phonopy.Phonopy object. Its band
#            structure (and DOS, if `mesh` was given) are already computed
#            and stored on it -- e.g. phonon.get_band_structure_dict(),
#            phonon.get_total_dos_dict(), or phonon.run_thermal_properties()
#            for further analysis beyond just plotting.
#   fig    : the matplotlib Figure if plot=True, else None.
#
#### What it does:
# 1) builds a phonopy Phonopy object for `atoms`, letting phonopy/spglib
#    auto-detect the primitive cell and space group (primitive_matrix=
#    'auto') -- this is what makes the function work unchanged for e.g.
#    cubic Li-metal, cubic (antifluorite) Li2O, and hexagonal Li2O2,
#    without any per-material configuration;
# 2) generates the minimal set of symmetry-inequivalent finite
#    displacements needed to build the full force-constant matrix;
# 3) computes forces on every displaced supercell with the given ASE
#    calculator;
# 4) builds the force constants from those forces;
# 5) computes the phonon band structure along the correct high-symmetry
#    q-point path for this structure's space group (auto-detected via
#    phonopy's seekpath integration -- again, no manual band path needed);
# 6) optionally computes the phonon DOS on a q-point mesh, and/or plots
#    everything.

def get_phonon_spectrum(atoms, calc=None, supercell_matrix=None,
                         min_supercell_length=15.0, displacement=0.01,
                         mesh=None, plot=True, savefig=None,
                         logfile=sys.stdout):

    log = pot_functions.Logger(logfile)

    # Fall back to pot_functions's shared global calculator if none was
    # given explicitly -- see the import comment at the top of this file
    # for why this is pot_functions.calc, not a bare `calc` imported once.
    if calc is None:
        calc = pot_functions.calc
    if calc is None:
        raise ValueError(
            "get_phonon_spectrum: no calculator given, and pot_functions.calc "
            "is None. Either pass calc=... explicitly, or call "
            "pot_functions.set_calculator(...) first."
        )

    # ---- automatic supercell sizing, if not given explicitly ----
    if supercell_matrix is None:
        supercell_matrix = _get_default_supercell_matrix(atoms, min_supercell_length)
        log(f"No supercell_matrix given, using automatic supercell (diag):\n{np.diag(supercell_matrix)}")

    # phonopy calls spglib internally at several points below (primitive-cell
    # detection in Phonopy(..., primitive_matrix='auto'), the seekpath-based
    # band path lookup in auto_band_structure(), and mesh symmetry reduction
    # in run_mesh()). On this environment's phonopy/spglib version pairing,
    # phonopy still reads SpglibDataset fields via the old dict-style
    # dataset['key'] interface, which newer spglib versions flag with a
    # DeprecationWarning on every single access -- harmless (phonopy still
    # gets the right symmetry data either way), but noisy enough to bury the
    # actual progress log below it. This is a phonopy/spglib version-skew
    # issue, not something wrong with the structure or the calculator, so
    # it's suppressed here rather than by touching installed package
    # versions. Scoped narrowly (this category+module+message combination
    # only, and only for the duration of this `with` block) so nothing else
    # in your notebook session has its warnings silenced by this.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"dict interface .* is deprecated",
            category=DeprecationWarning,
            module=r"spglib",
        )

        # ---- convert the ASE.Atoms into phonopy's own Atoms representation ----
        ph_atoms = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            cell=atoms.get_cell(),
            scaled_positions=atoms.get_scaled_positions(),
        )

        # primitive_matrix='auto': let phonopy/spglib find the primitive cell
        # from the symmetry of `atoms` itself, instead of assuming/hardcoding
        # one -- this is the key ingredient that makes this function "just work"
        # regardless of which crystal structure it's handed.
        phonon = Phonopy(ph_atoms, supercell_matrix=supercell_matrix, primitive_matrix='auto')

        # ---- generate the finite displacements phonopy needs ----
        phonon.generate_displacements(distance=displacement)
        supercells = phonon.get_supercells_with_displacements()
        log(f"Generated {len(supercells)} displaced supercell(s), computing forces...")

        # ---- compute forces on every displaced supercell with the ASE calculator ----
        force_sets = []
        for i, sc in enumerate(supercells):
            # phonopy hands back its own lightweight Atoms-like objects here;
            # convert each one to a real ASE.Atoms so the given ASE calculator
            # can be attached and used directly.
            sc_atoms = Atoms(
                symbols=sc.get_chemical_symbols(),
                scaled_positions=sc.get_scaled_positions(),
                cell=sc.get_cell(),
                pbc=True,
            )
            sc_atoms.calc = calc
            force_sets.append(sc_atoms.get_forces())
            log(f"  supercell {i + 1}/{len(supercells)} done.")

        phonon.forces = force_sets
        phonon.produce_force_constants()

        # ---- band structure along the automatically-detected high-symmetry path ----
        phonon.auto_band_structure(plot=False)

        # ---- optional DOS on a q-point mesh ----
        if mesh is not None:
            phonon.run_mesh(mesh)
            phonon.run_total_dos()

    fig = None
    if plot:
        if mesh is not None:
            fig = phonon.plot_band_structure_and_dos()
        else:
            fig = phonon.plot_band_structure()

        if savefig is not None:
            fig.savefig(savefig)
            log(f"Figure saved to {savefig}")

    log("Phonon spectrum calculation done.")

    return phonon, fig

##########################################
######## END OF get_phonon_spectrum() ####
##########################################



#################################
##### END OF phonon_functions.py #####
#################################

'''''
New packages (not in your pot_test.ipynb imports): phonopy (pip install phonopy) and its dependency seekpath, which phonopy uses internally to auto-detect the correct high-symmetry q-point path for the band structure — you never call it directly. Neither is installed in this sandbox, so I couldn't run it end-to-end here (same limitation as pymatgen/elastic/pyace earlier); the phonopy API used is stable and long-standing, but worth a quick smoke test on your side once installed.

Why phonopy rather than ASE's built-in ase.phonons.Phonons: with primitive_matrix='auto', phonopy/spglib detects the primitive cell and space group itself, so the same function handles cubic Li-metal, cubic Li2O, and hexagonal Li2O2 without you specifying a band path per material — that's the "universally applicable" part.

Usage, once you add from phonon_functions import * to your import cell:


phonon, fig = get_phonon_spectrum(bulk_strucs_dict['Li2O']['ase_atoms'])
A few things worth knowing:

It reuses whatever calculator you last set via pot_functions.set_calculator(...) — you don't need to pass one, unless you want a different calculator than your current global one.
The supercell size is auto-picked (each supercell lattice vector ≥ 15 Å by default, via min_supercell_length) so displaced atoms don't spuriously interact with their own periodic images; override with supercell_matrix= if you want control.
Pass mesh=[20,20,20] (or similar) to also get the phonon DOS plotted alongside the bands; omit it for band-structure-only.
Important: feed it an already cell+geometry-optimized structure (e.g. the output of your opt_cell()) — residual forces at the input geometry will show up as spurious soft/imaginary modes in the spectrum, since phonon theory assumes you're expanding around a true energy minimum.
Returns (phonon, fig) — phonon is the full phonopy.Phonopy object, so anything beyond plotting (thermal properties, raw band-structure/DOS data, etc.) is available on it directly via phonopy's own API if you need it later.
'''''