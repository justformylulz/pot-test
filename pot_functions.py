#General imports
import sys
import os
import copy
from collections import Counter
import fnmatch
import nglview as ngl
from invoke import run
import time
from contextlib import redirect_stdout

#Mathematical imports
import numpy as np
from numpy import array, linspace
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt


#Import from ase
import ase
from ase import units
from ase import Atoms
from ase.io import read, write
from ase.units import Rydberg, Bohr
from ase.units import _amu
from ase.io.trajectory import Trajectory
from ase.build import bulk
from ase.build import surface
from ase.optimize import BFGS
from ase.visualize import view
from ase.filters import FrechetCellFilter
from ase.constraints import FixAtoms

from pyace import PyACECalculator

import elastic
from elastic import get_pressure, BMEOS, get_strain
from elastic import get_elementary_deformations, scan_volumes
from elastic import get_BM_EOS, get_elastic_tensor

from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core.structure import Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.phase_diagram import PDPlotter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from mp_api.client import MPRester




########################################################
################ general functions #####################
########################################################

calc = None

def set_calculator(calculator):
    global calc
    calc = calculator



def flatten(xss):
    return [x for xs in xss for x in xs]


class Logger:
    def __init__(self, logfile=sys.stdout):
        self.logfile = logfile

    def __call__(self, *args, **kwargs):
        if self.logfile is not None:
            print(*args, file=self.logfile, **kwargs)



# returs a dictionary of the atoms object it takes as input

def lattice_ident(atoms):
    bravais = atoms.cell.get_bravais_lattice()
    lattice = { 'type' : bravais.name}
    
    lattice_constants = []
    for param in bravais.parameters:
        lattice[f"{param}"]= getattr(bravais, param)

    return lattice



def angle_between_vectors(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    
    # Clip the dot product to handle floating-point inaccuracies
    #dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    dot_product = np.dot(v1_u, v2_u)
    
    radians = np.arccos(dot_product)
    return np.degrees(radians)





def substitute_atoms(atoms, elements_list):

    symbols = atoms.get_chemical_symbols()
    counts = Counter(symbols)

    # sort elements by how many atoms of each there are (ascending),
    # so A is always the least abundant element, B next, etc.
    # tie-break alphabetically so equal counts are still deterministic
    elements = sorted(counts, key=lambda el: (counts[el], el))

    if len(elements) != len(elements_list):
        raise ValueError(
            f"Expected {len(elements_list)} elements, got {dict(counts)}"
        )

    # map each original element symbol to its replacement,
    # e.g. {'O': 'O', 'Li': 'Li'} for an AB2 structure with elements_list=['O', 'Li']
    elem_map = dict(zip(elements, elements_list))

    new_symbols = [elem_map[s] for s in symbols]  # replace every atom using the map

    new_atoms = atoms.copy()
    new_atoms.set_chemical_symbols(new_symbols)
    return new_atoms






########################################################
################ surface functions #####################
########################################################


##########################################
##### START OF shift_surface_atoms()  ####
##########################################
#### INPUTS:
# reference surface
# list of indeces
# shift in z-direction in Angstroms

def shift_surface_atoms(surf, idx_list, z_len):
    shift=[]
    shift_e=[]
    for i_at in idx_list:
        dmp = surf.copy()
        dmp.positions[i_at,2] = dmp.positions[i_at,2] + z_len
        #c = FixAtoms(indices=[atom.index for atom in dmp if atom.index != i_at])
        #dmp.set_constraint(c)
        dmp.calc = calc
        #opt = BFGS(dmp)
        #opt.run(fmax=0.01)
        #del dmp.constraints
        shift.append(dmp.copy())
        shift_e.append(dmp.get_potential_energy())
        
    best_shift = shift[ shift_e.index(min(shift_e)) ]
    best_shift.calc = calc
    return best_shift
        

def get_composition(atoms):
    #Return composition as a Counter, e.g.
    #Li2O -> {'Li': 2, 'O': 1}
    
    return Counter(atoms.get_chemical_symbols())


def get_layers(atoms, axis=2, tolerance=0.1):
    """
    Group atoms into layers according to their Cartesian
    coordinate along `axis`.

    Parameters
    ----------
    atoms : ASE Atoms
    axis : int
        Cartesian axis (0=x, 1=y, 2=z).
    tolerance : float
        Maximum distance between atoms to be considered
        part of the same layer.

    Returns
    -------
    layers : list[list[int]]
        Atomic indices grouped into layers, ordered
        from bottom to top.
    """

    positions = atoms.positions[:, axis]

    sorted_indices = np.argsort(positions)

    layers = []
    current_layer = [sorted_indices[0]]
    current_z = positions[sorted_indices[0]]

    for i in sorted_indices[1:]:
        if abs(positions[i] - current_z) <= tolerance:
            current_layer.append(i)
        else:
            layers.append(current_layer)
            current_layer = [i]
            current_z = positions[i]

    layers.append(current_layer)

    return layers


def layer_composition(atoms, layer):
    """
    Return the elemental composition of one atomic layer.
    """
    symbols = atoms.get_chemical_symbols()

    return Counter(symbols[i] for i in layer)


def surface_energy(surf, ref_bulk):
    s_e = surf.get_potential_energy()
    s_Area = np.linalg.norm( np.cross(surf.cell[0], surf.cell[1]) )
    b_e = ref_bulk.get_potential_energy()
    n_units = len(surf) / len(ref_bulk)
    gamma = (s_e - (n_units * b_e)) / (2*s_Area)
    return gamma



def layer_coordinates(surf, layer):
    layer_coords = []
    
    for i in layer:
            layer_coords.append([ i,
                                surf.get_chemical_symbols()[i],
                                surf.get_positions()[i][0],
                                surf.get_positions()[i][1]])
    #layer_coords.sort(key=lambda x: x[3])
    #layer_coords.sort(key=lambda x: x[2])
            
    return sorted( layer_coords, key=lambda x: (x[2], x[3]))
    #return layer_coords

def compare_layers(surf,layer1, layer2):
    layer_1 = layer_coordinates(surf,layer1)
    layer_2 = layer_coordinates(surf,layer2)
    if len(layer_1) == len(layer_2):
        return all(
                x[1] == y[1] 
                and abs(x[2] - y[2]) < 1e-4
                and abs(x[3] - y[3]) < 1e-4
                for x, y in zip(layer_1, layer_2))
    else:
        return False

def find_matching_layers(surf, layers):
    closest_matching_layers=[]
    for i in range(0, len(layers)):
        for j in range(i+1, len(layers)-1):
            if i-1 < 0 or j+1 > len(layers)-1:
                continue
            
            if (compare_layers(surf, layers[i-1], layers[j-1]) == True
                and compare_layers(surf, layers[i], layers[j]) == True
                and compare_layers(surf, layers[i+1], layers[j+1]) == True):
                    pairs = [ [x[0], y[0]] for x, y in zip(layer_coordinates(surf, layers[i]), layer_coordinates(surf, layers[j]) ) ]
                    closest_matching_layers.append([ abs(j-i), np.mean([abs(surf.positions[k, 2] - surf.positions[l, 2]) for k, l in pairs]) ] )

    return sorted(closest_matching_layers, key=lambda x: x[1])[0]





def find_stable_surface(atoms, h, k, l, nrep, periodic=True, opt=True, logfile=sys.stdout):
    ######################
    ### hexagonal {h,k,i,l} miller-bravais indices have to 
    ### transformed to (h,k,l) miller indices
    ### for pymat SlabGenerator to read them
    ### via i= -(h+k) 
    ### 
    ### ex: 
    ### {0001} -->	(0,0,1)
    ### {1̄(-1)00} -->	(1,-1,0)
    ### {11̄(-2)0} -->	(1,1,0)
    ######################
    log = Logger(logfile) 
    lat_type = lattice_ident(atoms)

    #check if cell is cubic or not and generate desired surface
    
    if lat_type['type'] == 'CUB':
        surf = surface(atoms, (h, k, l), nrep, periodic=periodic)
        z = surf.cell.lengths()[2]
        layers = get_layers(surf)
    else:
        structure = Structure.from_ase_atoms(atoms)
        sga = SpacegroupAnalyzer(structure)
        conventional = sga.get_conventional_standard_structure() # convert structure to conventional cell
        slabgen = SlabGenerator(
        conventional,
        miller_index=(h, k, l),
        min_slab_size=10,
        min_vacuum_size=1,
        center_slab=False,
        in_unit_planes = True,
        max_normal_search = max([h,k,l])
        )

        slabs = slabgen.get_slabs()
        slab = slabs[0]
        surf = slab.to_ase_atoms()
        x = surf.cell.lengths()[0]
        y = surf.cell.lengths()[1]
        if x < y:
            x_rep = round(y/x)
            surf = surf.repeat((x_rep,1,1))
        
        if x > y:
            y_rep = round(x/y)
            surf = surf.repeat((1,y_rep,1))
        else:
            surf = surf.repeat((2,2,1))

        layers = get_layers(surf)
        match = find_matching_layers(surf, layers)
        d_between_layers =  match[1] / match[0] # z = N_layers * distance between layers
        z_max = match[1] * nrep
        mask = surf.positions[:,2] > z_max
        del surf[np.where(mask)[0]]
        layers=get_layers(surf)
        z = len(layers) * d_between_layers
        surf.center(vacuum=0, axis=2)


    
    surf.calc=calc
    surfs_atoms=[]
    energies = []
    surfaces_dmp={}

    

    vac = 15
    cell = surf.cell
    cell[2] = cell[2] / np.linalg.norm(cell[2]) * (z+vac)
    surf.set_cell(cell)
    bot_comp_init =  layer_composition(surf,layers[0])
    top_comp_init =  layer_composition(surf,layers[-1])

    while True:
        surf = shift_surface_atoms(surf, layers[0], z)
        surf.calc=calc
        layers = get_layers(surf)
        bot_comp =  layer_composition(surf,layers[0])
        top_comp =  layer_composition(surf,layers[-1])
        if bot_comp == top_comp:
            surfs_atoms.append(surf.copy())
            energies.append(surf.get_potential_energy())
        if  bot_comp==bot_comp_init and top_comp==top_comp_init:
            break

    best_surf = surfs_atoms[ energies.index(min(energies)) ].copy()
    best_surf.calc=calc
    if opt == True:
        log(f"Low energy {h,k,l} surface found, starting geo-opt.")
        opt = BFGS(best_surf, logfile=None)
        opt.run(fmax=0.005)

    log(f" {h,k,l} done !")
    best_surf_dict = { 'ase_atoms' : best_surf,
                       'surface_energy' : surface_energy(best_surf, atoms) * 1000} # in meV/A**2
    
    return best_surf_dict







########################################################
########### formation energy functions #################
########################################################

##########################################
####### START OF get_bulk_ref_e()  #######
##########################################

#### INPUTS:
# dict of (optimized) bulk structures.
# this dict has to also include 
# the reference structure of gas phase molecules
# Example: Li-O systems -> O2 molecule has to be included
#### What it does:
# takes the highest energy/atom values for each element
# of your passed dictionary as reference energy
# 
# returns a dictionary that has the Element Symbol and the corresponding energy/atom as entries
# dict_dmp = {
############## 'elem1' : energy per atom
############## 'elem2' : energy per atom
############## ....
############## 'elemN' : energy per atom
############## }

def get_bulk_ref_e(bulk_dict):

    dict_dmp = {}

    energy_list = [
    [name, bulk_dict[name]['ase_atoms'].get_potential_energy() / len(bulk_dict[name]['ase_atoms'])  ]
    for name in bulk_dict
    ]

    energies_list = sorted(energy_list, key=lambda x: x[1], reverse=True)

    for i in energies_list:
        element = bulk_dict[i[0]]['ase_atoms'].get_chemical_symbols()[0]
        
        if element in dict_dmp:
            continue
        else:
            dict_dmp[element]= bulk_dict[i[0]]['ase_atoms'].get_potential_energy() / len(bulk_dict[name]['ase_atoms'])


    return dict_dmp
##########################################
####### END OF get_bulk_ref_e()  #########
##########################################


##########################################
#### START OF get_formation_energy()  ####
##########################################

#### INPUTS:
# ASE.Atoms (optimized structure)
# dictionary of reference elemental bulk energies
#### RETURNS:
# formation energy

def get_formation_energy(atoms, ref_bulk_dict,  logfile=sys.stdout):
    log = Logger(logfile)

    if 'n_units' in atoms.info:
        n_units = atoms.info['n_units']
    else:
        n_units = len(atoms)
        log("No number of units given, taking total number of atoms instead!")
    atoms.calc=calc
    en_unit = atoms.get_potential_energy() / n_units
    counts = Counter(atoms.get_chemical_symbols())
    ref_e = 0
    for element in counts:
        ref_e = ref_e + ( (counts[element]/n_units) * ref_bulk_dict[element] )
    
    Ef = en_unit - ref_e
    
    return Ef

##########################################
#### END OF get_formation_energy()  ######
##########################################






########################################################
################ cell-opt functions ####################
########################################################



##########################################
######## START OF opt_cell()  ############
##########################################

#### INPUTS:
# ASE.Atoms
# e_thr = 1e-5 : energy convergence threshold
# alat_thr = 1e-4 : a | a/c convergence threshold
# logfile = sys.stdout : where to print output
#### RETURNS:
# geometry and cell optimized ASE.Atoms

def opt_cell(atoms, e_thr = 1e-5, alat_thr = 1e-4, logfile=sys.stdout):

    log = Logger(logfile)

    dmp_atoms=atoms.copy()
    dmp_atoms.calc = calc
    a = dmp_atoms.get_cell()[0][0]
    c = dmp_atoms.get_cell()[2][2]

    # --------------------------------------------------------------------
    # Identify the crystal system via ASE's Bravais-lattice finder, reusing
    # the lattice_ident() helper defined earlier in this file. This used to
    # be done via "if a == c" / "if a != c", which is fragile for two reasons:
    #   1) it is a float equality check, so it can be thrown off by tiny
    #      numerical noise in the cell vectors;
    #   2) treating "a != c" as automatically hexagonal silently mis-handles
    #      every other non-cubic lattice (tetragonal, orthorhombic,
    #      monoclinic, triclinic, trigonal) as if it were hexagonal, giving
    #      a wrong result with no warning at all.
    # lattice_ident() returns the Bravais lattice name (e.g. 'CUB', 'HEX',
    # 'TET', ...) as determined by ase's cell.get_bravais_lattice(), which is
    # the same mechanism used for lattice_ident() elsewhere in this module.
    # --------------------------------------------------------------------
    lattice = lattice_ident(dmp_atoms)
    lattice_type = lattice['type']

    if lattice_type == 'CUB': #cubic
        log("====================================")
        log(f"d(E) \t  d(a) ")
        log("====================================")
        while True:
            a_0 = a
            e_0 = dmp_atoms.get_potential_energy()

            a = get_opt_alat(dmp_atoms)

            a_1 = a
            a_diff = a_1-a_0

            dmp_atoms.set_cell([a, a, a], scale_atoms=True)
            BFGS(dmp_atoms, logfile=None).run(fmax=0.0001)


            e_1 = dmp_atoms.get_potential_energy()
            e_diff = e_1-e_0

            log(f"{e_diff:.6f}  \t {(a_diff):.6f}")

            if abs(e_diff) < e_thr and abs(a_diff) < alat_thr:
                break

        BFGS(dmp_atoms, logfile=None).run(fmax=0.0001)
        log("Convergence has been achieved!")
        log("Returning optimized Atoms object.")


        return dmp_atoms


    elif lattice_type == 'HEX': #hexagonal
        log("====================================")
        log(f"d(E) \t  d(c/a)")
        log("====================================")
        while True:
            ca_0 = c/a
            e_0 = dmp_atoms.get_potential_energy()

            a,c = get_opt_alat(dmp_atoms)

            ca_1 = c/a
            ca_diff = ca_1-ca_0

            dmp_atoms.set_cell([a, a, c, 90, 90, 120], scale_atoms=True)
            BFGS(dmp_atoms, logfile=None).run(fmax=0.0001)


            e_1 = dmp_atoms.get_potential_energy()
            e_diff = e_1-e_0

            log(f"{e_diff:.6f}  \t {(ca_diff):.6f}")

            if abs(e_diff) < e_thr and abs(ca_diff) < alat_thr:
                break

        BFGS(dmp_atoms, logfile=None).run(fmax=0.00001)
        log("Convergence has been achieved!")
        log("Returning optimized Atoms object.")

        return dmp_atoms

    else:
        # ------------------------------------------------------------------
        # Generalized outer convergence loop for every other supported
        # lattice type (TET, ORC, RHL, MCL, TRI). Same idea as the CUB/HEX
        # loops above -- alternate a full parameter scan with a BFGS
        # position relaxation at the new cell, until both the energy and
        # every free lattice parameter stop changing -- just generalized to
        # however many free parameters the lattice has.
        #
        # free_params/values come from the same `lattice` dict used above
        # to pick this branch (lattice_ident()'s output), so there is no
        # need to call get_bravais_lattice() again here.
        #
        # IMPORTANT: unlike the CUB/HEX branches, this loop deliberately
        # does NOT call get_opt_alat(dmp_atoms) to do the scan, even though
        # that's what CUB/HEX do above. The reason: get_opt_alat() always
        # re-identifies the lattice type itself from scratch on whatever
        # Atoms object it is given. As this cell relaxes it can numerically
        # collapse towards a higher-symmetry lattice (e.g. an ORC cell
        # whose a,b,c happen to converge towards each other is eventually
        # reclassified by ASE as TET or even CUB, which have FEWER free
        # parameters). If get_opt_alat() were called fresh every iteration,
        # it could suddenly return a dict missing e.g. 'b', which would
        # then KeyError in _lattice_cellpar_from_values() below. Instead,
        # lattice_type/free_params are pinned ONCE here (same as CUB/HEX
        # implicitly pin their branch once) and _scan_and_fit_lattice_params()
        # is called directly with that fixed type/parameter set every
        # iteration, so the parameterization used cannot change mid-loop.
        # ------------------------------------------------------------------
        free_params = [key for key in lattice if key != 'type']
        values = {p: lattice[p] for p in free_params}

        log("====================================")
        log(f"d(E) \t  max d(param), over {free_params}")
        log("====================================")
        while True:
            values_0 = dict(values)
            e_0 = dmp_atoms.get_potential_energy()

            scan_atoms = dmp_atoms.copy()
            scan_atoms.calc = calc
            values = _scan_and_fit_lattice_params(scan_atoms, lattice_type, dict(values), free_params)

            new_cellpar = _lattice_cellpar_from_values(lattice_type, values)
            dmp_atoms.set_cell(new_cellpar, scale_atoms=True)
            BFGS(dmp_atoms, logfile=None).run(fmax=0.0001)

            e_1 = dmp_atoms.get_potential_energy()
            e_diff = e_1-e_0
            # Largest change of any one free parameter between this
            # iteration and the last, analogous to a_diff/ca_diff above but
            # generalized to an arbitrary number of parameters.
            param_diff = max(abs(values[p] - values_0[p]) for p in free_params)

            log(f"{e_diff:.6f}  \t {param_diff:.6f}")

            if abs(e_diff) < e_thr and param_diff < alat_thr:
                break

        BFGS(dmp_atoms, logfile=None).run(fmax=0.0001)
        log("Convergence has been achieved!")
        log("Returning optimized Atoms object.")

        return dmp_atoms

##########################################
########## END OF opt_cell()  ############
##########################################




##########################################
###### START OF _lattice_cellpar() #######
##########################################

# Small helper used only by the generalized (non-CUB/non-HEX) branch of
# get_opt_alat()/opt_cell() below.
#
# CUB and HEX are handled by hand above by building a 6-value cellpar list
# directly (e.g. [a, a, a, 90, 90, 90] or [a, a, c, 90, 90, 120]) -- this
# helper just does the same thing for the remaining lattice types that have
# a simple, orthogonal-ish conventional cell:
#   TET (a, c)                 -> [a, a, c, 90, 90, 90]
#   ORC (a, b, c)               -> [a, b, c, 90, 90, 90]
#   RHL (a, alpha)               -> [a, a, a, alpha, alpha, alpha]
#   MCL (a, b, c, alpha)         -> [a, b, c, alpha, 90, 90]
#   TRI (a, b, c, alpha,beta,gamma) -> [a, b, c, alpha, beta, gamma]  (nothing dependent)
#
# NOTE on scope: this intentionally does NOT cover the *centered* Bravais
# types (FCC, BCC, BCT, ORCF, ORCI, ORCC, MCLC). Those are defined by their
# PRIMITIVE cell, which is a skewed (non-orthogonal-box) shape -- e.g. the
# primitive FCC cell has 60-degree angles between its three edges, all of
# length a/sqrt(2), not a simple diagonal [a, a, a] box. In practice this is
# not a limitation for this codebase: get_bravais_lattice() only reports
# 'FCC'/'BCC'/etc. when the Atoms object's cell IS that skewed primitive
# cell. A conventional cell (e.g. bulk(..., cubic=True), which is what the
# rest of this module assumes -- see "Note: c has to be in z-direction"
# below, and the surface-building functions further up in this file) is
# geometrically just a simple cubic/tetragonal/orthorhombic box regardless
# of how many basis atoms it contains, so it is identified as CUB/TET/ORC,
# not FCC/BCT/ORCx. If you ever see one of the centered types raised as
# "not supported" below, the fix is to convert to a conventional cell first
# (e.g. via ase.build.bulk(..., cubic=True)), not to extend this helper.
#
# We deliberately build the cellpar list by hand (via ase.geometry's
# cellpar convention, same as CUB/HEX above) instead of instantiating
# ase.lattice's BravaisLattice subclasses (e.g. TRI(**values).tocell()):
# those classes reject "unconventional" angle combinations while scanning
# (raises ase.lattice.UnconventionalLattice), which would crash the scan
# the moment a trial angle wanders outside their strict convention. Plain
# cellpar values have no such restriction.

def _lattice_cellpar_from_values(lattice_type, values):
    if lattice_type == 'TET':
        return [values['a'], values['a'], values['c'], 90, 90, 90]
    elif lattice_type == 'ORC':
        return [values['a'], values['b'], values['c'], 90, 90, 90]
    elif lattice_type == 'RHL':
        return [values['a'], values['a'], values['a'],
                values['alpha'], values['alpha'], values['alpha']]
    elif lattice_type == 'MCL':
        return [values['a'], values['b'], values['c'], values['alpha'], 90, 90]
    elif lattice_type == 'TRI':
        return [values['a'], values['b'], values['c'],
                values['alpha'], values['beta'], values['gamma']]
    else:
        # Centered/primitive lattice types (FCC, BCC, BCT, ORCF, ORCI, ORCC,
        # MCLC) are out of scope -- see the long comment above.
        raise NotImplementedError(
            f"_lattice_cellpar_from_values: lattice type '{lattice_type}' is a "
            f"centered/primitive Bravais lattice, whose primitive cell is not "
            f"a simple orthogonal box. This helper (and therefore "
            f"get_opt_alat()/opt_cell()) only supports conventional cells. "
            f"Convert the structure to a conventional cell first, e.g. via "
            f"ase.build.bulk(..., cubic=True)."
        )

##########################################
######  END OF _lattice_cellpar()  #######
##########################################



##########################################
### START OF _scan_and_fit_lattice_params() ##
##########################################

# Shared by get_opt_alat()'s generalized branch and opt_cell()'s
# generalized outer loop. Scans each parameter in `free_params` in turn --
# +/-2.5% in 0.25% steps around its current value in `values`, exactly like
# the hardcoded 'a'/'c' loops in the CUB/HEX branches above -- relaxes
# atomic positions with BFGS at each trial cell, and fits a cubic through
# the resulting E(parameter) curve via fit_alat() to update that parameter
# to its optimum before moving on to the next one (sequential, not joint,
# optimization -- same as the existing HEX a-then-c logic).
#
# `lattice_type` and `free_params` are always passed in explicitly by the
# caller rather than being re-derived here via lattice_ident(): if this
# function looked the lattice type up itself from `atoms` on every call,
# a cell that numerically relaxes towards higher symmetry (e.g. ORC ->
# TET -> CUB as a,b,c converge together) would silently be reclassified
# mid-optimization, and `values` would suddenly have the wrong set of keys
# for whatever the caller expects. Pinning the type/parameters once, in
# the caller, avoids that.

def _scan_and_fit_lattice_params(dmp_atoms, lattice_type, values, free_params):
    for p in free_params:
        alat_list = []
        e_list = []
        p0 = values[p]

        for i in range(-10, 10, 1): #scan, -2.5% - +2.5% in 0.25% steps
            j = (i/4)/100
            p_new = p0 + p0*j

            trial_values = dict(values)
            trial_values[p] = p_new
            trial_cellpar = _lattice_cellpar_from_values(lattice_type, trial_values)

            dmp_atoms.set_cell(trial_cellpar, scale_atoms=True)
            alat_list.append(p_new)
            BFGS(dmp_atoms, logfile=None).run(fmax=0.0001)
            e_list.append(dmp_atoms.get_potential_energy())

        # Update this parameter to its fitted optimum before scanning the
        # next one, same as a_opt is fixed before the HEX branch above
        # goes on to scan c.
        values[p] = fit_alat(alat_list, e_list)

    return values

##########################################
#### END OF _scan_and_fit_lattice_params() ###
##########################################



##########################################
######## START OF get_opt_alat() #########
##########################################

#### INPUT:
# Atoms file
# Note: c has to be in z-direction
#### RETURNS:
# opt cell constants
# a for cubic
# a,c for hexagonal
# a dict of {param: value} for every other supported lattice type
# (TET, ORC, RHL, MCL, TRI) -- see _lattice_cellpar_from_values() above
# for exactly which lattice types are (and are not) supported.

def get_opt_alat(atoms,logfile=sys.stdout):
    log = Logger(logfile)

    alat_list=[]
    e_list=[]
    dmp_atoms=atoms.copy()
    dmp_atoms.calc = calc
    a = dmp_atoms.get_cell()[0][0]
    c = dmp_atoms.get_cell()[2][2]
    ca_ratio = c/a

    # --------------------------------------------------------------------
    # Same reasoning as in opt_cell(): identify the lattice type via
    # lattice_ident() (built on ase's get_bravais_lattice()) instead of the
    # old "a == c" / "a != c" float comparison. That old check couldn't tell
    # a hexagonal cell apart from e.g. a tetragonal or orthorhombic one --
    # it would just run the hexagonal a/c scan on any non-cubic cell, giving
    # a result that looks plausible but is not actually meaningful for that
    # lattice type.
    # --------------------------------------------------------------------
    lattice = lattice_ident(dmp_atoms)
    lattice_type = lattice['type']

    if lattice_type == 'CUB': #cubic
        for i in range(-14, 15, 1): #volume opt, -2.3% - +2.3% in 0.25% steps
            j=(i/4)/100
            a_new = a+a*j
            dmp_atoms.set_cell([a_new, a_new, a_new], scale_atoms=True)
            alat_list.append(a_new)
            BFGS(dmp_atoms, logfile=None).run(fmax=0.0001)

            e_list.append(dmp_atoms.get_potential_energy())

        a_opt = fit_alat(alat_list, e_list)
        return a_opt

    elif lattice_type == 'HEX': #hexagonal
        for i in range(-15, 15, 1): #volume opt, -3.5% - +3.5% in 0.25% steps
            j=(i/4)/100
            a_new = a+a*j
            dmp_atoms.set_cell([a_new, a_new, ca_ratio*a_new, 90, 90, 120], scale_atoms=True)


            alat_list.append(a_new)
            BFGS(dmp_atoms, logfile=None).run(fmax=0.0001)
            e_list.append(dmp_atoms.get_potential_energy())

        a_opt = fit_alat(alat_list, e_list)
        alat_list=[]
        e_list=[]

        for i in range(-14, 15, 1): #c opt, -3.5% - +3.5% in 0.25% steps
            j=(i/4)/100
            c_dmp = ca_ratio*a_opt
            c_new = c_dmp + c_dmp*j
            dmp_atoms.set_cell([a_opt, a_opt, c_new, 90, 90, 120], scale_atoms=True)

            alat_list.append(c_new)
            BFGS(dmp_atoms, logfile=None).run(fmax=0.0001)
            e_list.append(dmp_atoms.get_potential_energy())

        c_opt = fit_alat(alat_list, e_list)
        return a_opt, c_opt

    else:
        # ------------------------------------------------------------------
        # Generalized branch for every other supported lattice type (TET,
        # ORC, RHL, MCL, TRI -- centered/primitive types are rejected by
        # _lattice_cellpar_from_values(), see its docstring above for why).
        #
        # The idea is exactly the same as the CUB/HEX branches above --
        # scan one lattice parameter over +/-2.5% in 0.25% steps, relax the
        # atomic positions at each step, and fit a cubic through the
        # resulting E(parameter) curve via fit_alat() to find the minimum --
        # just generalized to however many independent parameters the
        # lattice actually has, instead of hardcoding 'a' (and 'c').
        #
        # lattice_ident() (called above) already used
        # bravais.parameters/getattr() to fill `lattice` with exactly the
        # free parameter names and their current values for this lattice
        # type (e.g. {'type': 'ORC', 'a':.., 'b':.., 'c':..} or
        # {'type': 'RHL', 'a':.., 'alpha':..}), so we just reuse that
        # instead of asking ase for the Bravais lattice a second time.
        #
        # Just like the HEX branch does for a and c, the parameters are
        # optimized SEQUENTIALLY, one at a time (holding all the others at
        # their latest known value), not jointly/simultaneously -- any
        # coupling left between them is ironed out across opt_cell()'s
        # outer while-loop iterations, exactly as it already is for HEX.
        #
        # The actual scan-and-fit loop lives in _scan_and_fit_lattice_params()
        # (defined just above this function) so that opt_cell()'s outer
        # loop can reuse the exact same logic without going through this
        # function's own lattice-type re-identification -- see the comment
        # in opt_cell()'s "else" branch for why that matters.
        # ------------------------------------------------------------------
        free_params = [key for key in lattice if key != 'type']
        values = {p: lattice[p] for p in free_params}

        return _scan_and_fit_lattice_params(dmp_atoms, lattice_type, values, free_params)

##########################################
######## END OF get_opt_alat() ###########
##########################################



##########################################
########### START OF fit_alat() ##########
##########################################

# returns fitted alat in the unit of the input
# takes alat_list and energy_corr_list as input

def fit_alat(alat_list, energy_corr_list):


    fit_coef = np.polyfit(alat_list, energy_corr_list, 3) # f = coef0*x**3 + coef1*x**2 + coef2*x +coef3
    fit = np.poly1d(fit_coef)

    df = np.polyder(fit)    # f' -> quadratic: f' = 3*coef0x**2 + 2*coef1*x + coef2

    a_df = 3*fit_coef[0]
    b_df = 2*fit_coef[1]
    c_df = fit_coef[2]

    # ------------------------------------------------------------------
    # Guard 1: the discriminant of f'(x) = 0 (the "Mitternachtsformel" /
    # quadratic formula) can be negative. That means the parabola f'(x)
    # never crosses zero, i.e. the cubic fit of E(a) has NO real extrema at
    # all -- it's monotonic across the whole scanned window. This happens
    # when the scanned lattice-parameter range is too narrow/wide, or the
    # energies are too noisy for the cubic fit to pick up any curvature.
    # Previously np.sqrt() of a negative number silently produced NaN here,
    # which then propagated into a_opt/c_opt and eventually into
    # dmp_atoms.set_cell(NaN, ...) with no error raised anywhere.
    # ------------------------------------------------------------------
    discriminant = b_df**2 - 4*a_df*c_df
    if discriminant < 0:
        raise ValueError(
            f"fit_alat: cubic fit of E(a) has no real extrema "
            f"(discriminant = {discriminant:.6g} < 0), so no lattice "
            f"constant could be extracted. This usually means the scanned "
            f"window ({min(alat_list):.6f} to {max(alat_list):.6f}) needs to "
            f"be widened/narrowed, or the energies are too noisy. "
            f"Scanned values: {alat_list}"
        )

    disc_df = np.sqrt(discriminant)

    # x_1 is analytically guaranteed to be the MINIMUM of the cubic fit
    # (not the maximum), because plugging it back into f''(x) = 2*a_df*x +
    # b_df always gives exactly +disc_df >= 0, regardless of the sign of
    # a_df. x_2 is therefore always the maximum. So returning x_1 below is
    # deliberate, not arbitrary.
    x_1 = (-b_df + disc_df)/(2*a_df) # extremum 1 of fit (the minimum)
    y_1 = fit(x_1)
    x_2 = (-b_df - disc_df)/(2*a_df) # extremum 2 of fit (the maximum)
    y_2 = fit(x_2)

    # ------------------------------------------------------------------
    # Guard 2: if the fitted minimum falls outside the range of lattice
    # parameters that were actually scanned, the cubic fit is being
    # extrapolated into a region with no supporting data. A degree-3
    # polynomial can diverge quickly just outside its fitted range, so such
    # an "extrapolated minimum" cannot be trusted -- the real minimum is
    # most likely just outside the scanned window, and the fix is to widen
    # the scan range in get_opt_alat(), not to accept this value.
    # ------------------------------------------------------------------
    a_min, a_max = min(alat_list), max(alat_list)
    if not (a_min <= x_1 <= a_max):
        raise ValueError(
            f"fit_alat: fitted minimum at x = {x_1:.6f} lies outside the "
            f"scanned range [{a_min:.6f}, {a_max:.6f}]. Refusing to return "
            f"an extrapolated value -- widen the scan range in "
            f"get_opt_alat() so the true minimum actually falls inside the "
            f"sampled window."
        )

    return x_1

    #ddf = np.polyder(df)  # f'' -> linear:   f''= 6*coef0*x + 2*coef1
    #ddfit = np.poly1d(ddf)

    #bmod_1 = ddfit(x_1)*0.5  # bmod from Bernds script, 0.5*(d^2 E/ d a^2 |_a0)
    #bmod_2 = ddfit(x_2)*0.5

    #B_1 = (2*bmod_1)/(9*x_1)*1.47108*10**4 # B in GPa
    #B_2 = (2*bmod_2)/(9*x_2)*1.47108*10**4
    
    
    #print(f"Extrema 1: x = {x_1}, y = {y_1}, bmod = {bmod_1}, B = {B_1}")
    #print(f"Extrema 2: x = {x_2}, y = {y_2}, bmod = {bmod_2}, B = {B_2}")

    #print(f"Lattice constant = {x_1*0.529177} Angstrom")

##########################################
########### END OF fit_alat() ############
##########################################




########################################################
############## elast const functions ###################
########################################################


        
##########################################
###### START OF calc_elast_const() #######
##########################################
 
# Returns all the elastic constants as a dictionary
# Takes the (optimized) cell as Atoms object input
# optional arguments: dl, dh, ds
# d is the change in cell parameters in percent
# if (dh-dl)/ds > 1, returns a dictionary entry for each d value
# dl, dh and ds are the minimum and maximum change and ds the stepsize
########################################
# Notes: 
###
### For Cubic:
# C_{11}, C_{12}, C_{44} returned from elastic tensor
# B is calculated from the BM-EoS 
# G = (c11_c12 + 3*c44) / 5
# E = 9*B*G / (3*B + G)
# v = 0.5*(3*B - 2*G)/(3*B + G)
### For hexagonal:
# C_{11}, C_{33}, C_{12}, C_{13}, C_{44} for hexagonal
# To-Do: check if BM-EoS also works for hexagonal
# To-Do: find out how G,E,v are calculated for hexagonal
#
### other crystals not implemented, just use get_elastic_tensor() from elastic 
# 

def calc_elast_const(atoms, dl=1.00, dh=1.00, ds=1.00):
    atoms.calc = calc
    vol_systems = scan_volumes(atoms, 0.85, 1.15, 20, scale_volumes=True)
    get_BM_EOS(atoms, vol_systems)


    c11_list = []
    c12_list = []
    c13_list = []
    c33_list = []
    c44_list = []
    c11_c12_list = []
    B_list = []
    G_list = []
    E_list = []
    v_list = []
    i_list = []
    steps=(dh-dl)/ds

#dyn = BFGS(atoms, logfile=None)
#dyn.run(fmax=0.000001)

    for j in range(int(steps+1)):
        i=dl+j*ds
# Calculate tensor and convert to GPa
        systems = get_elementary_deformations(atoms, n=10, d=i)
    #for sys in systems:
    #    dyn = BFGS(sys)
    #    dyn.run(fmax=0.000001)
#print(len(systems))
        cij_tmp, bij = get_elastic_tensor(atoms, systems=systems)
    
# Elastic only gives B, c11, c12, c44
        cij = cij_tmp  / units.GPa 
        B = atoms.bm_eos[1]  / units.GPa
        
# C_{11}, C_{33}, C_{12}, C_{13}, C_{44} for hexagonal
# C_{11}, C_{12}, C_{44}                 for cubic
        
        if len(cij) == 5:
            c11 = cij[0]
            c33 = cij[1]
            c12 = cij[2]
            c13 = cij[3]
            c44 = cij[4]
            
            c11_c12 = c11 - c12
            G = 'xxx'
            E = 'xxx'
            v = 'xxx'

        elif len(cij) == 3:
            c11 = cij[0]
            c12 = cij[1]
            c44 = cij[2]
            c33 = 0
            c13 = 0
            
            c11_c12 = c11 - c12
            G = (c11_c12 + 3*c44) / 5
            E = 9*B*G / (3*B + G)
            v = 0.5*(3*B - 2*G)/(3*B + G)
            
        else:
            c11 = 'xxx'
            c33 = 'xxx'
            c12 = 'xxx'
            c13 = 'xxx'
            c44 = 'xxx'
            
            c11_c12 = 'xxx'
            G = 'xxx'
            E = 'xxx'
            v = 'xxx'


        c11_list.append(c11)
        c12_list.append(c12)
        c13_list.append(c13)
        c33_list.append(c33)
        c44_list.append(c44)
        c11_c12_list.append(c11_c12)
        B_list.append(B)
        G_list.append(G)
        E_list.append(E)
        v_list.append(v)
        i_list.append(i)
        
    
# Print everything

        if steps > 1:
            elast_data = {
                'd': i_list,
                'c11': c11_list,
                'c12': c12_list,
                'c13': c13_list,
                'c33': c33_list,
                'c44': c44_list,
                'c11-c12': c11_c12_list,
                'B': B_list,
                'G': G_list,
                'E': E_list,
                'v': v_list,
                }
        else:
            elast_data = {
                'c11': c11,
                'c12': c12,
                'c13': c13,
                'c33': c33,
                'c44': c44,
                'c11-c12': c11_c12,
                'B': B,
                'G': G,
                'E': E,
                'v': v,
                }

    return(elast_data)

##########################################
######  END OF calc_elast_const()   ######
##########################################


##########################################
######  START OF get_lattice_type() ######
##########################################

# copied from elastic
# had to include it in here, 
# otherwise the function somehow wouldnt work
# but its not needed i think. 
# included just to be sure

def get_lattice_type(cryst):
    '''Find the symmetry of the crystal using spglib symmetry finder.

    Derive name of the space group and its number extracted from the result.
    Based on the group number identify also the lattice type and the Bravais
    lattice of the crystal. The lattice type numbers are
    (the numbering starts from 1):

    Triclinic (1), Monoclinic (2), Orthorombic (3),
    Tetragonal (4), Trigonal (5), Hexagonal (6), Cubic (7)

    :param cryst: ASE Atoms object

    :returns: tuple (lattice type number (1-7), lattice name, space group
                     name, space group number)
    '''

    # Table of lattice types and correcponding group numbers dividing
    # the ranges. See get_lattice_type method for precise definition.
    lattice_types = [
            [3,   "Triclinic"],
            [16,  "Monoclinic"],
            [75,  "Orthorombic"],
            [143, "Tetragonal"],
            [168, "Trigonal"],
            [195, "Hexagonal"],
            [231, "Cubic"]
        ]

    cell = (cryst.cell, cryst.get_scaled_positions(), cryst.numbers)
    dataset = spg.get_symmetry_dataset(cell)
    sg_name = dataset.international
    sg_nr = dataset.number

    for n, l in enumerate(lattice_types):
        if sg_nr < l[0]:
            bravais = l[1]
            lattype = n+1
            break

    return lattype, bravais, sg_name, sg_nr

##########################################
#######  END OF get_lattice_type() #######
##########################################

















