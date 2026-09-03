#General imports
import os
import copy
import sys
import time
import math
import fnmatch
import subprocess
from invoke import run

#Mathematical imports
import numpy as np
from numpy import array, linspace
import pandas as pd

#Import from ase
import ase
from ase import units
from ase import Atoms
from ase.io import read, write
from ase.units import Rydberg, Bohr
from ase.units import _amu





#################################
######  START OF run_pw() #######
#################################

class Logger:
    def __init__(self, logfile=sys.stdout):
        self.logfile = logfile

    def __call__(self, *args, **kwargs):
        if self.logfile is not None:
            print(*args, file=self.logfile, **kwargs)



# INPUTS:
#### atoms = structure as ASE_Atoms 
#### input_data = pw_inp keyword data as dictionary
#### calc_path = path to directory where you want to start the calculation
#### runbatch_path = path to the runbatch file of PW
#### pseudopotentials = pseudopotential dictionary
#### vdW_path = path to INP-vdW. if None is given, skips reading it. default = None
#
# RETURNS:
#### dict of job_id and corresponding directory

def run_pw(atoms, input_data, pseudopotentials, calc_path, runbatch_path, vdW_path=None):
    job={}
    # generate INP
    input_file_path = os.path.join(calc_path, "INP")                
    with open(input_file_path, mode = 'w') as f:
                write(f, atoms, format  = 'espresso-in', input_data=input_data, \
                pseudopotentials=pseudopotentials,\
                kpts=input_data['kpts'], koffset=input_data['koffset'])
    
    # copy runbatch to given directory    
    run(f"cp {runbatch_path} {os.path.join(calc_path, 'runbatch')}")

    # if D3-ext, copy INP-vdW to given directory
    if input_data['vdw_corr'] == 'D3-ext':
        if vdW_path != None:
            run(f"cp {vdW_path} {calc_path}")
        elif vdW_path == None:
            return print("ERROR: D3-ext given, but no vdW_path!")

    # start calc on 160 in given directory
    run(f"cd {calc_path}; sbatch runbatch > jobid")
    time.sleep(1)
    job_id = int(np.loadtxt(os.path.join(calc_path, "jobid"), usecols=3))

    #run(f"rm {os.path.join(calc_path, 'jobid')}")
    
    job = { 'job_id' : job_id,
            'directory' : calc_path}

    return job

#################################
#######  END OF run_pw() ########
#################################

#################################
##  START OF run_multiple_pw() ##
#################################

# INPUTS:
#### same as run_pw but runs multiple pw calcs all at once
#
# RETURNS: 
#### a nested dictionary of all jobids and directories

def run_multiple_pw(atoms_list, input_data, pseudopotentials, calc_path, runbatch_path, vdW_path=None):
    jobs={}
    for i in range(len(atoms_list)):
        calc_path_sub = os.path.join(calc_path, f"str_{i}")
        run(f"mkdir {calc_path_sub}")
        job = run_pw(atoms_list[i], input_data, pseudopotentials, calc_path_sub, runbatch_path, vdW_path)
        jobs[i] = job

    return jobs

#################################
### END OF run_multiple_pw()  ###
#################################

#################################
##### START OF is_nested()  #####
#################################

# INPUT:
#### dictionary
#
# RETURNS:
#### if dict is nested: TRUE
#### if dict is NOT nested: FALSE
#
# note: i just copied this from stack overflow, it works though :-D
def is_nested(dic):
    return isinstance(dic, dict) and any(isinstance(val, dict) for val in dic.values())

#################################
###### END OF is_nested()  ######
#################################



#################################
#####START OF read_pw_jobs()#####
#################################


# INPUTS:
#### dict of jobs
#
# RETURNS:
#### if multiple pw calcs were read: list of ASE.Atoms objects
#### if only one calc was read: ASE.Atoms object
#
# What does it do:
#### first checks if its a nested dict (aka multiple pw calcs) or not (singular pw calc)
#### then checks if jobid is still in queue 
#### if not, it checks if JOB DONE is in OUT
#### if no -> skip adding this atom and print error message
#### if yes -> read OUT as espresso-out and append to atoms_list
#


def read_pw_jobs(jobs, poll_interval=3):
    #Wait for a SLURM QE job to finish and verify JOB DONE
    jobid_list=[]
    dirs_list=[]
    atoms_list=[]
    
    # check if you only wait for one pw calc, or multiple
    
    if is_nested(jobs) == True:
        for key, val in jobs.items():
            jobid_list.append(val['job_id'])
            dirs_list.append(val['directory'])
    else:
        jobid_list.append(jobs['job_id'])
        dirs_list.append(jobs['directory'])


    for i in range(len(jobid_list)):
        calc_path = dirs_list[i]
        job_id=jobid_list[i]
        
        out_file = os.path.join(calc_path, "OUT")
        print(f"Waiting for SLURM job {job_id}...")

        while True:
            result = subprocess.run(
            ["squeue", "-j", str(job_id), "-h"],
            capture_output=True,
            text=True
        )

        # Job is no longer in the queue
            if not result.stdout.strip():
                break

        time.sleep(poll_interval)

        print(f"SLURM job {job_id} finished.")

    # Wait a little bit just in case
        while not os.path.exists(out_file):
            time.sleep(1)

    # Check for JOB DONE
        with open(out_file, "r") as f:
            output = f.read()

        if "JOB DONE" not in output:
            print(f"Job {job_id} finished, but 'JOB DONE' was not found in {out_file}.")
            print("Not returning this structure!")
        else:
            atoms = atoms = read(out_file, format="espresso-out")
            atoms_list.append(atoms)

    if len(atoms_list) > 1:
        return atoms_list
    else:
        return atoms_list[0]
#################################
##### END OF read_pw_jobs() #####
#################################

# k_per_inv_A: how many k-points per inverse Angstrom
# k_thresh: cell length after which the k-point in that direction is set to 1
# returns the k-point mesh as a list

def get_kpoint_mesh(atoms, k_per_inv_A=26, k_thresh=13):

    cell_lengths = atoms.cell.lengths()
    k_mesh = (1/cell_lengths) * k_per_inv_A
    
    for i in range(len(k_mesh)):
        if cell_lengths[i] >= k_thresh:
             k_mesh[i] = 1
        else:
            k_mesh[i] = math.floor(k_mesh[i] / 2) * 2
        
    return k_mesh





# --- vc_relax convex-hull workflow: restart-until-converged driver ---


def run_vc_relax_until_converged(strucs_dict, base_input_data, pseudopotentials,
                                  calc_base_dir, runbatch_path, vdW_path=None,
                                  max_restarts=8, poll_interval=3, logfile=sys.stdout):
    """
    Repeatedly runs vc_relax on every structure in strucs_dict, feeding each
    restart's relaxed geometry back in as the next restart's starting point,
    until each structure's vc_relax converges in a SINGLE ionic/cell step --
    i.e. the geometry handed in was already at the minimum for the freshly
    generated plane-wave basis of that cell, so no further restart changes
    anything.
 
    Returns a dict {index: {'name': ..., 'ase_atoms': ..., 'potential_energy': ...,
    'forces': ...}} for every structure that converged within max_restarts,
    where index is just a plain running number (0, 1, 2, ...) in the order
    structures converged -- NOT tied to the structure's name or its position
    in strucs_dict. The structure name is stored as a normal property inside
    each entry instead. Structures whose calculation crashed, or that didn't
    converge within max_restarts, are reported via `log` and simply left out
    of the returned dict rather than raising.

    After EVERY restart round, the results collected so far are also written
    to disk as a pandas DataFrame, pickled+gzipped to
    calc_base_dir/vc_relax_results.pckl.gzip. This way, whatever has already
    converged is saved even if the notebook/kernel dies partway through --
    you no longer have to build+pickle the DataFrame yourself after the call
    returns.

    If a job crashes (QE writes a CRASH file instead of finishing), that is
    now told apart from "simply not converged yet" in the log. The classic
    "Not enough space allocated for radial FFT" crash is handled specially:
    QE's own error message tells you to restart with a larger cell_factor,
    so this structure's cell_factor is bumped by +2.0 and the same restart
    round is retried with that larger value -- instead of blindly resubmitting
    the identical job (which would just crash again the same way).
    """
    log = Logger(logfile)

    # current_atoms holds each structure's LATEST known geometry -- the
    # starting point for its next restart. converged/n_steps/results track
    # per-structure progress across restart rounds.
    current_atoms = dict(strucs_dict)
    converged = {name: False for name in strucs_dict}
    results = {}
    result_index = 0  # plain running number used as the key in results, bumped each time a structure converges

    # cell_factor starts out the same for every structure (whatever
    # base_input_data says), but gets bumped per-structure if that
    # structure's job crashes with the "radial FFT too small" error.
    base_cell_factor = base_input_data.get('cell_factor', 4.0)
    cell_factor = {name: base_cell_factor for name in strucs_dict}

    # where the running results are (re-)saved after every restart round
    results_pickle_path = os.path.join(calc_base_dir, "vc_relax_results.pckl.gzip")

    for restart in range(max_restarts):
        # only (re)submit structures that haven't converged yet
        names_this_round = [name for name in current_atoms if not converged[name]]
        if not names_this_round:
            break
 
        log(f"=== restart round {restart}: {len(names_this_round)} structure(s) to (re)run ===")
 
        jobs = {}
        for name in names_this_round:
            atoms = current_atoms[name]
 
            # one directory per structure per restart round, e.g.
            # calc_base_dir/AB2_xyz/restart_0, restart_1, ...
            calc_path = os.path.join(calc_base_dir, name, f"restart_{restart}")
            run(f"mkdir -p {calc_path}")  # -p: name/ and restart_N/ may not exist yet
 
            input_data = copy.deepcopy(base_input_data)
            # unique prefix per structure+restart, so QE's scratch/output
            # files never collide between structures or between restarts
            input_data['prefix'] = f"{name}_r{restart}"
            # k-point mesh must be recomputed every restart: the cell (and
            # therefore the appropriate mesh) changes between restarts
            input_data['kpts'] = tuple(int(k) for k in get_kpoint_mesh(atoms))
            # use this structure's own cell_factor -- normally the base
            # value, but larger if an earlier restart crashed with the
            # "radial FFT too small" error (see crash handling below)
            input_data['cell_factor'] = cell_factor[name]

            jobs[name] = run_pw(atoms, input_data, pseudopotentials, calc_path,
                                 runbatch_path, vdW_path=vdW_path)
 
        # Block here until every job in this round has left the SLURM queue.
        # read_pw_jobs()'s own return value is not used: it silently drops
        # any job that never printed "JOB DONE", with no indication of WHICH
        # structure that was -- so each OUT file is re-read individually
        # below instead, keyed by structure name.
        read_pw_jobs(jobs, poll_interval=poll_interval)
 
        for name in names_this_round:
            calc_path = os.path.join(calc_base_dir, name, f"restart_{restart}")
            out_file = os.path.join(calc_path, "OUT")
            crash_file = os.path.join(calc_path, "CRASH")

            if not os.path.exists(out_file):
                log(f"  {name}: no OUT file found after restart {restart}, will retry.")
                continue

            with open(out_file) as f:
                output = f.read()

            if "JOB DONE" not in output:
                # QE writes a CRASH file when pw.x aborts -- that's a
                # different situation than "still needs more restarts",
                # so it gets its own message and (for the FFT case) an
                # actual fix instead of just resubmitting the same job.
                if os.path.exists(crash_file):
                    with open(crash_file) as f:
                        crash_text = f.read()

                    if "larger cell_factor" in crash_text:
                        # QE's own error message tells us the fix: the
                        # radial FFT grid it allocated (based on
                        # cell_factor) was too small for how much the
                        # cell grew during this restart. Bump it and
                        # retry -- otherwise this would crash the exact
                        # same way every single restart.
                        cell_factor[name] += 2.0
                        log(f"  {name}: CRASHED after restart {restart} "
                            f"(radial FFT grid too small), increasing "
                            f"cell_factor to {cell_factor[name]} and retrying.")
                    else:
                        log(f"  {name}: CRASHED after restart {restart}, "
                            f"see {crash_file} for details. Retrying with "
                            f"the same settings.")
                else:
                    log(f"  {name}: 'JOB DONE' not found after restart {restart} "
                        f"(no CRASH file -- job may have hit the walltime), will retry.")
                continue

            # ASE returns one image per ionic/cell step vc_relax performed.
            # Exactly one image means QE found the input geometry already
            # converged in its very first SCF cycle -- this structure is done.
            images = read(out_file, index=':', format='espresso-out')
            current_atoms[name] = images[-1]  # always carry the latest geometry forward
 
            if len(images) <= 2:
                converged[name] = True
                # key by a plain number instead of the structure name; name
                # itself just moves inside the dict as another property,
                # alongside ase_atoms/potential_energy/forces
                results[result_index] = { 'name' : name,
                                          'ase_atoms' : images[-1],
                                          'potential_energy' : images[-1].get_potential_energy(),
                                          'forces' : images[-1].get_forces()}
                result_index += 1  # next structure that converges gets the next number
                log(f"  {name}: converged after {restart + 1} restart(s).")
            else:
                log(f"  {name}: {len(images)} ionic/cell steps this round, restarting.")

        # Save whatever has converged so far after every restart round --
        # if the notebook/kernel dies before the function returns, this
        # file on disk is the only record of the work done up to now.
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_pickle(results_pickle_path, compression='gzip', protocol=4)
        log(f"  (saved {len(results)} converged structure(s) so far to {results_pickle_path})")

    still_running = [name for name in current_atoms if not converged[name]]
    if still_running:
        log(f"WARNING: {len(still_running)} structure(s) did not converge within "
            f"{max_restarts} restarts: {still_running}")

    return results






#################################
##### END OF pw_functions.py ##### 
#################################
