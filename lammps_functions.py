#################################
###### lammps_functions.py ######
#################################
#
# Helper functions to run (GR)ACE-potential LAMMPS jobs (NVT/NPT/NPH/NVE,
# a single ensemble or a chain of several) from the notebook -- the LAMMPS
# equivalent of run_pw()/run_vc_relax_until_converged() in pw_functions.py.
#
# ASE doesn't have anything like ase.io.write(..., format='espresso-in')
# for LAMMPS: it can write the STRUCTURE as a LAMMPS data file, but not a
# full LAMMPS input script (units/pair_style/fixes/run commands...) from a
# plain dict, because LAMMPS input files are a sequence of commands, not a
# namelist format like Quantum Espresso's. So here the input script is
# built from your template as a plain string, with only a handful of
# things filled in (paths, temperature/pressure, number of steps).

import os
import sys
import time
import subprocess
import numpy as np
from invoke import run
from ase.io import write

# reuse instead of duplicating: Logger just wraps print() so log messages
# can be silenced (logfile=None) or redirected; is_nested tells single
# jobs and dicts-of-jobs apart. Both come from pw_functions.py already.
from pw_functions import Logger, is_nested


#################################
## START OF write_lammps_data() #
#################################

# INPUTS:
#### atoms      : ASE Atoms object to write out
#### calc_path  : directory to write the data file into
#### elements   : list of chemical symbols giving the LAMMPS atom-type
####              order (type 1 = elements[0], type 2 = elements[1], ...).
####              If None, the elements present in `atoms` are used,
####              alphabetically sorted -- this becomes the "specorder"
####              that MUST also be used for the pair_coeff/dump lines,
####              so this function always returns whichever order it used.
#### filename   : name of the data file inside calc_path
#
# RETURNS:
#### (data_path, elements) -- full path to the data file, and the element
#### order actually used. Pass this straight into build_lammps_input() so
#### pair_coeff/dump element lists match the data file's atom types.

def write_lammps_data(atoms, calc_path, elements=None, filename="structure.data"):
    if elements is None:
        # sorted() makes this deterministic -- the same structure always
        # gets the same atom-type order, run after run
        elements = sorted(set(atoms.get_chemical_symbols()))

    data_path = os.path.join(calc_path, filename)
    write(data_path, atoms, format='lammps-data', specorder=elements,
          masses=True, atom_style='atomic', units='metal')

    return data_path, elements

#################################
### END OF write_lammps_data() ##
#################################


#################################
# START OF _ensemble_fix_line() #
#################################

# Builds the one "fix ens_fix all <ensemble> ..." line for a single MD
# stage. Internal helper, not meant to be called directly -- used by
# build_lammps_input().
#
# ensemble : one of 'nvt', 'npt', 'nph', 'nve' (case-insensitive)
# temp     : target temperature (used for nvt/npt, ignored for nph/nve)
# pressure : target pressure (used for npt/nph, ignored for nvt/nve)
# tdamp/pdamp : thermostat/barostat damping times, in LAMMPS "metal"
####            units (picoseconds)

def _ensemble_fix_line(ensemble, temp, pressure, tdamp, pdamp):
    ensemble = ensemble.lower()

    if ensemble == 'nvt':
        return f"fix ens_fix all nvt temp {temp} {temp} {tdamp}"
    elif ensemble == 'npt':
        return f"fix ens_fix all npt temp {temp} {temp} {tdamp} iso {pressure} {pressure} {pdamp}"
    elif ensemble == 'nph':
        return f"fix ens_fix all nph iso {pressure} {pressure} {pdamp}"
    elif ensemble == 'nve':
        return "fix ens_fix all nve"
    else:
        raise ValueError(f"Unknown ensemble '{ensemble}', must be one of "
                          f"'nvt', 'npt', 'nph', 'nve'.")

#################################
## END OF _ensemble_fix_line() ##
#################################


#################################
# START OF build_lammps_input() #
#################################

# Builds the full LAMMPS input script as one string, ready to be written
# to disk. Everything up to and including the box-relax minimization is
# the fixed prologue from your template (only the data-file path, the
# potential paths and the element list change there). After that comes
# the one-time velocity creation + the momentum fix, then one
# fix/run/unfix block per stage in `stages`.
#
# INPUTS:
#### data_path : path to the structure.data file (from write_lammps_data)
#### elements  : element/atom-type order used in the data file (from
####             write_lammps_data) -- used for pair_coeff and the dumps
#### pot_path  : directory containing the potential files
#### stages    : list of dicts, one per MD stage to run, e.g.
####             [{'ensemble': 'nvt', 'temp': 300, 'nsteps': 5000},
####              {'ensemble': 'npt', 'temp': 300, 'pressure': 0.0, 'nsteps': 20000}]
####             'pressure' defaults to 0.0 if a stage doesn't set it.
#### yaml_name/asi_name : filenames of the potential inside pot_path
#### timestep  : MD timestep, in ps (metal units)
#### tdamp/pdamp : thermostat/barostat damping times; if None, the usual
####             LAMMPS rule of thumb is used (100x / 1000x the timestep)
#### seed      : RNG seed for the initial "velocity create". If None, a
####             random one is drawn (and returned, so you can log/reuse it)
#### gamma_every : how often (steps) the extrapolation-gamma dump
####             (pace_dump) is written
#### traj_every  : how often (steps) the plain trajectory dump (dmp_trj)
####             is written
#
# RETURNS:
#### (input_text, seed) -- the finished input script text, and the seed
#### that was used (handy to log, since a random one may have been drawn).
#### This function only builds the text -- it doesn't write or submit
#### anything, that's what run_lammps_md() does.

def build_lammps_input(data_path, elements, pot_path, stages,
                        yaml_name="output_potential.yaml",
                        asi_name="output_potential.asi",
                        timestep=0.001, tdamp=None, pdamp=None, seed=None,
                        gamma_every=2000, traj_every=2000):

    if tdamp is None:
        tdamp = 100 * timestep
    if pdamp is None:
        pdamp = 1000 * timestep
    if seed is None:
        seed = int(np.random.randint(1, 1_000_000))

    yaml_path = os.path.join(pot_path, yaml_name)
    asi_path = os.path.join(pot_path, asi_name)
    elements_str = " ".join(elements)

    # temperature used for the one-time initial velocity creation: the
    # first stage that actually specifies one (nvt/npt stages do -- nph/nve
    # don't need one for their own fix, but the MD still needs a starting
    # velocity distribution to work with)
    init_temp = None
    for stage in stages:
        if stage.get('temp') is not None:
            init_temp = stage['temp']
            break
    if init_temp is None:
        raise ValueError("None of the stages specify a 'temp' -- at least "
                          "one is needed to generate the initial velocities.")

    # This is your template, verbatim, with only the bracketed bits filled
    # in. NOTE: the box-relax step always targets zero pressure (iso 0.0),
    # regardless of what pressure the actual MD stages target further down
    # -- that's what your template said, so it's kept fixed here too.
    header = f"""units metal
atom_style atomic
boundary p p p

read_data {data_path}

pair_style pace/extrapolation
pair_coeff * * {yaml_path} {asi_path} {elements_str}

fix pace_gamma all pair {gamma_every} pace/extrapolation gamma 1
compute max_pace_gamma all reduce max f_pace_gamma
variable dump_skip equal "(c_max_pace_gamma < 10) || (c_max_pace_gamma > 100)"
dump pace_dump all custom {gamma_every} extrapolative_structures.lammpstrj id element x y z f_pace_gamma
dump_modify pace_dump skip v_dump_skip
dump_modify pace_dump format line "%d %s %20.5g %20.15g %20.15g %20.15g" element {elements_str} first no sort id
dump dmp_trj all custom {traj_every} trj.lammpstrj id element xu yu zu f_pace_gamma
dump_modify dmp_trj format line "%d %s %20.5g %20.15g %20.15g %20.15g" element {elements_str} first no sort id

thermo 100
thermo_style custom step temp press vol etotal ke pe lz cpu
timestep {timestep}

variable T equal temp
variable P equal press
variable V equal vol
fix vpt_dump all ave/time 100 1 100 v_V v_P v_T file vpt.dat

min_style cg
fix rel_box all box/relax iso 0.0
minimize 1.0e-15 1.0e-15 1000000 1000000
run 101
unfix rel_box

velocity all create {init_temp} {seed} mom yes rot yes dist gaussian
fix mom_fix all momentum 1 linear 1 1 1
"""

    # one fix/run/unfix block per requested stage -- unfixing ens_fix
    # before the next stage defines it again is what lets you chain
    # different ensembles one after another
    stage_lines = []
    for stage in stages:
        ensemble = stage['ensemble']
        temp = stage.get('temp')
        pressure = stage.get('pressure', 0.0)
        nsteps = stage['nsteps']

        stage_lines.append(_ensemble_fix_line(ensemble, temp, pressure, tdamp, pdamp))
        stage_lines.append(f"run {nsteps}")
        stage_lines.append("unfix ens_fix")
        stage_lines.append("")  # blank line between stages, purely cosmetic

    input_text = header + "\n" + "\n".join(stage_lines)

    return input_text, seed

#################################
## END OF build_lammps_input() ##
#################################


#################################
##### START OF run_lammps_md() ##
#################################

# Submits one LAMMPS MD run (single ensemble, or a chain of several) for
# one ASE Atoms object. The LAMMPS equivalent of run_pw().
#
# INPUTS:
#### atoms        : ASE Atoms object to simulate
#### temp         : temperature. Either one number (used everywhere a
####                stage needs one) or a list with one entry per stage,
####                for a chain with a different temperature per stage.
#### pot_path     : directory containing output_potential.yaml/.asi
#### calc_path    : directory to run this job in (created if missing)
#### runbatch_path: path to your SLURM batch script. It must call LAMMPS
####                on "in.lammps", since that's the filename this
####                function writes the input script to (e.g.
####                `lmp -in in.lammps -log log.lammps`)
#### pressure     : like `temp`, but for pressure. Default 0.0, used by
####                npt/nph stages
#### ensemble     : one of 'nvt'/'npt'/'nph'/'nve', or a list of them for
####                a chain, e.g. ['nvt', 'npt']
#### nsteps       : number of MD steps per stage -- one number, or a list
####                with one entry per stage
#### elements     : LAMMPS atom-type order, see write_lammps_data(). If
####                None, it's worked out automatically from `atoms`.
#### timestep, tdamp, pdamp, seed, yaml_name, asi_name, gamma_every,
#### traj_every   : forwarded to build_lammps_input(), see there for what
####                they do.
#
# RETURNS:
#### dict {'job_id': ..., 'directory': ...} -- same shape as run_pw()
#### returns, so it can be handed straight to read_lammps_jobs().

def run_lammps_md(atoms, temp, pot_path, calc_path, runbatch_path,
                   pressure=0.0, ensemble='npt', nsteps=10000,
                   elements=None, timestep=0.001, tdamp=None, pdamp=None,
                   seed=None, yaml_name="output_potential.yaml",
                   asi_name="output_potential.asi",
                   gamma_every=2000, traj_every=2000, logfile=sys.stdout):

    log = Logger(logfile)

    # allow the simple case (one ensemble, one temp/pressure/nsteps) and
    # the chain case (lists) with the exact same arguments -- a bare
    # value just becomes a list of length 1
    ensembles = ensemble if isinstance(ensemble, list) else [ensemble]
    n_stages = len(ensembles)

    def broadcast(value, name):
        if isinstance(value, list):
            if len(value) != n_stages:
                raise ValueError(f"'{name}' has {len(value)} entries but "
                                  f"there are {n_stages} ensemble stage(s).")
            return value
        return [value] * n_stages

    temps = broadcast(temp, 'temp')
    pressures = broadcast(pressure, 'pressure')
    nsteps_list = broadcast(nsteps, 'nsteps')

    stages = [{'ensemble': ensembles[i], 'temp': temps[i],
               'pressure': pressures[i], 'nsteps': nsteps_list[i]}
              for i in range(n_stages)]

    run(f"mkdir -p {calc_path}")

    data_path, elements = write_lammps_data(atoms, calc_path, elements=elements)

    input_text, seed = build_lammps_input(
        data_path, elements, pot_path, stages,
        yaml_name=yaml_name, asi_name=asi_name, timestep=timestep,
        tdamp=tdamp, pdamp=pdamp, seed=seed,
        gamma_every=gamma_every, traj_every=traj_every)

    log(f"  seed used for velocity create: {seed}")

    input_file_path = os.path.join(calc_path, "in.lammps")
    with open(input_file_path, mode='w') as f:
        f.write(input_text)

    run(f"cp {runbatch_path} {os.path.join(calc_path, 'runbatch')}")

    # start calc on 160 in given directory -- identical pattern to run_pw()
    run(f"cd {calc_path}; sbatch runbatch > jobid")
    time.sleep(1)
    job_id = int(np.loadtxt(os.path.join(calc_path, "jobid"), usecols=3))

    job = {'job_id': job_id, 'directory': calc_path}

    return job

#################################
####### END OF run_lammps_md() ##
#################################


#################################
### START OF read_lammps_jobs() #
#################################

# Waits for one or several LAMMPS SLURM jobs (as returned by
# run_lammps_md()) to leave the queue, then checks whether each one
# actually finished cleanly: LAMMPS prints "Total wall time:" as its very
# last line on a normal exit -- if that's missing, something went wrong
# (crash, hit the walltime, ...).
#
# ASSUMPTION: your runbatch script writes LAMMPS' log to a file called
# "log.lammps" inside calc_path (LAMMPS' own default log filename). If
# your runbatch redirects it somewhere else (e.g. into an "OUT" file like
# the QE runs), pass that name as `log_filename`.
#
# INPUTS:
#### jobs : one job dict, or a nested dict of them (same shape read_pw_jobs()
####        takes)
#
# RETURNS:
#### nothing -- this just logs which jobs finished cleanly and which
#### didn't. Read the actual trajectory/log files afterwards yourself,
#### e.g. with read_lammps_trajectory() from frenkel.py.

def read_lammps_jobs(jobs, log_filename="log.lammps", poll_interval=3, logfile=sys.stdout):
    log = Logger(logfile)

    jobid_list = []
    dirs_list = []

    if is_nested(jobs):
        for key, val in jobs.items():
            jobid_list.append(val['job_id'])
            dirs_list.append(val['directory'])
    else:
        jobid_list.append(jobs['job_id'])
        dirs_list.append(jobs['directory'])

    for i in range(len(jobid_list)):
        calc_path = dirs_list[i]
        job_id = jobid_list[i]

        log(f"Waiting for SLURM job {job_id}...")

        while True:
            result = subprocess.run(
                ["squeue", "-j", str(job_id), "-h"],
                capture_output=True, text=True)

            # job is no longer in the queue
            if not result.stdout.strip():
                break

            time.sleep(poll_interval)

        log(f"SLURM job {job_id} finished.")

        log_path = os.path.join(calc_path, log_filename)

        if not os.path.exists(log_path):
            log(f"  no {log_filename} found in {calc_path} -- job probably "
                f"never started, check the runbatch's SLURM error file.")
            continue

        with open(log_path) as f:
            output = f.read()

        if "Total wall time" not in output:
            log(f"  {calc_path}: LAMMPS did not finish normally (no 'Total "
                f"wall time' found in {log_filename}) -- check {log_path} "
                f"for errors.")
        else:
            log(f"  {calc_path}: finished normally.")

#################################
#### END OF read_lammps_jobs() ##
#################################
