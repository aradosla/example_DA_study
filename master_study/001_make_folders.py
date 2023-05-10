# ==================================================================================================
# --- Imports
# ==================================================================================================
from tree_maker import initialize
import time
import os
import itertools
import numpy as np
import yaml
from user_defined_functions import generate_run_sh_htc

# ==================================================================================================
# --- Initial particle distribution parameters
# 
# Below, the user defines the parameters for the initial particles distribution.
# Path for the distribution config: master_study/master_jobs/000_build_distr_and_collider/config_distrib.yaml
# ==================================================================================================
# Radius of the initial particle distribution
r_min = 2
r_max = 10
n_r = 2*16*(r_max-r_min)

# Number of angles for the initial particle distribution
n_angles = 5

# Number of split for parallelization
n_split = 15

# ==================================================================================================
# --- Base collider parameters
# 
# Below, the user defines the base collider parameters. That is, only the parameters that subject
# to change are present here. More parameters can be added from the collider config.yaml, as needed.
# One needs to ensure that the remaining (default) parameters are set properly in the machine.
# Path for the collider config: master_study/master_jobs/000_build_distr_and_machine/config_collider.yaml
# ==================================================================================================
# Optic file path (round or flat)
optics_file = 'acc-models-lhc/flatcc/opt_flatvh_75_180_1500_thin.madx'

# Beam energy (for both beams)
beam_energy_tot = 7000

# Knobs at IPs
on_x1 = 250
on_sep1 = 0 
on_x2 = -170
on_sep2 = 0.138 
on_x5 = 250 
on_sep5 = 0 
on_x8h = 0.0
on_x8v = 170

# Crab cavities 
on_crab1: -190 
on_crab5: -190 

# Octupoles 
i_oct_b1 = 60.
i_oct_b2 = 60.

# Tunes and chromas (for both beams)
qx = 62.316
qy = 60.321
dqx = 15.
dqy = 15.

# Luminosity and particles
num_particles_per_bunch = 1.4e11
nemitt_x = 2.5e-6
nemitt_y = 2.5e-6
separation_in_sigma_ip2 = 5
luminosity_ip8 = 2.0e+33
num_colliding_bunches_ip8 = 1886

# ==================================================================================================
# --- Machine parameters being scanned
# 
# Below, the user defines the grid for the machine parameters that must be scanned to find the 
# optimal DA (e.g. tune, chroma, etc).
# ==================================================================================================
# Scan tune with step of 0.001 (need to round to correct for numpy numerical instabilities)
array_qx = np.arange(62.305, 62.330, 0.001)
array_qy = np.arange(60.305, 60.330, 0.001)

# ==================================================================================================
# --- Tracking parameters
# 
# Below, the user defines the parameters for the tracking.
# ==================================================================================================
n_turns = 1000000
delta_max = 27.e-5   # initial off-momentum

# ==================================================================================================
# --- Build base tree for the simulations
#
# The tree is built as a hierarchy of dictionnaries. We add a first generation (named as the 
# study being done) to the root. This first generation is used set the initial particle 
# distribution, and the parameters of the base machine which will later be used for simulations.
# ==================================================================================================
# Define study name
study_name = "example_HL_tunescan"

# Build empty tree: first generation (later added to the root), and second generation
children={study_name : {}}
children[study_name]["children"] = {}

# Add base machine parameters to the first generation
children[study_name]["config_mad"]["optics_file"] = optics_file
children[study_name]["config_mad"]["beam_config"]['lhcb1']['beam_energy_tot'] = beam_energy_tot
children[study_name]["config_mad"]["beam_config"]['lhcb2']['beam_energy_tot'] = beam_energy_tot

# Add all knobs at once
for knob in [on_x1, on_sep1, on_x2, on_sep2, on_x5, on_sep5, on_x8h, on_x8v, on_crab1, on_crab5, i_oct_b1, i_oct_b2]:
    children[study_name]["config_knobs_and_tuning"]["knob_settings"][f'{knob=}'.partition('=')[0]] = knob

# Add tunes and chromas
for knob in [qx, qy, dqx, dqy]:
    children[study_name]["config_knobs_and_tuning"][f'{knob=}'.partition('=')[0]]['lhcb1'] = knob
    children[study_name]["config_knobs_and_tuning"][f'{knob=}'.partition('=')[0]]['lhcb2'] = knob

# Add luminosity and particles configuration
children[study_name]["config_beambeam"]["num_particles_per_bunch"] = num_particles_per_bunch
children[study_name]["config_beambeam"]["nemitt_x"] = nemitt_x
children[study_name]["config_beambeam"]["nemitt_y"] = nemitt_y
children[study_name]["config_lumi_leveling"]["ip2"]["separation_in_sigmas"] = separation_in_sigma_ip2
children[study_name]["config_lumi_leveling"]["ip8"]["luminosity"] = luminosity_ip8
children[study_name]["config_lumi_leveling"]["ip8"]["num_colliding_bunches"] = num_colliding_bunches_ip8

# ==================================================================================================
# --- Generate second generation of the tree, with the machine parameters being scanned, and 
# tracking parameters being set.
# ==================================================================================================
track_array = np.arange(15)
for idx_job, (track, qx, qy) in enumerate(itertools.product(track_array, array_qx, array_qy)):
    # Ignore conditions below the upper diagonal as this can't exist in the real machine
    if qy < (qx - 2 + 0.005):
        continue
    children[study_name]["children"][f"madx_{idx_job:04}"] = {
        "qx0": float(qx),
        "qy0": float(qy),
        "particle_file": f"../../particles/{track:02}.parquet",
        "collider_file" : f"../../collider.json",
        "n_turns": int(1000000),
        "log_file": f"{os.getcwd()}/{study_name}/madx_{idx_job:04}/tree_maker.log",
    }

# ==================================================================================================
# --- Simulation configuration
# ==================================================================================================
# Load the tree_maker simulation configuration
config=yaml.safe_load(open('config.yaml'))

# Set the root children to the ones defined above, or to the default one
if config['root']['use_yaml_children']== False:
    config['root']['children'] = children
else:
    print('The default simulation configuration will be used.')

# Set miniconda environment path in the config
config['root']['setup_env_script'] = os.getcwd() + '/../miniconda/bin/activate'

# ==================================================================================================
# --- Build tree and write it to the filesystem
# ==================================================================================================
# Create tree object
start_time = time.time()
root = initialize(config)
print('Done with the tree creation.')
print("--- %s seconds ---" % (time.time() - start_time))

# From python objects we move the nodes to the filesystem.
start_time = time.time()
root.make_folders(generate_run_sh_htc)
print('The tree folders are ready.')
print("--- %s seconds ---" % (time.time() - start_time))
