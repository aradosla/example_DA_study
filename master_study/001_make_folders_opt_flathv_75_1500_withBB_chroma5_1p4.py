# ==================================================================================================
# --- Imports
# ==================================================================================================
from tree_maker import initialize
import time
import os
import itertools
import numpy as np
import yaml
import shutil
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
n_r = 2 * 16 * (r_max - r_min)

# Number of angles for the initial particle distribution
n_angles = 5

# Number of split for parallelization
n_split = 12

# ==================================================================================================
# --- Base collider parameters
#
# Below, the user defines the base collider parameters. That is, only the parameters that subject
# to change are present here. More parameters can be added from the collider config.yaml, as needed.
# One needs to ensure that the remaining (default) parameters are set properly in the machine.
# Path for the collider config: master_study/master_jobs/000_build_distr_and_machine/config_collider.yaml
# ==================================================================================================
# Optic file path (round or flat)
optics_file = "acc-models-lhc/flatcc/opt_flathv_75_180_1500_thin.madx"

# Filling scheme and bunch number (#! If one change the filling scheme, one needs to change the
# ! number of colliding bunches at IP8 accordingly)
pattern_fname = "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/master_jobs/filling_scheme/8b4e_1972b_1960_1178_1886_224bpi_12inj_800ns_bs200ns.json"
i_bunch_b1 = 1963
i_bunch_b2 = 1963
num_colliding_bunches_ip8 = 1886

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
on_crab1 = -190
on_crab5 = -190

# Octupoles
i_oct_b1 = 60.0
i_oct_b2 = 60.0

# Tunes and chromas (for both beams)
qx = 62.316
qy = 60.321
dqx = 5.0
dqy = 5.0

# Luminosity and particles
num_particles_per_bunch = 1.4e11
nemitt_x = 2.5e-6
nemitt_y = 2.5e-6
separation_in_sigma_ip2 = 5
luminosity_ip8 = 2.0e33

# Value to be added to linear coupling knobs
delta_cmr: 0.001
delta_cmi: 0.0

# ==================================================================================================
# --- Machine parameters being scanned
#
# Below, the user defines the grid for the machine parameters that must be scanned to find the
# optimal DA (e.g. tune, chroma, etc).
# ==================================================================================================
# Scan tune with step of 0.001 (need to round to correct for numpy numerical instabilities)
array_qx = np.round(np.arange(62.305, 62.330, 0.001), decimals=4)[-6:]
array_qy = np.round(np.arange(60.305, 60.330, 0.001), decimals=4)[-5:]

# ==================================================================================================
# --- Tracking parameters
#
# Below, the user defines the parameters for the tracking.
# ==================================================================================================
n_turns = 1000
delta_max = 27.0e-5  # initial off-momentum

# ==================================================================================================
# --- Build base tree for the simulations
#
# The tree is built as a hierarchy of dictionnaries. We add a first generation (named as the
# study being done) to the root. This first generation is used set the initial particle
# distribution, and the parameters of the base machine which will later be used for simulations.
# ==================================================================================================
# Define study name
study_name = "opt_flathv_75_1500_withBB_chroma5_1p4"

# Build empty tree: first generation (later added to the root), and second generation
children = {"base_collider": {"config_particles": {}, "config_collider": {}, "children": {}}}

# Add particles distribution parameters to the first generation
children["base_collider"]["config_particles"]["r_min"] = r_min
children["base_collider"]["config_particles"]["r_max"] = r_max
children["base_collider"]["config_particles"]["n_r"] = n_r
children["base_collider"]["config_particles"]["n_angles"] = n_angles
children["base_collider"]["config_particles"]["n_split"] = n_split

# Add base machine parameters to the first generation
children["base_collider"]["config_collider"]["config_mad"] = {
    "beam_config": {"lhcb1": {}, "lhcb2": {}},
    "optics_file": None,
}
children["base_collider"]["config_collider"]["config_mad"]["optics_file"] = optics_file
children["base_collider"]["config_collider"]["config_mad"]["beam_config"]["lhcb1"][
    "beam_energy_tot"
] = beam_energy_tot
children["base_collider"]["config_collider"]["config_mad"]["beam_config"]["lhcb2"][
    "beam_energy_tot"
] = beam_energy_tot

# Add all knobs to the first generation (prepare also dictionnaries for tune and chroma)
children["base_collider"]["config_collider"]["config_knobs_and_tuning"] = {
    "knob_settings": {},
    "qx": {},
    "qy": {},
    "dqx": {},
    "dqy": {},
}
children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["knob_settings"][
    "on_x1"
] = on_x1
children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["knob_settings"][
    "on_sep1"
] = on_sep1
children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["knob_settings"][
    "on_x2"
] = on_x2
children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["knob_settings"][
    "on_sep2"
] = on_sep2
children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["knob_settings"][
    "on_x5"
] = on_x5
children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["knob_settings"][
    "on_sep5"
] = on_sep5
children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["knob_settings"][
    "on_x8h"
] = on_x8h
children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["knob_settings"][
    "on_x8v"
] = on_x8v
children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["knob_settings"][
    "on_crab1"
] = on_crab1
children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["knob_settings"][
    "on_crab5"
] = on_crab5
children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["knob_settings"][
    "i_oct_b1"
] = i_oct_b1
children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["knob_settings"][
    "i_oct_b2"
] = i_oct_b2

# Add tunes and chromas to the first generation
for beam in ["lhcb1", "lhcb2"]:
    children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["qx"][beam] = qx
    children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["qy"][beam] = qy
    children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["dqx"][beam] = dqx
    children["base_collider"]["config_collider"]["config_knobs_and_tuning"]["dqy"][beam] = dqy

# Add luminosity configuration to the first generation
children["base_collider"]["config_collider"]["config_lumi_leveling"] = {"ip2": {}, "ip8": {}}
children["base_collider"]["config_collider"]["config_lumi_leveling"]["ip2"][
    "separation_in_sigmas"
] = separation_in_sigma_ip2
children["base_collider"]["config_collider"]["config_lumi_leveling"]["ip8"][
    "luminosity"
] = luminosity_ip8
children["base_collider"]["config_collider"]["config_lumi_leveling"]["ip8"][
    "num_colliding_bunches"
] = num_colliding_bunches_ip8

# Add beam beam configuration to the first generation
children["base_collider"]["config_collider"]["config_beambeam"] = {"mask_with_filling_pattern": {}}
children["base_collider"]["config_collider"]["config_beambeam"][
    "num_particles_per_bunch"
] = num_particles_per_bunch
children["base_collider"]["config_collider"]["config_beambeam"]["nemitt_x"] = nemitt_x
children["base_collider"]["config_collider"]["config_beambeam"]["nemitt_y"] = nemitt_y
children["base_collider"]["config_collider"]["config_beambeam"]["mask_with_filling_pattern"][
    "pattern_fname"
] = pattern_fname
children["base_collider"]["config_collider"]["config_beambeam"]["mask_with_filling_pattern"][
    "i_bunch_b1"
] = i_bunch_b1
children["base_collider"]["config_collider"]["config_beambeam"]["mask_with_filling_pattern"][
    "i_bunch_b2"
] = i_bunch_b2

# ==================================================================================================
# --- Generate second generation of the tree, with the machine parameters being scanned, and
# tracking parameters being set.
# Parameters are separated in three groups, depending how they affect the configuration of the
# collider:
# - group_1: parameters that require to retune and redo the levelling, and retune again
#            (e.g. crossing angle)
# - group_2: parameters that require to retune
#            (e.g. chromaticity, octupoles)
# - group_3: parameters that require to reconfigure bb lenses
#            (e.g. bunch_nb)
# Collider lines composition can not be reconfigured at this step, at least for now.
# ==================================================================================================
track_array = np.arange(n_split)
for idx_job, (track, qx, qy) in enumerate(itertools.product(track_array, array_qx, array_qy)):
    # Ignore conditions below the upper diagonal as this can't exist in the real machine
    if qy < (qx - 2 + 0.0039):  # 0.0039 instead of 0.04 to avoid rounding errors
        continue

    children["base_collider"]["children"][f"xtrack_{idx_job:04}"] = {
        "parameters_scanned": {"group_2": {"qx": float(qx), "qy": float(qy)}},
        "particle_file": f"../particles/{track:02}.parquet",
        "collider_file": f"../collider/collider.json",
        "n_turns": n_turns,
        "delta_max": delta_max,
        # "log_file": (
        #     f"{os.getcwd()}/scans/{study_name}/base_collider/xtrack_{idx_job:04}/tree_maker.log"
        # ),
        "log_file": f"tree_maker.log",
    }

# ==================================================================================================
# --- Simulation configuration
# ==================================================================================================
# Load the tree_maker simulation configuration
config = yaml.safe_load(open("config.yaml"))

# # Set the root children to the ones defined above, or to the default one
# if config["root"]["use_yaml_children"] == False:
config["root"]["children"] = children
# else:
#     print("The default simulation configuration will be used.")

# Set miniconda environment path in the config
config["root"]["setup_env_script"] = os.getcwd() + "/../miniconda/bin/activate"

# ==================================================================================================
# --- Build tree and write it to the filesystem
# ==================================================================================================
# Creade folder that will contain the tree
if not os.path.exists("scans/" + study_name):
    os.makedirs("scans/" + study_name)

# Move to the folder that will contain the tree
os.chdir("scans/" + study_name)

# Create tree object
start_time = time.time()
root = initialize(config)
print("Done with the tree creation.")
print("--- %s seconds ---" % (time.time() - start_time))

# From python objects we move the nodes to the filesystem.
start_time = time.time()
root.make_folders(generate_run_sh_htc)
print("The tree folders are ready.")
print("--- %s seconds ---" % (time.time() - start_time))

# Rename log files according to study
shutil.move("tree_maker.json", f"tree_maker_{study_name}.json")
shutil.move("tree_maker.log", f"tree_maker_{study_name}.log")
