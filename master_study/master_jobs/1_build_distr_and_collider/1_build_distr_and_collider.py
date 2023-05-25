# ==================================================================================================
# --- Imports
# ==================================================================================================
from cpymad.madx import Madx
import os
import xmask as xm
import xmask.lhc as xlhc
import shutil
import json
import ruamel.yaml
import tree_maker
import numpy as np
import itertools
import pandas as pd

# Import user-defined optics-specific tools
import optics_specific_tools_hlhc15 as ost
from gen_config_orbit_correction import generate_orbit_correction_setup

# ==================================================================================================
# --- Load configuration file
# ==================================================================================================
# Load configuration
with open("config.yaml", "r") as fid:
    configuration = fid.read()

# Convert to yaml object with comment preservation
yaml = ruamel.yaml.YAML()
configuration = yaml.load(configuration)

# Get configuration for the particles distribution and the collider separately
config_particles = configuration["config_particles"]
config_collider = configuration["config_collider"]

# Start tree_maker logging if log_file is present in config
if tree_maker is not None and "log_file" in configuration:
    tree_maker.tag_json.tag_it(configuration["log_file"], "started")

# ==================================================================================================
# --- Build particle distribution
# ==================================================================================================

# Define radius distribution
r_min = config_particles["r_min"]
r_max = config_particles["r_max"]
n_r = config_particles["n_r"]
radial_list = np.linspace(r_min, r_max, n_r, endpoint=False)

# Define angle distribution
n_angles = config_particles["n_angles"]
theta_list = np.linspace(0, 90, n_angles + 2)[1:-1]

# Define particle distribution as a cartesian product of the above
particle_list = [
    (particle_id, ii[0], ii[1])
    for particle_id, ii in enumerate(itertools.product(radial_list, theta_list))
]

# Split distribution into several chunks for parallelization
n_split = config_particles["n_split"]
particle_list = list(np.array_split(particle_list, n_split))

# Write distribution to parquet files
distributions_folder = "particles"
os.makedirs(distributions_folder, exist_ok=True)
for idx_chunk, my_list in enumerate(particle_list):
    pd.DataFrame(
        my_list,
        columns=["particle_id", "normalized amplitude in xy-plane", "angle in xy-plane [deg]"],
    ).to_parquet(f"{distributions_folder}/{idx_chunk:02}.parquet")

# ==================================================================================================
# --- Generate config correction files
# ==================================================================================================
correction_setup = generate_orbit_correction_setup()
os.makedirs("correction", exist_ok=True)
for nn in ["lhcb1", "lhcb2"]:
    with open(f"correction/corr_co_{nn}.json", "w") as fid:
        json.dump(correction_setup[nn], fid, indent=4)

# ==================================================================================================
# --- Build collider from mad model
# ==================================================================================================
config_mad_model = config_collider["config_mad"]

# Make mad environment
xm.make_mad_environment(links=config_mad_model["links"])

# Start mad
mad_b1b2 = Madx(command_log="mad_collider.log")
mad_b4 = Madx(command_log="mad_b4.log")

# Build sequences
ost.build_sequence(mad_b1b2, mylhcbeam=1)
ost.build_sequence(mad_b4, mylhcbeam=4)

# Apply optics (only for b1b2, b4 will be generated from b1b2)
ost.apply_optics(mad_b1b2, optics_file=config_mad_model["optics_file"])

# Build xsuite collider
collider = xlhc.build_xsuite_collider(
    sequence_b1=mad_b1b2.sequence.lhcb1,
    sequence_b2=mad_b1b2.sequence.lhcb2,
    sequence_b4=mad_b4.sequence.lhcb2,
    beam_config=config_mad_model["beam_config"],
    enable_imperfections=config_mad_model["enable_imperfections"],
    enable_knob_synthesis=config_mad_model["enable_knob_synthesis"],
    rename_coupling_knobs=config_mad_model["rename_coupling_knobs"],
    pars_for_imperfections=config_mad_model["pars_for_imperfections"],
    ver_lhc_run=config_mad_model["ver_lhc_run"],
    ver_hllhc_optics=config_mad_model["ver_hllhc_optics"],
)


# Remove all the temporaty files created in the process of building collider
os.remove("mad_collider.log")
os.remove("mad_b4.log")
shutil.rmtree("temp")
os.unlink("errors")
os.unlink("acc-models-lhc")

# ==================================================================================================
# --- Install beam-beam
# ==================================================================================================
config_bb = config_collider["config_beambeam"]

# Install beam-beam lenses (inactive and not configured)
collider.install_beambeam_interactions(
    clockwise_line="lhcb1",
    anticlockwise_line="lhcb2",
    ip_names=["ip1", "ip2", "ip5", "ip8"],
    delay_at_ips_slots=[0, 891, 0, 2670],
    num_long_range_encounters_per_side=config_bb["num_long_range_encounters_per_side"],
    num_slices_head_on=config_bb["num_slices_head_on"],
    harmonic_number=35640,
    bunch_spacing_buckets=config_bb["bunch_spacing_buckets"],
    sigmaz=config_bb["sigma_z"],
)

# ==================================================================================================
# ---Knobs and tuning
# ==================================================================================================
# Build trackers
collider.build_trackers()

# Read knobs and tuning settings from config file
conf_knobs_and_tuning = config_collider["config_knobs_and_tuning"]

# Set all knobs (crossing angles, dispersion correction, rf, crab cavities,
# experimental magnets, etc.)
for kk, vv in conf_knobs_and_tuning["knob_settings"].items():
    collider.vars[kk] = vv

# Tunings
for line_name in ["lhcb1", "lhcb2"]:
    knob_names = conf_knobs_and_tuning["knob_names"][line_name]

    targets = {
        "qx": conf_knobs_and_tuning["qx"][line_name],
        "qy": conf_knobs_and_tuning["qy"][line_name],
        "dqx": conf_knobs_and_tuning["dqx"][line_name],
        "dqy": conf_knobs_and_tuning["dqy"][line_name],
    }

    xm.machine_tuning(
        line=collider[line_name],
        enable_closed_orbit_correction=True,
        enable_linear_coupling_correction=True,
        enable_tune_correction=True,
        enable_chromaticity_correction=True,
        knob_names=knob_names,
        targets=targets,
        line_co_ref=collider[line_name + "_co_ref"],
        co_corr_config=conf_knobs_and_tuning["closed_orbit_correction"][line_name],
    )

# ==================================================================================================
# --- Compute the number of collisions in the different IPs (used for luminosity leveling)
# ==================================================================================================

# Get the filling scheme path (in json or csv format)
filling_scheme_path = config_bb["mask_with_filling_pattern"]["pattern_fname"]

# Load the filling scheme
if filling_scheme_path.endswith(".json"):
    with open(filling_scheme_path, "r") as fid:
        filling_scheme = json.load(fid)
else:
    raise ValueError(
        f"Unknown filling scheme file format: {filling_scheme_path}. It you provided a csv file, it"
        " should have been automatically convert when running the script 001_make_folders.py."
        " Something went wrong."
    )

# Extract booleans beam arrays
array_b1 = np.array(filling_scheme["beam1"])
array_b2 = np.array(filling_scheme["beam2"])

# Assert that the arrays have the required length, and do the convolution
assert len(array_b1) == len(array_b2) == 3564
n_collisions_ip1_and_5 = array_b1 @ array_b2
n_collisions_ip2 = np.roll(array_b1, -891) @ array_b2
n_collisions_ip8 = np.roll(array_b1, -2670) @ array_b2

# Update the configuration file with the number of collisions
configuration["config_collider"]["config_lumi_leveling"]["ip8"][
    "num_colliding_bunches"
] = n_collisions_ip8

# Write the updated configuration file
with open("config.yaml", "w") as file:
    yaml.dump(configuration, file)

# ==================================================================================================
# ---Levelling
# ==================================================================================================
if "config_lumi_leveling" in config_collider and not config_collider["skip_leveling_base_collider"]:
    # Read knobs and tuning settings from config file (already updated with the number of collisions)
    config_lumi_leveling = config_collider["config_lumi_leveling"]

    # Level luminosity
    xlhc.luminosity_leveling(
        collider, config_lumi_leveling=config_lumi_leveling, config_beambeam=config_bb
    )

    # Re-match tunes, and chromaticities
    for line_name in ["lhcb1", "lhcb2"]:
        knob_names = conf_knobs_and_tuning["knob_names"][line_name]
        targets = {
            "qx": conf_knobs_and_tuning["qx"][line_name],
            "qy": conf_knobs_and_tuning["qy"][line_name],
            "dqx": conf_knobs_and_tuning["dqx"][line_name],
            "dqy": conf_knobs_and_tuning["dqy"][line_name],
        }
        xm.machine_tuning(
            line=collider[line_name],
            enable_tune_correction=True,
            enable_chromaticity_correction=True,
            knob_names=knob_names,
            targets=targets,
        )
    else:
        print("No leveling is done as skip_leveling_base_collider is set to True.")
else:
    print(
        "No leveling is done as no configuration has been provided, or skip_leveling_base_collider"
        " is set to True."
    )
# ==================================================================================================
# ---Configure beam-beam
# ==================================================================================================

# The beam-beam configuration will be done in 2_tune_and_track.py, as the collider may have to be
# retuned for parameter scanning, and this must be done before configuring beam-beam.

# ==================================================================================================
# ---Save to json and log result
# ==================================================================================================
os.makedirs("collider", exist_ok=True)
collider.to_json("collider/collider.json")

if tree_maker is not None:
    tree_maker.tag_json.tag_it(configuration["log_file"], "completed")
