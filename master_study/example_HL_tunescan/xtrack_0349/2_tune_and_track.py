# ==================================================================================================
# --- Imports
# ==================================================================================================
import json
import yaml
import time
import logging
import numpy as np
import pandas as pd
import xtrack as xt
import xpart as xp
import tree_maker
import xmask as xm
import xmask.lhc as xlhc

# ==================================================================================================
# --- Read configuration files and tag start of the job
# ==================================================================================================
# Read configuration for simulations
with open("config.yaml", "r") as fid:
    configuration_sim = yaml.safe_load(fid)

# Read base collider configuration (one generation above)
with open("../config.yaml", "r") as fid:
    configuration_collider = yaml.safe_load(fid)

# Start tree_maker logging if log_file is present in config
if tree_maker is not None and "log_file" in configuration_sim:
    tree_maker.tag_json.tag_it(configuration_sim["log_file"], "started")
else:
    logging.warning("tree_maker loging not available")

# ==================================================================================================
# --- Check which parameters must be changed (beam-beam is always reconfigured)
# ==================================================================================================
# Read initial knobs and tuning settings from base collider configuration file
conf_knobs_and_tuning = configuration_collider["config_knobs_and_tuning"]

# Read initial levelling settings from base collider configuration file
config_lumi_leveling = configuration_collider["config_lumi_leveling"]

# Read inital beam-beam settings from base collider configuration file
config_bb = configuration_collider["config_beambeam"]

# Check if some parameters involve recomputing levelling (group 1)
start_from_levelling = False
for parameter_name, value in configuration_sim["parameters_scanned"]["group_1"]:
    if value is not None:
        if parameter_name in conf_knobs_and_tuning["knob_settings"]:
            conf_knobs_and_tuning["knob_settings"][parameter_name] = value
            start_from_levelling = True
        else:
            raise ValueError(
                f"The parameter {parameter_name} is assumed to be a knob belonging to"
                " conf_knobs_and_tuning['knob_settings']. Please update script accordingly."
            )
# Check if some parameters involve redoing the tuning (group 2)
start_from_tuning = False
for parameter_name, value in configuration_sim["parameters_scanned"]["group_2"]:
    if value is not None:
        if parameter_name in conf_knobs_and_tuning["knob_settings"]:
            conf_knobs_and_tuning["knob_settings"][parameter_name] = value
            start_from_tuning = True
        elif parameter_name in conf_knobs_and_tuning:
            conf_knobs_and_tuning[parameter_name] = value
            start_from_tuning = True
        else:
            raise ValueError(
                f"The parameter {parameter_name} is assumed to be a knob belonging to"
                " conf_knobs_and_tuning['knob_settings'] or conf_knobs_and_tuning. Please update"
                " script accordingly."
            )

# Check if some parameters involve resetting the beam-beam mask (group 3)
for parameter_name, value in configuration_sim["parameters_scanned"]["group_3"]:
    if value is not None:
        if parameter_name in config_bb["mask_with_filling_pattern"]:
            config_bb["mask_with_filling_pattern"][parameter_name] = value
        else:
            raise ValueError(
                f"The parameter {parameter_name} is assumed to be a knob belonging to"
                " config_bb['mask_with_filling_pattern']. Please update script accordingly."
            )

# ==================================================================================================
# --- Rebuild and tune collider
# ==================================================================================================
# Load collider and build trackers
collider = xt.Multiline.from_json(configuration_sim["collider_json"])
collider.build_trackers()

if start_from_levelling:
    ### Compute levelling
    xlhc.luminosity_leveling(
        collider, config_lumi_leveling=config_lumi_leveling, config_beambeam=config_bb
    )

if start_from_levelling or start_from_tuning:
    # Reset knobs that have been modified (for now, only octupoles are concerned)
    # ! Other knobs may require to recompute luminoisty leveling
    for kk, vv in conf_knobs_and_tuning["knob_settings"].items():
        if kk in configuration_sim["parameters_scanned"]["group_1"]:
            collider.vars[kk] = vv

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

### Configure beam-beam
print("Configuring beam-beam lenses...")
collider.configure_beambeam_interactions(
    num_particles=config_bb["num_particles_per_bunch"],
    nemitt_x=config_bb["nemitt_x"],
    nemitt_y=config_bb["nemitt_y"],
)

if "mask_with_filling_pattern" in config_bb:
    fname = config_bb["mask_with_filling_pattern"]["pattern_fname"]
    i_bunch_cw = config_bb["mask_with_filling_pattern"]["i_bunch_b1"]
    i_bunch_acw = config_bb["mask_with_filling_pattern"]["i_bunch_b2"]
    with open(fname, "r") as fid:
        filling = json.load(fid)

    collider.apply_filling_pattern(
        filling_pattern_cw=filling["beam1"],
        filling_pattern_acw=filling["beam2"],
        i_bunch_cw=i_bunch_cw,
        i_bunch_acw=i_bunch_acw,
    )

# ==================================================================================================
# --- Prepare particles distribution for tracking
# ==================================================================================================
particle_df = pd.read_parquet(configuration_sim["particle_file"])

r_vect = particle_df["normalized amplitude in xy-plane"].values
theta_vect = particle_df["angle in xy-plane [deg]"].values * np.pi / 180  # [rad]

A1_in_sigma = r_vect * np.cos(theta_vect)
A2_in_sigma = r_vect * np.sin(theta_vect)

particles = collider.lhcb1.build_particles(
    x_norm=A1_in_sigma,
    y_norm=A2_in_sigma,
    delta=configuration_sim["delta_max"],
    scale_with_transverse_norm_emitt=(configuration_sim["epsn_1"], configuration_sim["epsn_2"]),
)
particles.particle_id = particle_df.particle_id.values


# ==================================================================================================
# --- Build tracker and track
# ==================================================================================================
# Build tracker and optimize
tracker = xt.Tracker(
    line=collider.lhcb1,
    extra_headers=["#define XTRACK_MULTIPOLE_NO_SYNRAD"],
)
tracker.optimize_for_tracking()

# Save initial coordinates
pd.DataFrame(particles.to_dict()).to_parquet("input_particles.parquet")


# Track
num_turns = configuration_sim["n_turns"]
a = time.time()
tracker.track(particles, turn_by_turn_monitor=False, num_turns=num_turns)
b = time.time()

print(f"Elapsed time: {b-a} s")
print(f"Elapsed time per particle per turn: {(b-a)/particles._capacity/num_turns*1e6} us")

# ==================================================================================================
# --- Save output
# ==================================================================================================
pd.DataFrame(particles.to_dict()).to_parquet("output_particles.parquet")

if tree_maker is not None and "log_file" in configuration_sim:
    tree_maker.tag_json.tag_it(configuration_sim["log_file"], "completed")
