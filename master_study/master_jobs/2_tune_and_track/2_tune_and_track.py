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
    configuration_collider = yaml.safe_load(fid)["config_collider"]

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

# Read initial levelling settings from base collider configuration file, if levelling is done
if "config_lumi_leveling" in configuration_collider and not configuration_collider["skip_leveling"]:
    # Read knobs and tuning settings from config file
    config_lumi_leveling = configuration_collider["config_lumi_leveling"]
else:
    config_lumi_leveling = None


# Read inital beam-beam settings from base collider configuration file
config_bb = configuration_collider["config_beambeam"]

# Check if some parameters involve recomputing levelling (group 1)
start_from_levelling = False
for parameter_name, value in configuration_sim["parameters_scanned"]["group_1"].items():
    if value is not None:
        # Separation
        if parameter_name in conf_knobs_and_tuning["knob_settings"]:
            conf_knobs_and_tuning["knob_settings"][parameter_name] = value
            start_from_levelling = True
        # Bunch intensity
        elif parameter_name in config_bb:
            config_bb[parameter_name] = value
            start_from_levelling = True
        else:
            raise ValueError(
                f"The parameter {parameter_name} is assumed to be a knob belonging to"
                " conf_knobs_and_tuning['knob_settings']. Please update script accordingly."
            )
# Check if some parameters involve redoing the tuning (group 2)
start_from_tuning = False
for parameter_name, value in configuration_sim["parameters_scanned"]["group_2"].items():
    if value is not None:
        # Octupoles
        if parameter_name in conf_knobs_and_tuning["knob_settings"]:
            conf_knobs_and_tuning["knob_settings"][parameter_name] = value
            start_from_tuning = True
        # Tune and chroma
        elif parameter_name in conf_knobs_and_tuning:
            if "lhcb1" in conf_knobs_and_tuning[parameter_name]:
                conf_knobs_and_tuning[parameter_name]["lhcb1"] = value
                conf_knobs_and_tuning[parameter_name]["lhcb2"] = value
            else:
                conf_knobs_and_tuning[parameter_name] = value
            start_from_tuning = True
        else:
            raise ValueError(
                f"The parameter {parameter_name} is assumed to be a knob belonging to"
                " conf_knobs_and_tuning['knob_settings'] or conf_knobs_and_tuning. Please update"
                " script accordingly."
            )

# Check if some parameters involve resetting the beam-beam mask (group 3)
for parameter_name, value in configuration_sim["parameters_scanned"]["group_3"].items():
    if value is not None:
        # Bunch index
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
collider = xt.Multiline.from_json(configuration_sim["collider_file"])
collider.build_trackers()

if start_from_levelling:
    if config_lumi_leveling is not None:
        ### Compute levelling
        xlhc.luminosity_leveling(
            collider, config_lumi_leveling=config_lumi_leveling, config_beambeam=config_bb
        )
    else:
        print("WARNING: no levelling is being done, check that this is indeed what you want.")


# Reset knobs that might have been modified (e.g. octupoles, crossing-angle, etc)
# Knobs that have not been modified are not in configuration_sim["parameters_scanned"] and are
# therefore left untouched
# ! If adding other knobs to group_2, ensure that re-setting the knobs do no require a recomputing of the leveling
for kk, vv in conf_knobs_and_tuning["knob_settings"].items():
    if kk in configuration_sim["parameters_scanned"]["group_2"]:
        collider.vars[kk] = vv

# Add linear coupling as the target in the tuning of the base collider was 0
# (not possible to set it the target to 0.001 for now)
collider.vars["c_minus_re_b1"] += conf_knobs_and_tuning["delta_cmr"]
collider.vars["c_minus_re_b2"] += conf_knobs_and_tuning["delta_cmr"]

# Since we might have updated some knobs, we need to rematch tune and chromaticity
# This would been done in any case as we need to rematch after changing the linear coupling
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
        enable_linear_coupling_correction=False,
        knob_names=knob_names,
        targets=targets,
    )

# ==================================================================================================
# --- Assert that tune, chromaticity and linear coupling are correct before going further
# ==================================================================================================
for line_name in ["lhcb1", "lhcb2"]:
    tw = collider[line_name].twiss()
    assert np.isclose(tw.qx, conf_knobs_and_tuning["qx"][line_name], rtol=1e-3), (
        f"tune_x is not correct for {line_name}. Expected {conf_knobs_and_tuning['qx'][line_name]},"
        f" got {tw.qx}"
    )
    assert np.isclose(tw.qy, conf_knobs_and_tuning["qy"][line_name], rtol=1e-3), (
        f"tune_y is not correct for {line_name}. Expected {conf_knobs_and_tuning['qy'][line_name]},"
        f" got {tw.qy}"
    )
    assert np.isclose(
        tw.dqx,
        conf_knobs_and_tuning["dqx"][line_name],
        rtol=1e-3,
    ), (
        f"chromaticity_x is not correct for {line_name}. Expected"
        f" {conf_knobs_and_tuning['dqx'][line_name]}, got {tw.dqx}"
    )
    assert np.isclose(
        tw.dqy,
        conf_knobs_and_tuning["dqy"][line_name],
        rtol=1e-3,
    ), (
        f"chromaticity_y is not correct for {line_name}. Expected"
        f" {conf_knobs_and_tuning['dqy'][line_name]}, got {tw.dqy}"
    )
    assert np.isclose(
        tw.c_minus,
        conf_knobs_and_tuning["delta_cmr"],
        rtol=1e-1,
    ), (
        f"linear coupling is not correct for {line_name}. Expected"
        f" {conf_knobs_and_tuning['delta_cmr']}, got {tw.c_minus}"
    )

# ==================================================================================================
# --- Configure beam-beam
# ==================================================================================================
print("Configuring beam-beam lenses...")
collider.configure_beambeam_interactions(
    num_particles=config_bb["num_particles_per_bunch"],
    nemitt_x=config_bb["nemitt_x"],
    nemitt_y=config_bb["nemitt_y"],
)

# Configure filling scheme mask and bunch numbers
if "mask_with_filling_pattern" in config_bb:
    # Initialize filling pattern with empty values
    filling_pattern_cw = None
    filling_pattern_acw = None

    # Initialize bunch numbers with empty values
    i_bunch_cw = None
    i_bunch_acw = None

    if "pattern_fname" in config_bb["mask_with_filling_pattern"]:
        # Fill values if possible
        if config_bb["mask_with_filling_pattern"]["pattern_fname"] is not None:
            fname = config_bb["mask_with_filling_pattern"]["pattern_fname"]
            with open(fname, "r") as fid:
                filling = json.load(fid)
            filling_pattern_cw = filling["beam1"]
            filling_pattern_acw = filling["beam2"]

            # Only track bunch number if a filling pattern has been provided
            if "i_bunch_b1" in config_bb["mask_with_filling_pattern"]:
                i_bunch_cw = config_bb["mask_with_filling_pattern"]["i_bunch_b1"]
            if "i_bunch_b2" in config_bb["mask_with_filling_pattern"]:
                i_bunch_acw = config_bb["mask_with_filling_pattern"]["i_bunch_b2"]

            # Note that a bunch number must be provided if a filling pattern is provided
            # Apply filling pattern
            collider.apply_filling_pattern(
                filling_pattern_cw=filling["beam1"],
                filling_pattern_acw=filling["beam2"],
                i_bunch_cw=i_bunch_cw,
                i_bunch_acw=i_bunch_acw,
            )

# ==================================================================================================
# --- Save the final collider before tracking
# ==================================================================================================
collider.to_json("final_collider.json")

# ==================================================================================================
# --- Prepare particles distribution for tracking
# ==================================================================================================
particle_df = pd.read_parquet(configuration_sim["particle_file"])

r_vect = particle_df["normalized amplitude in xy-plane"].values
theta_vect = particle_df["angle in xy-plane [deg]"].values * np.pi / 180  # [rad]

A1_in_sigma = r_vect * np.cos(theta_vect)
A2_in_sigma = r_vect * np.sin(theta_vect)

# Assess that emittances in collider and in simulation configuration files are identical
if not np.allclose(
    configuration_sim["epsn_1"], config_bb["nemitt_x"], rtol=1e-3
) or not np.allclose(configuration_sim["epsn_2"], config_bb["nemitt_y"], rtol=1e-3):
    raise ValueError(
        "The emittances in the simulation configuration file and in the beam-beam configuration"
        " file are not identical. Please update script accordingly."
    )

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
# Optimize line for tracking
collider.lhcb1.optimize_for_tracking()

# Save initial coordinates
pd.DataFrame(particles.to_dict()).to_parquet("input_particles.parquet")

# Track
num_turns = configuration_sim["n_turns"]
a = time.time()
collider.lhcb1.track(particles, turn_by_turn_monitor=False, num_turns=num_turns)
b = time.time()

print(f"Elapsed time: {b-a} s")
print(f"Elapsed time per particle per turn: {(b-a)/particles._capacity/num_turns*1e6} us")

# ==================================================================================================
# --- Save output
# ==================================================================================================
pd.DataFrame(particles.to_dict()).to_parquet("output_particles.parquet")

if tree_maker is not None and "log_file" in configuration_sim:
    tree_maker.tag_json.tag_it(configuration_sim["log_file"], "completed")
