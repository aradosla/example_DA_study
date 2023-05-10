# ==================================================================================================
# --- Imports
# ==================================================================================================
from cpymad.madx import Madx
import xtrack as xt
import os
import xmask as xm
import xmask.lhc as xlhc
import shutil
import json

# Import user-defined optics-specific tools
from tools import optics_specific_tools_hlhc15 as ost

# ==================================================================================================
# --- Build collider from mad model
# ==================================================================================================

# Read config file
with open("config.yaml", "r") as fid:
    config = xm.yaml.load(fid)
config_mad_model = config["config_mad"]

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
config_bb = config["config_beambeam"]

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
with open("config.yaml", "r") as fid:
    config = xm.yaml.load(fid)
conf_knobs_and_tuning = config["config_knobs_and_tuning"]

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
# ---Levelling
# ==================================================================================================
# Build trackers
# collider.build_trackers()

# Read knobs and tuning settings from config file
config_lumi_leveling = config["config_lumi_leveling"]

xlhc.luminosity_leveling(
    collider, config_lumi_leveling=config_lumi_leveling, config_beambeam=config_bb
)

# Re-match tunes, and chromaticities
conf_knobs_and_tuning = config["config_knobs_and_tuning"]

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
    
# ==================================================================================================
# ---Configure beam-beam
# ==================================================================================================

# collider.build_trackers()

# Configure beam-beam lenses
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
# ---Save to json
# ==================================================================================================
collider.to_json("collider/collider_tuned_and_leveled_bb_on.json")