# %%
import pyarrow.dataset as ds
from tqdm import tqdm
import gc
import numpy as np
import pandas as pd
import ruamel.yaml
import tree_maker
import cupy as cp

import xmask as xm
import xpart as xp
import xmask.lhc as xlhc
import xobjects as xo
import xtrack as xt

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import json
import time
import itertools

params = {'xtick.labelsize': 28,
'ytick.labelsize': 28,
'font.size': 30,
'figure.autolayout': True,
'figure.figsize': (15, 10),
'axes.titlesize' : 35,
'axes.labelsize' : 35,
'lines.linewidth' : 2,
'lines.markersize' : 0.1,
'legend.fontsize': 28,
'mathtext.fontset': 'stix',
'font.family': 'STIXGeneral'}
plt.rcParams['figure.dpi'] = 100
plt.rcParams.update(params)

plt.rcParams['figure.dpi'] = 100

ft_mine = 20
lw_mine = 2
print('Modules imported')
#%%



def emit_formula_oneturn(data_concatenated, tw0, gamma_rel, betx_rel,turn):
    data_turn = data_concatenated[data_concatenated['at_turn'] == turn]
    sigma_delta = float(np.std(data_turn.pzeta_phys))

    sigma_x = float(np.std(data_turn.x_phys))
    sigma_y = float(np.std(data_turn.y_phys))
    geomx_emittance = (sigma_x**2-(tw0["dx"][0]*sigma_delta)**2)/tw0["betx"][0]
    normx_emittance = geomx_emittance*(gamma_rel*betx_rel)
    geomy_emittance = (sigma_y**2-(tw0["dy"][0]*sigma_delta)**2)/tw0["bety"][0]
    normy_emittance = geomy_emittance*(gamma_rel*betx_rel)
    return normx_emittance, normy_emittance

#files = glob.glob('/eos/user/a/aradosla/SWAN_projects/Separation_adjust/MD_4x36b_optics23_adjust_1m_particles_qgaus/*/*.parquet')
#files = glob.glob('/eos/user/a/aradosla/SWAN_projects/Separation_adjust/MD_4x36b_optics23_adjust_1m_particles_qgaus1.3/*/*.parquet')
files = glob.glob('/eos/user/a/aradosla/SWAN_projects/Separation_adjust/MD_4x36b_optics23_adjust_1m_particles_qgaus1.4/*/*.parquet')

files.sort()
print(len(files))
df = pd.read_parquet(files[0])

collider = xt.Multiline.from_json('/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/collider_final_2025_23_nonoise.json') 
collider.build_trackers()
#Built trackers
num_particles = 10000

nemitt_y = 1.8e-6
nemitt_x = 1.8e-6

num_particles = 5000
particle_ref = xp.Particles(
                        mass0=xp.PROTON_MASS_EV, q0=1, energy0=6800e9) 
#collider['lhcb1'].build_tracker()
particles = xp.generate_matched_gaussian_bunch(
            num_particles = num_particles, total_intensity_particles = 2.3e11,
            nemitt_x = nemitt_x, nemitt_y=nemitt_y, sigma_z = 7.8e-2,
            particle_ref = particle_ref,
            line = collider['lhcb1'])
print('Particles generated')

tw0 = collider['lhcb1'].twiss()
gamma_rel = particle_ref.gamma0
betx_rel = particle_ref.beta0
line = collider['lhcb1']
def collimator(x_data_turn, px_data_turn, tw0):
    survived = 1/2*(tw0.gamx[0]*x_data_turn**2 + 2*tw0.alfx[0]*x_data_turn*px_data_turn + tw0.betx[0]*px_data_turn**2)
    return survived 

def collimatory(x_data_turn, px_data_turn, tw0):
    survived = 1/2*(tw0.gamy[0]*x_data_turn**2 + 2*tw0.alfy[0]*x_data_turn*px_data_turn + tw0.bety[0]*px_data_turn**2)
    return survived 
results = []
lost_particles = set()
sigmax_col = np.sqrt(3.75e-6  / gamma_rel) #* tw0.betx[0])
sigmay_col = np.sqrt(3.75e-6  / gamma_rel) #* tw0.bety[0])
for turn in tqdm(df.at_turn.unique()[:], desc="Turns"):
    combined_df = pd.DataFrame()
    for file in files[:]:
        dataset = ds.dataset(file, format="parquet")
        table = dataset.to_table(filter=(ds.field("at_turn") == turn))
        df_chunk = table.to_pandas()
        if df_chunk.empty:
            continue
        df_chunk['source_file'] = file
        combined_df = pd.concat([combined_df, df_chunk], ignore_index=True)

    # Exclude already lost particles
    combined_df = combined_df[~combined_df.particles_id_all.isin(lost_particles)]
    if combined_df.empty:
        continue

    # Collimator cut
    survived_x = collimator(combined_df.x_phys, combined_df.px_phys, tw0).values < (6 * sigmax_col) ** 2
    survived_y = collimatory(combined_df.y_phys, combined_df.py_phys, tw0).values < (6 * sigmay_col) ** 2
    survived = survived_x & survived_y

    # Track lost
    lost_particles.update(combined_df[~survived].particles_id_all.tolist())

    df_survived = combined_df[survived]
    if df_survived.empty:
        continue

    assert df_survived['at_turn'].nunique() == 1, "Mixed turns found in survived particles!"

    try:
        normx_emit, normy_emit = emit_formula_oneturn(df_survived, tw0, gamma_rel, betx_rel, turn)
    except Exception as e:
        print(f"Error computing emittance at turn {turn}: {e}")
        continue

    results.append({
        'turn': turn,
        'normx_emittance': normx_emit,
        'normy_emittance': normy_emit,
        'beam_survived': len(df_survived),
    })

    # Clean up memory
    del combined_df, df_survived, survived_x, survived_y, survived
    gc.collect()

# Save results
df_out = pd.DataFrame(results).sort_values('turn')
#df_out.to_parquet("/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/emittance_and_losses_qgaussian6sigma.parquet", engine="pyarrow")  # this is the q = 1.5 
#df_out.to_parquet("/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/emittance_and_losses_qgaussian6sigma1.4.parquet", engine="pyarrow")  # this is the q = 1.3

print("✅ Done processing all chunks across all files.")
quit()


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import glob

turn_num = 70
# Load data
#df_out = pd.read_parquet('/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/emittance_and_losses_gaussian6sigma.parquet')
#df_out = pd.read_parquet('/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/emittance_and_losses_qgaussian6sigma1.3_new.parquet')
df_out = pd.read_parquet('/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/emittance_and_losses_qgaussian6sigma1.3.parquet')

# Calculate percentage of survived particles relative to turn 33
perc = np.array(df_out.beam_survived[turn_num:]) / np.array(df_out.beam_survived[turn_num]) * 100


# %%
fig, ax = plt.subplots()
ax.plot(df_out.turn[0:-turn_num],df_out.normx_emittance[turn_num:])
ax.plot(df_out.turn[0:-turn_num],df_out.normy_emittance[turn_num:])
ax.set_xlabel('Turn number')
ax.set_ylabel('Survived particles [%]')
ax1 = ax.twiny()
ax1.plot(np.linspace(0, 0.55, len(df_out.turn[33:])), df_out.normx_emittance[turn_num:])
ax1.set_xlabel('IP 1/5 separation [mm]')
# %%

fig, ax = plt.subplots()

# X-axis: IP separation
x_vals = np.linspace(0, 0.55, len(df_out.turn[turn_num:]))

# Y1: Survival %
perc = np.array(df_out.beam_survived[turn_num:]) / df_out.beam_survived[turn_num] * 100
line1, = ax.plot(x_vals, perc, label='Survival %', color='black', lw=3)

ax.set_xlabel('IP 1/5 separation [mm]')
ax.set_ylabel('Survived particles [%]')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
ax.grid(True)

# Y2: Normalized horizontal emittance
ax1 = ax.twinx()
line2, = ax1.plot(x_vals, df_out.normx_emittance[turn_num:], label=r'$\epsilon_x$', color='tab:blue')
ax1.set_ylabel('Normalized horizontal emittance [μm]')

# Combine legends
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax.legend(lines, labels, loc='best')


# %%
fig, ax = plt.subplots()

fill_nb = 10709
files = glob.glob(f'/eos/project-l/lhc-lumimod/LuminosityFollowUp/2025/rawdata/HX:FILLN={fill_nb}/HX:BMODE=ADJUST/*.parquet')
df = pd.read_parquet(files, engine='pyarrow')
keys = df.keys()
df_gaus = pd.read_parquet('/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/emittance_and_losses_gaussian6sigma.parquet')
df_qgaus = pd.read_parquet('/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/emittance_and_losses_qgaussian6sigma.parquet')
df_qgaus1 = pd.read_parquet('/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/emittance_and_losses_qgaussian6sigma1.4.parquet')

perc = np.array(df_gaus.beam_survived[turn_num:]) / np.array(df_gaus.beam_survived[turn_num]) * 100
percq = np.array(df_qgaus.beam_survived[0][turn_num:]) / np.array(df_qgaus.beam_survived[0][turn_num]) * 100
percq1 = np.array(df_qgaus1.beam_survived[turn_num:]) / np.array(df_qgaus1.beam_survived[turn_num]) * 100

#print(df_out.turn[turn_num], df_out.turn[-1])
# Plot survival percentage vs turn (normal direction)
#ax.plot(df_out.turn[:-turn_num], perc, label='Survival %', color='tab:blue')
ax.set_xlabel('Turns')
ax.set_ylabel('Survived particles [%]')
#ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%2f'))

# Twin x-axis (top) — reverse x and y for alignment
ax1 = ax.twiny()
x_ip_sep = np.linspace(0.007, 0.527, len(df_out.turn[turn_num:]))
x_ip_sep_DA = np.linspace(0.007, 0.527, len(df['LHC.BLM.LIFETIME:B1_BEAM_LIFETIME'].dropna()[turn_num:187]))
# Reverse both x and y to match the reversed axis

line1,  = ax1.plot(x_ip_sep[::-1], perc, label='Gaussian q = 1.0 at 6$\sigma$', alpha=1, lw=5)
line2, = ax1.plot(x_ip_sep[::-1], percq, label = 'qGaussian q = 1.5 at 6$\sigma$', lw = 5)
line3,  = ax1.plot(x_ip_sep[1:][::-1], percq1, label='qGaussian q = 1.4 at 6$\sigma$', alpha=1, lw=5)

ax2 = ax1.twinx()
line4, = ax2.plot(x_ip_sep_DA[::-1], 100*np.exp(-1/df['LHC.BLM.LIFETIME:B1_BEAM_LIFETIME'].dropna()[turn_num:187]), label = 'BLM lifetime', color = 'green', lw = 5, linestyle = '--')
ax2.set_ylabel('Intensity from BLM Lifetime [%]')
# Explicitly set xlim to make axis go from 0.55 to 0
ax1.set_xlim(0.527,0.007)
ax1.set_xlabel('IP 1/5 separation [mm]')
ax.set_xlim(0, df_out.turn[1537-turn_num])


lines = [line1, line2, line3, line4]
labels = [line.get_label() for line in lines]
ax.legend(lines, labels, loc='lower left')

plt.title('Beam Survival ADJUST')
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots()
ax.plot(pd.to_datetime(df['LHC.BLM.LIFETIME:B1_BEAM_LIFETIME'].dropna().index[52:185]),df['LHC.BLM.LIFETIME:B1_BEAM_LIFETIME'].dropna()[52:])
ax1 = ax.twinx()
ax1.plot(pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP5-SEP-H-MM:target'].dropna().index[10:]),df['LhcStateTracker:LHCBEAM:IP5-SEP-H-MM:target'].dropna()[10:])
# %%
#Plot cross section

# %%

# %%
from datetime import datetime
file = '/eos/project/l/lhc-lumimod/LuminosityFollowUp/2025/cycledata/HX:FILLN=10739/EffXsection_DBLM.parquet'
dff_c = pd.read_parquet(file, engine = 'pyarrow')

bunches =   [ 20,
        500,
        750,
        1250,
        1800,
        2250,
        2874
    ]  # from the idx_b1_b2.json
# Expand B1 as before
b1_expanded = dff_c['B1 effective cross section'].apply(pd.Series)
b1_expanded.index = dff_c.index
b1_expanded.columns = [f"B1_part_{i}" for i in b1_expanded.columns]

# Select only the components you want to plot
selected_indices = bunches
b1_expanded_selected = b1_expanded[[f"B1_part_{i}" for i in selected_indices]]

# Plot
b1_expanded_selected.plot(figsize=(15,8))
plt.xlabel("Time")
plt.ylabel("B1 effective cross section")
#plt.title("Selected B1 components over time")
#plt.legend(title="Component")
plt.ylim(1,300)
#plt.xlim(datetime(2025, 6, 20, 1, 50), datetime(2025, 6, 20, 1, 52))
#plt.yscale('log')
plt.show()
# %%blob:vscode-webview://1nom1j6jbick0j4nhhk38pvi83lnt18h3f4pnd8rfqpm8b5786mn/27d6e3f9-65f0-4222-93ae-6a0d00c74b98
