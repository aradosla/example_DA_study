# %%
# Import third-party modules
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

# %%
fill_nb = 10739
files = glob.glob(f'/eos/project-l/lhc-lumimod/LuminosityFollowUp/2025/rawdata/HX:FILLN={fill_nb}/HX:BMODE=ADJUST/*.parquet')
#files = glob.glob(f'/eos/project-l/lhc-lumimod/LuminosityFollowUp/2024/rawdata/HX:FILLN={10104}/HX:BMODE=SQUEEZE/*.parquet')
df = pd.read_parquet(files, columns = ['LhcStateTracker:LHCBEAM:IP1-SEP-V-MM:target', 'LhcStateTracker:LHCBEAM:IP2-SEP-H-MM:target', 'LhcStateTracker:LHCBEAM:IP5-SEP-H-MM:target', 'LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'], engine='pyarrow')
# %%
'''
fig, axs = plt.subplots(figsize = (10, 6))
plt.plot(pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP1-SEP-V-MM:target'].dropna().index), df['LhcStateTracker:LHCBEAM:IP1-SEP-V-MM:target'].dropna().values, label = 'LhcStateTracker:LHCBEAM:IP1-SEP-V-MM:target', lw = 5)


plt.plot(pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP5-SEP-H-MM:target'].dropna().index), df['LhcStateTracker:LHCBEAM:IP5-SEP-H-MM:target'].dropna().values, label = 'LhcStateTracker:LHCBEAM:IP5-SEP-H-MM:target', lw = 2 )

plt.plot(pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP2-SEP-H-MM:target'].dropna().index), df['LhcStateTracker:LHCBEAM:IP2-SEP-H-MM:target'].dropna().values, label = 'LhcStateTracker:LHCBEAM:IP2-SEP-H-MM:target', lw = 5)

plt.plot(pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'].dropna().index), df['LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'].dropna().values, label = 'LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target', lw = 5)
date_format = mdates.DateFormatter('%H:%M:%S')
axs.xaxis.set_major_formatter(date_format)
axs.tick_params(axis='both', which='major', labelsize=20)
axs.set_xlabel(f"UTC time {pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'].dropna().index[0]).day}/{pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'].dropna().index[0]).month}/{pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'].dropna().index)[0].year}", size = 20)

plt.ylabel('Separation [mm]', size = 20)
plt.legend()
plt.tick_params(labelsize = 18)
plt.grid(True)


fig, axs = plt.subplots(figsize = (10, 6))
plt.plot(pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP1-SEP-V-MM:target'].dropna().index), df['LhcStateTracker:LHCBEAM:IP1-SEP-V-MM:target'].dropna().values, '.', label = 'LhcStateTracker:LHCBEAM:IP1-SEP-V-MM:target', lw = 5)


plt.plot(pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP5-SEP-H-MM:target'].dropna().index), df['LhcStateTracker:LHCBEAM:IP5-SEP-H-MM:target'].dropna().values, '.', label = 'LhcStateTracker:LHCBEAM:IP5-SEP-H-MM:target', lw = 2 )

plt.plot(pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP2-SEP-H-MM:target'].dropna().index), df['LhcStateTracker:LHCBEAM:IP2-SEP-H-MM:target'].dropna().values, '.', label = 'LhcStateTracker:LHCBEAM:IP2-SEP-H-MM:target', lw = 5)

plt.plot(pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'].dropna().index), df['LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'].dropna().values, '.', label = 'LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target')
date_format = mdates.DateFormatter('%H:%M:%S')
axs.xaxis.set_major_formatter(date_format)
axs.tick_params(axis='both', which='major', labelsize=20)
axs.set_xlabel(f"UTC time {pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'].dropna().index[0]).day}/{pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'].dropna().index[0]).month}/{pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'].dropna().index)[0].year}", size = 20)

plt.ylabel('Separation [mm]', size = 20)
plt.legend(fontsize = 'xx-large')
plt.tick_params(labelsize = 18)
plt.grid(True)
'''
import numpy as np

def linear_fit(x, y):
    a, b = np.polyfit(x, y, 1)
    y_fit = a * np.array(x) + b
    return a, b, y_fit
fig, axs = plt.subplots(figsize = (10, 6))

min_val = 12
max_val = 61
all_fits_df = pd.DataFrame()
for key in df.keys():
    if 'LhcStateTracker:LHCBEAM:IP1-SEP-V-MM:target' in key or 'LhcStateTracker:LHCBEAM:IP5-SEP-H-MM:target' in key or 'LhcStateTracker:LHCBEAM:IP2-SEP-H-MM:target' in key or 'LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target' in key:
        x = pd.to_datetime(df[key].dropna().index[min_val:max_val]).astype(np.int64) // 10**9  # Convert to seconds
        y = df[key].dropna().values[min_val:max_val]
        a, b, y_fit = linear_fit(x, y)
        print(f"Key: {key}, Slope (a): {a}, Intercept (b): {b}")
        plt.plot(pd.to_datetime(df[key].dropna().index[min_val:max_val]), y_fit, label=f"Fit: {key}")
        all_fits_df[key] = pd.Series(data=y_fit)   # Here are all the interpolated values during separation
        #interpolated_values = pd.DataFrame({f'{key}': y_fit})


plt.plot(pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP1-SEP-V-MM:target'].dropna().index), df['LhcStateTracker:LHCBEAM:IP1-SEP-V-MM:target'].dropna().values, '.', label = 'LhcStateTracker:LHCBEAM:IP1-SEP-V-MM:target', lw = 5)


plt.plot(pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP5-SEP-H-MM:target'].dropna().index), df['LhcStateTracker:LHCBEAM:IP5-SEP-H-MM:target'].dropna().values, '.', label = 'LhcStateTracker:LHCBEAM:IP5-SEP-H-MM:target', lw = 2 )

plt.plot(pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP2-SEP-H-MM:target'].dropna().index), df['LhcStateTracker:LHCBEAM:IP2-SEP-H-MM:target'].dropna().values, '.', label = 'LhcStateTracker:LHCBEAM:IP2-SEP-H-MM:target', lw = 5)

plt.plot(pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'].dropna().index), df['LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'].dropna().values, '.', label = 'LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target')
date_format = mdates.DateFormatter('%H:%M:%S')
axs.xaxis.set_major_formatter(date_format)
axs.tick_params(axis='both', which='major', labelsize=20)
axs.set_xlabel(f"UTC time {pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'].dropna().index[0]).day}/{pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'].dropna().index[0]).month}/{pd.to_datetime(df['LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'].dropna().index)[0].year}", size = 20)

plt.ylabel('Separation [mm]', size = 20)
plt.legend()
plt.tick_params(labelsize = 18)
plt.grid(True)



# %%
#collider = xt.Multiline.from_json('/eos/user/a/aradosla/SWAN_projects/Colliders/extracted_correct_energy/collider20.json')
collider = xt.Multiline.from_json('/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/collider_final_2025_23_nonoise.json') # Collider from 1, no beam-beam effects, no errors etc. oct, chroma need to be added
# %%

# Check these and see about the long ranges!!!
#collider.vars['i_oct_b1'] = -500
#collider.vars['i_oct_b2'] = -500
#collider.vars['dqx'] = 20
#collider.vars['dqy'] = 20
keys = df.keys()
keys
#collider['lhcb1'][keys[1]]
#df[keys[1]].dropna()
# %%
nemitt_y = 1.8e-6
nemitt_x = 1.8e-6

num_particles = 100
particle_ref = xp.Particles(
                        mass0=xp.PROTON_MASS_EV, q0=1, energy0=6800e9) 
collider['lhcb1'].build_tracker()

particles = xp.generate_matched_gaussian_bunch(
            num_particles = num_particles, total_intensity_particles = 2.3e11,
            nemitt_x = nemitt_x, nemitt_y=nemitt_y, sigma_z = 7.8e-2,
            particle_ref = particle_ref,
            line = collider['lhcb1'])
'''
# Define radius distribution
r_min = 0
r_max = 10
#n_r =  2 * 16 * (r_max - r_min)
n_r = 30
radial_list = np.linspace(r_min, r_max, n_r, endpoint=False)

# Filter out particles with low and high amplitude to accelerate simulation
# radial_list = radial_list[(radial_list >= 4.5) & (radial_list <= 7.5)]

# Define angle distribution
n_angles = 40
theta_list = np.linspace(0, 90, n_angles + 2)[1:-1]
#theta_list = pd.DataFrame(theta_list)

# Define particle distribution as a cartesian product of the above
particle_list = [
    (particle_id, ii[1], ii[0])
    for particle_id, ii in enumerate(itertools.product(theta_list, radial_list))
]

particle_df = pd.DataFrame(particle_list, columns=["particle_id", "normalized amplitude in xy-plane", "angle in xy-plane [deg]"])


r_vect = particle_df["normalized amplitude in xy-plane"].values
theta_vect = particle_df["angle in xy-plane [deg]"].values * np.pi / 180  # type: ignore # [rad]

A1_in_sigma = r_vect * np.cos(theta_vect)
A2_in_sigma = r_vect * np.sin(theta_vect)

particles = collider['lhcb1'].build_particles(
    x_norm=A1_in_sigma,
    y_norm=A2_in_sigma,
    delta=0.,
    nemitt_x=(1.8e-6),
    nemitt_y=(1.8e-6)
)
'''
tw0 = collider['lhcb1'].twiss()
gamma_rel = particle_ref.gamma0
betx_rel = particle_ref.beta0

# %%

all_fits_df_used = all_fits_df[:] # Use every second row for the knobs
knobs = 10  # how many turns to apply the knob change
num_turns = knobs * (len(all_fits_df_used[keys[0]]) - 1) # total number of turns to track
ndata = 10 # how many turns to store the data
print(num_turns)
norm_intervals = num_turns // ndata
x_phys = np.zeros((norm_intervals, num_particles), dtype=cp.float64)
y_phys = np.zeros((norm_intervals, num_particles), dtype=cp.float64)
zeta_phys = np.zeros((norm_intervals, num_particles), dtype=cp.float64)
px_phys = np.zeros((norm_intervals, num_particles), dtype=cp.float64)
py_phys = np.zeros((norm_intervals, num_particles), dtype=cp.float64)
pzeta_phys = np.zeros((norm_intervals, num_particles), dtype=cp.float64)
state_all = np.zeros((norm_intervals, num_particles), dtype=cp.int32)
particles_id_all = np.zeros((norm_intervals, num_particles), dtype=cp.int32)
turns_totnorm = np.zeros((norm_intervals, num_particles), dtype=cp.int32)
# %%
knob_names = ['on_sep1', 'on_sep2h', 'on_sep5', 'on_sep8h']
df_fit = pd.read_parquet('/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/fits_sep_adjust.parquet')
keys = df_fit.keys()
for i in range(num_turns):
    c = time.time()
    
    collider['lhcb1'].track(particles, num_turns=1, turn_by_turn_monitor=True, freeze_longitudinal=False)

    ####
    if (i + 1) % knobs == 0:
        for kk in range(len(knob_names)):
            collider.vars[knob_names[kk]] = df_fit[keys[kk]].dropna().values[(i + 1) // knobs]

            if knob_names[kk] == 'on_sep1':
                collider.vars[knob_names[kk]] = - df_fit[keys[kk]].dropna().values[(i + 1) // knobs]
            #print(df[keys[kk]].dropna().values[(i + 1) // knobs])
            print(f'Applying knob {knob_names[kk]} from {keys[kk]} with value {df_fit[keys[kk]].dropna().values[(i + 1) // knobs]}')
            #collider.vars[keys[kk]] = df[keys[kk]].dropna().values[-1]
    ####

    if False:
        if (i + 1) % knobs == 0:
            for kk in range(len(keys)):
                #print(df[keys[kk]].dropna().values[(i + 1) // knobs])
                print(f'Applying knob {keys[kk]} with value {all_fits_df[keys[kk]].dropna().values[(i + 1) // knobs]}')
                #collider.vars[keys[kk]] = df[keys[kk]].dropna().values[-1]
                collider.vars[keys[kk]] = all_fits_df[keys[kk]].dropna().values[(i + 1) // knobs]

    # Store particle data every 1000 turns
    if (i + 1) % ndata == 0:
        interval_index = (i + 1) // ndata - 1

        # Transfer from GPU (cupy) to CPU (numpy)
        x_phys[interval_index, :] = particles.x
        y_phys[interval_index, :] = particles.y
        zeta_phys[interval_index, :] = particles.zeta
        px_phys[interval_index, :] = particles.px
        pzeta_phys[interval_index, :] = particles.delta
        state_all[interval_index, :] = particles.state
        turns_totnorm[interval_index, :] = particles.at_turn
        particles_id_all[interval_index, :] = particles.particle_id

        d = time.time()
        print(f'Turn {i+1}, time {d-c}s')


# %%
x_data = x_phys
y_data = y_phys
px_data = px_phys
py_data = py_phys
zeta_data = zeta_phys
pzeta_data = pzeta_phys
turns = turns_totnorm
particle_ids = particles_id_all
state = state_all

#Convert results to DataFrame
dff = pd.DataFrame({
    "x_phys": x_data.flatten(),
    "y_phys": y_data.flatten(),
    "zeta_phys": zeta_data.flatten(),
    "px_phys": px_data.flatten(),
    "py_phys": py_data.flatten(),
    "pzeta_phys": pzeta_data.flatten(),
    "at_turn": turns.flatten(),
    "particle_id_all": particle_ids.flatten(),
    "state": state.flatten()

})

#dff.to_parquet('/eos/user/a/aradosla/SWAN_projects/IBS/collider_2025_23_separation_adjust.parquet', index=False)
# %%
tw0 = collider['lhcb1'].twiss()
#dff = pd.read_parquet('/eos/user/a/aradosla/SWAN_projects/IBS/collider_2025_23_separation_adjust.parquet', engine='pyarrow')
dff = pd.read_parquet(f'/eos/user/a/aradosla/SWAN_projects/Separation_adjust/collider_2025_23_separation_adjust_32knobs_5000p3.parquet', engine='pyarrow')

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

# %%
normx_all = []
normy_all = []
for turn in dff.at_turn.unique()[::1]:
    norm_x, norm_y = emit_formula_oneturn(dff, tw0, gamma_rel, betx_rel, turn)
    normx_all.append(norm_x)
    normy_all.append(norm_y)
# %%
plt.plot(dff.at_turn.unique()[::1], normx_all, label = 'Norm. x emittance')
plt.plot(dff.at_turn.unique()[::1], normy_all, label = 'Norm. y emittance', alpha = 0.6)
plt.xlabel('Turn')
plt.ylabel('Normalized Emittance [m rad]')
plt.legend()
plt.show()
# %%
nemitt_y = 1.8e-6
nemitt_x = 1.8e-6

num_particles = 10000
particle_ref = xp.Particles(
                        mass0=xp.PROTON_MASS_EV, q0=1, energy0=6800e9) 
collider['lhcb1'].build_tracker()

particles = xp.generate_matched_gaussian_bunch(
            num_particles = num_particles, total_intensity_particles = 2.3e11,
            nemitt_x = nemitt_x, nemitt_y=nemitt_y, sigma_z = 7.8e-2,
            particle_ref = particle_ref,
            line = collider['lhcb1'])
line = collider['lhcb1']
def collimator(x_data_turn, px_data_turn, tw0):
    survived = 1/2*(tw0.gamx[0]*x_data_turn**2 + 2*tw0.alfx[0]*x_data_turn*px_data_turn + tw0.betx[0]*px_data_turn**2)
    return survived 

def collimatory(x_data_turn, px_data_turn, tw0):
    survived = 1/2*(tw0.gamy[0]*x_data_turn**2 + 2*tw0.alfy[0]*x_data_turn*px_data_turn + tw0.bety[0]*px_data_turn**2)
    return survived 

data_concatenated = dff
turns = np.unique(data_concatenated.at_turn)
beam_loss = []
gamx = tw0.gamx[0]
alfx = tw0.alfx[0]
alfy = tw0.alfy[0]
betx = tw0.betx[0]
bety = tw0.bety[0]
gamy = tw0.gamy[0]
print(gamx)
normx_all_std = []
normy_all_std = []
gamma_rel = particles.gamma0[0]
betx_rel = particles.beta0[0]
beam_size = nemitt_x/gamma_rel
beam_sizey = nemitt_y/gamma_rel

#sigmax_col = np.sqrt(3.75e-6  / gamma_rel * tw0.betx[0])
sigmax_col = np.sqrt(3.75e-6  / gamma_rel) #* tw0.betx[0])
sigmay_col = np.sqrt(3.75e-6  / gamma_rel) #* tw0.bety[0])
print(sigmax_col, sigmay_col)
lost_indices = []  # To track all lost particle indices
# %%

for turn in turns[1::1]:
    # Filter data for the current turn
    data_turn = data_concatenated[data_concatenated['at_turn'] == turn]
    
    # Exclude particles lost in previous turns
    data_turn = data_turn[~data_turn.particle_id_all.isin(lost_indices)]
    
    # Calculate the collimator condition for x and y
    survived_condition_x = collimator(data_turn.x_phys, data_turn.px_phys, tw0) < (4.5 * sigmax_col)**2
    survived_condition_y = collimatory(data_turn.y_phys, data_turn.py_phys, tw0) < (4.5 * sigmay_col)**2

    # Combine both conditions
    survived_condition = survived_condition_x & survived_condition_y
    # Track indices of lost particles
    lost_indices.extend(data_turn[~survived_condition].particle_id_all.tolist())
    data_turn_filtered = data_turn[survived_condition]  # Surviving particles only
    #print(data_turn_filtered)
    #normx_emittance, normy_emittance = emit_formula(data_turn_filtered, twiss, gamma_rel, betx_rel)
    
    # Append the results to lists
    #normx_all_std.append(np.array(normx_emittance))
    #normy_all_std.append(np.array(normy_emittance))
    beam_loss.append(len(data_turn_filtered))


# %%
fontsize = 20
states = line.record_last_track.state.T
states_turn = []
for turn in states:
    states_turn.append(sum(turn))
    
plt.figure(figsize = (10, 6))
plt.plot(beam_loss, label = 'Lost due to the collimation 6$\sigma$')
#plt.plot(states_turn, label = 'Lost due to DA') 
plt.xlabel('Turns', size = fontsize)
plt.ylabel('# surviving particles', size = fontsize)
plt.title('Beam Loss Over Turns', size = fontsize+2)
plt.tick_params(labelsize = fontsize-2)
plt.legend(fontsize = 'xx-large')
#plt.ylim(-10, 1050)
plt.grid(True)
# %%

for turn in np.arange(0,100,5)[10:30:2]:
    plt.figure(figsize = (10, 6))
    plt.hist(dff[dff.at_turn == int(turn)].x_phys/np.sqrt(tw0.betx[0]*nemitt_x)*np.sqrt(gamma_rel), bins=100, label = f'Turn {turn}', alpha = 0.5)
    #plt.plot(dff[dff.at_turn == turn].x/np.sqrt(tw0.betx[0]*nemitt_x)*np.sqrt(gamma_rel), dff[dff.at_turn == turn].y/np.sqrt(tw0.bety[0]*nemitt_y)*np.sqrt(gamma_rel), '.')
    plt.xlim(-6,6)
    plt.show()
# %%
for turn in np.arange(0,100,5)[10:30:2]:
    plt.figure(figsize = (10, 6))
    plt.hist(dff[dff.at_turn == int(turn)].y_phys/np.sqrt(tw0.bety[0]*nemitt_y)*np.sqrt(gamma_rel), bins=100, label = f'Turn {turn}', alpha = 0.5)
    #plt.plot(dff[dff.at_turn == turn].x/np.sqrt(tw0.betx[0]*nemitt_x)*np.sqrt(gamma_rel), dff[dff.at_turn == turn].y/np.sqrt(tw0.bety[0]*nemitt_y)*np.sqrt(gamma_rel), '.')
    plt.xlim(-6,6)
    plt.show()
# %%