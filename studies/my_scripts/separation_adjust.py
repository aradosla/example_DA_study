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
print('Modules imported')

fill_nb = 10739
files = glob.glob(f'/eos/project-l/lhc-lumimod/LuminosityFollowUp/2025/rawdata/HX:FILLN={fill_nb}/HX:BMODE=ADJUST/*.parquet')
#files = glob.glob(f'/eos/project-l/lhc-lumimod/LuminosityFollowUp/2024/rawdata/HX:FILLN={10104}/HX:BMODE=SQUEEZE/*.parquet')
df = pd.read_parquet(files, columns = ['LhcStateTracker:LHCBEAM:IP1-SEP-V-MM:target', 'LhcStateTracker:LHCBEAM:IP2-SEP-H-MM:target', 'LhcStateTracker:LHCBEAM:IP5-SEP-H-MM:target', 'LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target'], engine='pyarrow')

def linear_fit(x, y):
    a, b = np.polyfit(x, y, 1)
    y_fit = a * np.array(x) + b
    return a, b, y_fit
#fig, axs = plt.subplots(figsize = (10, 6))

min_val = 12
max_val = 61
all_fits_df = pd.DataFrame()
for key in df.keys():
    if 'LhcStateTracker:LHCBEAM:IP1-SEP-V-MM:target' in key or 'LhcStateTracker:LHCBEAM:IP5-SEP-H-MM:target' in key or 'LhcStateTracker:LHCBEAM:IP2-SEP-H-MM:target' in key or 'LhcStateTracker:LHCBEAM:IP8-SEP-H-MM:target' in key:
        x = pd.to_datetime(df[key].dropna().index[min_val:max_val]).astype(np.int64) // 10**9  # Convert to seconds
        y = df[key].dropna().values[min_val:max_val]
        a, b, y_fit = linear_fit(x, y)
        print(f"Key: {key}, Slope (a): {a}, Intercept (b): {b}")
        #plt.plot(pd.to_datetime(df[key].dropna().index[min_val:max_val]), y_fit, label=f"Fit: {key}")
        all_fits_df[key] = pd.Series(data=y_fit)   # Here are all the interpolated values during separation
        #interpolated_values = pd.DataFrame({f'{key}': y_fit})

print('linear fit done')

#collider = xt.Multiline.from_json('/eos/user/a/aradosla/SWAN_projects/Colliders/extracted_correct_energy/collider20.json')
collider = xt.Multiline.from_json('/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/collider_final_2025_23_nonoise.json') # Collider from 1, no beam-beam effects, no errors etc. oct, chroma need to be added
keys = df.keys()
print('Collider loaded')
#collider['lhcb1'][keys[1]]
#df[keys[1]].dropna()

nemitt_y = 1.8e-6
nemitt_x = 1.8e-6

num_particles = 5000
particle_ref = xp.Particles(
                        mass0=xp.PROTON_MASS_EV, q0=1, energy0=6800e9) 
#collider['lhcb1'].build_tracker()
device_number = None
# %%
context = xo.ContextCupy(device = device_number) 
collider.build_trackers(_context=context)
#Built trackers
particles = xp.generate_matched_gaussian_bunch(
            num_particles = num_particles, total_intensity_particles = 2.3e11,
            nemitt_x = nemitt_x, nemitt_y=nemitt_y, sigma_z = 7.8e-2,
            particle_ref = particle_ref,
            line = collider['lhcb1'], _context=context)
print('Particles generated')

tw0 = collider['lhcb1'].twiss()
gamma_rel = particle_ref.gamma0
betx_rel = particle_ref.beta0


all_fits_df_used = all_fits_df[:] # Use every second row for the knobs
knobs = 32000  # how many turns to apply the knob change
num_turns = knobs * (len(all_fits_df_used[keys[0]]) - 1) + 20000 # total number of turns to track
ndata = 1000 # how many turns to store the data
print(num_turns)
norm_intervals = num_turns // ndata
# %%
# Preallocate arrays for physical coordinates
x_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
y_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
zeta_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
px_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
py_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
pzeta_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
state_all = cp.empty((norm_intervals, num_particles), dtype=cp.int32)
particles_id_all = cp.empty((norm_intervals, num_particles), dtype=cp.int32)
turns_totnorm = cp.empty((norm_intervals, num_particles), dtype=cp.int32)

knob_names = ['on_sep1', 'on_sep2h', 'on_sep5', 'on_sep8h']
print('Tracking starting')
for i in range(num_turns):
    c = time.time()
    
    collider['lhcb1'].track(particles, num_turns=1, turn_by_turn_monitor=True, freeze_longitudinal=False)
    #if True:
    
    if (i + 1) % knobs == 0:
        for kk in range(len(knob_names)):
            collider.vars[knob_names[kk]] = all_fits_df[keys[kk]].dropna().values[(i + 1) // knobs]

            if knob_names[kk] == 'on_sep1':
                collider.vars[knob_names[kk]] = - all_fits_df[keys[kk]].dropna().values[(i + 1) // knobs]
            #print(df[keys[kk]].dropna().values[(i + 1) // knobs])
            print(f'Applying knob {knob_names[kk]} from {keys[kk]} with value {all_fits_df[keys[kk]].dropna().values[(i + 1) // knobs]}')
            #collider.vars[keys[kk]] = df[keys[kk]].dropna().values[-1]

    # Store particle data every 1000 turns
    if (i + 1) % ndata == 0:
        interval_index = (i + 1) // ndata - 1

       # Store physical coordinates
        x_phys[interval_index, :] = cp.asarray(particles.x)
        y_phys[interval_index, :] = cp.asarray(particles.y)
        zeta_phys[interval_index, :] = cp.asarray(particles.zeta)
        px_phys[interval_index, :] = cp.asarray(particles.px)
        py_phys[interval_index, :] = cp.asarray(particles.py)
        pzeta_phys[interval_index, :] = cp.asarray(particles.delta)
        state_all[interval_index, :] = cp.asarray(particles.state)
        turns_totnorm[interval_index, :] = cp.ones(num_particles, dtype=cp.int32) * (i + 1)
        particles_id_all[interval_index, :] = cp.asarray(particles.particle_id)

        d = time.time()
        print(f'Turn {i+1}, time {d-c}s')


# Convert results back to CPU and flatten arrays
x_phys = cp.asnumpy(x_phys).flatten()
y_phys = cp.asnumpy(y_phys).flatten()
zeta_phys = cp.asnumpy(zeta_phys).flatten()
px_phys = cp.asnumpy(px_phys).flatten()
py_phys = cp.asnumpy(py_phys).flatten()
pzeta_phys = cp.asnumpy(pzeta_phys).flatten()
state_all = cp.asnumpy(state_all).flatten()
turns_totnorm = cp.asnumpy(turns_totnorm).flatten()
particles_id_all = cp.asnumpy(particles_id_all).flatten()



print('Storing data')
#Convert results to DataFrame
result_phys = pd.DataFrame({
    "x_phys": x_phys,
    "y_phys": y_phys,
    "zeta_phys": zeta_phys,
    "px_phys": px_phys,
    "py_phys": py_phys,
    "pzeta_phys": pzeta_phys,
    "state": state_all,
    "at_turn": turns_totnorm, #"particle_id": particles_id_all #np.repeat(np.arange(1, (norm_intervals) * 1000, 1000), num_particles)
    "particle_id_all": particles_id_all
    #"particles_id_all": np.tile(particle_id, int(num_turns/ndata)) #particles_id_all
})

print('Data stored in DataFrame')
result_phys.to_parquet(f'/eos/user/a/aradosla/SWAN_projects/IBS/collider_2025_23_separation_adjust_{knobs}knobs_{num_particles}p3.parquet', index=False)
print('DataFrame saved to parquet file')
quit()
# %%
#dff = pd.read_parquet(f'/eos/user/a/aradosla/SWAN_projects/IBS/collider_2025_23_separation_adjust_{knobs}knobs_{num_particles}p3.parquet', engine='pyarrow')
#dff = pd.read_parquet(f'/eos/user/a/aradosla/SWAN_projects/IBS/collider_2025_23_separation_adjust_32000knobs_5000p3.parquet', engine='pyarrow')
dff = pd.DataFrame()
files = glob.glob(f'/eos/user/a/aradosla/SWAN_projects/Separation_adjust/MD_4x36b_optics23_adjust_1m_particles_qgaus/*/*.parquet')
print(f'Found {len(files)} files to concatenate')
for file in files[:]:
    print(file)
    dff = pd.concat([dff, pd.read_parquet(file, engine='pyarrow')], ignore_index=True)

# %%
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

collider.build_trackers()
#Built trackers
particles = xp.generate_matched_gaussian_bunch(
            num_particles = num_particles, total_intensity_particles = 2.3e11,
            nemitt_x = nemitt_x, nemitt_y=nemitt_y, sigma_z = 7.8e-2,
            particle_ref = particle_ref,
            line = collider['lhcb1'])
print('Particles generated')

tw0 = collider['lhcb1'].twiss()
gamma_rel = particle_ref.gamma0
betx_rel = particle_ref.beta0

# %%
normx_all = []
normy_all = []
for turn in dff.at_turn.unique()[:]:
    norm_x, norm_y = emit_formula_oneturn(dff, tw0, gamma_rel, betx_rel, turn)
    normx_all.append(norm_x)
    normy_all.append(norm_y)
# %%
plt.plot(dff.at_turn.unique(), normx_all, label = 'Norm. x emittance')
plt.plot(dff.at_turn.unique(), normy_all, label = 'Norm. y emittance', alpha = 0.6)
plt.xlabel('Turn')
plt.ylabel('Normalized Emittance [m rad]')
plt.legend()
plt.show()
plt.close()
# %%
line = collider['lhcb1']
def collimator(x_data_turn, px_data_turn, tw0):
    survived = 1/2*(tw0.gamx[0]*x_data_turn**2 + 2*tw0.alfx[0]*x_data_turn*px_data_turn + tw0.betx[0]*px_data_turn**2)
    return survived 

def collimatory(x_data_turn, px_data_turn, tw0):
    survived = 1/2*(tw0.gamy[0]*x_data_turn**2 + 2*tw0.alfy[0]*x_data_turn*px_data_turn + tw0.bety[0]*px_data_turn**2)
    return survived 
# %%
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

for turn in turns[:]:
    # Filter data for the current turn
    data_turn = data_concatenated[data_concatenated['at_turn'] == turn]
    
    # Exclude particles lost in previous turns
    data_turn = data_turn[~data_turn.particles_id_all.isin(lost_indices)]
    
    # Calculate the collimator condition for x and y
    survived_condition_x = collimator(data_turn.x_phys, data_turn.px_phys, tw0) < (6 * sigmax_col)**2
    survived_condition_y = collimatory(data_turn.y_phys, data_turn.py_phys, tw0) < (6 * sigmay_col)**2

    # Combine both conditions
    survived_condition = survived_condition_x & survived_condition_y
    # Track indices of lost particles
    lost_indices.extend(data_turn[~survived_condition].particles_id_all.tolist())
    data_turn_filtered = data_turn[survived_condition]  # Surviving particles only
    #print(data_turn_filtered)
    normx_emittance, normy_emittance = emit_formula_oneturn(data_turn_filtered, tw0, gamma_rel, betx_rel, turn)
    # Append the results to lists
    normx_all_std.append(np.array(normx_emittance))
    normy_all_std.append(np.array(normy_emittance))
    beam_loss.append(len(data_turn_filtered))
# %%
results = []
results.append({
        'turn': turns[:],
        'normx_emittance': np.array(normx_all_std).flatten(),
        'normy_emittance': np.array(normy_all_std).flatten(),
        'beam_survived': beam_loss,
    })
results = pd.DataFrame(results)
results.to_parquet('/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/emittance_and_losses_qgaussian6sigma.parquet')


# %%
plt.plot(dff.at_turn.unique()[1:], normx_all_std, label = 'Norm. x emittance')
plt.plot(dff.at_turn.unique()[1:], normy_all_std, label = 'Norm. y emittance', alpha = 0.6)
plt.xlabel('Turn')
plt.ylabel('Normalized Emittance [m rad]')
plt.legend()
plt.show()
plt.close()
# %%
fontsize = 20
states = line.record_last_track.state.T
states_turn = []
for turn in states:
    states_turn.append(sum(turn))
# %%
plt.figure(figsize = (10, 6))
plt.plot(dff.at_turn.unique()[1:1535], beam_loss[1:1535], label = 'Lost due to the collimation 10$\sigma$')
#plt.plot(5000-np.array(states_turn), label = 'Lost due to DA') 
plt.xlabel('Turns', size = fontsize)
plt.ylabel('# surviving particles', size = fontsize)
plt.title('Beam Loss Over Turns', size = fontsize+2)
plt.tick_params(labelsize = fontsize-2)
plt.legend(fontsize = 'xx-large')
#plt.ylim(29980, 30050)
#plt.xlim(30000, )
plt.grid(True)

# %%
#Plot against separation
plt.figure(figsize = (10, 6))
plt.plot(np.linspace(0.527, 0.007, len(beam_loss[1:1535])), np.array(beam_loss[1:1535])/np.array(beam_loss[1])*100, label = 'Lost due to the collimation 6$\sigma$', lw = 3)
#plt.plot(5000-np.array(states_turn), label = 'Lost due to DA') 
plt.xlabel('Separation IP1/5 [mm]', size = fontsize)
plt.ylabel('# surviving particles', size = fontsize)
plt.title('Beam Loss Over Turns', size = fontsize+2)
plt.tick_params(labelsize = fontsize-2)
plt.legend(fontsize = 'xx-large')
#plt.ylim(29980, 30050)
#plt.xlim(30000, )
plt.grid(True)

# %%
for turn in np.arange(3000, 50000, 10000):
    plt.plot(dff[dff.at_turn == turn].x_phys/np.sqrt(tw0.betx[0]*nemitt_x)*np.sqrt(gamma_rel), dff[dff.at_turn == turn].y_phys/np.sqrt(tw0.bety[0]*nemitt_y)*np.sqrt(gamma_rel), '.')
    plt.show()
    plt.close()
# %%
