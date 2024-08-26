# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import glob
import json
import logging
import os
import time

# Import third-party modules
import numpy as np
import pandas as pd
import ruamel.yaml
import tree_maker

# Import user-defined modules

import xmask as xm

import xobjects as xo
import xtrack as xt
import matplotlib.ticker as ticker
sns.set_theme(style="ticks")
# %%
#files = glob.glob('/eos/user/a/aradosla/SWAN_projects/Noise_sim_quad_300_12e-6/**/*norm.parquet')
#files = glob.glob('/eos/user/a/aradosla/SWAN_projects/Noise_sim_nonoise_ready/part*.parquet')
#files = glob.glob('/eos/user/a/aradosla/SWAN_projects/Noise_sim_nonoise_ready/**/*norm.parquet')
files_noise = glob.glob('/eos/user/a/aradosla/SWAN_projects/Noise_sim_dipol_try2_3kHz_6e-10/**/*norm.parquet')[1:]
files_new = glob.glob('/eos/user/a/aradosla/SWAN_projects/Noise_sim_no_noise/**/*norm.parquet')[1:]

# %%
assert len(files_new) == len(files_noise)

############################ Load the data ############################
df_all = pd.DataFrame()

#for num, file in enumerate(files_new):
for num, file in enumerate(files_noise):
    df = pd.read_parquet(file)
    df['particle_id'] = np.tile(np.arange(0 + num*len(df[df['at_turn']==1000]),len(df[df['at_turn']==1000]) + num*len(df[df['at_turn']==1000])), len(np.unique(df.at_turn)))
    df_all = pd.concat([df_all, df], ignore_index=True)

df_all_sorted = df_all.sort_values(by=['at_turn'])
df_all_new= df_all_sorted.reset_index(drop=True)
df_first = df
# %% 
df_noise = pd.DataFrame()
for file in files_noise:
    df = pd.read_parquet(file)
    df['particle_id'] = np.tile(np.arange(0 + num*len(df[df['at_turn']==1000]),len(df[df['at_turn']==1000]) + num*len(df[df['at_turn']==1000])), len(np.unique(df.at_turn)))
    df_noise = pd.concat([df_noise, df], ignore_index=True)


df_noise_sorted = df_noise.sort_values(by=['at_turn'])
df_noise_new= df_noise_sorted.reset_index(drop=True)
df_second = df

# %%

############################### Here we compute the beam loss ###############################

data_all = []
#df_copy = df_all_new.copy()
df_copy = df_all_new.copy()
#df_copy['particle_id'] = 
#df_copy['particle_id'] = np.tile(np.arange(0,len(df_copy[df_copy['at_turn']==1000])), len(np.unique(df_copy.at_turn)))
df_copy['lost'] = np.ones(len(df_copy))
current_aperture = 6.
spacing=1
turns = np.unique(df_copy.at_turn)[::spacing]
# Loop through each turn
for turn in turns[::spacing]:
    # Filter the DataFrame for the current turn
    if turn != turns[0]:
        lost_aperture = df_previous[((df_previous.x_norm)**2 + (df_previous.px_norm)**2) >= (current_aperture)**2]
        lost_aperture = lost_aperture[((lost_aperture.y_norm)**2 + (lost_aperture.py_norm)**2) >= (current_aperture)**2]
        survived = df_previous[((df_previous.x_norm)**2 + (df_previous.px_norm)**2) < (current_aperture)**2]
        survived = survived[((survived.y_norm)**2 + (survived.py_norm)**2) < (current_aperture)**2]
        lost_ids_aperture = lost_aperture.particle_id
        common_ids_aperture = survived.particle_id
        #common_ids = df_previous[df_previous['state'] == 1]['particle_id']
        #lost_ids = df_previous[df_previous['state'] == -1]['particle_id']
        df = df[df['particle_id'].isin(common_ids_aperture)]
        #df = df[df['particle_id'].isin(common_ids)]
        df = df_copy[(df_copy['at_turn'] == turn)]
        df = df.reset_index(drop=True)
        df_previous = df
        #df_copy.loc[(df_copy['particle_id'].isin(lost_ids))& (df_copy.at_turn >= turn), 'state']= -1
        df_copy.loc[(df_copy['particle_id'].isin(lost_ids_aperture)) & (df_copy.at_turn >= turn), 'state']= -1
    
    else:
        df = df_copy[df_copy['at_turn'] == turn]    
        df_previous = df
    
        
    
    # Identify particles with state = 1
    data = df[df['state'] == 1]['state'].values
    print(data)
    
    # Sum the state values where the state is 1
    data_sum = np.sum(data)
    print(data_sum)
    
    # Append the sum to the data_all list
    data_all.append(data_sum)
    
    # Set state to 0 for all particles in future turns if state == -1 in the current turn
    #particles_to_zero = df[df['state'] == -1].index
    #df_copy.loc[df_copy.index.isin(particles_to_zero) & (df_copy['at_turn'] > turn), 'state'] = -1
    
    # Plot the state values for the current turn
    plt.plot(df['state'])

# Display the plot
plt.show()
# %%
#data_all_new = data_all

# %%
#df_copy_new = df_copy.copy()
for turn in [turns[0], turns[-1]]:
    df = df_copy[df_copy['at_turn'] == turn]
    #df_new = df_copy_new[df_copy_new['at_turn'] == turn]
    df = df[df.state ==1]
    #df_new = df_new[df_new.state ==1]
    plt.hist(df.x_norm, bins =70, density  = False)
    #plt.hist(df_new.x_norm, bins =50, density  = True, alpha=0.5)
    #plt.yscale('log')
    
# %%
# First one is right
#plt.plot(np.linspace(0, 1e6, 1001)[1:],np.array(data_all_new)/len(df_first[df_first.at_turn==1000])*100)
plt.xlim(0,2e5)
plt.plot(np.linspace(0, 1e6, 1001)[1:],np.array(data_all)/len(df_all_new[df_all_new.at_turn==1000])*100)
#plt.plot(np.linspace(0, 1e6, 1001)[1:], 100 - np.array(all_sum)/len(df_all[df_all.at_turn==1000])*100)
#plt.plot(np.linspace(0, 1e6, 1001)[1:], 100 - np.array(data_sum)/len(df_all[df_all.at_turn==1000])*100)

# %%
# Initialize lists for storing IDs and values

############################### Basic check for the next cell ###############################
ids_all = []
values_all = []

df_copy = df_all_new.copy()

#df_copy['particle_id'] = 
#df_copy['particle_id'] = np.tile(np.arange(0,len(df_copy[df_copy['at_turn']==1000])), len(np.unique(df_copy.at_turn)))
df_copy['lost'] = np.ones(len(df_copy))
turns = np.unique(df_copy.at_turn)[:]
for turn in turns:
    ids = []
    values = []

    # Specify the turn you're interested in (e.g., turns[6])
    target_turn = turn

    # Filter DataFrame for the specific turn
    df_turn = df_copy[df_copy['at_turn'] == target_turn]

    # Calculate the condition (x_norm**2 + px_norm**2) and apply it
    condition = (df_turn['x_norm']**2 + df_turn['px_norm']**2 > 36) 

    # Iterate over the filtered DataFrame
    for _, row in df_turn[condition].iterrows():
        ids.append(row['particle_id'])
        values.append(row['x_norm']**2 + row['px_norm']**2)
    ids_all.append(ids)
    values_all.append(values)
# %%
#ids_all = ids_all[3:]
for i in range(len(ids_all[:50])):
#    print(ids_all[i])
#    print(ids_all[i+1])
    for j in ids_all[i]:
        if j not in ids_all[i+1]:
            print(j)
            ids_all[i+1].append(j)
            ids_all[i+1] =np.unique(np.sort(ids_all[i+1])).tolist()
    #print(ids_all[i+1])

# Print or use the collected IDs and values
print("Particle IDs:", ids)
print("Values:", values)


# %%
# BEAM LOSS FUNCTION
def loss(df):
    turns = np.unique(df.at_turn)[::spacing]
    num_survived_noise = []
    num_initial_particles_noise = len(df[df.at_turn == turns[0]])
    df_copy = df.copy()
    df_copy.index = np.arange(1, len(df_copy) + 1)
    beam_loss_noise = []
    lost = [] 
    df_copy['lost'] = np.ones(len(df_copy))
    survived_data_frame = pd.DataFrame()

    for turn in turns:
        # Update the dataframe for the current turn using the cumulative mask
        df_noise_turn = df_copy[(df_copy.at_turn == turn) & (df_copy.lost == 1)]
        
        survived_particles_noise = df_noise_turn[((df_noise_turn.x_norm)**2 + (df_noise_turn.px_norm)**2) < (current_aperture)**2]
        lost_particles_noise = df_noise_turn[((df_noise_turn.x_norm)**2 + (df_noise_turn.px_norm)**2) >= (current_aperture)**2]
        survived_data_frame = pd.concat([survived_data_frame, survived_particles_noise])
        
        #print(len(survived_particles_noise))
        for turn_new in np.arange(turn, len(turns)):
            #print(turn_new)
            if len(lost_particles_noise) != 0:
                df_copy =df_copy.loc[(lost_particles_noise.index - min(lost_particles_noise.index)+1)+ spacing*turn_new*num_initial_particles_noise, 'lost'] = -1
            
        #survived_particles_indices = survived_particles_noise.index 
        lost.append(len(lost_particles_noise))
        num_survived_noise.append(len(survived_particles_noise))
        #filtered_df = df_copy[(df_copy.at_turn == turn) & (df_copy.lost != 1)]

        beam_loss_noise.append(num_initial_particles_noise - len(survived_particles_noise))
    return survived_data_frame, beam_loss_noise, num_initial_particles_noise

# %%

spacing = 10
turns = np.unique(df.at_turn)[::spacing]
current_aperture = 6.  
survived_noise , loss_noise, init_noise = loss(df_noise_new)
survived_all, loss_all, init_all = loss(df_all_new)
# %%
# BEAM LOSS PLOT
plt.plot(turns, (init_all - np.array(loss_all))/init_all*100, label='No noise')
plt.plot(turns, (init_noise - np.array(loss_noise))/init_noise*100, label = 'Dipolar noise')
plt.legend()
plt.xlim(0,1e6)
plt.ylabel('Survival %, beam loss')
plt.xlabel('Turns number')
plt.grid()
plt.show()
# %%
plt.figure(figsize=(10, 6))
plt.plot(turns[:], np.array(loss_noise)[:]/init_noise, '.')
plt.xlabel('Turn Number')
plt.ylabel('Beam Loss')
plt.legend()
plt.title('Beam Loss as a Function of Turn')
plt.show()


# %%
survived_data_frame = survived_all # survived_noise
for turn in np.unique(survived_data_frame['at_turn'])[:10]:
    df = survived_data_frame[survived_data_frame.at_turn == turn]
    plt.plot(df.x_norm, df.px_norm, '.')
    plt.show()
    print(len(df))
# %%
path = 'Sim_pics_no_noise' #f'Sim_pics_dipol_noise_3kHz6e-10try2'
for turn in np.unique(survived_data_frame['at_turn'])[:]:
    df = survived_data_frame[survived_data_frame.at_turn == turn]
    plt.hist(df.x_norm, bins=100, alpha=0.5, label=f'Turn {turn}')
    plt.legend()
    plt.xlim(-4.5, 4.5)
    plt.ylim(0, 1400)
    plt.xlabel('x_norm')
    plt.ylabel('Number of particles')
    #plt.savefig(f'/eos/user/a/aradosla/SWAN_projects/{path}/normal_histograms_x/x_norm_hist_turn_{turn}.png')

    plt.show()
plt.legend()
# %%
for turn in np.unique(survived_data_frame['at_turn'])[:]:
    df = survived_data_frame[survived_data_frame.at_turn == turn]
    plt.hist(df.x_norm, bins=100, alpha=0.5, label=f'Turn {turn}')
    plt.legend()
    plt.xlim(-4.5, 4.5)
    #plt.ylim(0, 1200)
    plt.xlabel('x_norm')
    plt.ylabel('Number of particles')
    plt.yscale('log')
    #plt.savefig(f'/eos/user/a/aradosla/SWAN_projects/{path}/log/x_norm_hist_turn_{turn}_log.png')
    
    plt.show()
plt.legend()
# %%
for turn in np.unique(survived_data_frame['at_turn'])[::20]:
    df = survived_data_frame[survived_data_frame.at_turn == turn]
    plt.hist(df.x_norm, bins=100, alpha=0.5, label=f'Turn {turn}')
    plt.legend()
    plt.xlim(-4.5, 4.5)
    #plt.ylim(0, 1200)
    plt.xlabel('x_norm')
    plt.ylabel('Number of particles')
    #plt.yscale('log')
    #plt.savefig(f'/eos/user/a/aradosla/SWAN_projects/{path}/x_norm_hist_turn_5turns.png')

plt.legend()
# %%
for turn in np.unique(survived_data_frame['at_turn'])[::20]:
    df = survived_data_frame[survived_data_frame.at_turn == turn]
    plt.hist(df.x_norm, bins=100, alpha=0.5, label=f'Turn {turn}')
    plt.legend()
    plt.xlim(-4.5, 4.5)
    #plt.ylim(0, 1200)
    plt.xlabel('x_norm')
    plt.ylabel('Number of particles')
    plt.yscale('log')
    #plt.savefig(f'/eos/user/a/aradosla/SWAN_projects/{path}/x_norm_hist_turn_5turns_log.png')

plt.legend()

# %%
def plot_distributions(df, alpha = 0.5):
    path = 'Sim_pics_no_noise/phase_space'  # 'Sim_pics_dipol_noise_3kHz6e-10try2/phase_space' #
    for turn in np.unique(df['at_turn'])[:20:1]:
        df_new = df[df.at_turn == turn]
        #plt.hist(df_new.x_norm, bins=100, alpha=alpha, label=f'Turn {turn}')
        plt.plot(df_new.x_norm, df_new.px_norm, '.', alpha=alpha, label=f'Turn {turn}')
        plt.legend()
        plt.ylim(-6, 6)
        plt.xlim(-6, 6)
        #plt.ylim(0, 1200)
        plt.xlabel('x_norm')
        plt.ylabel('Number of particles')
        #plt.savefig(f'/eos/user/a/aradosla/SWAN_projects/{path}/phase_space_{turn}.png')
        plt.show()
    plt.legend()

#plot_distributions(survived_noise)
plot_distributions(survived_all, alpha=0.5)


# %%
