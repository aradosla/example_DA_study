#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# Import user-defined modules
import xmask as xm

import xobjects as xo
import xtrack as xt
import matplotlib.ticker as ticker
sns.set_theme(style="ticks")
import glob
import matplotlib.cm as cm
import matplotlib
import matplotlib.colors as mcolors

# %%
# ========================================================================================================
# Loading the collider used in the simulation

#collider = xt.Multiline.from_json('collider.json')
# %%


# In[2]:


files = glob.glob('/eos/user/a/aradosla/SWAN_projects/Colored_sim1e6turns1e5p/**/output_particles_norm.parquet')
data = pd.read_parquet(files[0])
turns = np.unique(data.at_turn)
turns = turns[turns%10000 == 0]
#turns = turns[turns>10000]
#turns = turns[::10]
print(turns)


# In[3]:


data_list = []

# Loop through each file and append the x_norm data to the list
for file in files:
    for turn in turns:
        data_part = pd.read_parquet(file)
        filtered_data = data_part[data_part.at_turn == turn]
        data_list.append(filtered_data)


# In[5]:


data_concatenated = pd.concat(data_list, ignore_index = True)
#display(data_concatenated)
data_concatenated.to_parquet('Colored1e6turns1e5pconcat.parquet')
#data_concatenated = pd.read_parquet('Gausian1e6turns1e5pconcat.parquet')
display(data_concatenated)
# In[207]:


plt.figure(figsize = (20,10))
cmap = matplotlib.colormaps['plasma']
norm_c = mcolors.Normalize(vmin=min(turns), vmax=max(turns))
scalar_map = cm.ScalarMappable(norm=norm_c, cmap=cmap)

ax = plt.gca()
#data_len = []
for turn in turns:
    color = scalar_map.to_rgba(turn)
    data_turn = data_concatenated[data_concatenated['at_turn'] == turn]
    #data_len.append(len(data_concatenated[data_concatenated['at_turn'] == turn].x_norm))
    plt.hist(data_turn.x_norm, bins=100, alpha=0.6, label=f'{turn}', density=False, color=color)
   
#And color bar to show the mapping from turn to color
cbr = plt.colorbar(scalar_map, ax = ax)
cbr.set_label('Turns', size=20)
cbr.ax.tick_params(labelsize=20)  # Set the font size of the colorbar ticks
cbr.ax.yaxis.set_tick_params(labelsize=20)  # Adjust top labels
plt.ylabel('Counts', size = 30)
plt.xticks(fontsize=20)  # Change font size of x-axis ticks
plt.yticks(fontsize=20)  # Change font size of y-axis ticks
plt.xlabel('Normalised x position', size = 30)
#plt.title('Tracking simulation ~1e5p for 1e6 turns, q-Gaussian distribution, q = 1.4', size = 28)
plt.title('Tracking simulation ~2e4p for 1e6 turns, Gaussian distribution', size = 28)


# In[208]:


plt.plot(data_len)


# In[ ]:


def emittance_from_action(x, y, px, py, zeta, pzeta):
    x = x_data.T
    y = y_data.T
    px = px_data.T
    py = py_data.T
    zeta = zeta_data.T
    pzeta = pz_data.T

    Jx = np.zeros((N_turns, N_particles))
    Jy = np.zeros((N_turns, N_particles)) 
    errorx = np.zeros(N_turns)
    errory = np.zeros(N_turns)

    betx_rel =particle_ref._beta0[0]
    gamma_rel = particle_ref._gamma0[0]
    W = line.twiss()['W_matrix'][0]

    W_inv = np.linalg.inv(W)
    tw_full_inverse = line.twiss(use_full_inverse=True)['W_matrix'][0]

    n_repetitions = x.shape[0]
    n_particles = x.shape[1]

    inv_w = W_inv

    phys_coord = np.array([x,px,y,py,zeta,pzeta])
    phys_coord = phys_coord.astype(float)
    phys_coord[phys_coord==0.]=np.nan
    norm_coord = np.zeros_like(phys_coord)
    for i in range(n_repetitions):
        norm_coord[:,i,:] = np.matmul(inv_w, (phys_coord[:,i,:]))

    for i in range(N_turns):
        Jx[i,:] = (pow(norm_coord[0, i, :],2)+pow(norm_coord[1, i, :],2))/2 
        Jy[i,:] = (pow(norm_coord[2, i, :],2)+pow(norm_coord[3, i, :],2))/2 

    emitx = np.nanmean(Jx, axis=1)*(betx_rel*gamma_rel)
    emity = np.nanmean(Jy, axis=1)*(betx_rel*gamma_rel)
    return emitx, emity

x = x_data.T
y = y_data.T
px = px_data.T
py = py_data.T
zeta = zeta_data.T
pzeta = pz_data.T

Jx = np.zeros((N_turns, N_particles))
Jy = np.zeros((N_turns, N_particles)) 
errorx = np.zeros(N_turns)
errory = np.zeros(N_turns)

betx_rel =particle_ref._beta0[0]
gamma_rel = particle_ref._gamma0[0]
W = line.twiss()['W_matrix'][0]

W_inv = np.linalg.inv(W)
tw_full_inverse = line.twiss(use_full_inverse=True)['W_matrix'][0]

n_repetitions = x.shape[0]
n_particles = x.shape[1]

inv_w = W_inv

phys_coord = np.array([x,px,y,py,zeta,pzeta])
phys_coord = phys_coord.astype(float)
phys_coord[phys_coord==0.]=np.nan
norm_coord = np.zeros_like(phys_coord)
for i in range(n_repetitions):
    norm_coord[:,i,:] = np.matmul(inv_w, (phys_coord[:,i,:]))

for i in range(N_turns):
    Jx[i,:] = (pow(norm_coord[0, i, :],2)+pow(norm_coord[1, i, :],2))/2 
    Jy[i,:] = (pow(norm_coord[2, i, :],2)+pow(norm_coord[3, i, :],2))/2 

emitx = np.nanmean(Jx, axis=1)*(betx_rel*gamma_rel)
emity = np.nanmean(Jy, axis=1)*(betx_rel*gamma_rel)

print(emitx, emity)

fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
plt.subplot(2,1,1)
plt.plot(emitx, label = 'Emittance x from action')
plt.xlabel('Turns', fontsize = fontsize)
plt.ylabel('Emittance x', fontsize = fontsize)
plt.hlines(2.5e-6, xmin = 0 , xmax = N_turns, colors = 'red', label = 'Ref value')
plt.subplots_adjust(hspace = 0.5)
plt.subplot(2,1,2)
plt.plot(emity, label = 'Emittance y from action')
plt.xlabel('Turns', fontsize = fontsize)
plt.ylabel('Emittance y', fontsize = fontsize)
plt.hlines(2.5e-6, xmin = 0 , xmax = N_turns, colors = 'red', label = 'Ref value')


# In[204]:
colored = pd.read_parquet('/afs/cern.ch/work/a/aradosla/private/example_DA_study_mine/master_study/master_jobs/1_build_distr_and_collider/mydistribution1e6htcondor_colored.parquet')
weights = colored.weights[5000:]
# %%

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit


# Create figure and colormap
fig = plt.figure(figsize=(20, 10))
cmap = plt.cm.get_cmap('plasma')
norm_c = mcolors.Normalize(vmin=min(turns), vmax=max(turns))
scalar_map = cm.ScalarMappable(norm=norm_c, cmap=cmap)

# Initialize empty histogram plot
ax = plt.gca()
hist = ax.hist([], bins=100, alpha=0.6, density=True)


def gaussian(x, mu, sigma, A):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Define the q-Gaussian function
def q_gaussian(x, beta, q, A):
    if q == 1:
        return A * np.exp(-beta * x**2)
    else:
        return A * (1 + (q - 1) * beta * x**2)**(1 / (1 - q))

# Update function for animation
def update(turn):
    ax.clear()  # Clear previous plot
    color = scalar_map.to_rgba(turn)
    data_turn = data_concatenated[data_concatenated['at_turn'] == turn]
    #data_turn = data[data['at_turn'] == turn]
    
    #data_turn = data_turn.replace([np.inf, -np.inf], np.nan).dropna()
    #hist_data = ax.hist(data_turn.x_norm[:], bins=100, alpha=0.6, density=True, color=color)
    hist_data = ax.hist(data_turn.x_norm[:20000], bins=100, alpha=0.6, density=True, weights = weights[:20000], color=color)
    
    # Fit the q-Gaussian
    bin_heights, bin_edges = hist_data[0], hist_data[1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initial guess for the parameters
    initial_guessq = [1, 1.4, 1]  # beta, q, A
    initial_guess = [0, 1, 1]  # mu, sigma, A

    # Fit the q-Gaussian to the histogram data
    paramsq, covarianceq = curve_fit(q_gaussian, bin_centers, bin_heights, p0=initial_guessq, maxfev = 10000)
    beta, q, Aq = paramsq
    params, covariance = curve_fit(gaussian, bin_centers, bin_heights, p0=initial_guess)
    mug, sigma, Ag = params
    
    # Plotting the q-Gaussian fit
    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 300)
    y_fitq = q_gaussian(x_fit, beta, q, Aq)
    y_fitg = gaussian(x_fit, mug, sigma, Ag)
    ax.plot(x_fit, y_fitq, 'b-', linewidth=2, label=f'q-Gaussian fit: β={beta:.2f}, q={q:.2f}, A={Aq:.2f}')
    ax.plot(x_fit, y_fitg, 'r-', linewidth=2, label=f'Gaussian fit: μ={mug:.2f}, σ={sigma:.2f}, A={Ag:.2f}')
    

    # Setting plot labels and titles
    ax.set_xlabel('Normalized x position', size=20)
    ax.set_ylabel('Density', size=20)
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_title(f'Tracking simulation for turn {turn} ~2e4p, colored distribution', size=20)
    #ax.set_title(f'Tracking simulation for turn {turn} ~1e5p, q-Gaussian distribution, q = 1.4', size=20)
    #ax.set_title(f'Tracking simulation for turn {turn} ~1e5p, Gaussian distribution', size = 20)

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=15)

cbar = plt.colorbar(scalar_map, ax=ax)
cbar.set_label('Turns', size=15)
cbar.ax.tick_params(labelsize=15)
# Create animation
ani = animation.FuncAnimation(fig, update, frames=turns, blit=False, repeat=False)
# Save animation as GIF
ani.save('tracking_simulation_c2e4.gif', writer='imagemagick')
plt.show()

# Create animation
plt.show()

# In[ ]:


from scipy.stats import norm
#plt.figure(figsize=(20, 10))
#plt.hist(concatenated_data0, bins=100, alpha=0.6, label='Initial distribution', density=True)
#plt.hist(concatenated_data999, bins=100, alpha=0.6, label='Initial distribution', density=True)
# Fit a Gaussian to the data
mu0, std0 = norm.fit(concatenated_data0)
mu999, std999 = norm.fit(concatenated_data999)

# Plotting the histograms
plt.figure(figsize=(20, 10))

# Histogram for concatenated_data0
plt.hist(concatenated_data0, bins=100, alpha=0.6, label='Initial distribution', density=True)

# Plotting the Gaussian fit for concatenated_data0
xmin0, xmax0 = plt.xlim()
x0 = np.linspace(xmin0, xmax0, 100)
p0 = norm.pdf(x0, mu0, std0)
plt.plot(x0, p0, 'b', linewidth=2, label=f'Fit for initial distribution') #: mu={mu0:.2f}, std={std0:.2f}')

# Histogram for concatenated_data999
plt.hist(concatenated_data999, bins=100, alpha=0.6, label='Final distribution', density=True)

# Plotting the Gaussian fit for concatenated_data999
xmin999, xmax999 = plt.xlim()
x999 = np.linspace(xmin999, xmax999, 100)
p999 = norm.pdf(x999, mu999, std999)
plt.plot(x999, p999, 'orange', linewidth=2, label=f'Fit for final distribution') #: mu={mu999:.2f}, std={std999:.2f}')

# Adding titles and labels
plt.title('Histogram with Gaussian Fit, 1e5p', size = 30)

plt.ylabel('Density', size =30)
plt.xticks(fontsize=20)  # Change font size of x-axis ticks
plt.yticks(fontsize=20)  # Change font size of y-axis ticks
plt.legend(loc='upper right', fontsize = 20)

# Display the plot
plt.show()


# In[24]:


np.sum(data_part0[data_part0['at_turn']==999]['state'])


# In[11]:


from scipy.optimize import curve_fit

# Define the q-Gaussian function
def q_gaussian(x, beta, q, A):
    if q == 1:
        return A * np.exp(-beta * x**2)
    else:
        return A * (1 + (q - 1) * beta * x**2)**(1 / (1 - q))


# In[110]:


# Compute histogram
#%matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'inline')

hist, bin_edges = np.histogram(concatenated_data0, bins=100, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Compute histogram1
hist1, bin_edges1 = np.histogram(concatenated_data999, bins=100, density=True)
bin_centers1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2


# Initial guess for the parameters
initial_guess = [1, 1.5, 1]

# Fit the q-Gaussian to the histogram data
params, covariance = curve_fit(q_gaussian, bin_centers, hist, p0=initial_guess)
params1, covariance1 = curve_fit(q_gaussian, bin_centers1, hist1, p0=initial_guess)

# Extract the parameters
beta, q, A = params
beta1, q1, A1 = params1

# Plotting the histograms and the fitted q-Gaussian
plt.figure(figsize=(20, 10))

# Histogram for the data
plt.hist(concatenated_data0, bins=100, alpha=0.6, label='Initial distribution, q-Gaussian, q = 1.4', density=True)
plt.hist(concatenated_data999, bins=100, alpha=0.6, label='Final distribution, q-Gaussian, q = 1.4', density=True)

# Plotting the q-Gaussian fit
x = np.linspace(bin_edges[0], bin_edges[-1], 1000)
y = q_gaussian(x, *params)

plt.plot(x0, p0, 'black', linewidth=2, label=f'Gaussian fit for initial distribution') #: mu={mu0:.2f}, std={std0:.2f}')
plt.plot(x999, p999, 'orange', linewidth=2, label=f'Gaussian fit for final distribution') #: mu={mu999:.2f}, std={std999:.2f}')

x1 = np.linspace(bin_edges1[0], bin_edges1[-1], 1000)
y1 = q_gaussian(x1, *params1)
plt.plot(x, y, 'b-', linewidth=2, label=f'q-Gaussian fit initial: beta={beta:.2f}, q={q:.2f}, A={A:.2f}')
plt.plot(x1, y1, 'red', linewidth=2, label=f'q-Gaussian fit final: beta={beta1:.2f}, q={q1:.2f}, A={A1:.2f}')
plt.hist(data_part0[data_part0['at_turn'] == 0].x_norm, bins=100, alpha=0.6, label='Initial distribution', density=True)
plt.hist(data_part0[data_part0['at_turn'] == 999].x_norm, bins=100, alpha=0.6, label='Final distribution', density=True)


# Adding titles and labels
plt.title('Histogram with q-Gaussian Fit, Gaussian, 1e5p', size = 30)

plt.ylabel('Density', size = 30)
plt.legend(loc='upper right', fontsize=18)  # Adjust fontsize as needed
plt.xticks(fontsize=20)  # Change font size of x-axis ticks
plt.yticks(fontsize=20)  # Change font size of y-axis ticks

#plt.xlim(-4, 0)

# Display the plot
plt.show()


# In[27]:


plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.hist(concatenated_data0, bins=100, alpha=0.6, label='1e5p Initial distribution', density=True)
plt.hist(data_part1[data_part1['at_turn'] == 0].x_norm, bins=100, alpha=0.6, label='2e4p Initial distribution', density=True)
plt.title('Histogram particles convergence', size = 30)
plt.ylabel('Density', size = 30)
plt.legend(loc='upper right', fontsize=18)  # Adjust fontsize as needed
plt.xticks(fontsize=20)  # Change font size of x-axis ticks
plt.yticks(fontsize=20)  # Change font size of y-axis ticks
plt.xlim(-4.5,4.5)

plt.subplot(2,1,2)
plt.hist(concatenated_data999, bins=100, alpha=0.6, label='1e5p Final distribution', density=True)
plt.hist(data_part1[data_part1['at_turn'] == 999].x_norm, bins=100, alpha=0.6, label='2e4p Final distribution', density=True)

plt.ylabel('Density', size = 30)
plt.legend(loc='upper right', fontsize=18)  # Adjust fontsize as needed
plt.xticks(fontsize=20)  # Change font size of x-axis ticks
plt.yticks(fontsize=20)  # Change font size of y-axis ticks
plt.xlim(-4.5,4.5)


# In[69]:


plt.figure(figsize=(20,10))
plt.suptitle('Convergence study particle number, gaussian', size = 30)
plt.subplot(2,2,1)

plt.hist(concatenated_data0, bins=100, alpha=0.6, label='1e5p Initial distribution', density=True)
plt.hist(data_part1[data_part1['at_turn'] == 0].x_norm, bins=100, alpha=0.6, label='2e4p Initial distribution', density=True)
plt.xticks(fontsize=15)  # Change font size of x-axis ticks
plt.yticks(fontsize=15)  # Change font size of y-axis ticks
plt.xlim(-4.5,4.5)
plt.legend(fontsize=16)
plt.ylabel('Density', size = 20)
plt.subplot(2,2,2)
plt.hist(concatenated_data0, bins=100, alpha=0.6, label='1e5p Initial distribution', density=True)
plt.hist(np.concatenate([data_part1[data_part1['at_turn'] == 0].x_norm, data_part2[data_part2['at_turn'] == 0].x_norm]) , bins=100, alpha=0.6, label='4e4p Initial distribution', density=True)
plt.xticks(fontsize=15)  # Change font size of x-axis ticks
plt.yticks(fontsize=15)  # Change font size of y-axis ticks
plt.xlim(-4.5,4.5)
plt.legend(fontsize=16)
plt.ylabel('Density', size = 20)
plt.subplot(2,2,3)
plt.hist(concatenated_data0, bins=100, alpha=0.6, label='1e5p Initial distribution', density=True)
plt.hist(np.concatenate([data_part1[data_part1['at_turn'] == 0].x_norm, data_part2[data_part2['at_turn'] == 0].x_norm, data_part3[data_part3['at_turn'] == 0].x_norm]), bins=100, alpha=0.6, label='6e4p Initial distribution', density=True)
plt.xticks(fontsize=15)  # Change font size of x-axis ticks
plt.yticks(fontsize=15)  # Change font size of y-axis ticks
plt.xlim(-4.5,4.5)
plt.legend(fontsize=16)
plt.ylabel('Density', size = 20)
plt.subplot(2,2,4)
plt.hist(concatenated_data0, bins=100, alpha=0.6, label='1e5p Initial distribution', density=True)
plt.hist(np.concatenate([data_part1[data_part1['at_turn'] == 0].x_norm, data_part2[data_part2['at_turn'] == 0].x_norm, data_part3[data_part3['at_turn'] == 0].x_norm,data_part4[data_part4['at_turn'] == 0].x_norm]), bins=100, alpha=0.6, label='8e4p Initial distribution', density=True)
plt.xticks(fontsize=15)  # Change font size of x-axis ticks
plt.yticks(fontsize=15)  # Change font size of y-axis ticks
plt.xlim(-4.5,4.5)
plt.legend(fontsize=16)
plt.ylabel('Density', size = 20)


# In[60]:


colored = pd.read_parquet('mydistribution_colored.parquet')
#display(colored)


# In[68]:


plt.figure(figsize=(20,10))
plt.suptitle('Convergence study particle number, pseudo KV(colored)', size = 30)
plt.subplot(2,2,1)

plt.hist(colored.x, bins=100, alpha=0.6, label='1e5p Initial distribution', weights = colored.weights, density=True)
plt.hist(colored.x[:20000], bins=100, alpha=0.6, label='2e4p Initial distribution', weights = colored.weights[:20000], density=True)
plt.xticks(fontsize=15)  # Change font size of x-axis ticks
plt.yticks(fontsize=15)  # Change font size of y-axis ticks
plt.xlim(-4.5,4.5)
plt.legend(fontsize=16)
plt.ylabel('Density', size = 20)
plt.subplot(2,2,2)
plt.hist(colored.x, bins=100, alpha=0.6, label='1e5p Initial distribution', weights = colored.weights, density=True)
plt.hist(colored.x[:40000], bins=100, alpha=0.6, label='4e4p Initial distribution', weights = colored.weights[:40000],density=True)
plt.xticks(fontsize=15)  # Change font size of x-axis ticks
plt.yticks(fontsize=15)  # Change font size of y-axis ticks
plt.xlim(-4.5,4.5)
plt.legend(fontsize=16)
plt.ylabel('Density', size = 20)
plt.subplot(2,2,3)
plt.hist(colored.x, bins=100, alpha=0.6, label='1e5p Initial distribution', weights = colored.weights,density=True)
plt.hist(colored.x[:60000], bins=100, alpha=0.6, label='6e4p Initial distribution',weights = colored.weights[:60000], density=True)
plt.xticks(fontsize=15)  # Change font size of x-axis ticks
plt.yticks(fontsize=15)  # Change font size of y-axis ticks
plt.xlim(-4.5,4.5)
plt.legend(fontsize=16)
plt.ylabel('Density', size = 20)
plt.subplot(2,2,4)
plt.hist(colored.x, bins=100, alpha=0.6, label='1e5p Initial distribution',weights = colored.weights, density=True)
plt.hist(colored.x[:80000], bins=100, alpha=0.6, label='8e4p Initial distribution', weights = colored.weights[:80000],density=True)
plt.xticks(fontsize=15)  # Change font size of x-axis ticks
plt.yticks(fontsize=15)  # Change font size of y-axis ticks
plt.xlim(-4.5,4.5)
plt.legend(fontsize=16)
plt.ylabel('Density', size = 20)


# In[ ]:




