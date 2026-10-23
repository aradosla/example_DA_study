# %%
import xtrack as xt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xpart as xp
import xobjects as xo
import yaml
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
import itertools
import nafflib as NAFFlib
import glob
from plotting_functions import *


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
##############################################3.3752b
######### Taken from https://github.com/SixTrack/SixDeskDB/blob/master/sixdeskdb/footprint.py #####################
##############################################
######## Modified 07/03/2018
"""module to plot resonance lines"""

mycolors = list('rgbcm')

 # %%

#Manual change of the path if needed
path = '/eos/user/a/aradosla/SWAN_projects/FMA_studies_FLATTOP/Injection_2026_example' 

files = glob.glob(f"{path}/**/fma.parquet")
dffs = pd.DataFrame()
for file in files[:]:
    dff = pd.read_parquet(file)
    dff = dff[(dff['at_turn'] < 2000) | (dff['at_turn'] > 8000)]
    print(file)
    dffs = pd.concat([dffs,dff], ignore_index=True)


# %%
turn = 1000
fig, ax = plt.subplots(figsize=(15,10), dpi = 300)
plt.scatter(dffs[dffs.at_turn == turn]['qx1'],dffs[dffs.at_turn == turn]['qy1'], s=6, edgecolors=None, c=dffs[dffs.at_turn == turn]['diffusion'], vmin=-7, vmax=-3,cmap='jet')
plt.xlim(0.20, 0.32)
plt.ylim(0.24, 0.33)
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
plt.xlabel(r"Horizontal tune, the $Q_x$")
plt.ylabel(r"Vertical tune, $Q_y$")
cbar=plt.colorbar(pad=0.01)
cbar.set_label(r'$\rm \log_{10}\left({\sqrt{\Delta Q_x^2 + \Delta Q_y^2}}\right)$',labelpad=45,rotation=270, fontsize=24)
#ootprint.lw_mine=0.5

plot_res_upto_order(16,c1 = 'darkgrey', c2 = 'darkgrey',c3='r',annotate=False)

# %%
