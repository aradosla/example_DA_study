# %%
import os
import itertools
import numpy as np
import pandas as pd

# tracking scans
r_min = 2
r_max = 10
#radial_list = np.linspace(r_min, r_max, 16*(r_max-r_min), endpoint=False)
radial_list = np.linspace(r_min, r_max, 2*16*(r_max-r_min), endpoint=False)

n_angles = 5
theta_list = np.linspace(0, 90, n_angles+2)[1:-1]

particle_list = [(particle_id, ii[0], ii[1]) for particle_id, ii in enumerate(itertools.product(radial_list, theta_list))]
particle_list = list(np.array_split(particle_list, 15))

distributions_folder = './distrib_abc'
os.makedirs(distributions_folder, exist_ok=True)
for ii, my_list in enumerate(particle_list):
   pd.DataFrame(my_list,
                columns=['particle_id','normalized amplitude in xy-plane',
                         'angle in xy-plane [deg]'])\
     .to_parquet(f'{distributions_folder}/{ii:03}.parquet')

