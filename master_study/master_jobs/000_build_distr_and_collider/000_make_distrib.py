# %%
import os
import itertools
import numpy as np
import pandas as pd
import yaml

try:
    with open('config.yaml','r') as fid:
        configuration = yaml.safe_load(fid)
except:
    from config import configuration

# Start tree_maker logging if log_file is present in config
try:
    import tree_maker
    if 'log_file' not in configuration.keys():
        tree_maker=None
except:
    tree_maker=None

if tree_maker is not None:
    tree_maker.tag_json.tag_it(configuration['log_file'], 'started')

# tracking scans
r_min = 2
r_max = 10
#radial_list = np.linspace(r_min, r_max, 16*(r_max-r_min), endpoint=False)
radial_list = np.linspace(r_min, r_max, 2*16*(r_max-r_min), endpoint=False)

n_angles = 5
theta_list = np.linspace(0, 90, n_angles+2)[1:-1]

particle_list = [(particle_id, ii[0], ii[1]) for particle_id, ii in
                 enumerate(itertools.product(radial_list, theta_list))]
particle_list = list(np.array_split(particle_list, 15))

distributions_folder = './particles'
os.makedirs(distributions_folder, exist_ok=True)
for ii, my_list in enumerate(particle_list):
   pd.DataFrame(my_list,
                columns=['particle_id','normalized amplitude in xy-plane',
                         'angle in xy-plane [deg]'])\
     .to_parquet(f'{distributions_folder}/{ii:03}.parquet')

# to debug
# import time
# time.sleep(600)

if tree_maker is not None:
    tree_maker.tag_json.tag_it(configuration['log_file'], 'completed')
