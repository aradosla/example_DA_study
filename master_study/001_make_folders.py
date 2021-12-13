# %%
import tree_maker
from tree_maker import NodeJob
from tree_maker import initialize
import time
import os
from pathlib import Path
import itertools
import numpy as np
import yaml
from udf import string_run

# %%
# Clearly for this easy task on can do all in the very same python kernel
# BUT here we want to mimic the typical flow
# 1. MADX for optics matching/error seeding
# 2. Tracking for FMA and or DA studies
# 3. simulation baby-sitting and
# 4. postprocessing

config=yaml.safe_load(open('config.yaml'))

# machine parameters scans
qx0 = np.arange(62.305, 62.330, 0.001)[::20]
qy0 = np.arange(60.305, 60.330, 0.001)[::20]

children={}
for optics_job, (myq1, myq2) in enumerate(itertools.product(qx0, qy0)):
    optics_children={}
    children[f'{optics_job:03}'] = {
                                    'qx0':float(myq1),
                                    'qy0':float(myq2),
                                    'children':optics_children}
    for track_job in range(15):
        optics_children[f'{track_job:03}'] = {
                    'particle_file': ('../../'
                                   f'distrib_abc/{track_job:03}.parquet'),
                    'xline_json': ('../xsuite_lines/'
                                  'line_bb_for_tracking.json'),
                    'n_turns': int(1000)}

#config['root']['children'] = children

# %%
"""
#### The root of the tree
"""
start_time = time.time()
root = initialize(config)
print('Done with the tree creation.')
print("--- %s seconds ---" % (time.time() - start_time))

# %%
"""
### Cloning the templates of the nodes
From python objects we move the nodes to the file-system.
"""
start_time = time.time()
root.clone()
print('The tree structure is moved to the file system.')
print("--- %s seconds ---" % (time.time() - start_time))

# %%
# Mutation
start_time = time.time() 
root.mutate_descendants()
print('The tree structure is mutated.')
print("--- %s seconds ---" % (time.time() - start_time))

# %%
# Prepare the run.sh
start_time = time.time() 
root.write_run_files(string_run)
print('The excutables are ready.')
print("--- %s seconds ---" % (time.time() - start_time))
