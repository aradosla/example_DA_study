# %%
import tree_maker
from tree_maker import NodeJob
import time
import os
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

# %%
# Clearly for this easy task on can do all in the very same python kernel
# BUT here we want to mimic the typical flow
# 1. MADX for optics matching/error seeding
# 2. Tracking for FMA and or DA studies
# 3. simulation baby-sitting and
# 4. postprocessing

setup_env_script = '/home/HPC/giadarol/miniconda3/bin/activate'

# machine parameters scans
qx0 = np.arange(62.305, 62.330, 0.001)[::20]
qy0 = np.arange(60.305, 60.330, 0.001)[::20]


# %%
"""
#### The root of the tree
"""
start_time = time.time()

# %%
#root
my_folder = os.getcwd() + '/study_001'
root = NodeJob(name='root', parent=None)
root.path = ''
root.dictionary = {'abs_path': my_folder}


# %%
"""
#### First generation of nodes
"""

# %%
#first generation
python_executable_first_gen = './master_codes/000_machine_model/000_pymask.py'
script_run_first_gen_path = my_folder + '/run_first_gen.sh'
script_run_first_gen_content = '\n'.join([
    f"source {setup_env_script}",
    f"python {Path(python_executable_first_gen).absolute()}"
    ])

for node in root.root.generation(0):
    children_list = []
    for child, (myq1, myq2) in enumerate(itertools.product(qx0, qy0)):
        path = f"{child:03}"
        template_path_rel = '../../master_codes/000_machine_model'
        children_list.append(NodeJob(name=f"{child:03}",
                             parent=node,
                             # these paths are relative to root
                             path=path,
                             template_path=template_path_rel,
                             # these paths are relative to the child path
                             # local run: submit_command=f'python {template_path_rel}/run.py &',
                             submit_command=(
                                 f"bsub -J {child:03} -n 2 -q hpc_acc "
                                 "-e %J.err -o %J.out "
                                 f"{Path(script_run_first_gen_path).absolute()} &"),
                             #log_file='log.json',
                             dictionary={'qx0':float(myq1),
                                         'qy0':float(myq2),
                                        }))

    node.children = children_list


# To combine different lists one can use the product or the zip functions    
#import itertools
#[[i, j, z] for i, j, z in itertools.product(['a','b'],['c','d'],[1,2,3])]
#[[i, j, z] for i, j, z in zip(['a','b'],['c','d'],[1,2,3])]

# %%
"""
#### Second generation of nodes
"""
python_executable_second_gen = './master_codes/002_tracking_job/000_track.py'
script_run_second_gen_path = my_folder + '/run_second_gen.sh'
script_run_second_gen_content = '\n'.join([
    f"source {setup_env_script}",
    f"python {Path(python_executable_second_gen).absolute()}"
    ])

distributions_folder = './master_codes/001_make_part_distribution/distrib_abc'
n_distrib_files = 15 # TODO to be automatized
# %%
#second generation
for node in root.root.generation(1):
    children_list = []
    for child in range(n_distrib_files):
        path = f"{node.path}/{child:03}"
        template_path_rel = '../../../master_codes/002_tracking_job'
        distributions_folder_rel = os.path.relpath(distributions_folder, path)
        children_list.append(NodeJob(name=f"{child:03}",
                             parent=node,
                             path=path,
                             template_path=template_path_rel,
                             #bsub -q hpc_acc -e %J.err -o %J.out cd $PWD && ./run.sh
                             submit_command = (f'bsub -J {node.name}/{child:03} '
                                 '-q hpc_acc -e %J.err -o %J.out '
                                 f'{script_run_second_gen_path} &'),
                             #submit_command = f'python {root.template_path}/multiply_it/run.py &',
                             #log_file = 'log.json', 
                             dictionary={'particle_file': f'../{distributions_folder_rel}/{child:03}.parquet',
                                         'xline_json': f'../xlines/line_bb_for_tracking.json',
                                         'n_turns': 1e6,
                                      }))
    node.children = children_list
root.to_json('tree.json')



print('Done with the tree creation.')
print("--- %s seconds ---" % (time.time() - start_time))


# %%
"""
### Cloning the templates of the nodes
From python objects we move the nodes to the file-system.
"""

# %%
# We map the pythonic tree in a >folder< tree
start_time = time.time()
root.clean_log()
root.rm_children_folders()
from joblib import Parallel, delayed

for depth in range(root.height):
#    [x.clone_children() for x in root.generation(depth)]
     Parallel(n_jobs=8)(delayed(x.clone_children)() for x in root.generation(depth))

# VERY IMPORTANT, tagging
root.tag_as('cloned')
print('The tree structure is moved to the file system.')
print("--- %s seconds ---" % (time.time() - start_time))

with open(script_run_first_gen_path, 'w') as fid:
    fid.write(script_run_first_gen_content)
with open(script_run_second_gen_path, 'w') as fid:
    fid.write(script_run_second_gen_content)
os.system(f"chmod u+x {script_run_first_gen_path}")
os.system(f"chmod u+x {script_run_second_gen_path}")

start_time = time.time() 
#Parallel(n_jobs=8)(delayed(node.mutate)() for node in root.descendants)
for node in root.descendants:
   node.mutate()
print('The tree structure is mutated.')
print("--- %s seconds ---" % (time.time() - start_time))

