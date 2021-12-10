# %%
import tree_maker
from tree_maker import NodeJob
import time
import os
from pathlib import Path
import itertools
import numpy as np
import yaml




# %%
# Clearly for this easy task on can do all in the very same python kernel
# BUT here we want to mimic the typical flow
# 1. MADX for optics matching/error seeding
# 2. Tracking for FMA and or DA studies
# 3. simulation baby-sitting and
# 4. postprocessing

config=yaml.safe_load(open('config.yaml'))
#setup_env_script = config['setup_env_script']

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
#my_folder = os.getcwd()

#root name
r=list(config.keys())[0]
#root node
root = NodeJob(name=list(config.keys())[0], 
               parent=None, 
               dictionary={x: config[r][x] for x in config[r].keys() 
                                           if x not in ['children']})

def get_children(node, node_dict):
    if 'children' in node_dict[node.name].keys():
        children_dict = node_dict[node.name]['children'] 
        for child in children_dict.keys():
            child_dict = children_dict[child]
            child_node = NodeJob(name=child,
                                 parent=node,                                    
                                 dictionary={x: child_dict[x]
                                             for x in child_dict.keys()
                                             if x not in ['children']})
            get_children(child_node, children_dict)    

get_children(root, config)

# %%
"""
#### First generation of nodes
"""

# %%
#first generation
#python_executable_first_gen = '../master_codes/000_machine_model/000_pymask.py'
#script_run_first_gen_path = my_folder + '/run_first_gen.sh'
#script_run_first_gen_content = '\n'.join([
#    f"source {setup_env_script}",
#    f"python 000_pymask.py"
#    ])

#for node in root.generation(0):
#    children_list = []
#    for child, (myq1, myq2) in enumerate(itertools.product(qx0, qy0)):
#        path = f"{child:03}"
#        children_list.append(NodeJob(name=f"{child:03}",
#                             parent=node,
                             # local run: submit_command=f'python {template_path_rel}/run.py &',
                             #submit_command=(
                             #    f"bsub -J {child:03} -n 2 -q hpc_acc "
                             #    "-e %J.err -o %J.out "
                             #    f"{Path(script_run_first_gen_path).absolute()} &"),
                             #log_file='log.json',
#                             dictionary={'qx0':float(myq1),
#                                         'qy0':float(myq2),
#                                        }))

#    node.children = children_list


# To combine different lists one can use the product or the zip functions    
#import itertools
#[[i, j, z] for i, j, z in itertools.product(['a','b'],['c','d'],[1,2,3])]
#[[i, j, z] for i, j, z in zip(['a','b'],['c','d'],[1,2,3])]

# %%
#"""
#### Second generation of nodes
#"""
#script_run_second_gen_path = my_folder + '/run_second_gen.sh'
#script_run_second_gen_content = '\n'.join([
#    f"source {config['setup_env_script']}",
#    f"python {config['generations'][2]['python_executable']}"
#    ])

#distributions_folder = '../master_codes/001_make_part_distribution/distrib_abc'
#n_distrib_files = 15 # TODO to be automatized
# %%
#second generation
#for node in root.generation(1):
#    children_list = []
#    for child in range(n_distrib_files):
#        distributions_folder_rel = os.path.relpath(distributions_folder, path)
#        children_list.append(NodeJob(name=f"{child:03}",
#                             parent=node,
#                             #bsub -q hpc_acc -e %J.err -o %J.out cd $PWD && ./run.sh
#                             #submit_command = (f'bsub -J {node.name}/{child:03} '
#                             #    '-q hpc_acc -e %J.err -o %J.out '
#                             #    f'{script_run_second_gen_path} &'),
#                             #submit_command = f'python {root.template_path}/multiply_it/run.py &',
#                             #log_file = 'log.json', 
#                             dictionary={'particle_file': f'../{distributions_folder_rel}/{child:03}.parquet',
#                                         'xline_json': f'../xsuite_lines/line_bb_for_tracking.json',
#                                         'n_turns': 1e3,
#                                      }))
#    node.children = children_list
root.to_json('tree_maker.json')



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

#for depth in range(root.height):
#    [x.clone_children() for x in root.generation(depth)]

for jj,ii in enumerate(root.dictionary['generations']):
    my_generation = root.dictionary['generations'][ii]
    Parallel(n_jobs=8)(delayed(x.clone_children)(my_generation['job_folder'],
						 files=(['config.yaml', my_generation['job_executable']]+
                                                        my_generation['files_to_clone']))  
                       for x in root.generation(jj))

# tagging
root.tag_as('cloned')
print('The tree structure is moved to the file system.')
print("--- %s seconds ---" % (time.time() - start_time))

# %%
# Mutation
start_time = time.time() 
for node in root.descendants:
   node.mutate()
print('The tree structure is mutated.')
print("--- %s seconds ---" % (time.time() - start_time))

# %%
# Prepare the run.sh

def string_run(node, generation_number):
	return f'''source {node.root.dictionary["setup_env_script"]}
cd {node.get_abs_path()}
python {node.root.dictionary["generations"][generation_number]["job_executable"]}'''


for generation_number in [1,2]:
    for node in root.generation(generation_number):
        file_name = node.get_abs_path()+'/run.sh'
        with open(file_name, 'w') as fid:
            fid.write(string_run(node, generation_number))
            os.system(f"chmod u+x {file_name}")
