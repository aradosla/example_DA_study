# **Tree**
<font size="4">This tutorial is made to understand several things:
- Why we use trees?
- How to use one that already works;
- How to create a new tree starting from the one that already works.

This tutorial is **super entry level** going step by step starting from building a miniconda to run your first lattice. If you are super skilled, sorry, you can skip this and try it yourself!
In particular we are trying to describe the know how to **use trees on HTCondor**, because this will be their primary usage for your studies.
## Why we use trees?
The first question might be "Do I really need trees? Can't I use my simple scripts that always worked in local? **It depends** .

If you are reading this you just started using HTCondor. If you just started your simulation could be very simple: generate particles, track them in some lattice, retrieve the results.
Now in this case **YOU DON'T NEED TREES**. 

Then why learn something new? Your study goes on and in two months you'll need to, for example, study how your distribution evolves in different lattices with small variations between them (tune scan for example). Will you submit 190 jobs manually by changing each config etc...?
Probably not. This is just one of the (infinitely) many examples that lead to the use of trees.

Learn it now on something easy and then extend it to something more complex. 

The first phase is to decide a structure for your simulation, this is  **very important**, before trying to change directories, write code etc... build a tree of your code with pen and paper.
Let's see some nomenclature and how to build a tree on pen and paper:
- **Simple Job** : a simple job is an atomic section of your big code that works on its own. This could be, for example, building particles or building the line or the tracking, etc... 

- **Generation** : given the simple job we take advantage of the tree structure to create "children" jobs from the simple one, with a twist: this children can have mutations! Example: father = particle distribution extraction. We have a simple job that generates a line, and we will use this to produce the children. We want to scan the Octupolar current (for example), by changing it by 10A on each different child. This is made possible by the tree and we can have 20 different lines each with a 10A variation on Octupolar current. So from ONE simple job we generated 20 different lines. This can then be applied again for the tracking for example, or could have been applied to the particle distribution extraction, so it's very powerful. You want 28 different distributions tracked in 7 different lattices with 11 different tracking conditions? Done!

Now what about our tree on pen and paper? A very easy one could be the following:
I want to random extract 10 sets of matched particles, use a line that is fixed, track the particles in this lattice. In this case my tree is very simple: it starts with 10 extractions and each of them will then be tracked in a separate job in HTCondor. In this way I have 10 separate jobs that run simultaneously. 

Now let's start with the practical part.
Some advices:

- HTCondor is connected to AFS, so everything that your simulation needs should be there;

- **ALWAYS** test the single jobs **BEFORE** building the tree;

- **THE PATHS MUST ALWAYS BE RELATIVE AND NOT ABSOLUTE**.

## **How to use a tree that already works?**

A useful (BUT COMPLICATED) template can be retrieved at https://github.com/xsuite/example_DA_study/tree/release/v0.1.1 (please notice that is not the master, is the v0.1.1). 

We don't need to understand all that stuff, we just need to go through the important concepts.

### **Understanding the /master_jobs**

The core of what you need to learn is inside master_study/master_jobs/, ignore everything else for now. Inside we can see three directories:
- 000_make_part_distribution
- 001_machine_model
- 002_tracking_job

**THIS STRUCTURE IS WHAT YOU SHOULD RECREATE FOR YOUR OWN TREE**: let's look into each directory.
#### 000_make_part_distribution
Three files, a clean_it.sh, useful to delete unwanted files when needed and the two crucial things, the config and the .py.
The config contains all the useful parameters for the .py and the tree_maker.log. This file will will contain informations regarding the status of this particular job, if it's running/completed etc...
```python
log_file: 'tree_maker.log'
```
The .py always needs to contain the lines that refer to the tree_maker.log at the start and at the end
So what is happening here it's easy: we are generating particles and writing on the log file the status of the operation.

#### 001_machine_model
In the example here is where things get complicated. The idea is easier. A clean_it.sh as before, and the config.yaml of lhcmask (with a new line of code added at the end, the log_file: 'tree_maker.log' as before). 001_copy_from_pymask_examples.py will create a line following the specifications of the config.yaml.

#### 002_tracking_job
Same concept, as you can see it's easy. Important things to notice: the config.yaml contains the paths to the directories where you generated the particles and the line and again a tree_maker.log.

**What is your goal for now?** Creating the same structure (master_jobs directory with the 000_make_part_distribution,001_machine_model,002_tracking_job subdirectories) with your own codes, and on your local pc run each program in each directory making sure that everything is working. If you have managed to accomplish this step we can move on.

### **Back to /master_study**
You should have noticed a problem: the config files in each directory are static, do I need to modify everything each time? Here in master_study lies the answer, that is obviously no.
#### config.yaml
Let's have a look at the config.yaml here, I'll comment it from here so that the explanation is easier to understand:

```python
'root':
  setup_env_script: '/home/sterbini/2022_08_08_wires/example_DA_study/miniconda/bin/activate' #we tell which python we want to use
  generations: #we define the generations! This is the tree!
    1: # Make the particle distribution #Everything starts by creating the particle distribution
      job_folder: 'master_jobs/000_make_part_distribution'
      job_executable: 000_make_distrib.py # has to be a python file
      files_to_clone: # relative to the template folder
        - clean_it.sh
      run_on: 'local_pc' #we want to run this section of the tree in local, we tipically track on htc and perform the rest in local, but it's up to you.
    2: # Launch the pymask and prepare the xlines
      job_folder: 'master_jobs/001_machine_model' 
      job_executable: 000_pymask.py # has to be a python file
      files_to_clone: # relative to the template folder
        - optics_specific_tools.py
        - clean_it.sh
      run_on: 'local_pc' #'htc'
      htc_job_flavor: 'microcentury' # optional parameter to define job flavor, default is espresso
    3: # Launch the tracking
      job_folder: 'master_jobs/002_tracking_job'
      job_executable: 000_track.py # has to be a python file
      files_to_clone: # relative to the template folder
        - clean_it.sh
      run_on: 'local_pc' #'htc'
      htc_job_flavor: 'tomorrow' # optional parameter to define job flavor, default is espress
  use_yaml_children: true # VERY IMPORTANT, I'll comment this after the code [1]
  # first generation
  children:
    '000_make_particle':
      children:
        '000_madx':
          qx0: 62.31
          qy0: 60.32
          test: 'test'
          children:
            '000_xsuite':
              particle_file: '../../particles/000.parquet' 
              xline_json:  '../xsuite_lines/line_bb_for_tracking.json'
              n_turns: 1000
            '001_xsuite':
              particle_file: '../../particles/000.parquet' 
              xline_json:  '../xsuite_lines/line_bb_for_tracking.json'
              n_turns: 2000
        '001_madx':
          qx0: 62.313
          qy0: 60.318
          children:
            '000_xsuite':
              particle_file: '../../particles/000.parquet' 
              xline_json:  '../xsuite_lines/line_bb_for_tracking.json'
              n_turns: 1000
```
**[1] :use_yaml_children:** If true is selected the lines from #first generation comment are used. This will create a "test" tree if you want, where you manually specified how to modify each config for each job. This is used for testing purposes. Let's see what happens in this case:
- We start by building the particles, this is the start of the tree;
- This father has two sons, 000_madx and 001_madx, two different lines with two different tunes configuration;
- 000_madx has two sons, a tracking with 1000 turns and one with 2000 turns, while 001_madx has just one son, a tracking with 1000 turns and every tracking is performed on just the 000.parquet.
As already explained this is just for testing purposes. We need to generate the whole tree with (possibly) tens or hundreds of children. This job is performed by make_folders.py.

#### 001_make_folders.py

In make_folders.py if use_yaml_children: true then the tree will be the test just described. If you select false then the complete system of directories will be created.
```python
# The user defines the variable to scan
# machine parameters scans
qx0 = np.arange(62.305, 62.330, 0.001)[:]
qy0 = np.arange(60.305, 60.330, 0.001)[:]

study_name = "HL_tunescan_20cm"

children={}
children[study_name] = {}
children[study_name]["children"] = {}

for optics_job, (myq1, myq2) in enumerate(itertools.product(qx0, qy0)):
    optics_children={}
    children[study_name]["children"][f'madx_{optics_job:03}'] = {
                                    'qx0':float(myq1),
                                    'qy0':float(myq2),
                                    'children':optics_children}
    for track_job in range(3):#in the original file this is 15, I switch to 3 just for readibility
        optics_children[f'xsuite_{track_job:03}'] = {
                    'particle_file': ('../../'
                                   f'distrib_abc/{track_job:03}.parquet'),
                    'xline_json': ('../xsuite_lines/'
                                  'line_bb_for_tracking.json'),
                    'n_turns': int(1000000)}
```
We create two arrays of tunes, and define a nested dictionary. In this case the particles are the same for each job, (and if you look back at the config of 000_make_part_distribution you can see that there are no specifications for the number of particles) so we just create the first children = {}.

Let's have a look at this dictionary, it's easier if we look at it (in the original for track_job in range(15), here for track_job in range(3) just to be clear again).
```python
{'HL_tunescan_20cm': {'children': {
  'madx_000': {'qx0': 62.305,
    'qy0': 60.305,
    'children': {'xsuite_000': {'particle_file': '../../distrib_abc/000.parquet',
      'xline_json': '../xsuite_lines/line_bb_for_tracking.json',
      'n_turns': 1000000},
     'xsuite_001': {'particle_file': '../../distrib_abc/001.parquet',
      'xline_json': '../xsuite_lines/line_bb_for_tracking.json',
      'n_turns': 1000000},
     'xsuite_002': {'particle_file': '../../distrib_abc/002.parquet',
      'xline_json': '../xsuite_lines/line_bb_for_tracking.json',
      'n_turns': 1000000}}},
   'madx_001': {'qx0': 62.305,
    'qy0': 60.306,
    'children': {'xsuite_000': {'particle_file': '../../distrib_abc/000.parquet',
      'xline_json': '../xsuite_lines/line_bb_for_tracking.json',
      'n_turns': 1000000},
     'xsuite_001': {'particle_file': '../../distrib_abc/001.parquet',
      'xline_json': '../xsuite_lines/line_bb_for_tracking.json',
      'n_turns': 1000000},
     'xsuite_002': {'particle_file': '../../distrib_abc/002.parquet',
      'xline_json': '../xsuite_lines/line_bb_for_tracking.json',
      'n_turns': 1000000}}},
   'madx_002': {'qx0': 62.305,
    'qy0': 60.306999999999995,
    'children': {'xsuite_000': {'particle_file': '../../distrib_abc/000.parquet',
      'xline_json': '../xsuite_lines/line_bb_for_tracking.json',
      'n_turns': 1000000},
     'xsuite_001': {'particle_file': '../../distrib_abc/001.parquet',
      'xline_json': '../xsuite_lines/line_bb_for_tracking.json',
      'n_turns': 1000000},
     'xsuite_002': {'particle_file': '../../distrib_abc/002.parquet',
      'xline_json': '../xsuite_lines/line_bb_for_tracking.json',
      'n_turns': 1000000}}},
      #....................................................................
      #....................................................................
      #....................................................................
      #....................................................................
    'madx_624': {'qx0': 62.328999999999944,
    'qy0': 60.328999999999944,
    'children': {'xsuite_000': {'particle_file': '../../distrib_abc/000.parquet',
      'xline_json': '../xsuite_lines/line_bb_for_tracking.json',
      'n_turns': 1000000},
     'xsuite_001': {'particle_file': '../../distrib_abc/001.parquet',
      'xline_json': '../xsuite_lines/line_bb_for_tracking.json',
      'n_turns': 1000000},
     'xsuite_002': {'particle_file': '../../distrib_abc/002.parquet',
      'xline_json': '../xsuite_lines/line_bb_for_tracking.json',
      'n_turns': 1000000}}}}}}
```
What have we done? We created a huge number of mutated children that, thanks to the relative paths and to the tree structure, cannot communicate between each other but just with their father/grandfather. This is the strength of the tree! 

In this case we call the study 'HL_tunescan_20cm', its own children are 624 different lines with different qx,qy. Now each of these own children are 3 (in the original 15) different trackings performed with that same line on the set of particle distributions created by the first job, going from 000.parquet to 002.parquet (014.parquet in the original).

From here on the hard part is finished. Again, this example is complicated, if you try to make a simple thing you will also understand better how the structure works but the idea should be clear by now.

#### 002_chronjob.py
This is how you launch the several generations! A simple procedure:

- python 002_chronjob.py;

- htop (local)/ condor_q(HTCondor) , so that you can see if the jobs are still running;

- if everything is finished in htop (local)/condor_q(HTCondor) do python 002_chronjob.py again, this will start the second generations;

- htop (local)/ condor_q(HTCondor) , and when everything is finished do python 002_chronjob.py again.

Notice: if everything is completed you will know it because 002_chronjob.py will tell you with a message. Notice that, thanks to the config.yaml we just need one program (002_chronjob.py) to launch either locally or on HTCondor each generation beacuse 001_make_folders.py already specified this in each folder thanks to the run_on: specification in the config.

The 003_postprocessing.py is just there as an example, you can postprocess in any way that you like.


## **How to create a new tree starting from the one that already works**

This procedure is just one option, it should make things easier. You can do whatever you want.
- Create a directory (maybe on afs so that it can be reached easily by the vm, HTCondor etc...) and build the miniconda you need there;

- git clone git@github.com:xsuite/example_DA_study.git -b release/v0.1.1 ;

- Create your own version of the master_jobs directory, with all the config.yaml etc...;

- Modify the general config.yaml to tell where the python environment is, define your children, the directories etc...; always test everything with 'run_on':'local_pc' and with use_yaml_children: True, so that the "simple" tree can be created and tested;

- Modify the 001_make_folders.py : this is obviously crucial, you need to define the dictionaries of your tree, so modify the for loop to your own needs;

- Optional: modify/add clean_it.sh files where needed to clean up the directories by the files that will be created by the jobs.

That's it, this will give you a working tree that can easily work both locally and on HTCondor. 

You can find some examples also at: https://github.com/xsuite/tree_maker/tree/release/v0.1.0/examples