# Dynamics aperture study boilerplate

This repository contains a boilerplate that allows users to compute the dynamics aperture of a collider
under different parametric scenarios. 

Jobs can be efficiently stored and parallelized using the
[Tree Maker](https://github.com/xsuite/tree_maker) package, while collider generation and particle tracking is performed using [X-Suite](https://github.com/xsuite/xsuite).

## Installation instructions

The simplest way to start is to clone the repository and install the dependencies using conda:

```bash
git clone git@github.com:ColasDroin/example_DA_study.git -b xmask_test
```

Then: 

```bash
cd example_DA_study
source make_miniconda.sh
```

This should install miniconda along with the required python modules. If something goes wrong, you can execute the commands in the ```make_miniconda.sh``` script manually, one line after the other.

**Warning**  
Please note that this example makes use of the HL-LHC optics files on the CERN AFS disk (e.g. ```/afs/cern.ch/eng/lhc/optics/HLLHCV1.5```). If you don't have access to AFS, you will have to install the files manually, [as done in the previous versions of this repository](https://github.com/xsuite/example_DA_study/blob/release/v0.1.1/make_miniconda.sh).

## Running a simple tune scan simulation

### Setting up the study

You can select the range of tunes you want to scan by editing the following lines in ```master_study/001_make_folders.py``` (here, only the first 10 tunes are selected, in order not to create too many jobs):

```python
array_qx = np.round(np.arange(62.305, 62.330, 0.001), decimals=4)[:10]
array_qy = np.round(np.arange(60.305, 60.330, 0.001), decimals=4)[:10]
```

Most likely, since this is a toy simulation, you also want to keep a low number of turns simulated (e.g. 200 instead of 1000000):
    
```python
n_turns = 200
```

Give the study you're doing the name of your choice by editing the following line:

```python
study_name = "example_HL_tunescan"
```

### Building the tree

You're now ready to create the folder structure (the _tree_) of your study. The tree structure can be checked in ```master_study/config.yaml```. As you can see, there are only 2 generations here :

- the first generation generates the particles distribution and build a collider with "base" parameters (parameters that are kept constant during the study)
- the second generation tunes the base collider(in here, this means changing the tunes) and tracks the particles for a given number of turns.

For now, you might want to keep the jobs running on your local machine to ensure everything runs fine. To do so, edit ```master_study/config.yaml``` to have, for both generations:

```yaml
run_on: 'local_pc'
```

If not already done, activate the conda environment:

```bash
source miniconda/bin/activate
```

Now, build the tree and write it on disk with:

```bash
python master_study/001_make_folders.py
```

This should create a folder named after ```study_name``` in ```master_study/scans```. This folder contains the tree structure of your study: the parent generation is in the subfolder ```base_collider```, while the subsequent children are in the ```xtrack_iiii```. The tree_maker ```.json``` and ```.log``` files are used by tree_maker to keep track of the jobs that have been run and the ones that are still to be run.

Each node of each generation contains a ```config.yaml``` file that contains the parameters used to run the corresponding job (e.g. the particle distributions parameters or the collider crossing-angle for the first generation, and, e.g. the tunes and number of turns simulated for the second generation).

You should be able to run each job individually by executing the following command in the corresponding folder:

```bash
source run.sh
```

Note that, to run without errors, children nodes will most likley need the files output by the parent nodes. Therefore, you should run the parent nodes first, and then the children nodes. However, this is all done automatically by the ```master_study/002_chronjob.py``` script (cf. upcoming section), such that running manually the ```run.sh``` should be done only for debugging purposes.

### Running the jobs

First, update the study name in ```master_study/002_chronjob.py```. You can now execute the script:

```bash
python master_study/002_chronjob.py
```

Here, this will run the first generation (```base_collider```), which consists of only one job (building the particles distribution and the base collider).

In a general way, once the script is finished running, executing it again will check that the jobs have been run successfully, and re-run the ones that failed. If no jobs have failed, it will run the jobs from the next generation. Therefore, executing it again should launch all the tracking jobs (several for each tune, as the particle distribution is split in several files).


## Using comuting clusters

The scripts in the repository allows for an easy deployment of the simulations on HTCondor (CERN cluster) and Slurm (CNAF.INFN cluste). Please consult the corresponding tutorials ([here](https://abpcomputing.web.cern.ch/guides/htcondor/), and [here](https://abpcomputing.web.cern.ch/computing_resources/hpc_cnaf/)) to set up the clusters on your machine.

Once, this is done, jobs can be executed on HTCondor by setting ```run_on: 'htc'``` instead of ```run_on: 'local_pc'``` in ```master_study/config.yaml```. Similarly, jobs can be executed on the CNAF cluster by setting ```run_on: 'slurm'```.

### Activate the environment
I suggest to open a `tmux` terminal and 

```bash
source /home/HPC/sterbini/py38/bin/activate
```
then move where you want to start lauch this DA study and make a clone of this repository

```
git clone https://github.com/sterbini/DA_study_example.git
cd DA_study_example
```

### Tree creation
With the command
```bash
python 001_make_folders.py
```
one creates the `study_000` tree in the `DA_study_example`.
It consists of a 21x21 folders (a tune scan using the pymask in `000_machine_model`).
Each folders has 15 subfolders (each is launching an `xtrack` jobs of 42 or 43 particles, in total the particles are 640).

All the details are in the code `001_make_folders.py`.

### Launching the simulation

One can launch the first generation of of jobs (441 pymasks) by 
```
python 002_chronjob.py
```
And you can repeat the same command to advance in the tree genealogy and launch the tracking (the code knows when is ready to launch the second generation).
In fact this could be implemented in a chron job.

One can monitor the status of the jobs by
```
bjobs -sum -u user_name 
```












