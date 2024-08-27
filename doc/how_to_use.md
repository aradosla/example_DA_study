
# How to use this template

## Running a simple parameter scan simulation

This section introduces the basic steps to run a simple parameter scan simulation. The simulation consists in tracking a set of particles for a given number of turns, and computing the dynamics aperture for each particle. To get a more refined understanding of what the scripts used below are actually doing, please check the section [What happens under the hood](#what-happens-under-the-hood).

### Setting up the study

You can select the range of parameters you want to scan by editing the ```studies/scripts/1_create_study.py``` script, under the section ```Machine parameters being scanned```. For example, you can edit the following lines to do a tune scan of your liking (here, only the first 6 tunes are selected, in order not to create too many jobs):

```python
array_qx = np.round(np.arange(62.305, 62.330, 0.001), decimals=4)[:6]
array_qy = np.round(np.arange(60.305, 60.330, 0.001), decimals=4)[:6]
```

Note that, if the parameter ```only_keep_upper_triangle``` is set to True, most of the jobs in the grid defined above will be automatically skipped as the corresponding working points are too close to resonance, or are unreachable in the LHC.

In addition, since this is a toy simulation, you also want to keep a low number of turns simulated (e.g. 200 instead of 1000000):

```python
n_turns = 200
```

you can give the study you're doing the name of your choice by editing the following line:

```python
study_name = "example_tunescan"
```

### Building the tree

You are now ready to create the folder structure (the *tree*) of your study. The tree structure can be checked in ```studies/scripts/config.yaml```. As you can see, there are only 2 generations here :

- the first generation generates the particles distribution and build a collider with just the optics (which we call "base collider" from now on).
- the second generation sets all the other collider parameters, including the ones that are being scanned, and tracks the particles for a given number of turns.

For now, you might want to keep the jobs running on your local machine, on the cpu, to ensure everything runs fine. To do so, ensure that in the file ```studies/scripts/config.yaml```, there is, for both generations:

```yaml
run_on: 'local_pc'
context: 'cpu'
```

If not already done, activate the conda environment:

```bash
poetry shell
```

Now, move to the master_study folder, and run to script to build the tree and write it on disk:

```bash
cd studies/scripts/
python 1_create_study.py
```

This should create a folder named after ```study_name``` in ```studies/scans```. This folder contains the tree structure of your study: the parent generation is in the subfolder ```base_collider```, while the subsequent children are in the ```xtrack_iiii```. The tree_maker ```.json``` and ```.log``` files are used by tree_maker to keep track of the jobs that have been run and the ones that are still to be run.

Each node of each generation contains a ```config.yaml``` file that contains the parameters used to run the corresponding job (e.g. the particle distributions parameters or the collider crossing-angle for the first generation, and, e.g. the tunes and number of turns simulated for the second generation).

You should be able to run each job individually by executing the following command in the corresponding folder:

```bash
source run.sh
```

Note that, to run without errors, children nodes will most likley need the files output by the parent nodes. Therefore, you should run the parent nodes first, and then the children nodes. However, this is all done automatically by the ```studies/scripts/2_run_jobs.py``` script (cf. upcoming section), such that running manually the ```run.sh``` should be done only for debugging purposes.

### Running the jobs

First, update the study name in ```studies/scripts/2_run_jobs.py```. You can now execute the script:

```bash
python 2_run_jobs.py
```

Here, this will run the first generation (```base_collider```), which consists of only one job (building the particles distribution and the base collider).

In a general way, once the script is finished running, executing it again will check that the jobs have been run successfully, and re-run the ones that failed. If no jobs have failed, it will run the jobs from the next generation. Therefore, executing it again should launch all the tracking jobs (several for each tune, as the particle distribution is split in several files).

⚠️ **If the generation of the simulation you're launching comprises many jobs, ensure that you're not running them on your local machine (i.e. you don't use ```run_on: 'local_pc'``` in ```studies/scripts/config.yaml```). Otherwise, as the jobs are run in parallel, you will most likely saturate the cores and/or the RAM of your machine.**

### Analyzing the results

Once all jobs of all generations have been computed, the results from the simulations can be gathered in a single dataframe by running the ```studies/scripts/3_postprocess.py``` script. First, make sure to update the study name in the script. Then, ensure that the jobs will be grouped by the variable that have been scanned (here, the tunes) by editing the following line:

```python
l_group_by_parameters = ["beam", "name base collider", "qx", "qy"]
```

Finally, run the script:

```bash
python 3_postprocess.py
```

This should output a parquet dataframe in ```studies/scans/study_name/```. This dataframe contains the results of the simulations (e.g. dynamics aperture for each tune), and can be used for further analysis. Note that, in the toy example above, since we simulate for a very small number of turns, the resulting dataframe will be empty as no particles will be lost during the simulation.

## What happens under the hood

The aim of this set of scripts is to run sets of simulations in a fast and automated way, while keeping the possibility to run each simulation individually.

### Tree structure

Since simulations all make use of the same base collider, the base collider only needs to be built once, as the optics is never "scanned". However, each simulation corresponds to a different set of parameters, meaning that the base collider needs to be tailored ("tuned") to each simulation. Therefore, the base collider will correspond to generation 1, and the subsequent tracking simulations with different parameters will correspond to generation 2.

This is described in the file ```studies/scripts/config.yaml```:

```yaml
'root':
  setup_env_script: 'none'
  # Following parameter is ignored when run_on is not htc_docker or slurm_docker
  singularity_image: "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cdroin/da-study-docker:7844a437"
  generations:
    1: # Build the particle distribution and base collider
      job_folder: "../../template_jobs/1_build_distr_and_collider"
      job_executable: 1_build_distr_and_collider.py
      files_to_clone:
        - optics_specific_tools.py
      run_on: "local_pc"
      context: "cpu"
      # Following parameter is ignored when run_on is not htc or htc_docker
      htc_job_flavor: 'tomorrow'
    2: # Launch the pymask and prepare the colliders
      job_folder: "../../template_jobs/2_configure_and_track"
      job_executable: 2_configure_and_track.py
      files_to_clone:
        - misc.py
      run_on: "local_pc"
      context: "cpu"
      # Following parameter is ignored when run_on is not htc or htc_docker
      htc_job_flavor: 'tomorrow' 
  # Children will be added below in the script 001_make_folders.py
  children:
```

This file defines the structure of the tree of jobs, which python files must be called at each generation, with which parameters, with which python distribution, and on which machine/cluster. More precisely:

- ```setup_env_script``` is the path to the conda environment that will be used to run Python in the simulations. By default, is set to ```none```, but is updated by the ```1_create_study.py``` script to point to the conda environment.
- ```singularity_image```: this is an optional parameter that can bmust be specified when running a simulation with ```htc_docker``` or ```slurm_docker```. This is useful to ensure reproducibility. See document [Clusters and GPUs](clusters_and_GPUs.md) for more information.
- ```job_folder```: for each generation, this describes the folder containing the files used to run the simulation. There should be at least a python script, and a ```config.yaml``` file. The python script then reads the parameters of the currunt simulation in the ```config.yaml file```, and runs the simulation accordingly.
- ```job_executable```, this is the name of the python script that will be run at each generation.
- ```files_to_clone```: this is a list of files that will be copied from the ```job_folder``` to the simulation folder. This is useful to copy files that are common to all simulations.
- ```run_on```: this is the machine/cluster on which the simulations will be run. At the moment, the following options are available (See document [Clusters and GPUs](clusters_and_GPUs.md) for more information.):
  - ```local_pc```: the simulations will be run on the local machine. This is useful when running small number of jobs, or debugging purposes.
  - ```htc```: the simulations will be run on the HTCondor cluster at CERN. This is useful to run large sets of simulations.
  - ```slurm```: the simulations will be run on the Slurm cluster at CNAF.INFN. This is useful to run large sets of simulations.
  - ```htc_docker```: the simulations will be run on the HTCondor cluster at CERN, using Docker images. This is useful to run large sets of simulations, and/or to ensure reproducibility.
  - ```slurm_docker```: the simulations will be run on the Slurm cluster at CNAF.INFN, using Docker images. This is useful to run large sets of simulations, and/or to ensure reproducibility.
- ```context```: this is the python distribution that will be used to run the simulations. At the moment, the following options are available:
  - ```cpu```: the simulations will be run on the CPU. This is the default option, and is useful when running simulations with small number of particles, or for debugging purposes.
  - ```cupy```: the simulations will be run on the GPU using CUDA. This is useful to run simulations with large number of particles (you need to have cupy installed), especially on HTCondor. However, note that simulations must use Docker when running on HTCondor with GPU.
  - ```opencl```: the simulations will be run on the GPU using OpenCL. This is useful to run simulations with large number of particles (you need to have pyopencl installed), especially on the Bologna cluster.
- ```htc_job_flavor```: this is an optional parameter that can be used to define the job flavor on HTCondor. Long jobs (>8h, <24h) should most likely use ```tomorrow```. See all flavours [here](https://batchdocs.web.cern.ch/local/submit.html).
- ```children```: this is a list of children for each generation. More precisely, this contains the set of parameters used by each job of each generation. This is generated by the ```1_create_study.py``` script, and should not be modified manually.

To get a better idea of how this file is used, you can check the json mutated version at the root of each scan (i.e. ```studies/scans/study_name/tree_maker.json```).

Here, only two generations are used, but it is possible to add more generations if needed. For instance, if one wants to run intricated grid searches, e.g. check the dynamics aperture for each tune and each crossing angle, one could build the tree such that the tunes are scanned at generation 2, and the crossing angles at generation 3. However, this is not implemented yet, you will have to modify the scripts yourself.

### Overall pipeline

When doing a parameter scan, the following steps are performed:

1. Running the ```1_create_study.py``` script. This creates the folder structure for the simulations, and generate the ```config.yaml``` files for each simulation. It also generates the ```tree_maker.json``` file at each generation of the scan. In this file, the following parameters are set:
    - the base parameters for the collider, that is, the parameters of the collider that might be changed from one study to the other (e.g. optics being used), but that will be the same for all simulations of the study (parameters being scanned excluded).
    ⚠️ **It is possible that you need to update other collider parameters (e.g. ```on_a5```). In this case, you can either update directly the master configuration file in ```studies/template_jobs/1_build_distr_and_collider/config.yaml```, or adapt the ```1_create_study.py``` script to update the collider parameters you need.**
    - the parameters for the initial particles distribution. One parameter that is important here is ```n_split```, as it sets how much a given working point will be split into different simulations, each containing a subset of the inital particles distribution. That is, ```n_split``` is actually responsible to a large extent for the parallelization of the simulations.
  
    All these parameters are added to the root of the main configuration file (```studies/scans/study_name/config.yaml```). The tree_maker package then takes care of providing the right set of parameters to the right python file for each generation. In practice, the template jobs (located in ```studies/template_jobs```) are copied to the simulation folders, and the corresponding ```config.yaml``` (e.g. ```studies/template_jobs/1_build_distr_and_collider/config.yaml```) file is adapted (mutated) for each generation and each simulation, according to the main tree_maker configuration file, which has been generated at the same time as the simulation folders (e.g. in ```studies/scans/study_name/tree_maker.json```).

2. Running the ```2_run_jobs.py``` script. This script will run the simulations in parallel, and output a file (e.g. a collider json file, or a dataframe containing the result of the tracking) for each simulation. In practice, it calls each ```run.sh``` script in each simulation folder, which in turn calls the python script defined in the ```job_executable``` parameter of the ```studies/scripts/config.yaml``` file. The python script makes use of the proper set of parameters, set in the mutated ```config.yaml``` files (one per job, e.g. ```studies/scans/study_name/base_collider/xtrack_0001/config.yaml```).
3. Running the ```3_postprocess.py``` script. This script will analyse the results of the simulations, and output a summary dataframe at the root of the study.
