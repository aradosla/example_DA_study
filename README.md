# FMA injection following example_DA_study
This repository contains a template that allows users to compute the FMA of a collider
under different parametric scenarios (this one is at injection but can be easily adapted.).

## Installation instructions

### Cloning the repository and its submodules

Run the following command to clone the repository and all the relevant submodules:

git clone --branch runIII_2026_injection_fma --single-branch https://github.com/aradosla/example_DA_study.git

### Python installation

1) You can create a miniforge environment, look for instructions: https://github.com/conda-forge/miniforge.
Or use:
```
cd
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b  -p ./miniforge -f
rm -f Miniforge3-Linux-x86_64.sh
source miniforge/bin/activate
```

4) In the example_DA_study repository there is a file requirements.txt 

Manual packages installation:

```bash
pip install -r requirements.txt
```
The requirements might be outdated, check if you have the packages...

3) We need to zip the envirnment for easier transfer later to the nodes. Store in eos, not afs!
Use:
```bash
pip install conda-pack
conda-pack miniforge envs.tar.gz (The name of the folder where your environment is stored. If you are using eos store it there! NOT on AFS please!)
```

Then you should be set.

### Paths modification

In the repository studies/scripts/ you need to modify the config.yaml. 

```
"root":
  setup_env_script: "none"
  # Following parameter is ignored when run_on is not htc_docker or slurm_docker
  singularity_image: "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cdroin/da-study-docker:74ed75ec"
  # use_eos_for_large_files: true
  eos_python: "root://eosuser.cern.ch//eos/user/a/aradosla/envs/envs.tar.gz" # PUT YOUR python path
  generations:
    1: # Build the particle distribution and base collider
      job_folder: "../../template_jobs/1_build_distr_and_collider"
      job_executable: 1_build_distr_and_collider.py
      files_to_clone: # relative to the template folder
        - optics_specific_tools.py
      files_to_copy_node:
        - 1_build_distr_and_collider.py
        - config.yaml
        - optics_specific_tools.py
        - "/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/optics/optics.zip"  # Generally you need a zipped optics.zip, in the current setup you can remove this line. The optics files are in: /afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/Proton_2025/NomH_RPV/opticsfile.46 for example.
      files_to_return:
        - collider.json.zip
        - output_python.txt
        - error_python.txt
        - tree_maker.json
        - tree_maker.log
        - particles
        - config_gen1.yaml
      run_on: "htc" # "local_pc" 'htc_docker' #'htc' #'slurm' #'slurm_docker'
      context: "cpu" # 'cupy' # opencl # how to run the simulation
      # Following parameter is ignored when run_on is not htc or htc_docker
      htc_job_flavor: "microcentury" # optional parameter to define job flavor, default is espresso
    2: # Launch the pymask and prepare the colliders
      job_folder: "../../template_jobs/2_configure_and_track"
      job_executable: 2_configure_and_track.py # has to be a python file
      files_to_clone:
        - misc.py
      files_to_copy_node:
        - 2_configure_and_track.py
        - config.yaml
        - misc.py
        - "/afs/cern.ch/work/a/aradosla/private/runIII_2025/studies/filling_scheme" # change with  ../../../../filling_scheme
        - "../config_gen1.yaml"
        - "../collider.json.zip"
        - "../particles"
      files_to_return:
        - output_particles.parquet
        - output_python.txt
        - error_python.txt
        - tree_maker.json
        - tree_maker.log
        - config_final.yaml
      context: "cpu" # 'cupy' # opencl # how to run the simulation
      run_on: "htc" # 'local_pc' # 'htc_docker' #'htc' #'slurm' #'slurm_docker'
      # Following parameter is ignored when run_on is not htc or htc_docker
      htc_job_flavor: "tomorrow" # optional parameter to define job flavor, default is espresso

```

## Running the study

- Change the parameters in studies/scripts/1_create_study.py and run. (Job creations. in studies/scan/)
- Run 2_run_jobs.py two times (First time we create the basic collider with the correct optics and the second time we track and obtain the results).
- In my_scripts open fma.py and change the path to the one where the simulation results are stored and run. 
