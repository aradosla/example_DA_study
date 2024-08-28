# Dynamics aperture study template

This repository contains a template that allows users to compute the dynamics aperture of a collider
under different parametric scenarios.

Jobs can be efficiently stored and parallelized using the
[Tree Maker](https://github.com/xsuite/tree_maker) package, while collider generation and particle tracking harness the power of [Xsuite](https://github.com/xsuite/xsuite).

ℹ️ If you do not need to do large parametric scans, this template is probably not what you're looking for! You should instead refer to the [simple DA study repo](https://github.com/ColasDroin/simple_DA_study), which adapts the template to run tracking simulation without doing any scan or parallelized cluster submission.

## Quick installation guide

For most scans (with GPUs not involved), the small guide below should be enough to get you started. Otherwise, or if you encounter problems, please refer to the [full installation guide](doc/installation_guide.md).

Ensure Python (at least 3.9) is available on your machine. If not, install it with, for instance, miniforge or miniconda. Since this Python version will be your base one, it is recommended to install it in a fast location (for CERN users, *not AFS*), such that it can be used by all your projects (creating virtual environment everytime).

For instance, to install Python with miniforge in your home directory, run the following commands:

```bash
cd
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b  -p ./miniforge -f
rm -f Miniforge3-Linux-x86_64.sh
source miniforge/bin/activate
```

If you plan to submit jobs to CERN HTCondor, now move to your AFS space (e.g. ```cd /afs/cern.ch/work/c/username/private```).

In any case, run the following command to clone the repository and all the relevant submodules.

```bash
git clone --recurse-submodules https://github.com/xsuite/DA_study_template.git my_DA_study
```

If not already done, install Poetry (system-wide) with the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Note that, at this point, you might need to add Poetry to your PATH. Simply follow what the Poetry installer tells you to do. For instance, for bash, you will have to run something like:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

If everything worked, impose the virtual environment to be created in the repository folder (needed so that jobs from HTCondor can access your Python distribution, especially if you don't plan on using a docker image) by running the following command:

```bash
poetry config virtualenvs.in-project true
```

Finally, move to your cloned repository and install the dependencies by running the following command:

```bash
cd my_DA_study
poetry install
```

You can make Xsuite faster with the following command:

```bash
poetry run xsuite-prebuild regenerate
```

In the future, don't forget to to activate the virtual environment when accessing the repository from a fresh terminal session with:

```bash
poetry shell
```

You can ensure that everything works by:
  
1. Creating a study with the following command:
  
    ```bash
    cd studies/scripts
    python 1_create_study.py
    ```

2. Running the jobs with the following command:
  
    ```bash
    python 2_run_jobs.py
    ```

    By default, you will need to run this command twice: first time, the base collider will be built locally, and the second time, the tracking jobs (by default, a small tune scan with only 50000 turns) will be run on HTCondor. You can check the job status by rerunning the command as many times as needed.

3. Postprocessing the results with the following command:
  
    ```bash
    python 3_postprocess.py
    ```

    At this point, you should have an output parquet file in the ```studies/scans/study_name``` folder.

## How to use this template

You can refer to the [how-to guide](doc/how_to_use.md) for more information on how to use this simulation template.

## Using computing clusters and/or GPUs

You can refer to [clusters and GPUs documentation](doc/clusters_and_GPUs.md) for more information on how to use computing clusters and GPUs.

## Previous documentation

More information, although possibly outdated, can be gathered from the explanations provided in previous versions of this repository, e.g. [in the previous README](https://github.com/xsuite/example_DA_study/blob/release/v0.1.1/README.md) and [the Tree Maker tutorial](https://github.com/xsuite/example_DA_study/blob/release/v0.1.1/tree_tutorial.md).

The code is now well formatted and well commented, such that any question should be relatively easily answered by looking at the code itself. If you have any question, do not hesitate to contact me (colas.noe.droin at cern.ch) or open an issue.

## Parameters that can be scanned

At the moment, all the collider parameters can be scanned without requiring extensive scripts modifications. This includes (but is not limited to):

- intensity (```num_particles_per_bunch```)
- crossing-angle (```on_x1, on_x5```)
- tune (```qx, qy```)
- chromaticity (```dqx, dqy```)
- octupole current (```i_oct_b1, i_oct_b2```)
- bunch being tracked (```i_bunch_b1, i_bunch_b2```)

At generation 1, the base collider is built with a default set of parameters for the optics (which are explicitely set in ```1_create_study.py```). At generation 2, the base collider is tailored to the parameters being scanned. That is, the tune and chroma are matched, the luminosity leveling is computed (if leveling is required), and the beam-beam lenses are configured.

It should be relatively easy to accomodate the scripts for other parameters.

## License

This repository is licensed under the MIT license. Please refer to the [LICENSE](LICENSE) file for more information.
