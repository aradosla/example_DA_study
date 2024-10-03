# Installation guide

## Installing Python

Python is probably already available on your machine. You can check by running the following commands:

```bash
which python # Tells you the path to the python executable
python --version # Tells you the version of Python
```

If Python (>3.8) is not available, you can install it with, for instance, miniforge or miniconda. Since this Python version will be your base one, it is recommended to install it in a fast location (for CERN users, *not AFS*), such that it can be used by all your projects (creating virtual environment everytime).

For instance, to install the latest version of Python with miniforge in your home directory, run the following commands:

```bash
cd
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b  -p ./miniforge -f
rm -f Miniforge3-Linux-x86_64.sh
source miniforge/bin/activate
```

## Cloning the repository and its submodules

Run the following command to clone the repository and all the relevant submodules. Note, that, if you plan to submit jobs to CERN HTCondor, you should move to your AFS space first (e.g. ```cd /afs/cern.ch/work/c/username/private```).:

```bash
git clone --recurse-submodules https://github.com/xsuite/example_DA_study.git
```

If you missed this step and clone the repository without the submodules, you can do *a posteriori*:

```bash
git submodule update --init --recursive
```

## Installing with Poetry

If not already done, install Poetry following the tutorial [here](https://python-poetry.org/docs/). Note that Poetry must have access to Python 3.9 or above for the rest of the tutorial to work. More importantly, the executable of Python must be accessible from a cluster node (e.g. located on AFS when submitting jobs to HTCondor) for a submission to work. Ideally, Poetry should use a Python distribution located on your local machine (shared acress all of your projects), and then use another Python distribution (e.g. miniforge or miniconda) on AFS for the simulations (which can be shared across projects or not, depending on the needs).

You can check the base executable of Python that Poetry is using by running the following command:

```bash
poetry env info
```

For easier submission later, impose the virtual environment to be created in the repository folder by running the following command:

```bash
poetry config virtualenvs.in-project true
```

If needed (for instance, if your Python base executable is not on AFS), you can change the exectutable with e.g:

```bash
poetry env use /path/to/python
```

If you're not interested in using GPUs, you can jump directly to the [Installing packages](#installing-packages) section. Otherwise, follow the next section.

## Installing with Poetry for GPUs

Using Poetry along with GPUs is a bit more complicated, as conda is not natively supported by Poetry. However, not all is lost as a simple trick allows to bypass this issue. First, from your (hopefully conda-compatible) Python environment, create a virtual environment with the following command:

```bash
conda create -n gpusim python=3.9
conda activate gpusim
```

⚠️ **Make sure that the Python version is 3.9 as, for now, a bug with Poetry prevents using 3.10 or above.**

Now configure Poetry to use the virtual environment you just created:
  
```bash
poetry config virtualenvs.in-project false
poetry config virtualenvs.path $CONDA_ENV_PATH
poetry config virtualenvs.create false
```

Where ```$CONDA_ENV_PATH``` is the path to the base envs folder (e.g. ```/home/user/miniforge3/envs```).  

You can then install the CUDA toolkit and the necessary packages (e.g. ```cupy```) in the virtual environment (from [Xsuite documentation](https://xsuite.readthedocs.io/en/latest/installation.html#gpu-multithreading-support) ):

```bash
conda install mamba -n base -c conda-forge
mamba install cudatoolkit=11.8.0
```

Finally, you need to modify the path to the virtual environment in the ```source_python.sh``` file.

Replace the line 17 ```source $SCRIPT_DIR/.venv/bin/activate``` with the path to the conda environment you just used to create the virtual environment (e.g. ```source /home/user/miniforge3/bin/activate```, and add below ```conda activate gpusim```).

You're now good to go with the next section, as Poetry will automatically detect that the conda virtual environment is activated and use it to install the packages.

## Installing packages

Finally, install the dependencies by running the following command:

```bash
poetry install
```

At this point, ensure that a `.venv` folder has been created in the repository folder (except if you modified the procedure to use GPUs, as explained above). If not, follow the fix described in the next section.

⚠️ **If you have a bug with nafflib installation, do the following:**
  
  ```bash
  poetry run pip install nafflib
  ```

  ⚠️ **If you have a bug with conda compilers, do the following:**

  ```bash
  poetry shell
  conda install compilers cmake
  ```

Finally, you can make xsuite faster by precompiling the kernel, with:

```bash
poetry run xsuite-prebuild regenerate
```

To run any subsequent Python command, either activate the virtual environment (activate a shell within Poetry) with:

```bash
poetry shell
```

or run the command directly with Poetry:

```bash
poetry run python my_script.py
```

## Fix the virtual environment path

If, for some reason, your virtual environment is not in a `.venv`folder inside of your repository, you will have to
update the submitting script to point to the correct virtual environment. To that end, run the following command:

```bash
poetry env list --full-path
```

Identify the virtual environment that is being used and copy the corresponding path. Now, open the file `source_python.sh` and replace the line `source $SCRIPT_DIR/.venv/bin/activate`with the path to the virtual environment you just found (e.g. `source /path/to/your/virtual/environment/bin/activate`).

## Installing without Poetry

It is strongly recommended to use Poetry as it will handle all the packages dependencies and the virtual environment for you. However, if you prefer to install the packages manually, you can do so by running the following commands (granted that you have Python installed along with pip):

```bash
pip install -r requirements.txt
```
