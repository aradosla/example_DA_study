# Using computing clusters and/or GPUs

## General procedure

The scripts in the repository allows for an easy deployment of the simulations on HTCondor (CERN cluster) and Slurm (CNAF.INFN cluster). Please consult the corresponding tutorials ([here](https://abpcomputing.web.cern.ch/guides/htcondor/), and [here](https://abpcomputing.web.cern.ch/computing_resources/hpc_cnaf/)) to set up the clusters on your machine.

Once, this is done, jobs can be executed on HTCondor by setting ```run_on: 'htc'``` instead of ```run_on: 'local_pc'``` in ```studies/scripts/config.yaml```. Similarly, jobs can be executed on the CNAF cluster by setting ```run_on: 'slurm'```.

## Using Docker images

For reproducibility purposes and/or limiting the load on AFS or EOS drive, one can use Docker images to run the simulations. A registry of Docker images is available at ```/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/```, and some ready-to-use for DA simulations Docker images are available at ```/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cdroin/da-study-docker``` (for now, this is the default directory for images in the ```2_run_jobs.py``` file). To learn more about building Docker images and hosting them on the CERN registry, please consult the [corresponding tutorial](https://abpcomputing.web.cern.ch/guides/docker_on_htcondor/) abd the [corresponding repository](https://gitlab.cern.ch/unpacked/sync).

An repository containing a working Docker image with both HTCondor and Bologna is available [here](https://gitlab.cern.ch/cdroin/da-study-docker).

### HTCondor

When running simulations on HTCondor, Docker images are be pulled directly from CVMFS through the submit file. No additional configuration is required, except for setting ```run_on: 'htc_docker'``` in ```studies/scripts/config.yaml```.

### Slurm

Things are a bit tricker with Slurm, as the Docker image must first be manually pulled from CVMFS, and then loaded on the node after Singularity-ize it. The pulling of the image is only needed the first time, and can be done with e.g. (for the image ```cdroin/da-study-docker```):
  
  ```bash
  singularity pull docker://gitlab-registry.cern.ch/cdroin/da-study-docker:6eb787ea
  ```

However, due to unknown reason, only some nodes of INFN-CNAF will correctly execute this command. For instance, it didn't work on the default CPU CERN node (```hpc-201-11-01-a```), but it did on an alternative one (```hpc-201-11-02-a```). We recommand using either ```hpc-201-11-02-a``` or a GPU node (e.g. ```hpc-201-11-35```) to pull the image. Once the image is pulled, it will be accessible from any node.

For testing purposes, one can then run the image with Singularity directly on the node (not required):
  
  ```bash
  singularity run da-study-docker_6eb787ea.sif
  ```

In practice, the ```2_run_jobs.py``` script will take care of mouting the image and running it on the node with the correct miniforge distribution. All you have to do is change the ```run_on``` parameter in ```studies/scripts/config.yaml``` to ```run_on: 'slurm_docker'```.

⚠️ **The slurm docker scripts are a still experimental**. For instance, at the time of writing this documentation, symlinks path in the front-end node of INFN-CNAF are currently broken, meaning that some temporary fixs are implemented. This will hopefully be fixed in the future, and the fixs should not prevent the scripts from running anyway.

## Using GPUs

When simulating massive number of particles, using GPUs can prove to be very useful for speeding up the simulations. The use of GPUs should be as transparent as possible to the user, through the parameter ```context``` in ```studies/scripts/config.yaml```. As explained above, (default) CPU computations are performed with ```context: 'cpu'```, while GPU computations are performed with ```context: 'cupy'``` or ```context: 'opencl'```. The last two cases require the installation of the corresponding packages (```cupy``` or ```pyopencl```).

It is strongly advised to use a Docker image to run simulations on clusters (i.e. ```run_on: htc_docker```or ```run_on: slurm_docker```), as this will ensure reproducibility and avoid any issue with the installation of the packages. In addition, the Docker image will automatically mount the GPUs on the node, and the simulations will be run on the GPU without any additional configuration.
