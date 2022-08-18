wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b  -p ./miniconda -f
source miniconda/bin/activate
python -m pip install -r requirements.txt
git clone -b release/v0.1.0 git@github.com:xsuite/tree_maker.git
python -m pip install -e tree_maker
git clone git@github.com:lhcopt/lhcmask.git
python -m pip install -e lhcmask
git clone git@github.com:lhcopt/lhcerrors.git
git clone git@github.com:lhcopt/lhctoolkit.git
git clone git@github.com:lhcopt/hllhc15.git
git clone $(whoami)@lxplus.cern.ch:/afs/cern.ch/eng/lhc/optics/runIII
cd master_study/master_jobs/001_machine_model/
python 001_copy_from_pymask_examples.py
cd ../../../
