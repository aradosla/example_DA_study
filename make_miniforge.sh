if [ $1 == 'use_cvmfs' ]
then
    . /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos8-gcc11-opt/setup.sh
else
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b  -p ./miniforge -f
    source miniforge/bin/activate
fi
python -m pip install ipython numpy scipy pandas psutil cpymad xsuite
mkdir modules
cd modules
git clone https://github.com/xsuite/tree_maker.git
python -m pip install -e tree_maker
git clone https://github.com/xsuite/xmask.git
pip install -e xmask
cd xmask/
git submodule init
git submodule update
cd ../../
xsuite-prebuild

