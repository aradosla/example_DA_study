. /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos8-gcc11-opt/setup.sh
export username=$(whoami)
export first_letter=${username:0:1}
PYTHONUSERBASE=/eos/user/$first_letter/$username/.local/ pip install --user -r requirements.txt
cd modules
PYTHONUSERBASE=/eos/user/$first_letter/$username/.local/ pip install --user tree_maker
PYTHONUSERBASE=/eos/user/$first_letter/$username/.local/ pip install --user xmask
cd ..
xsuite-prebuild

