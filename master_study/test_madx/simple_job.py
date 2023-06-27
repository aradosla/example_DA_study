from cpymad.madx import Madx

str_LHC_seq_b12 = """
! Get the toolkit
call,file="/afs/cern.ch/eng/lhc/optics/runIII/toolkit/macro.madx";
option,-echo,-warn;
call,file="/afs/cern.ch/eng/lhc/optics/runIII/lhc_acc-models-lhc.seq";
nrj=6800;  
beam,particle=proton,sequence=lhcb1,energy=nrj,npart=1.15E11,sige=4.5e-4;
beam,particle=proton,sequence=lhcb2,energy=nrj,bv = -1,npart=1.15E11,sige=4.5e-4;
"""

str_LHC_seq_b4 = """
option,-echo,-warn;
call,file="/afs/cern.ch/eng/lhc/optics/runIII/lhc_acc-models-lhc_b4.seq";
nrj=6800;  
beam,particle=proton,sequence=lhcb2,energy=nrj,bv = 1,npart=1.15E11,sige=4.5e-4;
"""

str_strength_seq_2023 = """
call,file="/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/Proton_2023/opticsfile.34"; 
"""

str_strength_seq_2024 = """
call,file="/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/Proton_2024/V0/opticsfile.34";
exec, myslice;
"""

str_twiss_b12 = """
use,sequence=lhcb1;
twiss;
value, table(twiss,IP1,betx),table(twiss,IP2,betx),table(twiss,IP5,betx),table(twiss,IP8,betx);
use,sequence=lhcb2;
twiss;
value, table(twiss,IP1,betx),table(twiss,IP2,betx),table(twiss,IP5,betx),table(twiss,IP8,betx);
"""

str_twiss_b4 = """
use,sequence=lhcb2;
twiss;
value, table(twiss,IP1,betx),table(twiss,IP2,betx),table(twiss,IP5,betx),table(twiss,IP8,betx);
"""


str_seq_2024_b12 = str_LHC_seq_b12 + str_strength_seq_2024 + str_twiss_b12
str_seq_2024_b4 = str_LHC_seq_b4 + str_strength_seq_2024 + str_twiss_b4
str_seq_2023_b12 = str_LHC_seq_b12 + str_strength_seq_2023 + str_twiss_b12
str_seq_2023_b4 = str_LHC_seq_b4 + str_strength_seq_2023 + str_twiss_b4

madx = Madx()
madx.input(str_seq_2024_b12)
