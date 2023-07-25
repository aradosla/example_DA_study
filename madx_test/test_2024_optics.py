from cpymad.madx import Madx


def my_slice(mad, slice_factor=3):
    mad.input(f"slicefactor = {slice_factor};")
    mad.input("""
        myslice: macro = {
        if (MBX.4L2->l>0) {
          select, flag=makethin, clear;
          select, flag=makethin, class=mb, slice=2;
          select, flag=makethin, class=mq, slice=2 * slicefactor;
          select, flag=makethin, class=mqxa,  slice=16 * slicefactor;  !old triplet
          select, flag=makethin, class=mqxb,  slice=16 * slicefactor;  !old triplet
          select, flag=makethin, class=mqxc,  slice=16 * slicefactor;  !new mqxa (q1,q3)
          select, flag=makethin, class=mqxd,  slice=16 * slicefactor;  !new mqxb (q2a,q2b)
          select, flag=makethin, class=mqxfa, slice=16 * slicefactor;  !new (q1,q3 v1.1)
          select, flag=makethin, class=mqxfb, slice=16 * slicefactor;  !new (q2a,q2b v1.1)
          select, flag=makethin, class=mbxa,  slice=4;   !new d1
          select, flag=makethin, class=mbxf,  slice=4;   !new d1 (v1.1)
          select, flag=makethin, class=mbrd,  slice=4;   !new d2 (if needed)
          select, flag=makethin, class=mqyy,  slice=4 * slicefactor;;   !new q4
          select, flag=makethin, class=mqyl,  slice=4 * slicefactor;;   !new q5
          select, flag=makethin, class=mbh,   slice=4;   !11T dipoles
          select, flag=makethin, pattern=mbx\.,    slice=4;
          select, flag=makethin, pattern=mbrb\.,   slice=4;
          select, flag=makethin, pattern=mbrc\.,   slice=4;
          select, flag=makethin, pattern=mbrs\.,   slice=4;
          select, flag=makethin, pattern=mbh\.,    slice=4;
          select, flag=makethin, pattern=mqwa\.,   slice=4 * slicefactor;
          select, flag=makethin, pattern=mqwb\.,   slice=4 * slicefactor;
          select, flag=makethin, pattern=mqy\.,    slice=4 * slicefactor;
          select, flag=makethin, pattern=mqm\.,    slice=4 * slicefactor;
          select, flag=makethin, pattern=mqmc\.,   slice=4 * slicefactor;
          select, flag=makethin, pattern=mqml\.,   slice=4 * slicefactor;
          select, flag=makethin, pattern=mqtlh\.,  slice=2 * slicefactor;
          select, flag=makethin, pattern=mqtli\.,  slice=2 * slicefactor;
          select, flag=makethin, pattern=mqt\.  ,  slice=2 * slicefactor;
          !thin lens
          option rbarc=false; beam;
          use,sequence=lhcb1; makethin,sequence=lhcb1,makedipedge=true,style=teapot;
          use,sequence=lhcb2; makethin,sequence=lhcb2,makedipedge=true,style=teapot;
          option rbarc=true;
        } else {
          print, text="Sequence is already thin";
        };
          is_thin=1;
        };
    """)


mad = Madx()
mad.input("""
option,-echo,-warn;
! call,file="/afs/cern.ch/eng/lhc/optics/runIII/toolkit/macro.madx";
call,file="/afs/cern.ch/eng/lhc/optics/runIII/lhc_acc-models-lhc.seq";
nrj=6800;  
beam,particle=proton,sequence=lhcb1,energy=nrj,npart=1.15E11,sige=4.5e-4;
beam,particle=proton,sequence=lhcb2,energy=nrj,bv = -1,npart=1.15E11,sige=4.5e-4;
call,file="/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/Proton_2024/V0/opticsfile.49";
""")
my_slice(mad)

mad.input("exec, myslice;")

mad.input("""
use,sequence=lhcb1;
twiss;
value, table(twiss,IP1,betx),table(twiss,IP2,betx),table(twiss,IP5,betx),table(twiss,IP8,betx);
use,sequence=lhcb2;
twiss;
value, table(twiss,IP1,betx),table(twiss,IP2,betx),table(twiss,IP5,betx),table(twiss,IP8,betx);
""")
