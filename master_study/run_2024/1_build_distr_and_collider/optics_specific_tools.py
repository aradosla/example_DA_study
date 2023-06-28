from xmask.lhc import install_errors_placeholders_hllhc


def build_sequence(
    mad,
    mylhcbeam,
    ignore_cycling=False,
    **kwargs,
):
    # Select beam
    mad.input(f"mylhcbeam = {mylhcbeam}")

    mad.input(f"""
      ! Get the toolkit
      call,file=
        "acc-models-lhc/toolkit/macro.madx";
      """)

    mad.input(f"""
      ! Build sequence
      option, -echo,-warn,-info;
      if (mylhcbeam==4){{
        call,file="acc-models-lhc/lhc_acc-models-lhc_b4.seq";
      }} else {{
        call,file="acc-models-lhc/lhc_acc-models-lhc.seq";
      }};
      option, -echo, warn,-info;
      """)

    mad.input("""
      ! Slice nominal sequence
      exec, myslice;
      """)

    mad.input(f"""
    nrj=6800;
    beam,particle=proton,sequence=lhcb1,energy=nrj,npart=1.15E11,sige=4.5e-4;
    beam,particle=proton,sequence=lhcb2,energy=nrj,bv = -1,npart=1.15E11,sige=4.5e-4;
    """)

    if not ignore_cycling:
        mad.input("""
        !Cycling w.r.t. to IP3 (mandatory to find closed orbit in collision in the presence of errors)
        if (mylhcbeam<3){
        seqedit, sequence=lhcb1; flatten; cycle, start=IP3; flatten; endedit;
        };
        seqedit, sequence=lhcb2; flatten; cycle, start=IP3; flatten; endedit;
        """)

    mad.input("""
        ! Set twiss formats for MAD-X parts (macro from opt. toolkit)
        exec, twiss_opt;
        """)


def apply_optics(mad, optics_file):
    mad.call(optics_file)
    # A knob redefinition
    mad.input("on_alice := on_alice_normalized * 7000./nrj;")
    mad.input("on_lhcb := on_lhcb_normalized * 7000./nrj;")
