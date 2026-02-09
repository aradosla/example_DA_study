# %%
import xtrack as xt
import numpy as np
import pandas as pd
# %%

collider = xt.Multiline.from_json('/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/collider_final_2025_23_nonoise.json')
line = collider['lhcb1']
twiss = line.twiss()
tw0 = twiss.to_pandas()
line2 = collider['lhcb2']
twiss2 = line2.twiss()
tw02 = twiss2.to_pandas()

# %%
collider_inj = xt.Multiline.from_json('/eos/user/a/aradosla/SWAN_projects/Colliders/collider_final_injection_2025.json')
line1_inj = collider_inj['lhcb1']
twiss_inj = line1_inj.twiss()
tw01_inj = twiss_inj.to_pandas()
betx_inj1h = tw01_inj[tw01_inj.name == 'bpmcs.7r4.b1'].betx
betx_inj1v = tw01_inj[tw01_inj.name == 'bpmcs.7l4.b1'].betx

betx_flat1h = tw0[tw0.name == 'bpmcs.7r4.b1'].betx
betx_flat1v = tw0[tw0.name == 'bpmcs.7l4.b1'].betx

line2_inj = collider_inj['lhcb2']
twiss_inj2 = line2_inj.twiss()
tw02_inj = twiss_inj2.to_pandas()
betx_inj2h = tw02_inj[tw02_inj.name == 'bpmcs.7r4.b2'].betx
betx_inj2v = tw02_inj[tw02_inj.name == 'bpmcs.7l4.b2'].betx
betx_flat2h = tw02[tw02.name == 'bpmcs.7r4.b2'].betx
betx_flat2v = tw02[tw02.name == 'bpmcs.7l4.b2'].betx
# %%

print('Beta opt inj B1H:', betx_inj1h.values)
print('Beta opt inj B1V:', betx_inj1v.values)
print('Beta opt flattop B1H:', betx_flat1h.values)
print('Beta opt flattop B1V:', betx_flat1v.values)

print('Beta opt inj B2H:', betx_inj2h.values)
print('Beta opt inj B2V:', betx_inj2v.values)
print('Beta opt flattop B2H:', betx_flat2h.values)
print('Beta opt flattop B2V:', betx_flat2v.values)
# %%
print('Position B1H:', 26658.73083276 - tw0[tw0.name == 'bpmcs.7r4.b1'].s.values)
print('Position B1V:', 26658.73083276 - tw0[tw0.name == 'bpmcs.7l4.b1'].s.values)
print('Position B2H:', tw02[tw02.name == 'bpmcs.7r4.b2'].s.values)
print('Position B2V:', tw02[tw02.name == 'bpmcs.7l4.b2'].s.values)
print('RF cavity B1:', 26658.73083276 - tw0[tw0.name == 'acsca.a5r4.b1'].s.values)
# %%
