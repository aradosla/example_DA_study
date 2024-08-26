# %%
import xtrack as xt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xpart as xp
import xobjects as xo
import yaml
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
# %%
ctx = xo.ContextCpu()
N_particles = 10000
bunch_intensity = 2.2e11
normal_emitt_x = 1.5e-6 #m*rad
normal_emitt_y = 1.5e-6 #m*rad
sigma_z = 7.5e-2 
particle_ref = xp.Particles(
                    mass0=xp.PROTON_MASS_EV, q0=1, energy0=6800e9)


collider = xt.Multiline.from_json('/afs/cern.ch/work/a/aradosla/private/example_DA_study_fma/studies/scans/example_tunescan_gpu/base_collider/collider.json')
collider['lhcb1'].build_tracker()
line = collider['lhcb1']
gaussian_bunch = xp.generate_matched_gaussian_bunch(
        num_particles = N_particles, total_intensity_particles = bunch_intensity,
        nemitt_x = normal_emitt_x, nemitt_y=normal_emitt_y, sigma_z = sigma_z,
        particle_ref = particle_ref,
        line = line)
# %%
line.track(gaussian_bunch, num_turns = 10, turn_by_turn_monitor=True)
# %%
coord = line.twiss().get_normalized_coordinates(gaussian_bunch, nemitt_x = normal_emitt_x ,
nemitt_y = normal_emitt_y)
# %%
plt.plot(coord.x_norm, coord.px_norm, '.')
# %%
