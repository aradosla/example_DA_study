# %%
import xtrack as xt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xpart as xp
import xmask as xm
import xobjects as xo
import time
import cupy as cp

import yaml
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
import itertools
import nafflib as NAFFlib

# %%

params = {'xtick.labelsize': 28,
'ytick.labelsize': 28,
'font.size': 30,
'figure.autolayout': True,
'figure.figsize': (15, 10),
'axes.titlesize' : 35,
'axes.labelsize' : 35,
'lines.linewidth' : 2,
'lines.markersize' : 0.1,
'legend.fontsize': 28,
'mathtext.fontset': 'stix',
'font.family': 'STIXGeneral'}
plt.rcParams['figure.dpi'] = 100
plt.rcParams.update(params)

plt.rcParams['figure.dpi'] = 100

ft_mine = 20
lw_mine = 2
##############################################3.3752b
######### Taken from https://github.com/SixTrack/SixDeskDB/blob/master/sixdeskdb/footprint.py #####################
##############################################
######## Modified 07/03/2018
"""module to plot resonance lines"""

mycolors = list('rgbcm')

def colorrotate():
    c = mycolors.pop(0)
    mycolors.append(c)
    return c

def getmn(order, kind='b'):
    """return resonance of order order of kind kind
    Parameters:
    order: order of resonance
    as list of tuples (m,n) of * resonances of order o
    kind: 't': all resonances
          'a': skew multipoles n=odd
          'b': normal multipoles n=even
          's': sum resonances (m>0,n>0), loss of beam
          'd': difference resonances (m<0,n>0) or (m>0,n<0), exchange between planes
    Returns:
    list of tuples (m,n) with |m|+|n|=order and mQx+nQy
    """
    out = []
    if 't' in kind:
        kind = 'ab'
    for m in range(0, order + 1):
        n = order - m
        if 'b' in kind and n % 2 == 0:
            out.append((m, n))
            if n > 0:
                out.append((m, -n))
        if 'a' in kind and n % 2 == 1 and m >= 0:
            out.append((m, n))
            if n > 0:
                out.append((m, -n))
        if 's' in kind and (n > 0 and m > 0):
            out.append((m, n))
        if 'd' in kind and (n > 0 and m > 0):
            out.append((m, -n))

    return list(set(out))

def find_res_xcross(m, n, q, xs, y1, y2, out):
    if n != 0:
        m, n, q, xs, y1, y2 = map(float, (m, n, q, xs, y1, y2))
        ys = (q - m * xs) / n
        if y1 <= ys <= y2:
            out.append((xs, ys))

def find_res_ycross(m, n, q, ys, x1, x2, out):
    if m != 0:
        m, n, q, ys, x1, x2 = map(float, (m, n, q, ys, x1, x2))
        xs = (q - n * ys) / m
        if x1 <= xs <= x2:
            out.append((xs, ys))

def get_res_box(m, n, l=0, qz=0, a=0, b=1, c=0, d=1):
    """get (x,y) coordinates of resonance lines with
    m, n, q:   resonance integers with mQx + nQy = q
    l, qz:    order l of resonance sqzideband with frequency qz
    a, b, c, d: box parameters=tune range, 
                explicitly a < qx < b and c < qy < d 
    """
    order = int(np.ceil(abs(m) * max(abs(a), abs(b)) + abs(n) * max(abs(c), abs(d))))
    out = []
    mnlq = []
    for q in range(-order, +order + 1):
        q = q - l * qz
        points = []
        find_res_xcross(m, n, q, a, c, d, points)  # find endpoint of line (a,ys) with c < ys < d
        find_res_xcross(m, n, q, b, c, d, points)  # find endpoint of line (b,ys) with c < ys < d
        find_res_ycross(m, n, q, c, a, b, points)  # find endpoint of line (xs,c) with a < xs < b
        find_res_ycross(m, n, q, d, a, b, points)  # find endpoint of line (xs,d) with a < xs < b
        points = list(set(points))
        if len(points) > 1:
            out.append(points)
            mnlq.append((m, n, l, q + l * qz))

    return out, mnlq

def plot_res_box(m, n, l=0, qz=0, a=0, b=1, c=0, d=1, color='b', linestyle='-'):
    """plot resonance (m, n, l) with sidesband of
    order l and frequency qz with qx in [a, b]
    and qy in [c, d]"""
    points, mnlq = get_res_box(m, n, l, qz, a, b, c, d)
    for p in points:
        x, y = zip(*p)
        plt.plot(x, y, color=color, linestyle=linestyle, linewidth=1.)

def annotate_res_order_box(o, l=0, qz=0, a=0, b=1, c=0, d=1):
    """annotate the resonance lines of order o
    where annotations are (m, n, l). If the same
    resonance line occurs multiple times, only
    the first one is plotted"""
    l_points = []
    l_mnlq = []
    for m, n in getmn(o, 't'):
        points, mnlq = get_res_box(m, n, l, qz, a, b, c, d)
        for pp, oo in zip(points, mnlq):
            if pp not in l_points:
                x, y = zip(*pp)
                (x1, x2) = x
                (y1, y2) = y
                (xp, yp) = (x1 + (x2 - x1) / 2., y1 + (y2 - y1) / 2.)
                theta = 90 if x2 - x1 == 0 else np.arctan((y2 - y1) / (x2 - x1)) * 360 / (2 * np.pi)
                plt.gca().annotate(s='%s,%s,%s' % (str(oo[0]), str(oo[1]), int(str(oo[2]))),
                                  xy=(xp, yp), xytext=(xp, yp), xycoords='data', rotation=theta,
                                  fontsize=ft_mine, color='k', horizontalalignment='center',
                                  verticalalignment='center', annotation_clip=True)
                l_points.append(pp)
                l_mnlq.append(oo)

def annotate_specific(m, n, l=0, qz=0, a=0, b=1, c=0, d=1, xy_all=[], theta_all=[], l_points=[], l_mnlq=[]):
    """annotate the resonance lines of order o
    where annotations are (m, n, l). If the same
    resonance line occurs multiple times, only
    the first one is plotted"""
    points, mnlq = get_res_box(m, n, l, qz, a, b, c, d)
    for pp, oo in zip(points, mnlq):
        if pp not in l_points:
            x, y = zip(*pp)
            (x1, x2) = x
            (y1, y2) = y
            (xp, yp) = (x1 + (x2 - x1) / 2., y1 + (y2 - y1) / 2.)
            if x2 - x1 == 0:
                theta = 90
                yp = max(y)
                ha = 'left' if xp <= a else 'right'
                va = 'top'
                label = "          (%s,%s,%s,%s)       " % (oo[0], oo[1], int(oo[2]), int(oo[3]))
            elif y2 - y1 == 0:
                theta = 0
                xp = min(x)
                ha = 'top'
                va = 'bottom' if yp <= c else 'top'
                label = "        (%s,%s,%s,%s)         " % (oo[0], oo[1], int(oo[2]), int(oo[3]))
            else:
                theta = np.arctan((y2 - y1) / (x2 - x1)) * 360 / (2 * np.pi)
                if theta > 0:
                    xp = min(x)
                    yp = y[np.argmin(x)]
                    ha = 'left'
                    va = 'left'
                    label = "                   (%s,%s,%s,%s)                        " % (oo[0], oo[1], int(oo[2]), int(oo[3]))
                else:
                    xp = max(x)
                    yp = y[np.argmax(x)]
                    ha = 'right'
                    va = 'right'
                    label = "(%s,%s,%s,%s)                  " % (oo[0], oo[1], int(oo[2]), int(oo[3]))

            annot = plt.gca().annotate(s=label, xy=(xp, yp), xytext=(xp, yp), xycoords='data', rotation=theta + 15,
                                      fontsize=ft_mine, color='k', horizontalalignment=ha, verticalalignment=va,
                                      annotation_clip=True)
            l_points.append(pp)
            l_mnlq.append(oo)
            xy_all.append(annot)
            theta_all.append(theta)

def plot_res_order_box(o, l=0, qz=0, a=0, b=1, c=0, d=1, c1='b', lst1='-', c2='b', lst2='--', c3='g', list=[], xy_total=[], theta_total=[], annotate=False, l_points=[], l_mnlq=[]):
    """plot resonance lines up to order o and 
    sidebands of order l for frequency qz
    which lie in the square described by
    x = [a, b] and y = [c, d]"""
    if not list:
        flag_specific = False
    else:
        flag_specific = True

    for m, n in getmn(o, 'b'):
        if ((abs(m), abs(n)) in list) or (flag_specific == False):
            plot_res_box(m, n, l=0, qz=0, a=a, b=b, c=c, d=d, color=c1, linestyle=lst1)
            if l != 0:  # sidebands
                for ll in +abs(l), -abs(l):
                    plot_res_box(m, n, l=ll, qz=qz, a=a, b=b, c=c, d=d, color=c3, linestyle=lst1)
            if annotate:
                annotate_specific(m, n, l, qz, a, b, c, d, xy_all=xy_total, theta_all=theta_total, l_points=l_points, l_mnlq=l_mnlq)

    for m, n in getmn(o, 'a'):
        if ((abs(m), abs(n)) in list) or (flag_specific == False):
            plot_res_box(m, n, l=0, qz=0, a=a, b=b, c=c, d=d, color=c2, linestyle=lst2)
            if l != 0:  # sidebands
                for ll in +abs(l), -abs(l):
                    plot_res_box(m, n, l=ll, qz=qz, a=a, b=b, c=c, d=d, color=c3, linestyle=lst2)
            if annotate:
                annotate_specific(m, n, l, qz, a, b, c, d, xy_all=xy_total, theta_all=theta_total, l_points=l_points, l_mnlq=l_mnlq)

def plot_res_order(o, l=0, qz=0, c1='b', lst1='-', c2='b', lst2='--', c3='g', annotate=False):
    """plot resonance lines of order o and sidebands
    of order l and frequency qz in current plot
    range"""
    a, b = plt.xlim()
    c, d = plt.ylim()
    plot_res_order_box(o, l, qz, a, b, c, d, c1, lst1, c2, lst2, c3)
    if annotate:
        annotate_res_order_box(o, l, qz, a, b, c, d)
    plt.xlim(a, b)
    plt.ylim(c, d)

def plot_res_upto_order(o, l=0, qz=0, c1='b', lst1='-', c2='b', lst2='--', c3='g', annotate=False):
    """plot resonance lines up to order o and sidebands
    of order l and frequency qz in current plot
    range"""
    for i in range(-o, +o + 1):
        plot_res_order(i, l, qz, c1, lst1, c2, lst2, c3, annotate)

def plot_res(m, n, l=0, qz=0, color='b', linestyle='-'):
    """plot resonance of order (m, n, l) where l is
    the order of the sideband with frequency qz in
    the current plot range"""
    a, b = plt.xlim()
    c, d = plt.ylim()
    points, order = get_res_box(m, n, l, qz, a, b, c, d)
    for c in points:
        x, y = zip(*c)
        plt.plot(x, y, color=color, linestyle=linestyle, linewidth = 1.)
    plt.xlim(a, b)
    plt.ylim(c, d)

def plot_res_order_specific(order, l=0, qz=0, c1='b', lst1='-', c2='b', lst2='--', c3='g', annotate=False, list=[]):
    """plot resonance lines of order o and sidebands
    of order l and frequency qz in current plot
    range, but the ones that are only specified in list,
    where list is a list of tuples"""
    a, b = plt.xlim()
    c, d = plt.ylim()
    xy_total = []
    theta_total = []
    l_points = []
    l_mnlq = []
    for o in order:
        plot_res_order_box(o, l, qz, a, b, c, d, c1, lst1, c2, lst2, c3, list=list, xy_total=xy_total, theta_total=theta_total, l_points=l_points, l_mnlq=l_mnlq, annotate=annotate)
    plt.xlim(a, b)
    plt.ylim(c, d)
# %%

#collider = xt.Multiline.from_json('/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/scans/example_tunescan_gpu/base_collider/collider.json')
#collider = xt.Multiline.from_json('/afs/cern.ch/work/a/aradosla/private/example_DA_study/studies/scans/example_tunescan/base_collider/xtrack_0000/collider_final.json')
#collider = xt.Multiline.from_json('/eos/user/a/aradosla/SWAN_projects/Realistic_noise_simulations/50_Hz_noise_simulation_50k/collider_final_60cm.json')
#collider = xt.Multiline.from_json('/afs/cern.ch/work/a/aradosla/private/example_DA_study_mine/master_study/master_jobs/2_configure_and_track/collider.json')
#collider = xt.Multiline.from_json('/eos/user/a/aradosla/SWAN_projects/50_Hz_120cm_correct_1Mturns_WITH_10excitation_WITH_beambeam_chroma20_oct400/collider_final.json')
#collider = xt.Multiline.from_json('/eos/user/a/aradosla/SWAN_projects/50_Hz_noise_simulationCORRECT_newnoise120cm_WITH_beam_beam_10excitation1M100k_6D_fft/collider_final_1timenoise.json')
#collider = xt.Multiline.from_json('collider_final_2025_23_nonoise.json')
collider = xt.Multiline.from_json('/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/my_scripts/collider_tryMD.json')
#collider = xt.Multiline.from_json('/eos/user/a/aradosla/SWAN_projects/Separation_adjust/FMA_1.6e11_2025_120cm/collider_final.json')
device_number = None
context = xo.ContextCupy(device=device_number)
#context = xo.ContextCpu()
collider.build_trackers(_context=context)

# Fits of the separation values at adjust
df_fit = pd.read_parquet('fits_sep_adjust.parquet')
keys = df_fit.keys()

# Check these and see about the long ranges!!!
#collider.vars['i_oct_b1'] = -500
#collider.vars['i_oct_b2'] = -500
#collider.vars['dqx'] = 20
#collider.vars['dqy'] = 20

tw=collider['lhcb1'].twiss().qy
print(tw)
# %%
#Functions for fma
def fma(result_phys):
    df = result_phys
    keys = []
    qx_tot1 = []
    qx_tot2 = []
    qy_tot1 = []
    qy_tot2 = []
    diffusions = []
    for key, group in df.groupby('particle_id_all'):
        qx1 = abs(NAFFlib.get_tune(group.x_phys.values[:2000], 2))
        qy1 = abs(NAFFlib.get_tune(group.y_phys.values[:2000], 2))
        qx2 = abs(NAFFlib.get_tune(group.x_phys.values[-2000:], 2))
        qy2 = abs(NAFFlib.get_tune(group.y_phys.values[-2000:], 2))
        qx_tot1.append(qx1)
        qy_tot1.append(qy1)
        qx_tot2.append(qx2)
        qy_tot2.append(qy2)
        keys.append(key)
        diffusion = np.sqrt( abs(qx1-qx2)**2 + abs(qy1-qy2)**2 )
        if diffusion==0.0:
            diffusion=1e-60
        diffusion = np.log10(diffusion)
        diffusions.append(diffusion)
    dff = pd.DataFrame({'particle_id_all': keys,'qx1': qx_tot1, 'qy1': qy_tot1, 'qx2':qx_tot2, 'qy2':qy_tot2, 'diffusion': diffusions} )
    dff = dff.merge(df, on='particle_id_all')
    return dff

# %%
#Particle distribution
# Define radius distribution
r_min = 0
r_max = 10
#n_r =  2 * 16 * (r_max - r_min)
n_r = 10
radial_list = np.linspace(r_min, r_max, n_r, endpoint=False)

# Filter out particles with low and high amplitude to accelerate simulation
# radial_list = radial_list[(radial_list >= 4.5) & (radial_list <= 7.5)]

# Define angle distribution
n_angles = 20
theta_list = np.linspace(0, 90, n_angles + 2)[1:-1]
#theta_list = pd.DataFrame(theta_list)

# Define particle distribution as a cartesian product of the above
particle_list = [
    (particle_id, ii[1], ii[0])
    for particle_id, ii in enumerate(itertools.product(theta_list, radial_list))
]

particle_df = pd.DataFrame(particle_list, columns=["particle_id", "normalized amplitude in xy-plane", "angle in xy-plane [deg]"])


r_vect = particle_df["normalized amplitude in xy-plane"].values
theta_vect = particle_df["angle in xy-plane [deg]"].values * np.pi / 180  # type: ignore # [rad]

A1_in_sigma = r_vect * np.cos(theta_vect)
A2_in_sigma = r_vect * np.sin(theta_vect)

particles = collider['lhcb1'].build_particles(
    x_norm=A1_in_sigma,
    y_norm=A2_in_sigma,
    delta=0.,
    nemitt_x=(1.8e-6),
    nemitt_y=(1.8e-6),
    _context = context
)

particle_id = particle_df.particle_id.values

# %%


#all_fits_df_used = all_fits_df[:] # Use every second row for the knobs
num_turns = 10000  # how many turns to apply the knob change
#num_turns = knobs * (len(all_fits_df_used[keys[0]]) - 1) # total number of turns to track
ndata = 1 # how many turns to store the data
print(num_turns)
num_particles = int(n_r * n_angles)  # number of particles in the distribution
norm_intervals = num_turns // ndata
# Preallocate arrays for physical coordinates
x_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
y_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
zeta_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
px_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
py_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
pzeta_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
state_all = cp.empty((norm_intervals, num_particles), dtype=cp.int32)
particles_id_all = cp.empty((norm_intervals, num_particles), dtype=cp.int32)
turns_totnorm = cp.empty((norm_intervals, num_particles), dtype=cp.int32)

print('Tracking starting')

knob_names = ['on_sep1', 'on_sep2h', 'on_sep5', 'on_sep8h']
print('Tracking starting')


for j in np.arange(len(df_fit[keys[0]].dropna()))[:1]:  
    x_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
    y_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
    zeta_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
    px_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
    py_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
    pzeta_phys = cp.empty((norm_intervals, num_particles), dtype=cp.float64)
    state_all = cp.empty((norm_intervals, num_particles), dtype=cp.int32)
    particles_id_all = cp.empty((norm_intervals, num_particles), dtype=cp.int32)
    turns_totnorm = cp.empty((norm_intervals, num_particles), dtype=cp.int32)
    particles_copy = particles.copy()
    #if True:
    for kk in range(len(knob_names)):
        #print(df[keys[kk]].dropna().values[(i + 1) // knobs])
        #print(f'Applying knob {knob_names[kk]} from {keys[kk]} with value {df_fit[keys[kk]].dropna().values[j]}')
        print(f'Applying knob {knob_names[kk]} from {keys[kk]} with value {df_fit[keys[kk]].dropna().values[-1]}')

        collider.vars[knob_names[kk]] = df_fit[keys[kk]].dropna().values[-1]
        #collider.vars[knob_names[kk]] = df_fit[keys[kk]].dropna().values[j]
        #collider.vars[knob_names[kk]] = 0.
        if knob_names[kk] == 'on_sep1':
            #collider.vars[knob_names[kk]] = - df_fit[keys[kk]].dropna().values[j]
            #collider.vars[knob_names[kk]] = 0.
            collider.vars[knob_names[kk]] = - df_fit[keys[kk]].dropna().values[-1]

    for i in range(num_turns):
        c = time.time()
        collider['lhcb1'].track(particles_copy, num_turns=1, turn_by_turn_monitor=True, freeze_longitudinal=True)

        # Store particle data every 1000 turns
        if (i + 1) % ndata == 0:
            interval_index = (i + 1) // ndata - 1

        # Store physical coordinates
            x_phys[interval_index, :] = cp.asarray(particles_copy.x)
            y_phys[interval_index, :] = cp.asarray(particles_copy.y)
            zeta_phys[interval_index, :] = cp.asarray(particles_copy.zeta)
            px_phys[interval_index, :] = cp.asarray(particles_copy.px)
            py_phys[interval_index, :] = cp.asarray(particles_copy.py)
            pzeta_phys[interval_index, :] = cp.asarray(particles_copy.delta)
            state_all[interval_index, :] = cp.asarray(particles_copy.state)
            turns_totnorm[interval_index, :] = cp.ones(num_particles, dtype=cp.int32) * (i + 1)
            particles_id_all[interval_index, :] = cp.asarray(particles_copy.particle_id)

            d = time.time()
            print(f'Turn {i+1}, time {d-c}s')


    # Convert results back to CPU and flatten arrays
    x_phys = cp.asnumpy(x_phys).flatten()
    y_phys = cp.asnumpy(y_phys).flatten()
    zeta_phys = cp.asnumpy(zeta_phys).flatten()
    px_phys = cp.asnumpy(px_phys).flatten()
    py_phys = cp.asnumpy(py_phys).flatten()
    pzeta_phys = cp.asnumpy(pzeta_phys).flatten()
    state_all = cp.asnumpy(state_all).flatten()
    turns_totnorm = cp.asnumpy(turns_totnorm).flatten()
    particles_id_all = cp.asnumpy(particles_id_all).flatten()



    print(f'Storing data {j} index knob')
    #Convert results to DataFrame
    result_phys = pd.DataFrame({
        "x_phys": x_phys,
        "y_phys": y_phys,
        "zeta_phys": zeta_phys,
        "px_phys": px_phys,
        "py_phys": py_phys,
        "pzeta_phys": pzeta_phys,
        "state": state_all,
        "at_turn": turns_totnorm, #"particle_id": particles_id_all #np.repeat(np.arange(1, (norm_intervals) * 1000, 1000), num_particles)
        "particle_id_all": particles_id_all
        #"particles_id_all": np.tile(particle_id, int(num_turns/ndata)) #particles_id_all
    })

    #result_phys.to_parquet(f'/eos/user/a/aradosla/SWAN_projects/Separation_adjust/fma_data.parquet_{j}indexknob.parquet')
    #dffs = pd.read_parquet('collider120_example_mine_1timenoise.parquet')

    dffs= fma(result_phys)
    dffs.to_parquet(f'/eos/user/a/aradosla/SWAN_projects/Separation_adjust/fma_data_{j}indexknob_try.parquet')
    #dffs.to_parquet(f'/eos/user/a/aradosla/SWAN_projects/Separation_adjust/fma_data_last_knob_new_filling_pattern.parquet')
    frev = 11245.5

    fig, ax = plt.subplots()
    print('Now plotting')
    plt.scatter(dffs[dffs['at_turn'] == 1]['qx1'],dffs[dffs['at_turn'] == 1]['qy1'], s=20, edgecolors=None, c=dffs[dffs['at_turn'] == 1]['diffusion'],vmin = -7, vmax = -3, cmap='jet')
    #plt.xlim(0.29, 0.315)
    #plt.ylim(0.30, 0.33)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.xlabel(r"Horizontal tune, $Q_x$")
    plt.ylabel(r"Vertical tune, $Q_y$")
    plt.title(f'120cm FMA ADJUST changing the separation index {j}')
    cbar=plt.colorbar(pad=0.01)
    x_min, x_max = 0.29, 0.4 # Define your x-axis limits

    #for i in np.arange(0, frev, 50):
    #    x_val = (frev - i) / frev  # Convert Hz to tune
    #    if x_min <= x_val <= x_max:  # Only plot if inside the x-range
    #        plt.axvline(x_val, color='black', linestyle='--', lw=2)
    #        plt.text(x_val, 0.324, f'{int(i)} [Hz]', rotation=90)
    #for i in flist:
    #    x_val = (frev - i) / frev  # Convert Hz to tune
    #    if x_min <= x_val <= x_max:  # Only plot if inside the x-range
    #        plt.axvline(x_val, color='red', linestyle='--', lw=2)
    #        plt.text(x_val, 0.324cat, f'{i:.0f} [Hz]', rotation=90)
    cbar.set_label(r'$\rm \log_{10}\left({\sqrt{\Delta Q_x^2 + \Delta Q_y^2}}\right)$',labelpad=45,rotation=270, fontsize=24)
    #ootprint.lw_mine=0.5
    #plt.title('FMA ADJUST LHC 120cm')
    #plt.xlim(0.30, 0.306)
    #plt.ylim(0.312, 0.323)
    plt.xlim(0.285, 0.315)
    plt.ylim(0.295, 0.323)
    plot_res_upto_order(12,c1 = 'darkgrey', c2 = 'darkgrey',c3='r',annotate=False)
    print(f'Saving figure for {j} index knob')
    plt.savefig(f'/eos/user/a/aradosla/SWAN_projects/Separation_adjust/fma_adjust_{j}_tryindexknob.png', dpi=300, facecolor='white')

# %%
import xtrack as xt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xpart as xp
import xmask as xm
import xobjects as xo
import time
import cupy as cp

import yaml
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
import itertools
import nafflib as NAFFlib
result_phys = pd.read_parquet('/eos/user/a/aradosla/SWAN_projects/Separation_adjust/fma_data.parquet_0indexknob.parquet')

dffs= fma(result_phys)
frev = 11245.5
# %%
fig, ax = plt.subplots()
plt.scatter(dffs[dffs['at_turn'] == 1]['qx1'],dffs[dffs['at_turn'] == 1]['qy1'], s=20, edgecolors=None, c=dffs[dffs['at_turn'] == 1]['diffusion'],vmin = -7, vmax = -3, cmap='jet')
#plt.xlim(0.29, 0.315)
#plt.ylim(0.30, 0.33)
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
plt.xlabel(r"Horizontal tune, $Q_x$")
plt.ylabel(r"Vertical tune, $Q_y$")
plt.title('')
cbar=plt.colorbar(pad=0.01)
x_min, x_max = 0.29, 0.4 # Define your x-axis limits

#for i in np.arange(0, frev, 50):
#    x_val = (frev - i) / frev  # Convert Hz to tune
#    if x_min <= x_val <= x_max:  # Only plot if inside the x-range
#        plt.axvline(x_val, color='black', linestyle='--', lw=2)
#        plt.text(x_val, 0.324, f'{int(i)} [Hz]', rotation=90)
#for i in flist:
#    x_val = (frev - i) / frev  # Convert Hz to tune
#    if x_min <= x_val <= x_max:  # Only plot if inside the x-range
#        plt.axvline(x_val, color='red', linestyle='--', lw=2)
#        plt.text(x_val, 0.324, f'{i:.0f} [Hz]', rotation=90)
cbar.set_label(r'$\rm \log_{10}\left({\sqrt{\Delta Q_x^2 + \Delta Q_y^2}}\right)$',labelpad=45,rotation=270, fontsize=24)
#ootprint.lw_mine=0.5
plt.title('FMA ADJUST LHC 120cm')
#plt.xlim(0.30, 0.306)
#plt.ylim(0.312, 0.323)
plt.xlim(0.285, 0.315)
plt.ylim(0.295, 0.323)
plot_res_upto_order(12,c1 = 'darkgrey', c2 = 'darkgrey',c3='r',annotate=False)
#plt.savefig(f'/eos/user/a/aradosla/SWAN_projects/Separation_adjust/fma_adjust_{j}indexknob.png', dpi=300, facecolor='white')

# %%
