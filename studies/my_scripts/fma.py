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
import itertools
import nafflib as NAFFlib
import glob

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
files = glob.glob('/eos/user/a/aradosla/SWAN_projects/Noise_sim_try_gpu_fma_without_noise/**/fma.parquet')
dffs = pd.DataFrame()
for file in files:
    dff = pd.read_parquet(file)
    dff = dff[(dff['at_turn'] < 2000) | (dff['at_turn'] > 8000)]
    print(file)
    dffs = pd.concat([dffs,dff], ignore_index=True)


# %%
fig, ax = plt.subplots()
plt.scatter(dffs['qx1'],dffs['qy1'], s=3, edgecolors=None, c=dffs['diffusion'], vmin=-7, vmax=-3,cmap='jet')
plt.xlim(0.20, 0.34)
plt.ylim(0.24, 0.4)
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
plt.xlabel(r"Horizontal tune, $Q_x$")
plt.ylabel(r"Vertical tune, $Q_y$")
cbar=plt.colorbar(pad=0.01)
cbar.set_label(r'$\rm \log_{10}\left({\sqrt{\Delta Q_x^2 + \Delta Q_y^2}}\right)$',labelpad=45,rotation=270, fontsize=24)
#ootprint.lw_mine=0.5

plot_res_upto_order(12,c1 = 'darkgrey', c2 = 'darkgrey',c3='r',annotate=False)
# %%# %%

files = glob.glob('/eos/user/a/aradosla/SWAN_projects/Noise_sim_try_gpu_fma_quad/**/fma.parquet')
dffs = pd.DataFrame()
for file in files:
    dff = pd.read_parquet(file)
    dff = dff[(dff['at_turn'] < 2000) | (dff['at_turn'] > 8000)]
    print(file)
    dffs = pd.concat([dffs,dff], ignore_index=True)

# %%
fig, ax = plt.subplots()
plt.scatter(dffs['qx1'],dffs['qy1'], s=3, edgecolors=None, c=dffs['diffusion'], vmin=-7, vmax=-3,cmap='jet')
plt.xlim(0.20, 0.34)
plt.ylim(0.24, 0.4)
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
plt.xlabel(r"Horizontal tune, $Q_x$")
plt.ylabel(r"Vertical tune, $Q_y$")
cbar=plt.colorbar(pad=0.01)
cbar.set_label(r'$\rm \log_{10}\left({\sqrt{\Delta Q_x^2 + \Delta Q_y^2}}\right)$',labelpad=45,rotation=270, fontsize=24)
#ootprint.lw_mine=0.5

plot_res_upto_order(12,c1 = 'darkgrey', c2 = 'darkgrey',c3='r',annotate=False)


# %%
#files = glob.glob('/eos/user/a/aradosla/SWAN_projects/Noise_sim_try_gpu_fma_quad_1000hz/*.parquet')
files = glob.glob('/eos/user/a/aradosla/SWAN_projects/Noise_first_working/Noise_sim_try_gpu_fma_dipol_6e-10/**/fma*.parquet')
#dff_phys = []
dffs = pd.DataFrame()
for file in files[:3]:
    dff = pd.read_parquet(file)
    #dff_phys.append(dff[dff.particle_id == 0].x_phys)
    #dff = dff[(dff['at_turn'] < 2000) | (dff['at_turn'] > 8000)]
    print(file)
    dffs = pd.concat([dffs,dff], ignore_index=True)

# %%
fig, ax = plt.subplots()
plt.scatter(dffs['qx1'],dffs['qy1'], s=3, edgecolors=None, c=dffs['diffusion'], vmin=-7, vmax=-3,cmap='jet')
plt.xlim(0.25, 0.30)
plt.ylim(0.26, 0.33)

ax.yaxis.set_major_locator(plt.MaxNLocator(3))
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
plt.xlabel(r"Horizontal tune, $Q_x$")
plt.ylabel(r"Vertical tune, $Q_y$")
cbar=plt.colorbar(pad=0.01)
cbar.set_label(r'$\rm \log_{10}\left({\sqrt{\Delta Q_x^2 + \Delta Q_y^2}}\right)$',labelpad=45,rotation=270, fontsize=24)
plot_res_upto_order(6,qz = 300/11245.5, l=1, c1 = 'darkgrey', c2 = 'darkgrey',c3='r',annotate=False)
plot_res_upto_order(6,qz = 300/11245.5, l=2, c1 = 'darkgrey', c2 = 'darkgrey',c3='r',annotate=False)

#ootprint.lw_mine=0.5

# %%
plot_res_upto_order(6,qz = 50/11245.5, l=7, c1 = 'darkgrey', c2 = 'darkgrey',c3='r',annotate=False)
plt.xlim(0.25, 0.30)
plt.ylim(0.26, 0.33)

# %%
