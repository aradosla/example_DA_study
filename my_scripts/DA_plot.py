# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

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

df_da = pd.read_parquet("/afs/cern.ch/work/a/aradosla/private/example_DA_study_50Hz/studies/scans/example_tunescan_full/da.parquet")

def plot_tunescan(df_da, title, var_x = 'qx', var_y = 'qy', var_z = 'normalized amplitude in xy-plane'):
    lw_mine=3
    ft_mine=25
    df_da[var_x] = [round(i,7) for i in df_da[var_x].values]
    df_da[var_y] = [round(i,7) for i in df_da[var_y].values]

    x  = np.unique(df_da[var_x].values)
    x = [float(i) for i in x]
    y  = np.unique(df_da[var_y].values)
    y = [float(i) for i in y]
    print(x,y)

    dx = x[-1]-x[-2]
    dy = y[-1]-y[-2]

    xx1, yy1 = np.meshgrid(x,y) 
    xx2, yy2 = np.meshgrid(np.append(x, x[-1]+dx)-dx/2., np.append(y,y[-1]+dy)-dy/2.)

    z  = griddata((df_da[var_x].values, df_da[var_y].values), df_da[var_z].values, (xx1, yy1), method='nearest') #interpolates missing points
    z1 = gaussian_filter(z, sigma=0.8)
    z2 = z#gaussian_filter(z, sigma=0.8)

    x1, y1 = xx1[0], [row[0] for row in yy1]
    x2, y2 = xx2[0], [row[0] for row in yy2]

    fig, ax = plt.subplots()

    #title1 = r'HL-LHC v1.5, no MS.10, $\rm N_b$=2.3$\rm \times 10^{11}$ ppb, $\rm \beta^{*}_{IP1/5}=1 \ m, \phi/2_{IP1/5}=250 \ \mu rad$'
    #title2 = r'$\rm \phi/2_{V, IP8}=170 \ \mu rad, \epsilon_n=2.5 \ \mu m, Q\prime=15, I_{MO} = 410 \ A, C^-=10^{-3}$' 
    #title = title1 + "\n" + title2
    plt.title(title, fontsize=18)
    #ax.set_ylabel(r"Horizontal tune $Q_x$", fontsize=30)
    #ax.set_xlabel(r"$I_{\rm oct}$ (A)", fontsize=30)
    ax.set_xlabel(r"Horizontal tune $Q_x$", fontsize=30)
    ax.set_ylabel(r"Vertical tune $Q_y$", fontsize=30)

    cf = plt.pcolormesh(x2,y2,z2, cmap=cm.RdBu)
    minDA = 3.0
    maxDA = 9.0
    minDA = 4.0
    maxDA = 14.0
    plt.clim(minDA, maxDA)
    cbar = plt.colorbar(cf,  pad=0.01)
    cbar.set_label(r'Minimum DA $(\rm \sigma)$', rotation=90)

    #add contour lines

    #levels = [2.0, 3.0, 4.0, 5.0, 5.5,6.0, 6.5, 7.0, 8.0, 9.0]
    levels = np.arange(4,14)

    ct = plt.contour(x1, y1, z1, levels, colors='k', linewidths=2, label=r'Dynamic Aperture ($\rm \sigma$)')
    #show_values(cf, fontsize=9)
    plt.clabel(ct, colors = 'k', fmt = '%2.1f')

    # Overlay 6σ contour in green
    ct6 = plt.contour(x1, y1, z1, levels=[6.0], colors=['green'], linewidths=3)
    plt.clabel(ct6, colors=['green'], fmt={6.0: '6.0'})




    x = np.linspace(0.3045, 0.3305, num=20)
    y = np.linspace(0.3045, 0.3305, num=20)

    
    for y in range(len(y1)):
        for x in range(len(x1)):
            #print(x,y)
            #continue
            plt.text(x1[x] , y1[y] , '%.2f' % z[y, x],
                    horizontalalignment='center',
                    verticalalignment='center',fontsize=8,color='k'
                    )
    
    x = np.linspace(0.305, 0.330, num=20)
    y = np.linspace(0.305, 0.330, num=20)
    plt.xlim(62.305-5e-4, 62.330-0.0005)
    plt.ylim(60.305-5e-4, 60.330-0.0005)
    plt.plot(x+62., y+60. , c='white', lw=3, linestyle='-')
    plt.plot(x+62., y+60.  , c='white', lw=3, linestyle='--')

    plt.plot(x+62., y+60. + 5e-3, c='b', lw=2, linestyle='--')
    plt.plot(x+62., y+60. - + 5e-3, c='b', lw=2, linestyle='--')
    #plt.plot(x+62., y+60. + 1e-3, c='gold', lw=1, linestyle='--')
    #plt.plot(x+62., y+60. - + 1e-3, c='gold', lw=1, linestyle='--')

    plt.show()
    #fig.savefig('IPAC_tunescan_noMS10.png')
    #fig.savefig('/home/skostogl/Desktop/fellowship/HL_LHC/DA/final/EOL_c0_noerr.png')
    return fig, ax

plot_tunescan(df_da, title="Example DA scan")


# %%
