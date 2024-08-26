# %%
import numpy as np
import nafflib
import xobjects as xo
import xtrack as xt
import xpart as xp
import matplotlib.pyplot as plt
import xobjects as xo
# %%

ctx = xo.ContextCpu()
N_particles = 10000
bunch_intensity = 1.5e11
normal_emitt_x = 1.5e-6 #m*rad
normal_emitt_y = 1.5e-6 #m*rad
sigma_z = 7.5e-2 
particle_ref = xp.Particles(
                   mass0=xp.PROTON_MASS_EV, q0=1, energy0=450e9)
num_turns = 1000

# %%
#line = xt.Line()

qx = 0.27
qy = 0.295


elements = {
    'segment_map': xt.LineSegmentMap(_context=ctx,
            qx=qx, qy=qy, det_xx = 1000)
}
line = xt.Line(elements=elements,
               element_names=['segment_map'])
#line.twiss_default['method'] = '4d'

sampling_frequency = 11245.5
total_time = num_turns / sampling_frequency
time = np.arange(0, total_time, 1/sampling_frequency)

stdv = 0.8

np.random.seed(0)
samples = np.random.normal(0, stdv, len(time))

exciter = xt.Exciter(_context = ctx,
    samples = samples,
    sampling_frequency = sampling_frequency,
    duration= num_turns/sampling_frequency,
    frev = sampling_frequency,
    knl = [1.]
)

line.insert_element(
    element = exciter,
    name = 'white_noise_exciter',
    index = 0,
)


# sextupole = xt.Multipole(_context=ctx,order = 4, knl=[1000])
# line.insert_element(element=sextupole, name='sextupole', index=0)

line.build_tracker(_context = ctx)

# %%

line.twiss_default['method'] ='4d'
x_in_sigmas, px_in_sigmas = xp.generate_2D_gaussian(N_particles)
y_in_sigmas, py_in_sigmas = xp.generate_2D_gaussian(N_particles)

gaussian_bunch = xp.build_particles(line = line, particle_ref=particle_ref,
            x_norm=x_in_sigmas, px_norm=px_in_sigmas,
            y_norm=y_in_sigmas, py_norm=py_in_sigmas, method = '4d') 

p0 = gaussian_bunch.copy()
# %%

x_data = np.zeros((N_particles, num_turns))
y_data = np.zeros((N_particles, num_turns))
px_data = np.zeros((N_particles, num_turns))
py_data = np.zeros((N_particles, num_turns))

plt.plot(p0.x, p0.px, '.')

for i in range (num_turns):
    line.track(gaussian_bunch, freeze_longitudinal=True)
    x_data[:, i] = gaussian_bunch.copy().x
    y_data[:, i] = gaussian_bunch.copy().y
    px_data[:, i] = gaussian_bunch.copy().px
    py_data[:, i] = gaussian_bunch.copy().py
    plt.plot(x_data[:, i], px_data[:, i], '.')
  
x_data = x_data.T
y_data = y_data.T
px_data = px_data.T
py_data = py_data.T

# %%
plt.hist(x_data[0], bins ='auto')
plt.hist(x_data[-1], bins ='auto')

bins, edges = np.histogram(x_data[0], bins=100)
# %%
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import scipy.special

def q_gaussian(x, beta, q, A):
    if q == 1:
        return A * np.exp(-beta * x**2)
    else:
        return A * (1 + (q - 1) * beta * x**2)**(1 / (1 - q))
def qGaussfit(x):
    hist, bin_edges = np.histogram(x[0]-np.mean(x[0]), bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute histogram1
    hist1, bin_edges1 = np.histogram(x[-1]-np.mean(x[-1]), bins=100, density=True)
    bin_centers1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2

    # Initial guess for the parameters
    initial_guess = [1, 1.5, 1]

    # Fit the q-Gaussian to the histogram data
    params, _ = curve_fit(q_gaussian, bin_centers, hist, p0=initial_guess, maxfev = 20000)
    params1, _ = curve_fit(q_gaussian, bin_centers1, hist1, p0=initial_guess)

    # Extract the parameters
    beta, q, A = params
    beta1, q1, A1 = params1

    x_plot = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    y_plot = q_gaussian(x_plot, *params)
    plt.plot(x_plot, y_plot, 'g-', linewidth=2, label=f'q-Gaussian fit initial: beta={beta:.2f}, q={q:.2f}, A={A:.2f}')
    plt.hist(x[0]-np.mean(x[0]), bins=100, density=True, alpha=0.6, color='g', label='Initial distribution')
    plt.legend()

    x1 = np.linspace(bin_edges1[0], bin_edges1[-1], 1000)
    y1 = q_gaussian(x1, *params1)
    plt.plot(x1, y1, 'y-', linewidth=2, label=f'q-Gaussian fit final: beta={beta1:.2f}, q={q1:.2f}, A={A1:.2f}')
    plt.hist(x[-1]-np.mean(x[-1]), bins=100, density=True, alpha=0.6, color='yellow', label='Final distribution')
    plt.legend(loc ='upper left')
    plt.xlabel('x')
    plt.ylabel('Counts')
    #plt.title(f'q-Gaussian fit white noise stdv={stdv}, Gaussian distr 10k particles, 1k turns')
    #plt.title(f'q-Gaussian fit no noise, Gaussian distr 10k particles, 1k turns')
    plt.grid(True)
    return params, params1
params, params1 = qGaussfit(x_data)

def emittance(params, params1):
    popt_q = params
    print(popt_q)
    beta_optics = 1
    disp = 0
    gamma = 479.57 
    dpp = 0
    my_fit_sigma = np.sqrt(1.0/(popt_q[2] * (5.0 - 3.0*popt_q[1])))
    qgaussian_sigma = my_fit_sigma

    print(qgaussian_sigma)
    qgaussian_q = popt_q[1]
    qgaussian_emit = ((qgaussian_sigma**2 - (dpp*disp)**2)/beta_optics)# * gamma *1e-6
    
    popt_q1 = params1
    print(popt_q1)
    beta_optics = 1
    disp = 0
    gamma = 479.57 
    dpp = 0
    my_fit_sigma1 = np.sqrt(1.0/(popt_q1[2] * (5.0 - 3.0*popt_q1[1])))
    qgaussian_sigma1 = my_fit_sigma1 
    print(qgaussian_sigma1)
    qgaussian_q1 = popt_q1[1]
    qgaussian_emit1 = ((qgaussian_sigma1**2 - (dpp*disp)**2)/beta_optics) #* gamma *1e-6
    print(f'Initial emittance: {qgaussian_emit}')
    print(f'Final emittance: {qgaussian_emit1}')
    
    
emittance(params, params1)
# %%
def qGaussemit(x):

    qgaussian_emit_all = []
    qgaussian_q_all = []
    for turn in range(len(x)):
        hist, bin_edges = np.histogram(x[turn]-np.mean(x[turn]), bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        A0 = np.max(hist)
        mu0 = bin_centers[np.argmax(hist)]
        sigma0 = np.std(x[turn]-np.mean(x[turn]))
        initial_guess = [1., 1., A0] 

        # Fit the q-Gaussian to the histogram data
        params, _ = curve_fit(q_gaussian, bin_centers, hist, p0=initial_guess, maxfev = 200000)
    
        # Extract the parameters
        beta, q, A = params
        popt_q = params
        beta_optics = 1
        disp = 0
        gamma = 479.57 
        dpp = 0
        my_fit_sigma = np.sqrt(1.0/(popt_q[0] * (5.0 - 3.0*popt_q[1])))
        qgaussian_sigma = my_fit_sigma
        #print(qgaussian_sigma)
        qgaussian_q = popt_q[1]
        qgaussian_emit = ((qgaussian_sigma**2 - (dpp*disp)**2)/beta_optics) #* gamma *1e-6
        qgaussian_emit_all.append(qgaussian_emit)
        qgaussian_q_all.append(qgaussian_q)   
  

    
    return qgaussian_emit_all, qgaussian_q_all
emit_all5, q = qGaussemit(x_data)


# %% 
# Gaussian fit
dpp = sigma_z
beta_optics = 1
disp = 1
gamma = 479.57 

def gaussianemit(x):
    gaussemit = []
    for i in range(len(x)):
        sigma_x = float(np.std(x[i]))    
        geomx_emittance = (sigma_x**2-((dpp*disp)**2)/beta_optics)# * gamma *1e-6
        gaussemit.append(geomx_emittance)
  
    return gaussemit

gaussemit_try = gaussianemit(x_data)


# %%
plt.plot(gaussemit_try)
plt.plot(emit_all5)
# %%
#gaussian_emit_01 = gaussemit_try
#emit_all5_01 = emit_all5
#gaussian_emit_02= gaussemit_try
#emit_all5_02 = emit_all5
#gaussian_emit_03= gaussemit_try
#emit_all5_03 = emit_all5
#gaussian_emit_04= gaussemit_try
#emit_all5_04 = emit_all5
#gaussian_emit_05= gaussemit_try
#emit_all5_05 = emit_all5
#gaussian_emit_06= gaussemit_try
#emit_all5_06 = emit_all5
#gaussian_emit_07= gaussemit_try
#emit_all5_07 = emit_all5
#gaussian_emit_08= gaussemit_try
#emit_all5_08 = emit_all5

gaussemit_try = [gaussian_emit_01, gaussian_emit_02, gaussian_emit_03, gaussian_emit_04, gaussian_emit_05, gaussian_emit_06, gaussian_emit_07, gaussian_emit_08]
emit_all5 = [emit_all5_01, emit_all5_02, emit_all5_03, emit_all5_04, emit_all5_05, emit_all5_06, emit_all5_07, emit_all5_08]
noise_all = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
df = pd.DataFrame({'gaussian_emit': gaussemit_try, 'emit_all5': emit_all5, 'noise_level': noise_all, 'slope_gaussemit': slope_gaussemit, 'slope_qgausemit': slope_emit, 'slope_gaussemit_fit': slope_gaussemit_fit, 'slope_qemit_fit': slope_emit_fit})
# %%

# Fit the slope
import numpy as np
import matplotlib.pyplot as plt



slopes_gaussemit = []
slopes_emit = []

plt.figure()

for level in range(len(noise_all)):
    turns = np.arange(len(gaussemit_try[level]))  # Assuming emit_all5[level] has the same length as gaussemit_try[level]
    
    # Perform a linear fit for gaussemit_try[level]
    slope_gaussemit, intercept_gaussemit = np.polyfit(turns, gaussemit_try[level], 1)
    slopes_gaussemit.append(slope_gaussemit)
    
    # Perform a linear fit for emit_all5[level]
    slope_emit, intercept_emit = np.polyfit(turns, emit_all5[level], 1)
    slopes_emit.append(slope_emit)
    
    # Plot the data and the linear fit for gaussemit_try[level]
    plt.plot(turns, gaussemit_try[level], label=f'Gaussemit, Noise Level {level}')
    plt.plot(turns, slope_gaussemit * turns + intercept_gaussemit, '--', label=f'Fit Gaussemit Slope {slope_gaussemit:.2f}')
    
    # Plot the data and the linear fit for emit_all5[level]
    plt.plot(turns, emit_all5[level], label=f'Emit, Noise Level {level}')
    plt.plot(turns, slope_emit * turns + intercept_emit, '--', label=f'Fit Emit Slope {slope_emit:.2f}')
    
plt.xlabel('Turns')
plt.ylabel('Emittance')
#plt.legend()
plt.grid()
plt.tight_layout()
plt.title('Emittance evolution for different noise levels and a linear fit')
plt.show()

# Print out the slopes for each noise level
for level in range(len(noise_all)):
    print(f'Noise Level {level}: Gaussemit Slope = {slopes_gaussemit[level]:.6f}, Emit Slope = {slopes_emit[level]:.6f}')

# Perform a quadratic fit for slopes
coeffs_gaussemit = np.polyfit(noise_all, slopes_gaussemit, 2)
slope_gaussemit_fit = np.polyval(coeffs_gaussemit, noise_all)

# Perform a quadratic fit for emit_all5[level]
coeffs_emit = np.polyfit(noise_all, slopes_emit, 2)
slope_emit_fit = np.polyval(coeffs_emit, noise_all)

#plt.plot(np.array(noise_all), slopes_gaussemit, '.', label = 'Gaussian beam, gaussian emittance')
#plt.plot(np.array(noise_all), slopes_emit, '.', label = 'Gaussian beam, q-Gaussian emittance')

# Plot the data and the quadratic fit for emit_all5[level]


plt.plot(noise_all, slopes_gaussemit, '.', label = 'Gaussian beam, gaussian emittance')
plt.plot(noise_all, slopes_emit, '.', label = 'Gaussian beam, q-Gaussian emittance')
plt.plot(np.array(noise_all),slope_gaussemit_fit, '-', label=f'Gaussian emit fit $x^2$, Gaussian beam', markersize=8)
plt.plot(np.array(noise_all), slope_emit_fit, '--', label=f'q-Gaussian emit fit $x^2$, Gaussian beam')
plt.xlabel('Noise Level')
plt.ylabel('Slope')
plt.legend()
# %%
