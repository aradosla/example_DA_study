# %%
import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp
import matplotlib.pyplot as plt
import xobjects as xo
import scipy.special
import scipy.integrate
import scipy.optimize
from scipy.interpolate import interp1d
import pandas as pd
from scipy.optimize import curve_fit

# %%
#======================== q-Gaussian distribution for q < 1 and q > 1 ==========================
num_particles = 10000

def _Cq2(q):
    Gamma = scipy.special.gamma
    if q < 1:
        return (2 * np.sqrt(np.pi)) / ((3 - q) * np.sqrt(1 - q)) * (Gamma(1 / (1 - q))) / (Gamma((3 - q) / (2 * (1 - q))))
    elif q == 1:
        return np.sqrt(np.pi)
    elif q < 3:
        return (np.sqrt(np.pi)) / (np.sqrt(q - 1)) * (Gamma((3 - q) / (2 * (q - 1)))) / (Gamma(1 / (q - 1)))
    else:
        return 0

def _eq2(x, q):
    if q == 1:
        return np.exp(x)
    else:
        # Ensure that the argument inside the power remains non-negative
        result = (1 + (1 - q) * x)
        # Take the real part if result is complex, otherwise return result
        return np.real(result)**(1 / (1 - q)) if np.all(result >= 0) else 0

def qGauss(x, A, mu, q, b, offset):
    result = A * np.sqrt(b) / _Cq2(q) * _eq2(-b * (x - mu)**2, q) + offset
    result = np.where(np.isnan(result), offset, result)
    return result

def qGauss_CDF(x, A, mu, q, b, offset):
    pdf = lambda t: qGauss(t, A, mu, q, b, offset)
    cdf_values = [scipy.integrate.quad(pdf, -10, xi, limit=100)[0] for xi in x]  # Use a finite range
    return np.array(cdf_values)

def inverse_qGauss_CDF(p, A, mu, q, b, offset):
    # Define the range of x values
    x_values = np.linspace(-10, 10, 1000)
    # Calculate the CDF values for the range of x values
    cdf_values = qGauss_CDF(x_values, A, mu, q, b, offset)
    # Interpolate the CDF values
    cdf_interp = interp1d(cdf_values, x_values, kind='linear', bounds_error=False, fill_value=(x_values[0], x_values[-1]))
    # Evaluate the interpolated function at the given probabilities
    inv_cdf_values = cdf_interp(p)
    return inv_cdf_values

x_values = np.linspace(-10, 10, num_particles)

# Generate uniform random numbers between 0 and 1
uniform_samples_x = np.random.uniform(0, 1, num_particles)
uniform_samples_px = np.random.uniform(0, 1, num_particles)
uniform_samples_y = np.random.uniform(0, 1, num_particles)
uniform_samples_py = np.random.uniform(0, 1, num_particles)

# Define the parameters of the q-Gaussian distribution
A = 1
mu = 0
q = 0.65 # q < 1 case
b = 0.5
offset = 0

# Generate samples using the interpolation method
pdf_values = qGauss(x_values, A, mu, q, b, offset)
x_q = inverse_qGauss_CDF(uniform_samples_x, A, mu, q, b, offset)
y_q = inverse_qGauss_CDF(uniform_samples_y, A, mu, q, b, offset)
px_q = inverse_qGauss_CDF(uniform_samples_px, A, mu, q, b, offset)
py_q = inverse_qGauss_CDF(uniform_samples_py, A, mu, q, b, offset)

# Plot the histogram of generated samples for x_q
plt.hist(x_q, bins=100, alpha=0.5)
plt.title(f"Histogram of q-Gaussian distributed samples (q = {q})")
plt.xlabel("x")
plt.ylabel("Frequency")
plt.show()


# %%
# =============================== Some parameter definitions ==========================
ctx = xo.ContextCpu()
N_particles = num_particles
bunch_intensity = 1.5e11
normal_emitt_x = 1.5e-6 #m*rad
normal_emitt_y = 1.5e-6 #m*rad
sigma_z = 7.5e-2 
particle_ref = xp.Particles(
                   mass0=xp.PROTON_MASS_EV, q0=1, energy0=450e9)
num_turns = 1000

# %%
# =============================== Line ===============================

qx = 0.27
qy = 0.295
# Line generation
elements = {
    'segment_map': xt.LineSegmentMap(_context=ctx,
            qx=qx, qy=qy, det_xx=1000, betx=1)
}
line = xt.Line(elements=elements,
               element_names=['segment_map'])
line.twiss_default['method'] = '4d'

sampling_frequency = 11245.5
total_time = num_turns / sampling_frequency
time = np.arange(0, total_time, 1/sampling_frequency)

stdv = 0.09
# Generate white noise
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

# Insert a nonlinear element
sextupole = xt.Multipole(_context=ctx,order = 3, knl=[0,0, 0.1])
#line.insert_element(element=sextupole, name='sextupole', index=0)

line.build_tracker(_context = ctx)
betx = line.twiss(particle_ref=particle_ref).betx[0]
gamma = line.twiss(particle_ref=particle_ref).gamma0
# %%

line.twiss_default['method'] ='4d'
x_in_sigmas, px_in_sigmas = xp.generate_2D_gaussian(N_particles)
y_in_sigmas, py_in_sigmas = xp.generate_2D_gaussian(N_particles)
'''
# Normal Gaussian distribution
gaussian_bunch = xp.build_particles(line = line, particle_ref=particle_ref,
            x_norm=x_in_sigmas, px_norm=px_in_sigmas,
            y_norm=y_in_sigmas, py_norm=py_in_sigmas, method = '4d')#, nemitt_x=normal_emitt_x, nemitt_y=normal_emitt_y) 
            #y_norm=y_in_sigmas, py_norm=py_in_sigmas, method = '4d') 

# q-Gaussian distribution
'''
gaussian_bunch = xp.build_particles(line = line, particle_ref=particle_ref,
            x_norm=x_q, px_norm=px_q,
            y_norm=y_q, py_norm=py_q, method = '4d') 

# %%
line.track(gaussian_bunch, num_turns=num_turns, turn_by_turn_monitor=True, freeze_longitudinal=True)

# %%
# =============================== Tracking ==========================
x_data = np.zeros((N_particles, num_turns))
y_data = np.zeros((N_particles, num_turns))
px_data = np.zeros((N_particles, num_turns))
py_data = np.zeros((N_particles, num_turns))


for i in range (num_turns):
    line.track(gaussian_bunch, freeze_longitudinal=True)
    x_data[:, i] = gaussian_bunch.copy().x
    y_data[:, i] = gaussian_bunch.copy().y
    px_data[:, i] = gaussian_bunch.copy().px
    py_data[:, i] = gaussian_bunch.copy().py
    plt.plot(x_data[:, i], px_data[:, i], '.')
   
    #coord = line.twiss(particle_ref=particle_ref).get_normalized_coordinates(gaussian_bunch)
    #x_norm = coord.x_norm
    #px_norm = coord.px_norm
    #plt.plot(x_norm, px_norm, '.')
    plt.xlabel('x')
    plt.ylabel('px')
x_data = x_data.T
y_data = y_data.T
px_data = px_data.T
py_data = py_data.T


#x_norm = x_data/np.sqrt(betx*normal_emitt_x/gamma)

#========================= Until here is the phase space evolution of the particles =========================

# %%
# =============================== q-Gaussian fit ==========================
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
    initial_guess = [0.5, 1.2, 1]

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
# %%

# =============================== Emittance calculation initial and final ==========================
# Calculate the emittance using fitted parameters
def emittance(x,  params_x, params1_x,  betx):
    popt_q_x = params_x
    
    beta_optics = betx  # This could change based on your optics setup
    disp = 0  # Dispersion
    gamma = line.twiss(particle_ref=particle_ref).gamma0 # Relativistic factor for the beam
    dpp = 0  # Momentum spread

    # Initial emittance for x
    qgaussian_sigma_x = np.sqrt(1.0 / (popt_q_x[0] * (5.0 - 3.0 * popt_q_x[1])))

    # Geometric emittance
    #qgaussian_emit = qgaussian_sigma_x**2/ betx
    qgaussian_emit = ((qgaussian_sigma_x**2 - (dpp*disp)**2)/beta_optics)# * gamma *1e-6

    print(f'Initial emittance: {qgaussian_emit}')

    # Final emittance for x and px
    popt_q1_x = params1_x
   
    qgaussian_sigma1_x = np.sqrt(1.0 / (popt_q1_x[0] * (5.0 - 3.0 * popt_q1_x[1])))
    

    # Final emittance
    #qgaussian_emit1 = qgaussian_sigma1_x**2/ betx
    qgaussian_emit1 = ((qgaussian_sigma1_x**2 - (dpp*disp)**2)/beta_optics) #* gamma *1e-6
    print(f'Final emittance: {qgaussian_emit1}')

    return qgaussian_emit, qgaussian_emit1

# Example usage with both x and px data
params_x, params1_x = qGaussfit(x_data)

initial_emit, final_emit = emittance(x_data, params_x, params1_x, betx)

# %%
# =============================== q- Gaussian Emittance evolution ==========================
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

# Example usage (assuming x_data and px_data are available)
qemit_all, q_all = qGaussemit(x_data)


# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean)**2) / (2 * stddev**2))

def Gaussemit(x, betx):
    gaussian_emit_all = []

    for turn in range(len(x)):
        # Calculate histogram
        hist, bin_edges = np.histogram(x[turn] - np.mean(x[turn]), bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        A0 = np.max(hist)
        mu0 = bin_centers[np.argmax(hist)]
        sigma0 = np.std(x[turn] - np.mean(x[turn]))
        initial_guess = [A0, mu0, sigma0]  # Amplitude, Mean, StdDev

        # Fit the Gaussian to the histogram data
        try:
            params, _ = curve_fit(gaussian, bin_centers, hist, p0=initial_guess, maxfev=200000)
        except RuntimeError as e:
            print(f"Error fitting turn {turn}: {e}")
            continue  # Skip this turn if fitting fails

        # Extract the parameters
        amplitude, mean, stdv_fit = params

        # Calculation for emittance
        gaussian_sigma = stdv_fit
        disp = 0  # Assuming this is defined elsewhere
        gamma = 479.57  # Check if this is the correct value
        dpp = 0  # Assuming this is defined elsewhere
        gaussian_emit = ((gaussian_sigma**2 - (dpp * disp)**2) / betx)  # * gamma * 1e-6

        gaussian_emit_all.append(gaussian_emit)

        # Plotting
        plt.hist(x[turn] - np.mean(x[turn]), bins=100, density=True, alpha=0.6)
        plt.plot(bin_centers, gaussian(bin_centers, *params), 'r-')

    plt.xlabel('x (normalized)')
    plt.ylabel('Density')
    plt.title('Gaussian Fit to Histogram')
    plt.show()
    
    return gaussian_emit_all

# Example usage (assuming x_data and betx are available)
gemit_all = Gaussemit(x_data[::10], betx)

# %% 
# Variance fit
dpp = 0
disp = 1
gamma = 479.57 

def gaussianemit(x):
    gaussemit = []
    for i in range(len(x)):
        sigma_x = float(np.std(x[i]))    
        geomx_emittance = (sigma_x**2-((dpp*disp)**2)/betx)# * gamma *1e-6
        gaussemit.append(geomx_emittance)
  
    return gaussemit

gaussemit_try = gaussianemit(x_data)


# %%
#plt.figure(figsize=(10, 7))
plt.plot(gaussemit_try, label = 'q-Gaussian beam q = 0.7, emittance from variance')
plt.plot(qemit_all, label = 'q-Gaussian beam q = 0.7, q-Gaussian emittance')
plt.xlabel('Turns')
#
#plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylabel('Emittance [$\sigma$]')
#plt.plot(np.arange(10, 1001, 10), gemit_all, label = 'Gaussian beam Gaussian emittance', alpha = 0.5)
plt.legend()

# %%
plt.figure(figsize=(10, 7))
plt.plot(q_all)
plt.xlabel('Turns', fontsize = 20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylabel('q', fontsize=20)
# %%


# =============================== Here not important ==========================
#gaussian_emit_01 = gaussemit_try
#emit_all5_01 = qemit_all
#gaussian_emit_02= gaussemit_try
#emit_all5_02 = qemit_all
#gaussian_emit_03= gaussemit_try
#emit_all5_03 = qemit_all
#gaussian_emit_04= gaussemit_try
#emit_all5_04 = qemit_all
#gaussian_emit_05= gaussemit_try
#emit_all5_05 = qemit_all
#gaussian_emit_06= gaussemit_try
#emit_all5_06 = emit_all5
#gaussian_emit_07= gaussemit_try
#emit_all5_07 = qemit_all
#gaussian_emit_08= gaussemit_try
#emit_all5_08 = qemit_all
#gaussian_emit_09= gaussemit_try
#emit_all5_09 = qemit_all


#gaussemit_try = [gaussian_emit_01, gaussian_emit_02, gaussian_emit_03, gaussian_emit_04, gaussian_emit_05, gaussian_emit_06, gaussian_emit_07, gaussian_emit_08]
#emit_all5 = [emit_all5_01, emit_all5_02, emit_all5_03, emit_all5_04, emit_all5_05, emit_all5_06, emit_all5_07, emit_all5_08]
noise_all = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
gaussemit_try = [gaussian_emit_01, gaussian_emit_06, gaussian_emit_09] 
qemit_all = [emit_all5_01, emit_all5_05, emit_all5_09]
betx = [1,2,3,4,5,6,7,8]
import pandas as pd
#df = pd.DataFrame({'gaussian_emit': gaussemit_try, 'qemit_all': qemit_all, 'betx': betx, 'slope_gaussemit': slopes_gaussemit, 'slope_qgausemit': slopes_emit, 'slope_gaussemit_fit': slope_gaussemit_fit, 'slope_qemit_fit': slope_emit_fit})
# %%

# Fit the slope
import numpy as np
import matplotlib.pyplot as plt

#betx = [1,2,3,4,5,6,7,8]
betx = [1, 5, 9]
noise_all = betx
slopes_gaussemit = []
slopes_emit = []

plt.figure()

for level in range(len(noise_all)):
    turns = np.arange(len(gaussemit_try[level]))  # Assuming qemit_all[level] has the same length as gaussemit_try[level]
    
    # Perform a linear fit for gaussemit_try[level]
    slope_gaussemit, intercept_gaussemit = np.polyfit(turns, gaussemit_try[level], 1)
    slopes_gaussemit.append(slope_gaussemit)
    
    # Perform a linear fit for qemit_all[level]
    slope_emit, intercept_emit = np.polyfit(turns, qemit_all[level], 1)
    slopes_emit.append(slope_emit)
    
    # Plot the data and the linear fit for gaussemit_try[level]
    plt.plot(turns, gaussemit_try[level], label=f'Gaussemit, Noise Level {level}')
    plt.plot(turns, slope_gaussemit * turns + intercept_gaussemit, '--', label=f'Fit Gaussemit Slope {slope_gaussemit:.2f}')
    
    # Plot the data and the linear fit for qemit_all[level]
    plt.plot(turns, qemit_all[level], label=f'Emit, Noise Level {level}')
    plt.plot(turns, slope_emit * turns + intercept_emit, '--', label=f'Fit Emit Slope {slope_emit:.2f}')
    
plt.xlabel('Turns')
plt.ylabel('Emittance')
#plt.legend()
plt.grid()
plt.tight_layout()
plt.title('Emittance evolution for different noise levels and a linear fit')
plt.title('Emittance evolution for different betx and a linear fit')
plt.show()

# Print out the slopes for each noise level
for level in range(len(noise_all)):
    print(f'Noise Level {level}: Gaussemit Slope = {slopes_gaussemit[level]:.6f}, Emit Slope = {slopes_emit[level]:.6f}')

# Perform a quadratic fit for slopes
coeffs_gaussemit = np.polyfit(noise_all, slopes_gaussemit, 2)
slope_gaussemit_fit = np.polyval(coeffs_gaussemit, noise_all)

# Perform a quadratic fit for qemit_all[level]
coeffs_emit = np.polyfit(noise_all, slopes_emit, 2)
slope_emit_fit = np.polyval(coeffs_emit, noise_all)

#plt.plot(np.array(noise_all), slopes_gaussemit, '.', label = 'Gaussian beam, gaussian emittance')
#plt.plot(np.array(noise_all), slopes_emit, '.', label = 'Gaussian beam, q-Gaussian emittance')

# Plot the data and the quadratic fit for qemit_all[level]


plt.plot(noise_all, slopes_gaussemit, '.', label = 'Gaussian beam, gaussian emittance')
plt.plot(noise_all, slopes_emit, '.', label = 'Gaussian beam, q-Gaussian emittance')
plt.plot(np.array(noise_all),slope_gaussemit_fit, '-', label=f'Gaussian emit fit $x^2$, Gaussian beam', markersize=8)
plt.plot(np.array(noise_all), slope_emit_fit, '--', label=f'q-Gaussian emit fit $x^2$, Gaussian beam')
#plt.xlabel('Noise Level')
plt.xlabel('Betx')
plt.ylabel('Slope')
plt.legend()
# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import xtrack as xt
import xpart as xp
import xobjects as xo

# Example q-Gaussian function (replace with the actual function if different)


def gaussianemit(x, betx):
    gaussemit = []
    for i in range(len(x)):
        sigma_x = float(np.std(x[i]))    
        geomx_emittance = (sigma_x ** 2 - ((dpp * disp) ** 2)) / betx
        gaussemit.append(geomx_emittance)
    return gaussemit

# Initialize parameters for noise and betx values
betx_values = np.linspace(1, 10, 30)
noise_levels = np.linspace(0.1, 1, 30)

# Arrays to store slope values
slope_matrix = np.zeros((len(betx_values), len(noise_levels)))

for i, betx in enumerate(betx_values):
    for j, noise_level in enumerate(noise_levels):
        # Initialize the line with current betx value
        elements = {
            'segment_map': xt.LineSegmentMap(_context=ctx, qx=0.27, qy=0.295, det_xx=1000, betx=betx)
        }
        line = xt.Line(elements=elements, element_names=['segment_map'])
        
        # Generate noise
        stdv = noise_level
        np.random.seed(0)
        samples = np.random.normal(0, stdv, len(time))
        
        exciter = xt.Exciter(_context=ctx,
            samples=samples,
            sampling_frequency=sampling_frequency,
            duration=num_turns/sampling_frequency,
            frev=sampling_frequency,
            knl=[1.] 
        )
        
        line.insert_element(
            element=exciter,
            name='white_noise_exciter',
            index=0,
        )
        
        # Build tracker
        line.build_tracker(_context=ctx)
        line.twiss_default['method'] = '4d'
        
        # Generate initial particle distribution
        x_in_sigmas, px_in_sigmas = xp.generate_2D_gaussian(N_particles)
        y_in_sigmas, py_in_sigmas = xp.generate_2D_gaussian(N_particles)
        
        gaussian_bunch = xp.build_particles(line=line, particle_ref=particle_ref,
            x_norm=x_in_sigmas, px_norm=px_in_sigmas,
            y_norm=y_in_sigmas, py_norm=py_in_sigmas, method='4d')
        
        p0 = gaussian_bunch.copy()
        
        # Arrays to store data
        x_data = np.zeros((N_particles, num_turns))
        
        # Track particles
        for k in range(num_turns):
            line.track(gaussian_bunch, freeze_longitudinal=True)
            x_data[:, k] = gaussian_bunch.copy().x
        
        # Compute emittance and fit slopes
        gaussemit_try = gaussianemit(x_data.T, betx)
        plt.plot(gaussemit_try)
        
        turns = np.arange(len(gaussemit_try))
        
        # Perform a linear fit for gaussemit_try
        slope_gaussemit, _ = np.polyfit(turns, gaussemit_try, 1)
        #plt.plot(gaussemit_try)
        #plt.show()
        # Store the slope value in the matrix
        slope_matrix[i, j] = slope_gaussemit  # You can also use slope_emit if desired

# Plot colormap
plt.figure(figsize = (5,3),dpi = 300)
plt.pcolormesh(noise_levels, betx_values, slope_matrix.T, shading='auto', cmap='viridis')
plt.colorbar(label=r'$d\epsilon/dt$')
plt.xlabel('Noise level')
plt.ylabel(r'$\beta_{s}$')

#plt.title('Slope of Emittance Evolution')
plt.show()

# %%
#=========================== Prediction ==============================
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def model(log_noise_betx, a, b, c):
    log_noise, log_betx = log_noise_betx
    return a * log_noise + b * log_betx + c

log_noise_levels = np.log(noise_levels)
log_betx_values = np.log(betx_values)

log_slope_matrix = np.log(slope_matrix)

# Generate a grid of all possible (log_noise_level, log_betx_value) pairs
log_noise_grid, log_betx_grid = np.meshgrid(log_noise_levels, log_betx_values)

# Flatten the matrices to create 1D arrays for fitting
log_noise_betx = np.vstack((log_noise_grid.flatten(), log_betx_grid.flatten()))
log_slopes = log_slope_matrix.flatten()

# Fit the model
popt, pcov = curve_fit(model, log_noise_betx, log_slopes)

# Extract the parameters
a, b, c = popt
print(f"Fitted parameters:\n a = {a}\n b = {b}\n c = {c}")

predicted_log_slopes = model(log_noise_betx, a, b, c)
predicted_slopes = np.exp(predicted_log_slopes)

# Reshape to original matrix form
predicted_slope_matrix = predicted_slopes.reshape(slope_matrix.shape)

plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.pcolormesh(noise_levels, betx_values, slope_matrix.T, shading='auto', cmap='viridis')
plt.colorbar(label='Observed Slope')
plt.xlabel('Noise Level')
plt.ylabel('Betx')
plt.title('Observed Slopes')

plt.subplot(1, 2, 2)
plt.pcolormesh(noise_levels, betx_values, predicted_slope_matrix.T, shading='auto', cmap='viridis')
plt.colorbar(label='Predicted Slope')
plt.xlabel('Noise Level')
plt.ylabel('Betx')
plt.title('Predicted Slopes')

plt.tight_layout()
plt.show()

# %%
import nafflib as NAFFlib
tbt_x = line.record_last_track.x.flatten() # particle_id, turns
tbt_px = line.record_last_track.px.flatten() # particle_id, turns
tbt_y = line.record_last_track.y.flatten() # particle_id, turns
tbt_py = line.record_last_track.py.flatten() # particle_id, turns
particles_id = line.record_last_track.particle_id.flatten()
turns =line.record_last_track.at_turn.flatten()
df = pd.DataFrame({'particle_id': particles_id, 'x': tbt_x, 'y': tbt_y, 'turn': turns} )
# %%
keys = []
qx_tot1 = []
qx_tot2 = []
qy_tot1 = []
qy_tot2 = []
diffusions = []
for key, group in df.groupby('particle_id'):
    qx1 = abs(NAFFlib.get_tune(group.x.values[:2000], 2))
    qy1 = abs(NAFFlib.get_tune(group.y.values[:2000], 2))
    qx2 = abs(NAFFlib.get_tune(group.x.values[-2000:], 2))
    qy2 = abs(NAFFlib.get_tune(group.y.values[-2000:], 2))
    
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
dff = pd.DataFrame({'particle_id': keys,'qx1': qx_tot1, 'qy1': qy_tot1, 'qx2':qx_tot2, 'qy2':qy_tot2, 'diffusion': diffusions} )
#dff = dff.merge(particle_df, on='particle_id')
dff.to_parquet(f'fma.parquet')
gamx = line.twiss(particle_ref=particle_ref).gamx
betx = line.twiss(particle_ref=particle_ref).betx
alfx = line.twiss(particle_ref=particle_ref).alfx
# %%
Jx = 1/2*(gamx[0]*tbt_x[::num_turns]**2 + 2*alfx[0]*tbt_px[::num_turns]*tbt_x[::num_turns] + betx[0]*tbt_px[::num_turns]**2)
plt.plot(Jx, dff['qx1'], '.')
# %%
# Scatter plot of initial tunes qx1 vs. qy1

plt.figure(figsize=(10, 6))
plt.scatter(dff['qx1'], dff['qy1'], c='blue', label='Initial Tunes', alpha=0.5)
plt.xlabel('qx1')
plt.ylabel('qy1')
plt.title('Initial Tunes qx1 vs. qy1')
plt.legend()
plt.grid(True)
# %%
# Scatter plot of final tunes qx2 vs. qy2

plt.scatter(dff['qx2'], dff['qy2'], c='red', label='Final Tunes', alpha=0.1)
plt.xlabel('qx')
plt.ylabel('qy')
plt.title('Tunes qx vs. qy')
plt.legend()
plt.grid(True)
#plt.xlim(0.2, 0.35)
#plt.ylim(0.24, 0.4)
plt.show()

# Diffusion plot
plt.figure(figsize=(10, 6))
plt.scatter(dff['particle_id'], dff['diffusion'], c='green', label='Diffusion', alpha=0.5)
plt.xlabel('Particle ID')
plt.ylabel('Log10 Diffusion')
plt.title('Log10 Diffusion of Tunes')
plt.legend()
plt.grid(True)

plt.show()

# %%
