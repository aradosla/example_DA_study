# %%

alpha_x = 1.0
beta_x = 100.0
alpha_y = 2.0
beta_y = 10.0
K2 = 0.1
lmbd = K2 * beta_x**(3.0/2.0) / 2.0
K3 = -5.0 * 3.0 * K2**2 * beta_x / 2.0
test_context = xo.ContextCpu()
N = 1000
x_n = np.random.normal(N)
px_n = np.random.normal(N)
x = x_n / lmbd * np.sqrt(beta_x)
px = - alpha_x * x_n / np.sqrt(beta_x) / lmbd + px_n / np.sqrt(beta_x) / lmbd

p_n = xp.Particles(x=x_n, px=px_n, _context=test_context)
p = xp.Particles(x=x, px=px, _context=test_context)

henon_n = xt.Henonmap(omega_x = 2 * np.pi * 0.334,
                        omega_y = 2 * np.pi * 0.279,
                        n_turns = 1, 
                        twiss_params = [0., 1., 0., 1.],
                        multipole_coeffs = [2.0, -30.0],
                        norm = True)
line_n = xt.Line(elements=[henon_n], element_names=["henon_n"])
line_n.build_tracker(_context=test_context)
henon = xt.Henonmap(omega_x = 2 * np.pi * 0.334,
                    omega_y = 2 * np.pi * 0.279,
                    n_turns = 1, 
                    twiss_params = [alpha_x, beta_x, alpha_y, beta_y],
                    multipole_coeffs = [K2, K3],
                    norm = False)
line = xt.Line(elements=[henon], element_names=["henon"])
line.build_tracker(_context=test_context)

nTurns = 100
x_n_all = np.zeros(N * nTurns)
x_all = np.zeros(N * nTurns)
px_n_all = np.zeros(N * nTurns)
px_all = np.zeros(N * nTurns)
for n in range(nTurns):
    line_n.track(p_n)
    line.track(p)
    x_n_all[n*N:(n+1)*N] = test_context.nparray_from_context_array(p_n.x)
    x_all[n*N:(n+1)*N] = test_context.nparray_from_context_array(p.x)
    px_n_all[n*N:(n+1)*N] = test_context.nparray_from_context_array(p_n.px)
    px_all[n*N:(n+1)*N] = test_context.nparray_from_context_array(p.px)
x_all_n = x_all * lmbd / np.sqrt(beta_x)
px_all_n = alpha_x * x_all / np.sqrt(beta_x) * lmbd + px_all * np.sqrt(beta_x) * lmbd

#assert np.all(np.isclose(x_n_all, x_all_n, atol=1e-10, rtol=1e-8))
#assert np.all(np.isclose(px_n_all, px_all_n, atol=1e-10, rtol=1e-8))
# %%
x_in_test  = np.asarray([1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 3, 3, 3, 0, 4]) * 0.01
px_in_test = np.asarray([0, 1, 0, 0, 2, 0, 0, 2, 2, 0, 3, 3, 0, 3, 4]) * 0.01
y_in_test  = np.asarray([0, 0, 1, 0, 0, 2, 0, 2, 0, 2, 3, 0, 3, 3, 4]) * 0.01
py_in_test = np.asarray([0, 0, 0, 1, 0, 0, 2, 0, 2, 2, 0, 3, 3, 3, 4]) * 0.01
N = len(x_in_test)
omega_x = 2 * np.pi / 3.0
omega_y = 2 * np.pi / 8.0

sin_omega_x = np.sin(omega_x)
cos_omega_x = np.cos(omega_x)
sin_omega_y = np.sin(omega_y)
cos_omega_y = np.cos(omega_y)

x_out_test = cos_omega_x * x_in_test + sin_omega_x * (px_in_test + (x_in_test**2 - y_in_test**2))
px_out_test = -sin_omega_x * x_in_test + cos_omega_x * (px_in_test + (x_in_test**2 - y_in_test**2))
y_out_test = cos_omega_y * y_in_test + sin_omega_y * (py_in_test - 2 * x_in_test * y_in_test)
py_out_test = -sin_omega_y * y_in_test + cos_omega_y * (py_in_test - 2 * x_in_test * y_in_test)

p_test = xp.Particles(x=x_in_test, px=px_in_test, y=y_in_test, py=py_in_test, _context=test_context)

henon_test = xt.Henonmap(omega_x = omega_x,
                            omega_y = omega_y,
                            n_turns = 1, 
                            twiss_params = [0.0, 1.0, 0.0, 1.0],
                            multipole_coeffs = [2.0],
                            norm = True)
line_test = xt.Line(elements=[henon_test], element_names=["henon"])
line_test.build_tracker(_context=test_context)


x_out = np.zeros(N * nTurns)
px_out = np.zeros(N * nTurns)
y_out = np.zeros(N * nTurns)
py_out = np.zeros(N * nTurns)

for n in range(nTurns):
    line_test.track(p_test)
    x_out[n*N:(n+1)*N] = test_context.nparray_from_context_array(p_test.x)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]
    px_out[n*N:(n+1)*N] = test_context.nparray_from_context_array(p_test.px)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]
    y_out[n*N:(n+1)*N] = test_context.nparray_from_context_array(p_test.y)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]
    py_out[n*N:(n+1)*N] = test_context.nparray_from_context_array(p_test.py)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]
    x_out_old = test_context.nparray_from_context_array(p_test.x)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]
    px_out_old = test_context.nparray_from_context_array(p_test.px)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]
    y_out_old = test_context.nparray_from_context_array(p_test.y)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]
    py_out_old = test_context.nparray_from_context_array(p_test.py)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]

assert np.all(np.isclose(x_out_old, x_out_test, atol=1e-15, rtol=1e-10))
assert np.all(np.isclose(px_out_old, px_out_test, atol=1e-15, rtol=1e-10))
assert np.all(np.isclose(y_out_old, y_out_test, atol=1e-15, rtol=1e-10))
assert np.all(np.isclose(py_out_old, py_out_test, atol=1e-15, rtol=1e-10))

assert np.all(np.isclose(x_out[-2*N:-1*N], x_out_test, atol=1e-15, rtol=1e-10))
assert np.all(np.isclose(px_out[-2*N:-1*N], px_out_test, atol=1e-15, rtol=1e-10))
assert np.all(np.isclose(y_out[-2*N:-1*N], y_out_test, atol=1e-15, rtol=1e-10))
assert np.all(np.isclose(py_out[-2*N:-1*N], py_out_test, atol=1e-15, rtol=1e-10))

p_inv_test = xp.Particles(x=x_out, px=px_out, y=y_out, py=py_out, _context=test_context)

line_test.track(p_inv_test, backtrack=True)

x_in_test_inv = test_context.nparray_from_context_array(p_inv_test.x)[np.argsort(test_context.nparray_from_context_array(p_inv_test.particle_id))]
px_in_test_inv = test_context.nparray_from_context_array(p_inv_test.px)[np.argsort(test_context.nparray_from_context_array(p_inv_test.particle_id))]
y_in_test_inv = test_context.nparray_from_context_array(p_inv_test.y)[np.argsort(test_context.nparray_from_context_array(p_inv_test.particle_id))]
py_in_test_inv = test_context.nparray_from_context_array(p_inv_test.py)[np.argsort(test_context.nparray_from_context_array(p_inv_test.particle_id))]

assert np.all(np.isclose(x_in_test_inv, x_in_test, atol=1e-15, rtol=1e-10))
assert np.all(np.isclose(px_in_test_inv, px_in_test, atol=1e-15, rtol=1e-10))
assert np.all(np.isclose(y_in_test_inv, y_in_test, atol=1e-15, rtol=1e-10))
assert np.all(np.isclose(py_in_test_inv, py_in_test, atol=1e-15, rtol=1e-10))
# %%


import numpy as np
import matplotlib.pyplot as plt
plot_params = {
    'figsize': (10, 6),
    'dpi': 100,
    'title_fontsize': 16,
    'label_fontsize': 14,
    'legend_fontsize': 12,
    'line_width': 2,
    'marker_size': 6,
    'grid': True,
    'color_map': 'viridis',
}

# Henon map function
def henon_map(x, p, nu, lam, beta, alpha):
  
    alpha0 = alpha
    beta0 = beta
    cos_nu = np.cos(2 * np.pi * nu)
    sin_nu = np.sin(2 * np.pi * nu)
    a11 = np.sqrt(beta/beta0) * (cos_nu + alpha0 * sin_nu)
    a12 = np.sqrt(beta0*beta) * sin_nu
    a21 = ((alpha0 - alpha) * cos_nu - (1 + alpha*alpha0) * sin_nu)/np.sqrt(beta*beta0)
    a22 = np.sqrt(beta0/beta) *( cos_nu - alpha * sin_nu)

    #p_new = p - lam * x**2
    p_new = p
    x_new = a11 * x + a12 * p_new
    p_new = a21 * x + a22 * p_new

    return x_new, p_new

def sextupolar_kick(x, p, lam):
    p_new = p - lam * x**2
    return x, p_new

def white_noise_kick(x, p, noise_std_dev):
    noise_x = np.random.normal(0, noise_std_dev, size=x.shape)
    noise_p = np.random.normal(0, noise_std_dev, size=p.shape)
    x_new = x + noise_x
    p_new = p + noise_p
    return x_new, p_new

'''
means_x = [0.01, 0.01, 0.01]  # Means fsize=num_particlesr px-coordinates
noise_std_dev = 0.1  # Standard deviation of the white noise

# Standard deviations (example values, adjust as needed)
std_devs_x = [0.05, 0.05, 0.05]  # Standard deviations for x-coordinates
std_devs_px = [0.01, 0.01, 0.01]  # Standard deviations for px-coordinates

'''
"""
# Gaussian distribution
means_x = [0.1, 0.3, 0.5]  # Means for x-coordinates
means_px = [0.0, 0.0, 0.0]  # Means for px-coordinates
noise_std_dev = 0.1  # Standard deviation of the white noise

# Standard deviations (example values, adjust as needed)
std_devs_x = [0.05, 0.05, 0.05]  # Standard deviations for x-coordinates
std_devs_px = [0.01, 0.01, 0.01]  # Standard deviations for px-coordinates

# Number of particles to generate
n_particles = 1000

# Generate Gaussian-distributed initial conditions
x_coords = []
px_coords = []

for mean_x, std_dev_x in zip(means_x, std_devs_x):
    x_coords.append(np.random.normal(loc=mean_x, scale=std_dev_x, size=n_particles))

for mean_px, std_dev_px in zip(means_px, std_devs_px):
    px_coords.append(np.random.normal(loc=mean_px, scale=std_dev_px, size=n_particles))

x_coords = np.concatenate(x_coords)
px_coords = np.concatenate(px_coords)

initial_conditions = np.vstack((x_coords, px_coords)).T
"""

# %%

import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_particles = 100000       # Number of particles
n_iterations = 1000      # Number of iterations for the map
nu = 0.27                # Angle ν for rotation (in fractions of 1 turn, e.g., 0.27 of a full circle)
lam = 1                  # Coefficient for the nonlinear part (if used)
noise_std_dev = 0.5      # Standard deviation of the white noise

# Generate Gaussian-distributed initial conditions
initial_conditions = np.vstack((np.random.normal(size=n_particles), 
                                np.random.normal(size=n_particles))).T

# Preallocate arrays to store results
x_all = np.zeros((n_iterations + 1, n_particles))
px_all = np.zeros((n_iterations + 1, n_particles))

# Assign initial conditions
x_all[0, :] = initial_conditions[:, 0]
px_all[0, :] = initial_conditions[:, 1]

# Rotation map function
def rotation_map(x, p, nu):
    cos_nu = np.cos(2 * np.pi * nu)
    sin_nu = np.sin(2 * np.pi * nu)
    x_new = x * cos_nu - p * sin_nu
    p_new = x * sin_nu + p * cos_nu
    return x_new, p_new

# Iterate over the number of iterations
for j in range(1, n_iterations + 1):
    x_all[j, :], px_all[j, :] = rotation_map(x_all[j-1, :], px_all[j-1, :], nu)
    # Optionally, apply noise or other kicks if needed
    # x_all[j, :], px_all[j, :] = white_noise_kick(x_all[j, :], px_all[j, :], noise_std_dev)
    # x_all[j, :], px_all[j, :] = sextupolar_kick(x_all[j, :], px_all[j, :], lam)

# Plot the trajectories for all particles
plt.figure(figsize=(12, 12))
plt.plot(x_all, px_all, '.', markersize=0.5)
plt.title(f'Trajectories for ν = {nu}')
plt.xlabel('x')
plt.ylabel('px')
plt.grid()
plt.tight_layout()
plt.show()

# %% 
import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_particles = 10000       # Number of particles
n_iterations = 1000      # Number of iterations for the map
nu = 0.27                # Angle ν for rotation (in fractions of 1 turn, e.g., 0.27 of a full circle)
lam = 1                  # Nonlinear coefficient (if used)
beta = 1.0               # Beta function
alpha = 0.0              # Alpha function
noise_std_dev = 0.1      # Standard deviation of the white noise
noise_all = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5,2, 5, 10] 
# Generate Gaussian-distributed initial conditions

x_all_noise = []
px_all_noise = []
initial_conditions = np.vstack((np.random.normal(size=n_particles), 
                                np.random.normal(size=n_particles))).T

# Preallocate arrays to store results
x_all = np.zeros((n_iterations + 1, n_particles))
px_all = np.zeros((n_iterations + 1, n_particles))

# Assign initial conditions
x_all[0, :] = initial_conditions[:, 0]
px_all[0, :] = initial_conditions[:, 1]

# Hénon-like map function
def henon_map(x, p, nu, lam, beta, alpha):
    alpha0 = alpha
    beta0 = beta
    cos_nu = np.cos(2 * np.pi * nu)
    sin_nu = np.sin(2 * np.pi * nu)
    a11 = np.sqrt(beta / beta0) * (cos_nu + alpha0 * sin_nu)
    a12 = np.sqrt(beta0 * beta) * sin_nu
    a21 = ((alpha0 - alpha) * cos_nu - (1 + alpha * alpha0) * sin_nu) / np.sqrt(beta * beta0)
    a22 = np.sqrt(beta0 / beta) * (cos_nu - alpha * sin_nu)

    # p_new = p - lam * x**2  # Uncomment this line if nonlinear effects are needed
    p_new = p
    x_new = a11 * x + a12 * p_new
    p_new = a21 * x + a22 * p_new

    return x_new, p_new


noise_x = np.random.normal(0,noise_std_dev, size = n_iterations)
noise_px = np.random.normal(0,noise_std_dev, size = n_iterations)

# White noise function
def white_noise_kick(x, p, noise_std):
    noise_x = np.random.normal(0, noise_std, size=x.shape)
    noise_p = np.random.normal(0, noise_std, size=p.shape)
    x_new = x + noise_x
    p_new = p + noise_p
    return x_new, p_new

for noise in noise_all:
    print(noise)
        # Preallocate arrays to store results
    x_all = np.zeros((n_iterations + 1, n_particles))
    px_all = np.zeros((n_iterations + 1, n_particles))

    # Assign initial conditions
    x_all[0, :] = initial_conditions[:, 0]
    px_all[0, :] = initial_conditions[:, 1]
    # Iterate over the number of iterations
    for j in range(1, n_iterations + 1):
        x_all[j, :], px_all[j, :] = henon_map(x_all[j-1, :], px_all[j-1, :], nu, lam, beta, alpha)
        #print(j)
        x_all[j, :], px_all[j, :] = white_noise_kick(x_all[j, :], px_all[j, :], noise)
        #x_all[j, :] += noise_x[j-1]
        #px_all[j, :]+= noise_px[j-1]
    x_all_noise.append(x_all)
    px_all_noise.append(px_all)

# Plot the trajectories for all particles
plt.figure(figsize=(12, 12))
plt.plot(x_all, px_all, '.', markersize=0.5)
plt.title(f'Trajectories with Hénon-like Map and White Noise')
plt.xlabel('x')
plt.ylabel('px')
plt.grid()
plt.tight_layout()
plt.show()

# %%
#Fitting Gaussian. qGaussian, emittance
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import scipy.special

def q_gaussian(x, beta, q, A):
    if q == 1:
        return A * np.exp(-beta * x**2)
    else:
        return A * (1 + (q - 1) * beta * x**2)**(1 / (1 - q))
def qGaussfit(x):
    hist, bin_edges = np.histogram(x[0], bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute histogram1
    hist1, bin_edges1 = np.histogram(x[-1], bins=100, density=True)
    bin_centers1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2

    # Initial guess for the parameters
    initial_guess = [1, 1.5, 1]

    # Fit the q-Gaussian to the histogram data
    params, _ = curve_fit(q_gaussian, bin_centers, hist, p0=initial_guess)
    params1, _ = curve_fit(q_gaussian, bin_centers1, hist1, p0=initial_guess)

    # Extract the parameters
    beta, q, A = params
    beta1, q1, A1 = params1

    x_plot = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    y_plot = q_gaussian(x_plot, *params)
    plt.plot(x_plot, y_plot, 'g-', linewidth=2, label=f'q-Gaussian fit initial: beta={beta:.2f}, q={q:.2f}, A={A:.2f}')
    plt.hist(x_all[0], bins=100, density=True, alpha=0.6, color='g', label='Initial distribution')
    plt.legend()

    x1 = np.linspace(bin_edges1[0], bin_edges1[-1], 1000)
    y1 = q_gaussian(x1, *params1)
    plt.plot(x1, y1, 'y-', linewidth=2, label=f'q-Gaussian fit final: beta={beta1:.2f}, q={q1:.2f}, A={A1:.2f}')
    plt.hist(x_all[-1], bins=100, density=True, alpha=0.6, color='yellow', label='Final distribution')
    plt.legend(loc ='upper left')
    plt.xlabel('x')
    plt.ylabel('Counts')
    plt.title(f'q-Gaussian fit white noise stdv={noise_std_dev}, Gaussian distr 10k particles, 1k turns')
    #plt.title(f'q-Gaussian fit no noise, Gaussian distr 10k particles, 1k turns')
    plt.grid(True)
    return params, params1
params, params1 = qGaussfit(x_all)

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
    qgaussian_emit = ((qgaussian_sigma**2 - (dpp*disp)**2)/beta_optics) * gamma *1e-6
    
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
    qgaussian_emit1 = ((qgaussian_sigma1**2 - (dpp*disp)**2)/beta_optics) * gamma *1e-6
    print(f'Initial emittance: {qgaussian_emit}')
    print(f'Final emittance: {qgaussian_emit1}')
    
    
emittance(params, params1)
# %%
def qGaussemit(x):

    qgaus_noise_emit = []
    qgaus_noise_q = []
    for level in range(len(x_all_noise)):
        qgaussian_emit_all = []
        qgaussian_q_all = []
        for turn in range(len(x_all)):
            hist, bin_edges = np.histogram(x_all_noise[level][turn], bins=100, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            A0 = np.max(hist)
            mu0 = bin_centers[np.argmax(hist)]
            sigma0 = np.std(x_all[turn])
            initial_guess = [1., 1., A0] 

            # Fit the q-Gaussian to the histogram data
            params, _ = curve_fit(q_gaussian, bin_centers, hist, p0=initial_guess)
        
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
            qgaussian_emit = ((qgaussian_sigma**2 - (dpp*disp)**2)/beta_optics) * gamma *1e-6
            qgaussian_emit_all.append(qgaussian_emit)
            qgaussian_q_all.append(qgaussian_q)   
        qgaus_noise_emit.append(qgaussian_emit_all)
        qgaus_noise_q.append(qgaussian_q_all)
             
    
    return qgaus_noise_emit, qgaus_noise_q

emit_all5, q = qGaussemit(x_all)


# %% 
# Gaussian fit
beta_optics = 1
disp = 0
gamma = 479.57 
dpp = 0
def gaussianemit(x):
    gaus_noise_emit = []
    for level in range(len(x_all_noise)):
        gaussemit = []
        for i in range(len(x_all)):
            sigma_x = float(np.std(x_all_noise[level][i]))    
            geomx_emittance = (sigma_x**2-((dpp*disp)**2)/beta_optics) * gamma *1e-6
            gaussemit.append(geomx_emittance)
        gaus_noise_emit.append(gaussemit)
    return gaus_noise_emit

gaussemit_try = gaussianemit(x_all)

for level in range(len(x_all_noise)):
    plt.plot(gaussemit_try[level])
    #plt.plot(emit_all5[level])
    plt.xlabel('Turns')
    plt.ylabel('Emittance')

# %%

# Fit the slope
import numpy as np
import matplotlib.pyplot as plt



slopes_gaussemit = []
slopes_emit = []

plt.figure(figsize=(12, 6))

for level in range(len(x_all_noise)):
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
    #plt.plot(turns, emit_all5[level], label=f'Emit, Noise Level {level}')
    #plt.plot(turns, slope_emit * turns + intercept_emit, '--', label=f'Fit Emit Slope {slope_emit:.2f}')
    
plt.xlabel('Turns')
plt.ylabel('Emittance')
#plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Print out the slopes for each noise level
for level in range(len(x_all_noise)):
    print(f'Noise Level {level}: Gaussemit Slope = {slopes_gaussemit[level]:.6f}, Emit Slope = {slopes_emit[level]:.6f}')

plt.plot(noise_all, slopes_gaussemit, '.')
#plt.plot(noise_all, slopes_emit, '.')
plt.xlabel('Noise Level')
plt.ylabel('Slope')
# %%
#OLD SEXTUPOL
def henon_map(x, p, nu, lam, beta, alpha):
  
    alpha0 = alpha
    beta0 = beta
    cos_nu = np.cos(2 * np.pi * nu)
    sin_nu = np.sin(2 * np.pi * nu)
    a11 = np.sqrt(beta/beta0) * (cos_nu + alpha0 * sin_nu)
    a12 = np.sqrt(beta0*beta) * sin_nu
    a21 = ((alpha0 - alpha) * cos_nu - (1 + alpha*alpha0) * sin_nu)/np.sqrt(beta*beta0)
    a22 = np.sqrt(beta0/beta) *( cos_nu - alpha * sin_nu)

    p_new = p - lam * x**2
    #p_new = p
    x_new = a11 * x + a12 * p_new
    p_new = a21 * x + a22 * p_new

    return x_new, p_new

def sextupolar_kick(x, p, lam):
    p_new = p - lam * x**2
    return x, p_new

# Parameters
n_iterations = 1000   # Number of iterations for the map
lam = 1               # Coefficient for the nonlinear part

# Gaussian distribution
means_x = [0.1, 0.3, 0.5]  # Means for x-coordinates
means_px = [0.0, 0.0, 0.0]  # Means for px-coordinates

# Standard deviations (example values, adjust as needed)
std_devs_x = [0.05, 0.05, 0.05]  # Standard deviations for x-coordinates
std_devs_px = [0.01, 0.01, 0.01]  # Standard deviations for px-coordinates

# Number of particles to generate
n_particles = 1000

# Generate Gaussian-distributed initial conditions
x_coords = []
px_coords = []

for mean_x, std_dev_x in zip(means_x, std_devs_x):
    x_coords.append(np.random.normal(loc=mean_x, scale=std_dev_x, size=n_particles))

for mean_px, std_dev_px in zip(means_px, std_devs_px):
    px_coords.append(np.random.normal(loc=mean_px, scale=std_dev_px, size=n_particles))

x_coords = np.concatenate(x_coords)
px_coords = np.concatenate(px_coords)

initial_conditions = np.vstack((x_coords, px_coords)).T

angles = [0.25/8, 0.3, 0.4, 0.5]  # Different values of the angles ν

# Plot the trajectories
plt.figure(figsize=(12, 12))
for i, nu in enumerate(angles):
    plt.subplot(2, 2, i + 1)
    plt.title(f'Trajectories for ν = {nu}')
    for x0, p0 in initial_conditions[:]:
        x, p = [x0], [p0]
        for _ in range(n_iterations):
            x_new, p_new = henon_map(x[-1], p[-1], nu, lam, 1, 0)
            x.append(x_new)
            p.append(p_new)
        plt.plot(x, p, '.', label=f'x0={x0}, p0={p0}')

    plt.xlabel('x')
    plt.ylabel('px')
    #plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()

# %%
import numpy as np
import xobjects as xo
import xpart as xp
import xtrack as xt
import matplotlib.pyplot as plt
import pandas as pd


# Define the Henon map as a custom element in xsuite
class HenonMap_mine(xt.BeamElement):

    _xofields = {
        'nu_x': xo.Float64,
        'nu_y': xo.Float64,
        'lam': xo.Float64,
    }

    def __init__(self, *, nu_x, nu_y, lam, _sin_rot_s=-999, _cos_rot_s=-999, _shift_x=0, _shift_y=0, _shift_s=0, _xobject=None, **kwargs):
        if _xobject is not None:
            super().__init__(_xobject=_xobject)
        else:
            super().__init__(
                nu_x=nu_x, nu_y=nu_y, lam=lam,
                _sin_rot_s=_sin_rot_s, _cos_rot_s=_cos_rot_s,
                _shift_x=_shift_x, _shift_y=_shift_y, _shift_s=_shift_s,
                **kwargs
            )
 
    def track(self, particles, num_turns=1):
        # Extract particle coordinates
        x_all = []
        y_all = []
        px_all = []
        py_all = []
        at_turn = []
        for _ in range(num_turns):   
            x = particles.x
            px = particles.px
            y = particles.y
            py = particles.py
            x_all.append(particles.copy().x)
            px_all.append(particles.copy().px)
            y_all.append(particles.copy().y)
            py_all.append(particles.copy().py)

            # Rotation matrix components for x-px and y-py spaces
            cos_nu_x = np.cos(2 * np.pi * self.nu_x)
            sin_nu_x = np.sin(2 * np.pi * self.nu_x)
            cos_nu_y = np.cos(2 * np.pi * self.nu_y)
            sin_nu_y = np.sin(2 * np.pi * self.nu_y)

            # Apply the nonlinear part (thin sextupole kick) for x and y
            px -= self.lam * x**2
            py -= self.lam * y**2

            # Apply the linear rotation for x-px space
            x_new = cos_nu_x * x + sin_nu_x * px
            px_new = -sin_nu_x * x + cos_nu_x * px

            # Apply the linear rotation for y-py space
            y_new = cos_nu_y * y + sin_nu_y * py
            py_new = -sin_nu_y * y + cos_nu_y * py

            # Update particle coordinates
            particles.x = x_new
            particles.px = px_new
            particles.y = y_new
            particles.py = py_new
            turn = _*np.ones_like(particles.x)
            
            at_turn.append(turn)
        print(at_turn)
        #at_turn = at_turn.flatten()
        df = pd.DataFrame({'x': x_all, 'px': px_all, 'y': y_all, 'py': py_all, 'at_turn': at_turn})
        return df

# Example of defining and using the HenonMap in an xsuite simulation

# Parameters
nu_x = 0.25
nu_y = 0.3
lam = 1.4

sigma_x = 1e-3       # Standard deviation for x
sigma_px = 1e-3     # Standard deviation for px
sigma_y = 1e-3       # Standard deviation for y
sigma_py = 1e-3      # Standard deviation for py
n_particles = 100

particle_ref = xp.Particles(
                    mass0=xp.PROTON_MASS_EV, q0=1, energy0=7000e9)

particles = xp.Particles(
    x=np.random.normal(0, sigma_x, n_particles),
    px=np.random.normal(0, sigma_px, n_particles),
    y=np.random.normal(0, sigma_y, n_particles),
    py=np.random.normal(0, sigma_py, n_particles),
    zeta=np.zeros(n_particles),
    delta=np.zeros(n_particles), particle_ref = particle_ref
)
# Create the HenonMap element
henon_map_element = HenonMap_mine(nu_x=nu_x, nu_y=nu_y, lam=lam)


elements = {
    'd1.1':  xt.Drift(length=1)
}



# Build the ring
line = xt.Line(elements=elements,
               element_names=['d1.1'])

# Define reference particle
line.particle_ref = xt.Particles(p0c=1.2e9, mass0=xt.PROTON_MASS_EV)


line.insert_element(
    element=henon_map_element,
    name='Henon', 
    index='d1.1',
)
# %%
# Build the tracker
line.build_tracker()
# %%

sigma_x = 1e-3       # Standard deviation for x
sigma_px = 1e-4      # Standard deviation for px
sigma_y = 1e-3       # Standard deviation for y
sigma_py = 1e-4      # Standard deviation for py
n_particles = 100

particle_ref = xp.Particles(
                    mass0=xp.PROTON_MASS_EV, q0=1, energy0=7000e9)

particles = xp.Particles(
    x=np.random.normal(0, sigma_x, n_particles),
    px=np.random.normal(0, sigma_px, n_particles),
    y=np.random.normal(0, sigma_y, n_particles),
    py=np.random.normal(0, sigma_py, n_particles),
    zeta=np.zeros(n_particles),
    delta=np.zeros(n_particles), particle_ref = particle_ref
)

# Track the particles
n_turns = 1000
x_coords = []
px_coords = []
y_coords = []
py_coords = []



for _ in range(n_turns):
    line.track(particles)
    x_coords.append(particles.x.copy())
    px_coords.append(particles.px.copy())
    y_coords.append(particles.y.copy())
    py_coords.append(particles.py.copy())

# Convert coordinates to numpy arrays for easy plotting
x_coords = np.array(x_coords)
px_coords = np.array(px_coords)
y_coords = np.array(y_coords)
py_coords = np.array(py_coords)

# Plot the trajectories in the x-px and y-py planes
plt.figure(figsize=(12, 6))

# Subplot for x-px plane
plt.subplot(1, 2, 1)
plt.title('Trajectories in x-px Plane')
for i in range(x_coords.shape[1]):
    plt.plot(x_coords[:, i], px_coords[:, i], '.', label=f'Particle {i+1}')

plt.xlabel('x')
plt.ylabel('p_x')
#plt.legend()
plt.grid()

# Subplot for y-py plane
plt.subplot(1, 2, 2)
plt.title('Trajectories in y-py Plane')
for i in range(y_coords.shape[1]):
    plt.plot(y_coords[:, i], py_coords[:, i], '.', label=f'Particle {i+1}')
plt.xlabel('y')
plt.ylabel('p_y')
#plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# %%
