# %%
import numpy as np
import pandas as pd
import yaml
import scipy

def load_configuration(config_path="config.yaml"):
    # Load configuration
    with open(config_path, "r") as fid:
        configuration = yaml.safe_load(fid)

    # Get configuration for the particles distribution and the collider separately
    config_particles = configuration["config_particles"]

    return configuration, config_particles

def parameters(n_part):
    #n_sigma = 6.0
    n_sigma = 5.0
    x = np.zeros(n_part)
    px = np.zeros(n_part)
    y = np.zeros(n_part)
    py = np.zeros(n_part)
    z = np.zeros(n_part)

    #sigma_d = 7.5e-2
    sigma_d = 7.5e-4
    dp = np.random.uniform(0.1 * sigma_d, 3.1 * sigma_d, n_part)
    return n_sigma, x, px, y, py, z, dp

def cmp_weights(df):
    r2 = df['x']**2 + df['px']**2 + df['y']**2 + df['py']**2
    w = np.exp(-r2/2.)
    w /= np.sum(w)
    return w

def generate_pseudoKV_xpyp(n_sigma):
    not_generated = True
    while not_generated:
        u = np.random.normal(size=4)
        r = np.sqrt(np.sum(u**2))
        u *= n_sigma / r
        v = np.random.normal(size=4)
        r = np.sqrt(np.sum(v**2))
        v *= n_sigma / r
        R2 = u[0]**2 + u[1]**2 + v[0]**2 + v[1]**2
        if R2 <= n_sigma**2:
            return u[0], u[1], v[0], v[1]

def df_colored_func(n_part):
    n_sigma, x, px, y, py, z, dp = parameters(n_part)
    pseudo_results = [generate_pseudoKV_xpyp(n_sigma) for _ in range(n_part)]
    
    for i, result in enumerate(pseudo_results):
        x[i], px[i], y[i], py[i] = result
    df = pd.DataFrame({'x': x, 'y': y, 'px': px, 'py': py, 'z': z, 'dp': dp})
    df['weights'] = cmp_weights(df)
    #print(df)
    return df


###### Load configuration ######
configuration, config_particles = load_configuration(config_path = 'config.yaml')
num_particles = int(float(config_particles["N_particles"]))


###### qGaussian new ######

import numpy as np
import scipy.special
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
        return (1 + (1 - q) * x)**(1 / (1 - q))

def qGauss(x, A, mu, q, b, offset):
    result = A * np.sqrt(b) / _Cq2(q) * _eq2(-b * (x - mu)**2, q) + offset
    result = np.where(np.isnan(result), offset, result)
    return result

def qGauss_CDF(x, A, mu, q, b, offset):
    pdf = lambda t: qGauss(t, A, mu, q, b, offset)
    cdf_values = [scipy.integrate.quad(pdf, -np.inf, xi)[0] for xi in x]
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
q = 1.4
b = 1
offset = 0

# Generate samples using the interpolation method
#samples = inverse_qGauss_CDF(uniform_samples, A, mu, q, b, offset)

pdf_values = qGauss(x_values, A, mu, q, b, offset)
x_q = inverse_qGauss_CDF(uniform_samples_x, A, mu, q, b, offset)
y_q = inverse_qGauss_CDF(uniform_samples_y, A, mu, q, b, offset)
px_q = inverse_qGauss_CDF(uniform_samples_px, A, mu, q, b, offset)
py_q = inverse_qGauss_CDF(uniform_samples_py, A, mu, q, b, offset)
df_q = pd.DataFrame({'x': x_q, 'y': y_q, 'px': px_q, 'py': py_q})
df_q.to_parquet('mydistribution.parquet')

# %%

'''
# Plotting the results
plt.figure(figsize=(8, 6))
plt.hist(samples, bins=30, alpha=0.6, color='g', label='Generated Samples', density=True)
plt.plot(x_values, pdf_values, 'r-', lw=2, label='q-Gaussian PDF')
plt.title('Samples from q-Gaussian Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.legend()
plt.show()



###### qGaussian distribution ######
from scipy.interpolate import interp1d
import scipy

def _Cq2(q):
    Gamma = scipy.special.gamma
    if q < 1:
        return (2*np.sqrt(np.pi))/((3-q)*np.sqrt(1-q))*(Gamma((1)/(1-q)))/(Gamma((3-q)/(2*(1-q))))
    elif q == 1:
        return np.sqrt(np.pi)
    elif q < 3:
        return (np.sqrt(np.pi))/(np.sqrt(q-1))*(Gamma((3-q)/(2*(q-1))))/(Gamma((1)/(q-1)))
    else:
        return 0

def _eq2(x, q):
    if q == 1:
        return np.exp(x)
    else: 
        if (1 - q) * np.any(x) < -1:
            return np.nan
        return (1 + (1 - q) * x) ** (1 / (1 - q))

def qGauss_samples(size, A, mu, q, b, offset):
    # Generate uniform random numbers
    u = np.random.rand(size)
    
    # Calculate normalization constant
    Cq = _Cq2(q)
    
    # Calculate inverse transform
    x = mu + np.sqrt(1/b * (_eq2(np.log((A*np.sqrt(b))/(Cq*u)), q)))  # Inverse of _eq2 function
    
    return x

# Define parameters for the q-Gaussian distribution
A = 1
mu = 0
q = 1.5  # Entropic index, q > 0
b = 1  # Related to variance
offset = 0

# Compute the PDF values for the q-Gaussian distribution
pdf_values = qGauss_samples(num_particles, A, mu, q, b, offset)

# Compute the cumulative sum of the PDF values
cumulative_sum = np.cumsum(pdf_values)

# Normalize the CDF
cdf = cumulative_sum / cumulative_sum[-1]
x_values = np.linspace(-5, 5, num_particles)

# Interpolate the CDF to create the inverse CDF
inverse_cdf = interp1d(cdf, x_values,  bounds_error=False, fill_value=(x_values[0], x_values[-1]))

# Generate uniform random numbers
uniform_random_numbers = np.random.rand(num_particles)

# Use the inverse CDF to get samples
samples = inverse_cdf(uniform_random_numbers)
'''

# %%


###### Colored distribution ######
df_colored = df_colored_func(num_particles)
#df_colored.to_parquet('mydistribution.parquet')

###### Gaussian distribution ######
x_norm = np.random.normal(size=num_particles)
px_norm = np.random.normal(size=num_particles)
y_norm = np.random.normal(size=num_particles)
py_norm = np.random.normal(size=num_particles)
        
#df_gaus = pd.DataFrame({'x': x_norm, 'y': y_norm, 'px': px_norm, 'py': py_norm})
#print(df_gaus)
#df_gaus.to_parquet('mydistribution.parquet')

# %%
