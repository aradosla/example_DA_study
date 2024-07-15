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

# Number of samples
num_samples = 1000
x_values = np.linspace(-10, 10, num_samples)

# Define the parameters of the q-Gaussian distribution
A = 1
mu = 0
q = 1.5
b = 1
offset = 0


# Generate uniform random numbers between 0 and 1
uniform_samples = np.random.uniform(0, 1, num_samples)

# Define the parameters of the q-Gaussian distribution
A = 1
mu = 0
q = 1.5
b = 1
offset = 0

# Generate samples using the interpolation method
samples = inverse_qGauss_CDF(uniform_samples, A, mu, q, b, offset)

pdf_values = qGauss(x_values, A, mu, q, b, offset)

plt.figure(figsize=(8, 6))
plt.hist(samples, bins=30, alpha=0.6, color='g', label='Generated Samples', density=True)
plt.plot(x_values, pdf_values, 'r-', lw=2, label='q-Gaussian PDF')
plt.title('Samples from q-Gaussian Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.legend()
plt.show()