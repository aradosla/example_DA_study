import numpy as np
import pandas as pd
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
    print(df)
    return df
