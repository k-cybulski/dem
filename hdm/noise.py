import numpy as np
from scipy.stats import multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import sympy as sym

from functools import cache

## Generating autocorrelated noise
# Method 1: Just make a covariance matrix from a desired kernel, and generate it all at once.

# Let's try to generate autocorrelated noise, and see if its precision in
# generalized coordinates is similar to the target
def gauss_kern(a, b, sig):
    # it's not normalized to integrate to 1
    # but it has gauss_kern(0, 0, h) = 0
    return np.exp(- ((a - b)/sig)**2/2)

def noise_cov(n, dt, var, sig, kern=gauss_kern):
    matr = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dist = 0
            else:
                dist = np.abs(i - j) * dt
            matr[i, j] = var * kern(0, dist, sig)
    return matr


def generate_noise(n, dt, var, sig, kern=gauss_kern, random_state=None):
    distr = multivariate_normal(np.zeros(n), noise_cov(n, dt, var, sig, kern))
    return distr.rvs(random_state=random_state)

# Method 2: Generate noise in generalized coordinates, use inverse Taylor trick to get simulated noise
# note: It's complete disaster hehe.
def noise_cov_gen(n, sig):
    # Covariance of noise in generalized coordinates
    # see eq. 52 of Friston's DEM paper
    x = sym.Symbol('x')
    sig_ = sym.Symbol('sigma')
    kern_sym = sym.exp(- ((x)/sig)**2/4)

    matr = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 1:
                matr[i, j] = 0
            else:
                diff_to = i + j
                diffed = sym.diff(kern_sym, (sym.Symbol('x'), diff_to))
                sign = -2 * ((i % 2) == 1) + 1
                matr[i, j] = sign * diffed.evalf(subs={x: 0, sig_: sig})
    return matr

# Method 3: Generate noise from a Gaussian process
# kernels: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF
def generate_noise_gp(n, dt, var, sig, kern=None, n_samples=1, random_state=None):
    if kern is None:
        kern = RBF(length_scale=sig) * ConstantKernel(constant_value=var)
    gp = GaussianProcessRegressor(kernel=kern)
    xs = (np.arange(n) * dt).reshape((-1, 1))
    return gp.sample_y(xs, n_samples=n_samples, random_state=random_state)


# Method 4: Generate white noise, convolve with a Gaussian kernel
def gaussian_conv_kern(kern_size, dt, sig):
    xs = np.arange(kern_size) * dt
    xs = xs - np.median(xs)
    ys = gauss_kern(0, xs, sig)
    ys = ys / np.sum(ys)
    return ys

def generate_noise_conv(n, dt, var, sig, kern_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    kern = gaussian_conv_kern(kern_size, dt, sig)
    white_size = n + kern_size - 1
    white = np.random.normal(np.zeros(white_size), 1)
    conved = np.convolve(white, kern, mode='valid')
    # Adjust variance following Eq. (11) from Example 7.4 of
    #   D. R. Cox and H. M. Miller, The Theory of Stochastic Processes.
    #   Boca Raton: Routledge, 1965. doi: 10.1201/9780203719152.
    std_adj = np.sqrt(var / np.sum(kern ** 2))
    return conved * std_adj
