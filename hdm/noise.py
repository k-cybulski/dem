from math import ceil

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

## Generating autocorrelated noise
# Method 1: Generate noise from a Gaussian process
# kernels: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF
def generate_noise_gp(n, dt, var, sig, kern=None, n_samples=1, random_state=None):
    if kern is None:
        kern = RBF(length_scale=sig) * ConstantKernel(constant_value=var)
    gp = GaussianProcessRegressor(kernel=kern)
    xs = (np.arange(n) * dt).reshape((-1, 1))
    return gp.sample_y(xs, n_samples=n_samples, random_state=random_state)


# Method 2: Generate white noise, convolve with a Gaussian kernel
def gauss_kern(a, b, sig):
    # it's not normalized to integrate to 1
    # but it has gauss_kern(0, 0, h) = 0
    return np.exp(- ((a - b)/sig)**2/2)

def gaussian_conv_kern(kern_size, dt, sig):
    xs = np.arange(kern_size) * dt
    xs = xs - np.median(xs)
    ys = gauss_kern(0, xs, sig)
    ys = ys / np.sum(ys)
    return ys

def generate_noise_conv(n, dt, var, sig, kern_size=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    if kern_size is None:
        kern_size = ceil(150 / dt) # rule of thumb for relatively good approximation
    kern = gaussian_conv_kern(kern_size, dt, sig)
    white_size = n + kern_size - 1
    white = np.random.normal(np.zeros(white_size), 1)
    conved = np.convolve(white, kern, mode='valid')
    # Adjust variance following Eq. (11) from Example 7.4 of
    #   D. R. Cox and H. M. Miller, The Theory of Stochastic Processes.
    #   Boca Raton: Routledge, 1965. doi: 10.1201/9780203719152.
    std_adj = np.sqrt(var / np.sum(kern ** 2))
    return conved * std_adj
