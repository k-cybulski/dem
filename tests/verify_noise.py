import sympy as sym
import numpy as np
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot

from dem.core import taylor_mat
from dem.noise import (generate_noise_gp, generate_noise_conv,
                       autocorr_gaussian, autocorr_friston,
                       noise_cov_gen_theoretical)

## Theoretical covariances of noise vectors in generalized coordinates
# See Eq. 52 of Friston's DEM paper

def plot_noise_sample(sample, xs, name=None):
    # noise_samples expected to have one sequence of noise per column

    name_str = f'({name})' if name is not None else ''

    # Visualize one sample
    plt.plot(xs, sample)
    plt.suptitle(f"Example noise sequence {name_str}")
    plt.show()

    autocorrelation_plot(sample)
    plt.suptitle(f"Example noise autocorrelation {name_str}")
    plt.show()

def check_noise_statistics(sample, name=None):
    if name is not None:
        print(f"Statistics of {name}:")
    print(f"  Variance: {np.var(sample):.3f}\n  Mean: {np.mean(sample):.3f}")

def check_generalized_noise_covariance(samples, n, dt, sig, autocorr, name=None, k=9):
    # Check for whether the covariance of the generalized noise actually
    # follows the desired distribution

    name_str = f'({name})' if name is not None else ''
    xs = (np.arange(n) * dt).reshape((-1, 1))

    ## Taylor inversion
    mat = taylor_mat(k - 1, dt)
    gen_samples = []
    for i in range(0, samples.shape[0] - k):
        gen_sample = np.linalg.inv(mat) @ samples[i:(i + k), :]
        gen_samples.append(gen_sample)
    gen_samples = np.hstack(gen_samples)

    cov_est = np.cov(gen_samples)
    plt.imshow(cov_est, norm='symlog')
    plt.suptitle(f"Values of the estimated covariance matrix {name_str}")
    plt.colorbar()
    plt.show()

    cov_target = noise_cov_gen_theoretical(k - 1, sig, autocorr=autocorr)
    plt.imshow(np.abs(cov_est - cov_target), norm='log')
    plt.suptitle(f"Estimation error for generalized noise covariance {name_str}")
    plt.colorbar()
    plt.show()

    cov_target = noise_cov_gen_theoretical(k - 1, sig, autocorr=autocorr)
    plt.imshow(np.abs(cov_est - cov_target) / np.max(np.abs(np.array([cov_est, cov_target])), axis=0))
    plt.suptitle(f"Relative estimation error for generalized noise covariance {name_str}")
    plt.colorbar()
    plt.show()


def test_noise_gp():
    # Test if the noise sampled from a Gaussian Process has the properties we assume
    n = 1000
    dt = 0.1
    random_state = 5
    n_samples = 1000
    xs = (np.arange(n) * dt).reshape((-1, 1))

    # Is it similar to white noise for low sigma?
    var = 5
    sig = 0.000001
    samples = generate_noise_gp(n, dt, var, sig, n_samples=n_samples, random_state=random_state)
    sample = samples[:, 0]

    plot_noise_sample(sample, xs, name='hopefully non-autocorrelated GP noise')
    check_noise_statistics(samples, name='hopefully non-autocorrelated GP noise')

    # How about higher sigma?
    sig = 5
    samples = generate_noise_gp(n, dt, var, sig, n_samples=n_samples, random_state=random_state)
    sample = samples[:, 0]

    plot_noise_sample(sample, xs, name='autocorrelated GP noise')
    check_noise_statistics(samples, name='autocorrelated GP noies')

    # Does the generalized noise precision follow the desired formula?
    var = 1
    sig = 1
    n_samples = 5000
    samples = generate_noise_gp(n, dt, var, sig, n_samples=n_samples, random_state=random_state)

    check_generalized_noise_covariance(samples, n, dt, sig, autocorr=autocorr_gaussian(), name='autocorrelated GP noise')

def test_noise_conv():
    n = 1000
    dt = 0.1
    xs = np.arange(n) * dt
    random_state = 5
    var = 5
    sig = 1
    rng = np.random.default_rng(random_state)
    kern_size = 200
    n_samples = 1000

    samples = np.vstack([generate_noise_conv(n, dt, var, sig, kern_size, rng=rng) for _ in range(n_samples)]).T
    check_noise_statistics(samples, name='smoothed noise')

    var = 1
    n_samples = 5000

    samples = np.vstack([generate_noise_conv(n, dt, var, sig, kern_size, rng=rng) for _ in range(n_samples)]).T
    check_generalized_noise_covariance(samples, n, dt, sig, autocorr=autocorr_friston(), name='smoothed noise')


if __name__ == '__main__':
    print("In all tests printed below, we hope for a variance of 5 and mean of 1")
    test_noise_gp()
    test_noise_conv()
