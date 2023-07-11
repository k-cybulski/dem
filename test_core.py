import numpy as np
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from scipy.stats import multivariate_normal
import sympy as sym

from stochastic.processes.noise import ColoredNoise

from hdm.core import taylor_mat
from hdm.noise import noise_cov_gen, generate_noise_gp


# Sanity check: Can we convert from generalized coordinates to normal coordinates?
def sin_gen(n, t):
    matr = np.zeros((n, 1))
    for i in range(n):
        i_mod = np.mod(i, 4)
        match i_mod:
            case 0:
                matr[i, 0] = np.sin(t)
            case 1:
                matr[i, 0] = np.cos(t)
            case 2:
                matr[i, 0] = -np.sin(t)
            case 3:
                matr[i, 0] = -np.cos(t)
    return matr

def cos_gen(n, t):
    sin_gen_ = sin_gen(n + 1, t)
    return sin_gen_[1:]

def combine_gen(gen1, gen2):
    # https://stackoverflow.com/a/5347492
    out = np.empty((gen1.shape[0] + gen2.shape[0], 1), dtype=gen1.dtype)
    out[0::2] = gen1
    out[1::2] = gen2
    return out


def plot_taylor_approx_for_sin():
    """
    Sanity check to see if the taylor_mat method works for a single variable function
    """
    t_start = 0
    t_end = 20
    t_span = (t_start, t_end)
    dt = 0.1
    ts = np.arange(start=t_start, stop=t_end, step=dt)
    xs = np.sin(ts)

    ts_to_check = [5, 10, 15]
    p = 20

    plt.plot(ts, xs, color='red')

    for t in ts_to_check:
        x_gen = sin_gen(p + 1, t)
        x_appr = taylor_mat(p, dt) @ x_gen
        ts_for_appr = t - np.ceil(p/2) * dt + ((np.arange(p + 1)) * dt)
        plt.plot(ts_for_appr, x_appr, color='green')

    plt.show()

def plot_taylor_approx_for_sin_cos():
    """
    Sanity check to see if the taylor_mat method works for a two variable function
    """
    t_start = 0
    t_end = 20
    t_span = (t_start, t_end)
    dt = 0.1
    ts = np.arange(start=t_start, stop=t_end, step=dt)
    x1s = np.sin(ts)
    x2s = np.cos(ts)

    ts_to_check = [5, 10, 15]
    p = 20

    plt.plot(ts, x1s, color='red')
    plt.plot(ts, x2s, color='pink')

    for t in ts_to_check:
        x1_gen = sin_gen(p + 1, t)
        x2_gen = cos_gen(p + 1, t)
        x_gen = combine_gen(x1_gen, x2_gen)
        t_mat = np.kron(taylor_mat(p, dt), np.eye(2))
        x_appr = t_mat @ x_gen
        x1_appr = x_appr[0::2]
        x2_appr = x_appr[1::2]
        ts_for_appr = t - np.ceil(p/2) * dt + ((np.arange(p + 1)) * dt)
        plt.plot(ts_for_appr, x1_appr, color='green')
        plt.plot(ts_for_appr, x2_appr, color='blue')

    plt.suptitle("Approximate and actual function values")
    plt.show()


def plot_taylor_inv_for_sin():
    """
    Sanity check to see if the inverse taylor_mat method works for computing derivatives.

    This little experiment also shows the magnitude of error in computing derivatives.
    """
    t_start = 0
    t_end = 20
    t_span = (t_start, t_end)
    dt = 0.1
    ts = np.arange(start=t_start, stop=t_end, step=dt)
    xs = np.sin(ts)

    ps_to_test = [2, 4, 8, 16, 32, 40]
    for p in ps_to_test:

        n0 = 100

        nstart = int(n0 - np.ceil(p / 2))
        xs_to_inv = xs[nstart:nstart + p + 1].reshape((-1, 1))
        x_gen_appr = np.linalg.inv(taylor_mat(p, dt)) @ xs_to_inv

        t0 = ts[n0]
        x_gen_target = sin_gen(p + 1, t0)

        # Where do the derivatives differ?
        plt.plot(np.abs(x_gen_appr - x_gen_target), label=f'p = {p}')

    plt.semilogy()
    plt.suptitle("Log absolute difference between true derivative and estimate")
    plt.xlabel("Which derivative")
    plt.legend()
    plt.show()


def test_noise_gen():
    n = 5
    dt = 0.1
    sig = 1

    cov_gen = noise_cov_gen(n, sig)
    distr = multivariate_normal(np.zeros(n), cov_gen)
    noise_gen = distr.rvs()

    noises = np.linalg.inv(taylor_mat(n - 1, dt)) @ noise_gen


def noise_cov_gen_theoretical(n, sig):
    # Theoretical covariance of noise in generalized coordinates, based on
    # symbolic computations of derivatives
    #
    # see eq. 52 of Friston's DEM paper
    # (though also note that their kernel formula is a bit different)
    x = sym.Symbol('x')
    sig_ = sym.Symbol('sigma')
    kern_sym = sym.exp(-((x)/sig)**2/2)

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



def test_noise_gp_gen():
    # Test if the noise sampled from a Gaussian Process has the properties we assume
    n = 1000
    dt = 0.1
    random_state = 5
    xs = (np.arange(n) * dt).reshape((-1, 1))


    # Is it similar to white noise for low sigma?
    var = 5
    sig = 0.000001
    sample = generate_noise_gp(n, dt, var, sig, random_state=random_state)

    plt.plot(xs, sample)
    plt.suptitle("Plot of (hopefully) non-autocorrelated noise")
    plt.show()
    autocorrelation_plot(sample)
    plt.suptitle("Autocovariance of (hopefully) non-autocorrelated noise")
    plt.show()

    print("For non-autocorrelated noise...")
    print(f"Variance: {np.var(sample):.3f}, Mean: {np.mean(sample):.3f}")

    # How about higher sigma?
    sig = 5
    sample = generate_noise_gp(n, dt, var, sig, random_state=random_state)

    plt.plot(xs, sample)
    plt.suptitle("Plot of autocorrelated noise")
    plt.show()
    autocorrelation_plot(sample)
    plt.suptitle("Autocovariance of autocorrelated noise")
    plt.show()

    print("For autocorrelated noise...")
    print(f"Variance: {np.var(sample):.3f}, Mean: {np.mean(sample):.3f}")

    # Does the generalized noise precision follow the desired formula?
    var = 1
    sig = 1
    n_samples = 5000
    samples = generate_noise_gp(n, dt, var, sig, n_samples=n_samples, random_state=random_state)

    ## Taylor inversion
    k = 9
    mat = taylor_mat(k - 1, dt)
    gen_samples = []
    for i in range(0, samples.shape[0] - k):
        gen_sample = np.linalg.inv(mat) @ samples[i:(i + k), :]
        gen_samples.append(gen_sample)
    gen_samples = np.hstack(gen_samples)

    cov_est = np.cov(gen_samples)
    plt.imshow(cov_est, norm='symlog')
    plt.suptitle("Values of the estimated covariance matrix")
    plt.colorbar()
    plt.show()

    cov_target = noise_cov_gen_theoretical(k, sig)
    plt.imshow(np.abs(cov_est - cov_target), norm='log')
    plt.suptitle("Estimation error for generalized noise covariance")
    plt.colorbar()
    plt.show()


    cov_target = noise_cov_gen_theoretical(k, sig)
    plt.imshow(np.abs(cov_est - cov_target) / np.max(np.abs(np.array([cov_est, cov_target])), axis=0))
    plt.suptitle("Relative estimation error for generalized noise covariance")
    plt.colorbar()
    plt.show()



if __name__ == '__main__':
    plot_taylor_approx_for_sin_cos()
    plot_taylor_inv_for_sin()
