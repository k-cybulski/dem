import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

from hdm.dummy import sin_gen, cos_gen, combine_gen
from hdm.core import taylor_mat


# Sanity check: Can we convert from generalized coordinates to normal coordinates?
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
        x_gen = sin_gen(p, t)
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
        x1_gen = sin_gen(p, t)
        x2_gen = cos_gen(p, t)
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

    ps_to_test = [2, 4, 5, 8, 9, 12, 16, 32, 40]
    for p in ps_to_test:

        n0 = 100

        nstart = int(n0 - np.ceil(p / 2))
        xs_to_inv = xs[nstart:nstart + p + 1].reshape((-1, 1))
        x_gen_appr = np.linalg.inv(taylor_mat(p, dt)) @ xs_to_inv

        t0 = ts[n0]
        x_gen_target = sin_gen(p, t0)

        # Where do the derivatives differ?
        plt.plot(np.abs(x_gen_appr - x_gen_target), label=f'p = {p}')

    plt.semilogy()
    plt.suptitle("Log absolute difference between true derivative and estimate")
    plt.xlabel("Which derivative")
    plt.legend()
    plt.show()

    for p in ps_to_test:

        n0 = 100

        nstart = int(n0 - np.ceil(p / 2))
        xs_to_inv = xs[nstart:nstart + p + 1].reshape((-1, 1))
        x_gen_appr = np.linalg.inv(taylor_mat(p, dt)) @ xs_to_inv

        t0 = ts[n0]
        x_gen_target = sin_gen(p, t0)

        # Where do the derivatives differ?
        plt.plot(np.abs(x_gen_appr - x_gen_target) / np.abs( x_gen_target), label=f'p = {p}')

    plt.ylim(-0.05, 1.05)
    plt.suptitle("Relative difference between true derivative and estimate")
    plt.xlabel("Which derivative")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    plot_taylor_approx_for_sin_cos()
    plot_taylor_inv_for_sin()
