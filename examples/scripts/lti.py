"""
This script runs a benchmark on the LTI example from [1]. It should more or
less exactly reproduce the same results as in the paper, up to the random noise
generated differently. Note that it takes a lot of memory, around 20 GB of RAM.

[1] A. Anil Meera and M. Wisse, “Dynamic Expectation Maximization Algorithm for
    Estimation of Linear Systems with Colored Noise,” Entropy (Basel), vol. 23,
    no. 10, p. 1306, Oct. 2021, doi: 10.3390/e23101306.
"""

import pickle
from pathlib import Path
from time import time

import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from dem.algo import DEMInput, DEMState, extract_dynamic
from dem.dummy import simulate_colored_lti
from jax import config
from matplotlib import pyplot as plt
from tabulate import tabulate
from tqdm import tqdm

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

##########
########## Test setup
##########


OUTPUT_DIR = Path("out/lti")

# lti model definition
A = np.array([[0.0484, 0.7535], [-0.7617, -0.2187]])
B = np.array([[0.3604], [0.0776]])
C = np.array(
    [[0.2265, -0.4786], [0.4066, -0.2641], [0.3871, 0.3817], [-0.1630, -0.9290]]
)
D = np.array([[0], [0], [0], [0]])

x0 = np.zeros(2)
t_max = 32
dt = 0.1
# input pulse
vs = np.exp(-0.25 * (np.arange(0, t_max, dt) - 12) ** 2).reshape((-1, 1))
# noises
rng = np.random.default_rng(215)
noise_temporal_sig = 0.5
## noise precisions in the paper were exp(8)
lam_w = 8
lam_z = 8
noise_prec = np.exp(8)
noise_var = 1 / noise_prec
noise_sd = np.sqrt(noise_var)
w_sd = noise_sd
z_sd = noise_sd
# simulate
ts, xs, ys, ws, zs = simulate_colored_lti(
    A, B, C, D, x0, dt, vs, w_sd, z_sd, noise_temporal_sig, rng
)
## the outputs look similar to Figure 5.1
# plt.plot(ts, xs)

##
## Model definition
##

# embedding order
p = 6  # for states
d = 2  # for inputs

m_x = 2
m_v = 1
m_y = 4


def ABC_from_params(params):
    shapes = ((m_x, m_x), (m_x, m_v), (m_y, m_x))
    cursor = 0
    matrs = []
    for rows, cols in shapes:
        size = rows * cols
        start = cursor
        end = cursor + size
        matr = params[start:end].reshape((rows, cols))
        cursor = end
        matrs.append(matr)

    A, B, C = matrs
    return A, B, C


true_params = np.concatenate([A.reshape(-1), B.reshape(-1), C.reshape(-1)])

known_value_exp = 32


def f_jax(x, v, params):
    A = params[0 : (m_x * m_x)].reshape((m_x, m_x))
    B = params[(m_x * m_x) : (m_x * m_x + m_x * m_v)].reshape((m_x, m_v))
    return jnp.matmul(A, x) + jnp.matmul(B, v)


def g_jax(x, v, params):
    C = params[(m_x * m_x + m_x * m_v) : (m_x * m_x + m_x * m_v + m_y * m_x)].reshape(
        (m_y, m_x)
    )
    return jnp.matmul(C, x)


## Priors

p_v = np.exp(known_value_exp).reshape((1, 1))
eta_v = vs.copy()

eta_theta = np.concatenate([rng.uniform(-2, 2, m_x * m_x + m_x * m_v), C.reshape(-1)])
p_theta_diag = np.concatenate(
    [
        np.full(m_x * m_x + m_x * m_v, np.exp(6)),
        np.full(m_y * m_x, np.exp(known_value_exp)),
    ]
)
p_theta = np.diag(p_theta_diag)

eta_lambda = np.zeros(2)
p_lambda = np.eye(2) * np.exp(3)

omega_w = np.eye(m_x)
omega_z = np.eye(m_y)

dem_input = DEMInput(
    dt=dt,
    m_x=m_x,
    m_v=m_v,
    m_y=m_y,
    p=p,
    d=d,
    ys=jnp.array(ys).astype(jnp.float64),
    eta_v=jnp.array(vs).astype(jnp.float64),
    p_v=p_v,
    eta_theta=eta_theta,
    p_theta=p_theta,
    eta_lambda=eta_lambda,
    p_lambda=p_lambda,
    f=f_jax,
    g=g_jax,
    noise_temporal_sig=noise_temporal_sig,
)

dem_state = DEMState.from_input(dem_input, x0)


lr_dynamic = 1
lr_theta = 10  # from the matlab code
lr_lambda = 1
iter_lambda = 8  # from the matlab code
m_min_improv = 0.01
num_iter = 25

print("Running initial D step")
t0 = time()
dem_state.step_d(lr_dynamic)
t_dynamic_first = time() - t0

trajectories = [[np.array(v) for v in extract_dynamic(dem_state)]]
param_estimates = [dem_state.mu_theta]
f_bars = []
f_bar_diagnostics = []


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_convergence_table(A, B, C, mu_thetas, f_bars):
    rows = []
    with np.printoptions(precision=3, suppress=True):
        for iter, (mu_theta, f_bar) in enumerate(zip(mu_thetas, f_bars)):
            A_est, B_est, C_est = ABC_from_params(mu_theta)
            A_diff = A - A_est
            B_diff = B - B_est
            C_diff = C - C_est
            row = [
                iter,
                f_bar.item(),
                str(A_diff),
                str(B_diff),
                str(C_diff),
                str(np.linalg.norm(A_diff)),
                str(np.linalg.norm(B_diff)),
                str(np.linalg.norm(C_diff)),
            ]
            rows.append(row)
    print(
        tabulate(
            rows,
            headers=(
                "Iter",
                "Free action",
                "A err",
                "B err",
                "C err",
                "A err norm",
                "B err norm",
                "C err norm",
            ),
        )
    )


times = {k: [] for k in ["D", "E", "M", "Precision"]}

times["D"].append(t_dynamic_first)

for i in tqdm(range(num_iter), desc="Running DEM..."):
    print("Step D")
    if i > 0:  # we already did a D step before
        t0 = time()
        dem_state.step_d(lr_dynamic)
        times["D"].append(time() - t0)
    print("Step M")
    t0 = time()
    dem_state.step_m(lr_lambda, iter_lambda, min_improv=m_min_improv, low_memory=False)
    times["M"].append(time() - t0)
    print("Step E")
    t0 = time()
    dem_state.step_e(lr_theta, low_memory=3)
    times["E"].append(time() - t0)
    print("Step precision")
    t0 = time()
    dem_state.step_precision()
    times["Precision"].append(time() - t0)

    param_estimates.append(dem_state.mu_theta)
    trajectories.append([np.array(v) for v in extract_dynamic(dem_state)])
    f_bar, extr = dem_state.free_action(diagnostic=True)

    f_bars.append(f_bar)
    extr = {key: item for key, item in extr.items()}
    f_bar_diagnostics.append(extr)

    demo_state = {
        "trajectories": trajectories,
        "param_estimates": param_estimates,
        "dem_state": dem_state,
        "f_bars": f_bars,
        "times": times,
    }

    with open(OUTPUT_DIR / "experiment_state.pkl", "wb") as file_:
        pickle.dump(demo_state, file_)

    pdict = {
        key: (jnp.linalg.norm(item), item.max(), item.min())
        for key, item in extr.items()
        if isinstance(item, jnp.ndarray)
    }
    print(
        tabulate(
            [(key, *item) for key, item in pdict.items()],
            headers=("variable", "norm", "max", "min"),
            floatfmt=".3f",
        )
    )
    print_convergence_table(A, B, C, param_estimates, f_bars)
    print(tabulate(times, headers="keys", floatfmt=".3f"))

##
## Analysis and plots
##

# in case this is opened in the REPL without having executed the experiment itself above
with open(OUTPUT_DIR / "experiment_state.pkl", "rb") as file_:
    demo_state = pickle.load(file_)

# Plot parameter estimate convergence
param_estimates = demo_state["param_estimates"]
param_estimates_by_param = jnp.array(param_estimates).T

fig, ax = plt.subplots()
for est_seq, target in zip(
    param_estimates_by_param[: (m_x * m_x + m_x * m_v), :],
    true_params[: (m_x * m_x + m_x * m_v)],
):
    line = ax.plot(est_seq)[0]
    ax.hlines(target, 0, len(est_seq) - 1, color=line.get_color(), linestyle="--")
# fig.suptitle("Parameter value estimates over course of DEM")
ax.set_xlabel("DEM iteration")
line_true = mpl.lines.Line2D(
    [], [], color="grey", marker="", markersize=15, label="True value", linestyle="--"
)
line_estimate = mpl.lines.Line2D(
    [], [], color="grey", marker="", markersize=15, label="Estimate", linestyle="-"
)
ax.legend(handles=[line_true, line_estimate])
fig.savefig(OUTPUT_DIR / "params_over_time.pdf")

# Plot trajectory at the start of DEM, and at the end
fig, ax = plt.subplots()
mu_xs, sig_xs, mu_vs, sig_vs, ts_est = demo_state['trajectories'][0]
for idx, (mu_x, x, color) in enumerate(zip(mu_xs.T, xs.T, ['red', 'blue'])):
    ax.plot(ts_est, mu_x, linestyle='--', color=color)
    ax.plot(ts, x, linestyle='-', color=color)
# plt.suptitle("Initial trajectory estimate")
ax.set_xlabel("time (s)")
ax.set_ylim(-1.5, 1.5)
line_true = mpl.lines.Line2D(
    [], [], color="grey", marker="", markersize=15, label="True value", linestyle="-"
)
line_estimate = mpl.lines.Line2D(
    [], [], color="grey", marker="", markersize=15, label="Estimate", linestyle="--"
)
ax.legend(handles=[line_true, line_estimate])
fig.savefig(OUTPUT_DIR / "trajectory_initial.pdf")


# Plot trajectory at the start of DEM, and at the end
fig, ax = plt.subplots()
mu_xs, sig_xs, mu_vs, sig_vs, ts_est = demo_state['trajectories'][-1]
for idx, (mu_x, x, color) in enumerate(zip(mu_xs.T, xs.T, ['red', 'blue'])):
    ax.plot(ts_est, mu_x, linestyle='--', color=color)
    ax.plot(ts, x, linestyle='-', color=color)
# plt.suptitle("Final trajectory estimate")
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("time (s)")
line_true = mpl.lines.Line2D(
    [], [], color="grey", marker="", markersize=15, label="True value", linestyle="-"
)
line_estimate = mpl.lines.Line2D(
    [], [], color="grey", marker="", markersize=15, label="Estimate", linestyle="--"
)
ax.legend(handles=[line_true, line_estimate])
fig.savefig(OUTPUT_DIR / "trajectory_final.pdf")
