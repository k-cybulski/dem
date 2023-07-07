import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

plot = False

# Part 1: Simulate some data
# generate data with a simple model that we can invert with DEM

## Define the model
# x' = f(x) + w
# y  = g(x) + z

## Simplest case
# x' = Ax
# y  = Ix + z
# for white noise z

x0 = np.array([0, 1])
A = np.array([[0, 1], [-1, 0]])

noise_sd = 0.1

def f(t, x):
    return A @ x

# Simulate the data
t_start = 0
t_end = 20
t_span = (t_start, t_end)
dt = 0.1
ts = np.arange(start=t_start, stop=t_end, step=dt)
out = solve_ivp(f, t_span, x0, t_eval=ts)

xs = out.y
ys = np.random.normal(xs, noise_sd)

if plot:
    plt.plot(ts, ys[0, :])
    plt.plot(ts, ys[1, :])
    plt.show()


# Part 2: Invert the model

## Extract generalized y, i.e. y_tilde
# we will use the inverted Taylor trick, using 6 derivatives

# say we focus on the 30th timestep
p = 6
t_step = 30


# x0 = np.array([[0], [1]])
# A = np.array([[0, 1], [-1, 0]])

# def evolve_system(x_0, A, iter=100):
#     # Evolves a system by Euler method
#     xs = x_0
#     x_i = x_0
#     for i in range(1, iter):
#         x_i = A @ x_i
#         xs = np.hstack([xs, x_i])
#     return xs

# xs = evolve_system(x_0, A)
