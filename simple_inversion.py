"""
The goal of this script is to illustrate inversion of a simple model by DEM.
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import lti, cont2discrete, lsim

from hdm.noise import generate_noise_conv

# Simulating a system
# we will consider a simple, single variable model defined by
# x' = alpha * x + beta * u + w
# y = x + z

def simple_lti(alpha, beta):
    A = np.array([alpha])
    B = np.array([[beta, 1]]) # 1 for noise
    C = np.array([1])
    D = np.array([[0, 0]]) # 0 for noise
    return lti(A, B, C, D)

# Properties of the system
alpha = -0.5
beta = 5


sys = simple_lti(alpha, beta)
# Properties of noise
var = 2.5
sig = 1


# Properties of the simulation
dt = 0.1
n = 500
t = np.arange(n) * dt
u = np.zeros(n)
u[100:150] = 1
x0 = np.array([100])

# Run the simulation and generate noise
noise = generate_noise_conv(n, dt, var, sig)
u_with_noise = np.stack([u, noise], axis=1)
_, yout, xout = lsim(sys, u_with_noise, t, x0, interp=False)
plt.plot(t, yout, label="System output")
plt.plot(t, noise, label="Noise")
plt.legend()
plt.show()
