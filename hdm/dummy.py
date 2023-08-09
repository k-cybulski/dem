"""
Various helper functions for testing.
"""

import numpy as np
from scipy.integrate import solve_ivp
from .noise import generate_noise_conv

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


def simulate_colored_lti(A, B, C, D, x0, n, dt, vs,
                         w_sd, z_sd, noise_temporal_sig, rng):
    """
    Simulates an LTI system with colored noise.

    x' = Ax + Bv + w
    y  = Cx + Dv + z

    Args:
        A, B, C, D: numpy matrices defining the LTI system
        x0: initial state x
        t_span: tuple (time_start, time_end)
        dt: sampling rate
        vs: inputs. Each column is separate feature, each row is separate timestep.
        w_sd, z_sd: standard deviations of noises (w is for state noise, z is for output noise)
        noise_temporal_sig: bandwidth of the temporal smoothing kernel
        rng: random number generator. Seed or np.random.Generator
    """
    assert all(len(matr.shape) == 2 for matr in (A, B, C, D)) # all are 2d matrices
    assert A.shape[0] == A.shape[1] # A is square
    assert A.shape[0] == B.shape[0] # B transforms inputs v into same shape as states x
    assert C.shape[1] == A.shape[0] # C accepts state vectors x
    assert B.shape[1] == D.shape[1] # B and D accept input vectors v
    assert C.shape[0] == D.shape[0] # C and D output the same shape of vectors

    t_start = 0
    t_end = n * dt
    t_span = (t_start, t_end)
    ts = np.arange(start=t_start, stop=t_end, step=dt)
    n = int((t_end - t_start) / dt)
    m_x = A.shape[0]
    m_v = B.shape[1]
    m_y = C.shape[0]

    # generate noises
    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    ws = np.vstack([
            generate_noise_conv(n, dt, w_sd ** 2, noise_temporal_sig, rng=rng)
            for _ in range(m_x)
        ]).T
    zs = np.vstack([
            generate_noise_conv(n, dt, z_sd ** 2, noise_temporal_sig, rng=rng)
            for _ in range(m_y)
        ]).T

    # function to run the systems with noise
    def f(t, x):
        # need to interpolate noise at this t separately for each feature
        # the interpolation points should overlap with the ts sampled by the
        # ODE solver, but we're doing this interpolation just in case
        noises = []
        for col in range(ws.shape[1]):
            noise_col = np.interp(t, ts, ws[:,col])
            noises.append(noise_col)

        vsin = []
        for col in range(vs.shape[1]):
            v_col = np.interp(t, ts, vs[:,col])
            vsin.append(noise_col)
        return A @ x + B @ np.array(vsin) + np.array(noises)

    out = solve_ivp(f, t_span, x0, t_eval=ts)

    xs = out.y.T
    ys = (C @ xs.T).T + (D @ vs.T).T + zs

    return ts, xs, ys, ws, zs

def dummy_lti(m_x, m_v, m_y, n, dt,
              v_sd, v_temporal_sig,
              w_sd, z_sd, noise_temporal_sig, rng):
    """
    Generates and simulates a random LTI with colored noise. The inputs to the
    system are generated as a Gaussian process.
    """
    A_shape = (m_x, m_x)
    B_shape = (m_x, m_v)
    C_shape = (m_y, m_x)
    D_shape = (m_y, m_v)
    A = rng.normal(np.zeros(A_shape))
    B = rng.normal(np.zeros(B_shape))
    C = rng.normal(np.zeros(C_shape))
    D = rng.normal(np.zeros(D_shape))
    x0 = rng.normal(np.zeros(m_x))
    vs = np.vstack([
        generate_noise_conv(n, dt, v_sd ** 2, v_temporal_sig, rng=rng)
        for _ in range(m_v)
    ]).T
    ts, xs, ys, ws, zs = simulate_colored_lti(A, B, C, D, x0, n, dt, vs,
                                              w_sd, z_sd, noise_temporal_sig, rng)
    return A, B, C, D, x0, ts, vs, xs, ys, ws, zs
