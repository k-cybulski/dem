"""
Various helper functions for testing.
"""

import numpy as np
from scipy.integrate import solve_ivp
from .noise import generate_noise_conv

def sin_gen(p, t):
    n = p + 1
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

def cos_gen(p, t):
    sin_gen_ = sin_gen(p + 1, t)
    return sin_gen_[1:]

def combine_gen(gen1, gen2):
    # https://stackoverflow.com/a/5347492
    out = np.empty((gen1.shape[0] + gen2.shape[0], 1), dtype=gen1.dtype)
    out[0::2] = gen1
    out[1::2] = gen2
    return out


def wrap_with_innovations(ts, ws, vs):
    """
    Adds linearly interpolated noises w and inputs v to a function f(x, v), and
    returns a function f(t, x) which can be put in an ODE solver. This allows
    for simulating a convolution of noisy innovations. In other words, it
    allows for simulating a system

        x' = f(x, v) + w
    """
    def wrapper(func):
        def f(t, x):
            # FIXME OPT: Vectorize?
            noises = []
            for col in range(ws.shape[1]):
                noise_col = np.interp(t, ts, ws[:,col])
                noises.append(noise_col)
            w = np.array(noises)

            vsin = []
            for col in range(vs.shape[1]):
                v_col = np.interp(t, ts, vs[:,col])
                vsin.append(noise_col)
            v = np.array(vsin)
            return func(x, v) + w
        return f
    return wrapper


def simulate_system(f, g, x0, dt, vs, w_sd, z_sd, noise_temporal_sig, rng=None):
    """
    Simulates a system defined by

        x' = f(x, v) + w
        g  = g(x, v) + z

    for colored noises w and z.

    The functions take and return 1-dimensional numpy arrays.

    x0 is shape (m_x,)
    vs is shape (n, m_v)
    """
    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    n = vs.shape[0]
    t_start = 0
    t_end = n * dt
    t_span = (t_start, t_end)
    ts = np.arange(start=t_start, stop=t_end, step=dt)
    n = int((t_end - t_start) / dt)
    m_x = len(x0)
    m_v = vs.shape[1]

    out_of_0 = g(x0, vs[0]) # used to check shapes of output
    if np.ndim(out_of_0) == 0:
        m_y = 1
    else:
        m_y = out_of_0.shape[0]

    ws = np.vstack([
            generate_noise_conv(n, dt, w_sd ** 2, noise_temporal_sig, rng=rng)
            for _ in range(m_x)
        ]).T
    zs = np.vstack([
            generate_noise_conv(n, dt, z_sd ** 2, noise_temporal_sig, rng=rng)
            for _ in range(m_y)
        ]).T

    @wrap_with_innovations(ts, ws, vs)
    def ode_f(x, v):
        return f(x, v)

    out = solve_ivp(ode_f, t_span, x0, t_eval=ts)
    xs = out.y.T

    gs = np.array([
        g(x, v).reshape(-1) for x, v in zip(xs, vs)
    ])
    ys = gs + zs

    return ts, xs, ys, ws, zs


def simulate_colored_lti(A, B, C, D, x0, dt, vs,
                         w_sd, z_sd, noise_temporal_sig, rng):
    """
    Simulates an LTI system with colored noise.

    x' = Ax + Bv + w
    y  = Cx + Dv + z

    The number of timesteps is decided by length of inputs v

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

    def f(x, v):
        return A @ x + B @ v

    def g(x, v):
        return C @ x + D @ v
    return simulate_system(f, g, x0, dt, vs, w_sd, z_sd, noise_temporal_sig, rng=rng)

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
    ts, xs, ys, ws, zs = simulate_colored_lti(A, B, C, D, x0, dt, vs,
                                              w_sd, z_sd, noise_temporal_sig, rng)
    return A, B, C, D, x0, ts, vs, xs, ys, ws, zs
