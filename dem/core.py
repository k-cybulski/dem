import numpy as np
import torch
from scipy.special import factorial


def taylor_mat(p, dt, m=1, inv=False):
    """
    Taylor approximation matrix. Used for a Taylor polynomial approximation to
    move from time coordinates to generalized coordinates and vice versa. See
    Eq. 8 and 9 from [1].

    To approximate the function over p timesteps, do

        y_approx = taylor_mat @ y_generalized

    To estimate the generalized coordinates given p timesteps, do

        y_generalized_approx = inv(taylor_mat) @ y

    Args:
        p:  order of approximation. The generalized coordinates vector has p +
            1 entries, for p derivatives and for value at central timestep.
        dt: delta time of the time data.
        m:  [TODO] how many separate variables are described in y? y is expected


    [1] A. Anil Meera and M. Wisse, “Dynamic Expectation Maximization Algorithm
        for Estimation of Linear Systems with Colored Noise,” Entropy (Basel),
        vol. 23, no. 10, p. 1306, Oct. 2021, doi: 10.3390/e23101306.
    """
    # Eq 8 from Meera and Wisse
    matr = np.empty((p + 1, p + 1))
    for i in range(1, p + 2):
        for j in range(1, p + 2):
            matr[i - 1, j - 1] = np.power(
                (i - np.ceil((p + 1) / 2)) * dt, j - 1
            ) / factorial(j - 1)
    if inv:
        matr = np.linalg.inv(matr)
    if m > 1:
        # Doing this would follow the notation of [1] (Eq. 8)
        # however, this requires that the input is just a vector of interleaved features like
        # [x1 y1 x2 y2 x3 y3 ...].T but it's not clear that is the notation we should stick to?
        raise NotImplementedError("Don't use this.")
        matr = np.kron(matr, np.eye(m))
    return matr


def weave_gen(matr):
    """
    Transforms a matrix where each column is a vector in generalized
    coordinates into one, tall vector in generalized coords which is
    'interweaved' from the columns. So, it turns a matrix

        x1  x2  x3
        x1' x2' x3'

    into vector [x1 x2 x3 x1' x2' x3'].T
    """
    # NOTE: This might all be kind of pointless?
    # https://stackoverflow.com/a/5347492
    # out = np.empty((matr.shape[0] * matr.shape[1], 1), dtype=matr.dtype)
    # for col in range(matr.shape[1]):
    #     out[col::2, 0] = matr[:, col]
    # return out
    return matr.reshape((matr.shape[0] * matr.shape[1], 1))


def iterate_generalized(y, dt, p, p_comp=None):
    """
    Generate approximate vectors of y in generalized coordinates from a
    timeseries of y. The vectors are of size m * (p + 1), corresponding to
    m variables * (p derivatives + 1 value at midpoint).

    Args:
        y (torch.Tensor): Timeseries of data, where each column contains the
            timeseries of one covariate.
        dt (float): Sampling frequency.
        p (int): Embedding order in output vectors. The output vectors contain
            m * (1 + p) values.
        p_comp (int or None): Embedding order in computed vectors. Must be
            higher than or equal to p, so that the output vectors are
            truncated. The idea is that this can improve the estimation
            accuracy for lower derivatives, without having to include noisy
            high-level deriatives.
    """
    if p_comp is None:
        p_comp = p

    assert p_comp >= p

    # TODO: vectorize?
    if y.ndim == 1:
        y = y.reshape((-1, 1))

    y.shape[1]

    mat = taylor_mat(p_comp, dt, inv=True)
    if isinstance(y, torch.Tensor):
        mat = torch.from_numpy(mat).to(dtype=y.dtype, device=y.device)

    for i in range(0, y.shape[0] - p_comp - 1):
        weaved = weave_gen((mat @ y[i : (i + p_comp + 1), :])[: (p + 1), :])
        yield weaved


def len_generalized(n, p_comp):
    """How many generalized samples are we getting out of a size n sample?"""
    return n - p_comp - 1


def deriv_mat(p, n):
    """
    Block derivative operator.

    Args:
        p: number of derivatives
        n: number of terms
    """
    return np.kron(np.eye(p + 1, k=1), np.eye(n))
