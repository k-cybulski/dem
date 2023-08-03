"""
Various helper functions for testing.
"""

import numpy as np

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
