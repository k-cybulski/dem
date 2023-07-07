import numpy as np
from scipy.special import factorial

def taylor_mat(p, dt):
    # Eq 8 from Meera and Wisse
    matr = np.zeros((p + 1, p + 1))
    for i in range(1, p + 2):
        for j in range(1, p + 2):
            matr[i - 1, j - 1] = np.power((i - np.ceil((p + 1)/2)) * dt, j - 1) / factorial(j - 1)
    return matr
