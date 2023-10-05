# Dynamic Expectation Maximization

**Branch note**: This branch contains PyTorch implementations of DEM. However, they were considerably slower than the JAX implementation, and somewhat of a burden to maintain. They are kept here for completeness.

This project provides implementations of Dynamic Expectation Maximization (DEM) [1] in Python, in JAX and in PyTorch. It is largely based on [2]  and extended to allow for non-linear state transition and cause functions as in [1].

[1] K. J. Friston, N. Trujillo-Barreto, and J. Daunizeau, “DEM: A variational treatment of dynamic systems,” NeuroImage, vol. 41, no. 3, pp. 849–885, Jul. 2008, doi: 10.1016/j.neuroimage.2008.02.054.

[2] A. Anil Meera and M. Wisse, “Dynamic Expectation Maximization Algorithm for Estimation of Linear Systems with Colored Noise,” Entropy (Basel), vol. 23, no. 10, p. 1306, Oct. 2021, doi: 10.3390/e23101306.
