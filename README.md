# Dynamic Expectation Maximization

This repo contains an implementation of Dynamic Expectation Maximization (DEM) [1], largely based on [2].

[1] K. J. Friston, N. Trujillo-Barreto, and J. Daunizeau, “DEM: A variational treatment of dynamic systems,” NeuroImage, vol. 41, no. 3, pp. 849–885, Jul. 2008, doi: 10.1016/j.neuroimage.2008.02.054.

[2] A. Anil Meera and M. Wisse, “Dynamic Expectation Maximization Algorithm for Estimation of Linear Systems with Colored Noise,” Entropy (Basel), vol. 23, no. 10, p. 1306, Oct. 2021, doi: 10.3390/e23101306.

Two PyTorch implementations are kept in the branch `legacy-torch`. These were removed from the main branch because they were considerably slower than the JAX implementation, to the extent that they weren't really usable. They became a burden to maintain, so they diverged somewhat from the JAX implementation.
