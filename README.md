# Dynamic Expectation Maximization

This repo contains an implementation of Dynamic Expectation Maximization (DEM) [1], largely based on [2].

[1] K. J. Friston, N. Trujillo-Barreto, and J. Daunizeau, “DEM: A variational treatment of dynamic systems,” NeuroImage, vol. 41, no. 3, pp. 849–885, Jul. 2008, doi: 10.1016/j.neuroimage.2008.02.054.

[2] A. Anil Meera and M. Wisse, “Dynamic Expectation Maximization Algorithm for Estimation of Linear Systems with Colored Noise,” Entropy (Basel), vol. 23, no. 10, p. 1306, Oct. 2021, doi: 10.3390/e23101306.

## Usage

The DEM algorithm API is accessible via two classes:

- `dem.algo.DEMInput` which contains input data as well as priors. It is static over the course of DEM.
- `dem.algo.DEMState` which contains parameter estimates. It contains all of the terms which vary over the course of DEM. It also includes methods for actually running the algorithm.

Example notebooks in `examples/notebooks` illustrate how these can be applied. For clarification on what all of the parameters mean, have a look at the documentation of these two classes.

## PyTorch implementations

Two PyTorch implementations are kept in the branch `legacy-torch`. These were removed from the main branch because they were considerably slower than the JAX implementation, to the extent that they weren't really usable. The cause of this is that PyTorch compilation does not, as of writing, support all of the operations used in these implementations.
