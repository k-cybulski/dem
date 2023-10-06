# Dynamic Expectation Maximization

This repo contains an implementation of Dynamic Expectation Maximization (DEM) [1] relying on [JAX](https://jax.readthedocs.io/en/latest/) for fast automatic differentiation. It largely follows the derivation from [2].

## Usage

To try out the package, get into your favourite Python virtual environment and run
```
pip install .
```
To run the example notebooks, also install
```
pip install jupyter matplotlib tqdm tabulate
```

The DEM algorithm API is accessible via two classes:

- `dem.algo.DEMInput` which contains input data as well as priors. It is static over the course of DEM.
- `dem.algo.DEMState` which contains parameter estimates. It contains all of the terms which vary over the course of DEM. It also includes methods for actually running the algorithm.

Example notebooks in `examples/notebooks` illustrate how these can be applied. For clarification on what all of the parameters mean, have a look at the documentation of these two classes.

## Caveats

The package more or less directly implements of Algorithm 1 from [2], without too many smart optimizations to speed up the procedure or lower memory usage. The purpose of this was to make the implementation relatively simple. For example, Hessians of the massive free action function with respect to dynamic states, parameters, and hyperparameters are computed directly by calling `jax.hessian` on it.

Unfortunately, this causes large memory usage, and makes the procedure slower than it could otherwise be. In case the memory runs out, some functions support a `low_memory` argument that lets them use a slower but less memory-intensive algorithm.

Some ways to speed it up and to make it more memory efficient:

- Implementing optimizations from [SPM](https://github.com/spm/) or [dempy](https://github.com/johmedr/dempy).
- Implementing optimizations from Algorithm 2 of [2] to the extent that they allow for nonlinear state transition and output functions.

### PyTorch version

Two PyTorch implementations of DEM are kept in the branch `legacy-torch`. These were removed from the main branch because they were considerably slower than the JAX implementation, to the extent that they weren't really usable. The cause of this is that PyTorch compilation does not, as of writing, support all of the operations used in these implementations.

## Alternative implementations of DEM

- [Statistical Parametric Mapping (SPM)](https://github.com/spm/) - the most extensive package implementing DEM, written in MATLAB
- [dempy](https://github.com/johmedr/dempy) - a Python implementation based on SPM, using [symengine](https://github.com/symengine/symengine.py) for symbolic differentiation and [NumPy](https://numpy.org/) for numerical operations.
- [DEM_LTI](https://github.com/ajitham123/DEM_LTI/) - a MATLAB implementation of DEM for LTI systems used in [2].

## References

[1] K. J. Friston, N. Trujillo-Barreto, and J. Daunizeau, “DEM: A variational treatment of dynamic systems,” NeuroImage, vol. 41, no. 3, pp. 849–885, Jul. 2008, doi: 10.1016/j.neuroimage.2008.02.054.

[2] A. Anil Meera and M. Wisse, “Dynamic Expectation Maximization Algorithm for Estimation of Linear Systems with Colored Noise,” Entropy (Basel), vol. 23, no. 10, p. 1306, Oct. 2021, doi: 10.3390/e23101306.
