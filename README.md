# DEM in hierarchical dynamical models

This repo contains a reimplementation of Dynamic Expectation Maximization [1], largely based on [2].

[1] K. J. Friston, N. Trujillo-Barreto, and J. Daunizeau, “DEM: A variational treatment of dynamic systems,” NeuroImage, vol. 41, no. 3, pp. 849–885, Jul. 2008, doi: 10.1016/j.neuroimage.2008.02.054.

[2] A. Anil Meera and M. Wisse, “Dynamic Expectation Maximization Algorithm for Estimation of Linear Systems with Colored Noise,” Entropy (Basel), vol. 23, no. 10, p. 1306, Oct. 2021, doi: 10.3390/e23101306.

## Structure
The repository is very much a work in progress, so it's still quite messy.

- Directory `benchmarks` contains a number of alternative implementations of the key functions, as well as tests to check whether they are valid and benchmarks for speed comparison.
- Implementations of DEM meant for reuse are in `hdm/dem`
- `demo` scripts are meant to illustrate DEM on examples, but they're very much unfinished.

## Profiling
pytest-profiling allows for finding particularly slow functions. To run it, for example try
```bash
python -m pytest benchmarks/batched.py --profile
```

Running the above will save some cProfile outputs to `prof`. These can be inspected by, for example:
```python
import pstats
p = pstats.Stats('prof/combined.prof')
p.sort_stats('cumulative').print_stats(50)
```
