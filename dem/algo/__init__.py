"""
This module contains implementations of DEM. These implementations differ in
how optimized they are. The default and most effective by far implementation is
in JAX.
"""

from .jax.algo import DEMInputJAX, DEMStateJAX, extract_dynamic

# JAX implementation is the default
DEMInput = DEMInputJAX
DEMState = DEMStateJAX
