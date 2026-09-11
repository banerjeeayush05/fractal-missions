"""M2 — differentiable geometry core.

Importing ``m2`` switches JAX to fp64 (PRD §7.5). Every module that touches level-set data
imports this package first, so fp32 can never be the silent default.
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
