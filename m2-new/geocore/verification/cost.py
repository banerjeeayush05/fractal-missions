"""Memory and time cost of the adjoint. PRD §7.4, decision I5(a), finding I4.

M2.3 MEASURES these; it does not gate on them. They exist to size M2.4's checkpoint schedule, where
segment length `L* = sqrt(N / k)` comes from the MEASURED residual factor k, never from `sqrt(N)` and
never hand-picked -- a hand-picked L becomes a knob that makes a memory gate pass.

**k, the residual factor.** Reverse mode stores intermediate arrays ("residuals") from the forward
pass so the backward pass can use them. k is how many phi-sized fields' worth of residuals one step
leaves behind. A run of N steps without checkpointing then needs about `N * k` fields at peak.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
from jax import Array


def residual_factor(fn: Callable[[Array], Array], phi: Array) -> float:
    """Bytes of residuals `jax.vjp(fn, phi)` keeps alive, in units of `phi.nbytes`.

    Counts the array leaves closed over by the VJP function. Scalars and Python constants are
    excluded; they do not scale with the grid.
    """
    _, vjp_fn = jax.vjp(fn, phi)
    stored = sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(vjp_fn)
                 if hasattr(leaf, "nbytes") and getattr(leaf, "ndim", 0) > 0)
    return stored / phi.nbytes


def optimal_segment_length(n_steps: int, k: float) -> float:
    """Two-level checkpointing peak is `N/L + L*k`, minimised at `L* = sqrt(N/k)` (PRD §7.4)."""
    return (n_steps / k) ** 0.5
