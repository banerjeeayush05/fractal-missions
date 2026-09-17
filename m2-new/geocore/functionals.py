"""Smooth scalar functionals of the level set. PRD §5.7, decision A2.

`solid_volume` exists for ONE job: the §8.7 diagnostic decomposition. Run V14 on it and on an
extracted quantity like CD; if volume passes and CD fails, the bug is in extraction, and if both
fail it is upstream in advection, reinitialisation or checkpointing. That localises a gradient
failure in minutes. No product consumes this number.

It is smooth BY CONSTRUCTION -- a mollified Heaviside summed over cells -- which is exactly what
makes it a clean V14 target: nothing in it switches discretely as the interface crosses a grid
node, unlike a sub-cell crossing (whose derivative jumps there).

Named `solid_volume`, not "etched volume" (decision A2): with phi < 0 in solid, the integral of
`1 - H(phi)` is the material REMAINING. For the diagnostic the sign is irrelevant -- Taylor passes
or fails identically on J and -J.

In 2D it is an area per unit depth, in nm^2; in 3D a volume in nm^3.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from geocore.band import smoothstep
from geocore.constants import HEAVISIDE_WIDTH_CELLS
from geocore.schema import Grid


def heaviside(phi: Array, grid: Grid) -> Array:
    """Mollified Heaviside: 0 for phi <= -w, 1 for phi >= +w, C2 between. `w = 1.5 dx`.

    The quintic smoothstep, not the more common sine form, for consistency with the band weights
    and because it is C2 -- V14 fits a quadratic remainder, so the second derivative must exist.
    `w` is fixed in `constants.py` so the §8.7 diagnostic means the same thing in every run.
    """
    w = HEAVISIDE_WIDTH_CELLS * grid.spacing_nm
    return smoothstep((phi + w) / (2.0 * w))


def solid_volume(phi: Array, grid: Grid) -> Array:
    """Integral of (1 - H(phi)) over the domain."""
    return jnp.sum(1.0 - heaviside(phi, grid)) * grid.spacing_nm ** grid.ndim
