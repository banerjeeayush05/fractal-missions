"""Scalar outputs of a solve, differentiable end to end (PRD §5.7).

`solid_volume` is the smooth volumetric functional. No product uses it: it exists as a diagnostic
instrument, because it is smooth by construction, and running the gradient test on both it and an
extracted CD localises a failure in minutes (§8.7):

    volume pass, CD pass  -> the chain is sound
    volume pass, CD fail  -> the bug is in extraction
    volume fail, CD fail  -> the bug is in advection, reinitialisation or checkpointing
    volume fail, CD pass  -> something is badly wrong with the functional itself; stop and ask

Naming (decision A2): with φ < 0 inside solid, ∫(1 − H(φ)) dV is the **remaining solid**, not the
volume etched. It is named for what it measures. For its only job the sign is irrelevant — a Taylor
test passes or fails identically on J and −J — but a quantity named for the opposite of what it
computes is how sign errors survive review.

The mollification width is fixed in `constants.py` at 1.5·dx, not exposed to config, so it cannot be
tuned to make a gradient test pass (§11).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import m2  # noqa: F401  (enables fp64)
from m2.constants import HEAVISIDE_WIDTH_CELLS
from m2.schema import Grid


def mollified_heaviside(phi: jax.Array, grid: Grid) -> jax.Array:
    """H(φ) → 1 in the open volume, 0 in solid, smooth across a band of 1.5·dx.

    `tanh` rather than a polynomial ramp: it is C^∞, so it adds no kink anywhere in the
    differentiated path, and its derivative never becomes exactly zero, which keeps the gradient
    informative slightly away from the interface.
    """
    width = HEAVISIDE_WIDTH_CELLS * grid.spacing_nm
    return 0.5 * (1.0 + jnp.tanh(phi / width))


def solid_volume(phi: jax.Array, grid: Grid) -> jax.Array:
    """∫(1 − H(φ)) dV — the remaining solid, in nm^d."""
    cell = grid.spacing_nm**grid.ndim
    return jnp.sum(1.0 - mollified_heaviside(phi, grid)) * cell


def solid_volume_in_window(phi: jax.Array, grid: Grid, window: jax.Array) -> jax.Array:
    """Solid volume weighted by a static window, for measuring away from a domain seam.

    The window is a fixed array, so it carries no parameter dependence and cannot move under
    differentiation; it narrows *where* the functional looks, never *what* it measures.
    """
    cell = grid.spacing_nm**grid.ndim
    return jnp.sum((1.0 - mollified_heaviside(phi, grid)) * window) * cell


def open_volume(phi: jax.Array, grid: Grid) -> jax.Array:
    """∫H(φ) dV — the complement, for when the open side is the natural quantity."""
    cell = grid.spacing_nm**grid.ndim
    return jnp.sum(mollified_heaviside(phi, grid)) * cell


def etched_volume(phi: jax.Array, phi_reference: jax.Array, grid: Grid) -> jax.Array:
    """∫[H(φ) − H(φ_ref)] dV — the volume removed since a reference state, in nm^d.

    Decision A2 offered exactly this as the alternative to renaming: "define etched volume as the
    integral of H(phi) − H(phi_0)". It is the better-conditioned form, and the reason matters for
    the gradient tests.

    `solid_volume` sums a quantity of order 1 over every cell in the domain, so its rounding error
    is about sqrt(cells)·eps·|J| — with |J| ~ 1e5 nm² that is a noise floor near 1e-9, which eats
    the bottom decades of the Taylor test's sweep. Here the summand is exactly zero wherever the
    interface has not passed, so both the magnitude and the number of contributing terms collapse to
    the swept band. The noise floor drops by roughly three orders of magnitude, and the Taylor window
    can sit where the quadratic model actually holds (findings I1, I3).

    `phi_reference` is a constant field, so subtracting it changes no derivative: the gradient of
    this and of `solid_volume` are identical in exact arithmetic.
    """
    cell = grid.spacing_nm**grid.ndim
    return jnp.sum(mollified_heaviside(phi, grid) - mollified_heaviside(phi_reference, grid)) * cell
