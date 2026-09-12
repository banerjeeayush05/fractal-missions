"""Finite differences, boundaries, and the Godunov Hamiltonian (PRD §5.1, §5.2).

Conventions (`m2.constants`): φ < 0 inside solid, axis 0 is vertical and increases toward the
plasma, lateral axes are periodic and the vertical axis is Neumann (zero normal derivative).

Everything here is first order in space (Godunov upwind). WENO5 is a §5.2 flag and is not built:
first-order upwind with adequate resolution is enough to verify gradients, and the extra stencil is
a place for bugs to hide.

The NaN trap: ∇φ/|∇φ| blows up where |∇φ| → 0, which happens on the medial axis — the centre of
V1's disk, the centreline of a trench. `jnp.where` alone does not save you, because the untaken
branch is still differentiated and a NaN there poisons the whole gradient. Every division here uses
the double-where pattern.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import m2  # noqa: F401  (enables fp64)
from m2.schema import Grid

# Below this, |∇φ| is treated as degenerate and the normal is taken to be zero rather than infinite.
GRAD_FLOOR = 1e-12


def shift(field: jax.Array, axis: int, offset: int, periodic: bool) -> jax.Array:
    """Neighbour lookup along one axis: periodic wrap, or Neumann (edge-replicating) clamp."""
    if periodic:
        return jnp.roll(field, shift=-offset, axis=axis)
    n = field.shape[axis]
    idx = jnp.clip(jnp.arange(n) + offset, 0, n - 1)  # zero normal derivative at the ends
    return jnp.take(field, idx, axis=axis)


def one_sided_differences(phi: jax.Array, grid: Grid) -> tuple[list[jax.Array], list[jax.Array]]:
    """Backward (D⁻) and forward (D⁺) differences per axis, in nm⁻¹."""
    dx = grid.spacing_nm
    minus, plus = [], []
    for axis, periodic in enumerate(grid.periodic):
        minus.append((phi - shift(phi, axis, -1, periodic)) / dx)
        plus.append((shift(phi, axis, +1, periodic) - phi) / dx)
    return minus, plus


def godunov_grad_norm(phi: jax.Array, grid: Grid, speed: jax.Array) -> jax.Array:
    """|∇φ| for the Hamiltonian, upwinded against the direction of motion (PRD §5.2).

    For φ_t + F|∇φ| = 0 the upwind choice depends on the sign of F: take the branch that looks in
    the direction information is coming from. M2 advects φ_t − R|∇φ| = 0 with R an etch rate
    (decision §1), so the caller passes F = −R.

    `max(x, 0)²` is C¹, so this is differentiable everywhere despite the max — §11 forbids kinks in
    the differentiated path, not the word `max`.
    """
    minus, plus = one_sided_differences(phi, grid)
    total = jnp.zeros_like(phi)
    for dm, dp in zip(minus, plus):
        forward = jnp.maximum(dm, 0.0) ** 2 + jnp.minimum(dp, 0.0) ** 2  # F > 0
        backward = jnp.minimum(dm, 0.0) ** 2 + jnp.maximum(dp, 0.0) ** 2  # F < 0
        total = total + jnp.where(speed >= 0, forward, backward)
    # Double-where around the sqrt itself, not just around a later division: d(sqrt)/dx is infinite
    # at zero, and 0 * inf is NaN, so a masked branch is not enough to keep the gradient finite.
    safe = jnp.where(total > GRAD_FLOOR**2, total, 1.0)
    return jnp.where(total > GRAD_FLOOR**2, jnp.sqrt(safe), 0.0)


def central_gradient(phi: jax.Array, grid: Grid) -> jax.Array:
    """∇φ by central differences, stacked on a leading axis: shape (ndim, *grid.shape)."""
    dx = grid.spacing_nm
    comps = [
        (shift(phi, axis, +1, periodic) - shift(phi, axis, -1, periodic)) / (2.0 * dx)
        for axis, periodic in enumerate(grid.periodic)
    ]
    return jnp.stack(comps, axis=0)


def normals(phi: jax.Array, grid: Grid) -> tuple[jax.Array, jax.Array]:
    """Unit normal n = ∇φ/|∇φ| and |∇φ|. n points from solid into the open volume (§5.1).

    Double-where: the denominator is replaced *before* the division, so the untaken branch never
    produces a NaN to differentiate. On the medial axis the normal is zero, not infinite.
    """
    grad = central_gradient(phi, grid)
    square = jnp.sum(grad**2, axis=0)
    live = square > GRAD_FLOOR**2
    # Guard the sqrt as well as the division: at the medial axis square == 0, where d(sqrt)/dx is
    # infinite. Masking afterwards does not help, because 0 * inf is NaN in the backward pass.
    safe_square = jnp.where(live, square, 1.0)
    norm = jnp.where(live, jnp.sqrt(safe_square), 0.0)
    safe_norm = jnp.where(live, norm, 1.0)
    unit = jnp.where(live, grad / safe_norm, 0.0)
    return unit, norm


def z_hat(grid: Grid) -> jax.Array:
    """The direction toward the plasma: +e₀, since axis 0 increases toward the plasma (§5.1)."""
    from m2.constants import Z_AXIS, Z_HAT_SIGN

    return jnp.zeros(grid.ndim).at[Z_AXIS].set(float(Z_HAT_SIGN))


def cell_centres(grid: Grid) -> jax.Array:
    """Coordinates of every cell centre in nm, shape (ndim, *grid.shape).

    The origin is the centre of cell 0 along every axis, so position and index agree: a point at
    index i on axis a sits at i·dx. Extraction and the coupon geometry share this convention.
    """
    axes = [jnp.arange(n, dtype=jnp.float64) * grid.spacing_nm for n in grid.shape]
    return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=0)
