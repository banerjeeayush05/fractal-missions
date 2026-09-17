"""Finite differences on the level set. PRD §5.2.

One code path serves 2D and 3D: every function loops over `range(grid.ndim)` and the vertical
axis is 0 in both, so nothing here knows which dimension it is in.

**Boundaries.** PRD §5.1: lateral axes periodic, vertical Neumann. `_shift` is the only place
that knows the difference — a neighbour lookup either wraps or replicates the edge value — and
everything else is written as if the array were infinite.

**The Godunov upwind Hamiltonian.** For `phi_t + c|grad phi| = 0`, information flows from the
side the front is coming from, and a centred difference would average in the side it is going to
(which produces oscillations). So each axis contributes the ONE-SIDED difference on the upwind
side, selected by the sign of `c`:

    c > 0:  sum over axes of  max(D-, 0)^2 + min(D+, 0)^2
    c < 0:  sum over axes of  min(D-, 0)^2 + max(D+, 0)^2

Only `sign(c)` is used, which is what lets one function serve two callers: advection passes the
propagation coefficient, and reinitialisation passes the smoothed sign of phi.

**Two traps, both already paid for.**

The squaring is fine. `max(x, 0)^2` is C-1 -- its derivative is `2*max(x, 0)`, which is continuous
through zero. PRD §11 forbids kinks in the differentiated path, not the word `max`.

The square ROOT is not fine. At a Godunov "valley" cell, every selected branch is exactly zero,
the sum is exactly zero, and `d/dx sqrt(x)` is infinite at x = 0. One such cell makes `jax.grad`
return NaN for every parameter in the run. `GRAD_MAG_EPS` therefore sits INSIDE the sqrt, where
it keeps the argument off zero. Adding it outside, or clipping afterwards, does nothing: the
infinite derivative has already been taken.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from geocore.constants import GRAD_MAG_EPS, WENO_BOUNDARY_FALLBACK_CELLS, WENO_EPS
from geocore.schema import Grid


def _axis_slice(arr: Array, axis: int, start: int | None, stop: int | None) -> Array:
    index: list[slice] = [slice(None)] * arr.ndim
    index[axis] = slice(start, stop)
    return arr[tuple(index)]


def _shift(phi: Array, axis: int, offset: int, periodic: bool) -> Array:
    """`phi` evaluated at index `i + offset` along `axis`, with the boundary condition applied.

    The ONLY place in the module that knows a boundary exists.

    * periodic (lateral): the neighbour wraps around, which is exactly `jnp.roll`.
    * Neumann (vertical): zero normal derivative, imposed by replicating the edge value so the
      one-sided difference across the boundary is zero. A periodic vertical axis would wrap the
      plasma side onto the substrate side — `Grid` refuses to construct one.
    """
    if offset not in (-1, 1):
        raise ValueError(f"only nearest-neighbour shifts are supported, got {offset}")
    if periodic:
        return jnp.roll(phi, -offset, axis=axis)
    if offset == 1:
        return jnp.concatenate(
            [_axis_slice(phi, axis, 1, None), _axis_slice(phi, axis, -1, None)], axis=axis
        )
    return jnp.concatenate(
        [_axis_slice(phi, axis, 0, 1), _axis_slice(phi, axis, None, -1)], axis=axis
    )


def _shift_by(phi: Array, axis: int, offset: int, periodic: bool) -> Array:
    """`phi` at index `i + offset` for any offset, same boundary rules as `_shift`.

    WENO5 reads three cells each way, so it needs more than nearest neighbours. Periodic wraps;
    Neumann clamps the index to the edge, which is the same edge-value replication `_shift` does.
    """
    n = phi.shape[axis]
    index = jnp.arange(n) + offset
    index = index % n if periodic else jnp.clip(index, 0, n - 1)
    return jnp.take(phi, index, axis=axis)


def forward_difference(phi: Array, axis: int, grid: Grid) -> Array:
    """D+ = (phi[i+1] - phi[i]) / dx."""
    return (_shift(phi, axis, 1, grid.periodic[axis]) - phi) / grid.spacing_nm


def backward_difference(phi: Array, axis: int, grid: Grid) -> Array:
    """D- = (phi[i] - phi[i-1]) / dx."""
    return (phi - _shift(phi, axis, -1, grid.periodic[axis])) / grid.spacing_nm


def central_difference(phi: Array, axis: int, grid: Grid) -> Array:
    """(phi[i+1] - phi[i-1]) / (2 dx). For NORMALS, never for the Hamiltonian.

    Second-order and symmetric, which is what a normal direction wants. It is the wrong choice
    for advection: a centred stencil averages in the downwind side and oscillates.
    """
    return (
        _shift(phi, axis, 1, grid.periodic[axis]) - _shift(phi, axis, -1, grid.periodic[axis])
    ) / (2.0 * grid.spacing_nm)


def central_gradient(phi: Array, grid: Grid) -> Array:
    """All components of grad(phi) by central differences, stacked on a NEW LEADING axis.

    Leading rather than trailing so `gradient[0]` is the vertical component in both 2D and 3D,
    matching `VERTICAL_AXIS = 0`.
    """
    return jnp.stack([central_difference(phi, ax, grid) for ax in range(grid.ndim)], axis=0)


SPATIAL_SCHEMES = ("godunov", "weno5")


def _weno5_combine(v1, v2, v3, v4, v5, eps):
    """Jiang & Peng's fifth-order HJ-WENO combination of five first differences.

    Three third-order candidate stencils, weighted by how smooth each is. On smooth data the
    weights approach the ideal 0.1 / 0.6 / 0.3 and the result is fifth order; across a kink the
    rough stencils are suppressed and it degrades gracefully instead of oscillating.
    """
    s1 = 13.0 / 12.0 * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - 4 * v2 + 3 * v3) ** 2
    s2 = 13.0 / 12.0 * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2
    s3 = 13.0 / 12.0 * (v3 - 2 * v4 + v5) ** 2 + 0.25 * (3 * v3 - 4 * v4 + v5) ** 2
    a1 = 0.1 / (s1 + eps) ** 2
    a2 = 0.6 / (s2 + eps) ** 2
    a3 = 0.3 / (s3 + eps) ** 2
    total = a1 + a2 + a3
    return (a1 * (v1 / 3 - 7 * v2 / 6 + 11 * v3 / 6)
            + a2 * (-v2 / 6 + 5 * v3 / 6 + v4 / 3)
            + a3 * (v3 / 3 + 5 * v4 / 6 - v5 / 6)) / total


def _boundary_fallback_mask(grid: Grid, axis: int) -> Array:
    """True within `WENO_BOUNDARY_FALLBACK_CELLS` of a non-periodic edge along `axis` (finding J3).
    Broadcastable to the grid. A function of the index only."""
    n = grid.shape[axis]
    shape = [1] * grid.ndim
    shape[axis] = n
    if grid.periodic[axis]:
        return jnp.zeros(shape, dtype=bool)
    i = jnp.arange(n)
    near = (i < WENO_BOUNDARY_FALLBACK_CELLS) | (i >= n - WENO_BOUNDARY_FALLBACK_CELLS)
    return near.reshape(shape)


def one_sided_differences(phi: Array, axis: int, grid: Grid, scheme: str = "godunov",
                          weno_eps: float = WENO_EPS) -> tuple[Array, Array]:
    """(D-, D+) along one axis under the chosen spatial scheme.

    This is the ONLY thing WENO5 changes. The Godunov upwind SELECTION in `grad_mag_godunov` sits on
    top and is identical for both schemes, so the upwinding logic has one place to be.
    """
    d_minus = backward_difference(phi, axis, grid)
    d_plus = forward_difference(phi, axis, grid)
    if scheme == "godunov":
        return d_minus, d_plus
    if scheme != "weno5":
        raise ValueError(f"spatial scheme must be one of {SPATIAL_SCHEMES}, got {scheme!r}")

    periodic, dx = grid.periodic[axis], grid.spacing_nm

    def backward_at(offset):
        return (_shift_by(phi, axis, offset, periodic)
                - _shift_by(phi, axis, offset - 1, periodic)) / dx

    def forward_at(offset):
        return (_shift_by(phi, axis, offset + 1, periodic)
                - _shift_by(phi, axis, offset, periodic)) / dx

    w_minus = _weno5_combine(backward_at(-2), backward_at(-1), backward_at(0),
                             backward_at(1), backward_at(2), weno_eps)
    w_plus = _weno5_combine(forward_at(2), forward_at(1), forward_at(0),
                            forward_at(-1), forward_at(-2), weno_eps)
    fallback = _boundary_fallback_mask(grid, axis)
    return jnp.where(fallback, d_minus, w_minus), jnp.where(fallback, d_plus, w_plus)


def grad_mag_godunov(phi: Array, speed: Array, grid: Grid, eps: float = GRAD_MAG_EPS,
                     scheme: str = "godunov") -> Array:
    """|grad phi| by Godunov upwinding. Only `sign(speed)` is used.

    `speed` is the coefficient `c` in `phi_t + c|grad phi| = 0`, broadcastable to phi's shape:

    * advection passes `UPWIND_SELECTOR_SIGN * R`, i.e. `-R`, since M2 solves
      `phi_t - R|grad phi| = 0`;
    * reinitialisation passes the smoothed sign of phi, `phi0 / sqrt(phi0^2 + dx^2)`.

    `eps` is inside the sqrt on purpose. See the module docstring — this is the single most
    expensive line in the file to get wrong, and it fails as NaN in every parameter rather than
    as a wrong number in one.
    """
    upwind = jnp.zeros_like(phi)
    downwind = jnp.zeros_like(phi)
    for axis in range(grid.ndim):
        d_minus, d_plus = one_sided_differences(phi, axis, grid, scheme)
        upwind = upwind + jnp.maximum(d_minus, 0.0) ** 2 + jnp.minimum(d_plus, 0.0) ** 2
        downwind = downwind + jnp.minimum(d_minus, 0.0) ** 2 + jnp.maximum(d_plus, 0.0) ** 2
    return jnp.sqrt(jnp.where(speed > 0.0, upwind, downwind) + eps)
