"""Finite differences, boundaries, and the Godunov Hamiltonian (PRD §5.1, §5.2).

Conventions (`m2.constants`): φ < 0 inside solid, axis 0 is vertical and increases toward the
plasma, lateral axes are periodic and the vertical axis is Neumann (zero normal derivative).

Two spatial schemes, selected by `M2Config.spatial_scheme` (§5.2):

- **`godunov`** — first-order upwind one-sided differences. The default through M2.3.
- **`weno5`** — fifth-order HJ-WENO one-sided differences (Jiang & Peng 1999), landed after M2.3
  by owner decision H1. Only the *reconstruction* of D⁻ and D⁺ changes; the Godunov upwind
  selection on top of them is identical, which is why `godunov_grad_norm` still carries that name.

Why WENO5 was deferred rather than skipped: new numerics belong where V14/V15/V16 can verify them,
and that harness did not exist at M2.2. The cost of waiting was near zero because the adjoint is
automatic — changing the forward scheme means re-running the gradient checks, not rewriting an
adjoint.

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


# HJ-WENO (Jiang & Peng 1999). The ideal weights of the three candidate stencils, and the
# regulariser that keeps the weight denominators away from zero.
WENO_IDEAL = (0.1, 0.6, 0.3)
WENO_EPS = 1e-6
# **A deliberate deviation from the textbook formulation, for differentiability.** Jiang & Peng set
# ε = 1e-6·max(v₁²…v₅²) + 1e-99, a data-dependent scale that makes the scheme invariant to the
# magnitude of φ. That `max` over the stencil is a kink in the differentiated path, and §11 forbids
# those. A fixed ε is safe here because the smoothness indicators are built from v = Δφ/dx, which is
# |∇φ| — order 1 for a signed distance, whatever dx and whatever units φ carries. So the fixed and
# the scaled ε agree to within the amount |∇φ| strays from 1, which reinitialisation bounds.
# `tests/test_weno.py` pins the consequence: the observed order is unchanged if ε is moved by ±100×.


def _weno5_derivative(v1, v2, v3, v4, v5):
    """One WENO5-reconstructed one-sided derivative from five consecutive divided differences.

    No double-where is needed anywhere in here, which is worth stating because every other division
    in this file does need one. The weight denominators are `(β + ε)²` with β ≥ 0 a sum of squares
    and ε > 0, so they are bounded below by ε² > 0; their sum is bounded below by the smallest
    ideal weight over the largest denominator, so it too is strictly positive. There is no branch,
    no `sqrt`, and no path on which a zero can reach a denominator.
    """
    beta = (
        13.0 / 12.0 * (v1 - 2.0 * v2 + v3) ** 2 + 0.25 * (v1 - 4.0 * v2 + 3.0 * v3) ** 2,
        13.0 / 12.0 * (v2 - 2.0 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2,
        13.0 / 12.0 * (v3 - 2.0 * v4 + v5) ** 2 + 0.25 * (3.0 * v3 - 4.0 * v4 + v5) ** 2,
    )
    alpha = [ideal / (b + WENO_EPS) ** 2 for ideal, b in zip(WENO_IDEAL, beta)]
    total = alpha[0] + alpha[1] + alpha[2]
    candidates = (
        v1 / 3.0 - 7.0 * v2 / 6.0 + 11.0 * v3 / 6.0,
        -v2 / 6.0 + 5.0 * v3 / 6.0 + v4 / 3.0,
        v3 / 3.0 + 5.0 * v4 / 6.0 - v5 / 6.0,
    )
    return sum(a / total * c for a, c in zip(alpha, candidates))


def _divided_differences(phi: jax.Array, grid: Grid, axis: int, periodic: bool) -> dict:
    """Dₖ = (φ_{i+k} − φ_{i+k−1})/dx for k ∈ [−2, 3]: everything both WENO stencils need."""
    dx = grid.spacing_nm
    values = {k: shift(phi, axis, k, periodic) for k in range(-3, 4)}
    return {k: (values[k] - values[k - 1]) / dx for k in range(-2, 4)}


WENO_BOUNDARY_CELLS = 3  # the stencil half-width: cells this close to a hard edge cannot use it


def _away_from_hard_boundary(grid: Grid, axis: int) -> jax.Array:
    """Static mask: True where the full WENO stencil fits without reading a clamped value.

    **This exists because of a real failure, not as a precaution.** `shift` implements the Neumann
    condition by replicating the edge value. WENO reads that flat run as a perfectly smooth region,
    hands it the maximum ideal weight, and extrapolates from data that is an artefact of the
    boundary rather than of φ. Under Godunov the two-point stencil never sees past the edge and the
    scheme's own diffusion erases whatever leaks in; under WENO5 the error persists and accumulates.

    Measured on Zalesak at 100², one revolution: without this mask, φ at the bottom-edge corner
    cell (99, 4) drifts from +67 to **−2.96** by step 588 — a *spurious solid blob manufactured at
    the domain edge*, 67 cells from anything real. It then enters the velocity band and, under the
    rotation model whose rate grows with radius, drives the CFL number from 0.34 to 0.70 and trips
    the assertion. A phantom feature at a boundary would be a serious bug in a production run, so
    the CFL trip was the solver catching a real defect rather than a tuning problem.

    The mask is a function of the grid index alone — no data, no parameters — so it is a compile-time
    constant. It introduces no branch that can move under differentiation and no kink: it is exactly
    the "static window" pattern already used by V14c's seam window and the material fields.
    """
    n = grid.shape[axis]
    index = jnp.arange(n)
    inside = (index >= WENO_BOUNDARY_CELLS) & (index < n - WENO_BOUNDARY_CELLS)
    shape = [1] * grid.ndim
    shape[axis] = n
    return inside.reshape(shape)


def one_sided_differences(phi: jax.Array, grid: Grid, scheme: str = "godunov"
                          ) -> tuple[list[jax.Array], list[jax.Array]]:
    """Backward (D⁻) and forward (D⁺) differences per axis, in nm⁻¹.

    `scheme="weno5"` replaces the two-point differences with the fifth-order HJ-WENO
    reconstruction. Nothing downstream changes: the Godunov upwind selection, the reinitialisation
    Hamiltonian and the band machinery all consume the same two lists.

    Within `WENO_BOUNDARY_CELLS` of a non-periodic edge the reconstruction falls back to the
    two-point difference, because the wide stencil would otherwise read clamped values and treat
    them as smooth data. See `_away_from_hard_boundary` for the failure that motivated it. The
    fallback is first order, so the observed order degrades in those cells; that is the vertical
    axis only, where the interface is many cells away in every case M2 runs. If a run ever puts the
    interface within three cells of the top or bottom, its order claim is no longer valid there.
    """
    if scheme == "godunov":
        dx = grid.spacing_nm
        minus, plus = [], []
        for axis, periodic in enumerate(grid.periodic):
            minus.append((phi - shift(phi, axis, -1, periodic)) / dx)
            plus.append((shift(phi, axis, +1, periodic) - phi) / dx)
        return minus, plus

    if scheme != "weno5":
        raise ValueError(f"unknown spatial scheme {scheme!r}: expected 'godunov' or 'weno5' (§5.2)")

    minus, plus = [], []
    for axis, periodic in enumerate(grid.periodic):
        d = _divided_differences(phi, grid, axis, periodic)
        # D⁻ reads the backward-biased stencil; D⁺ reads its mirror image. Both reduce to the
        # two-point difference when the ideal weights are recovered, which is the smooth case.
        dm = _weno5_derivative(d[-2], d[-1], d[0], d[1], d[2])
        dp = _weno5_derivative(d[3], d[2], d[1], d[0], d[-1])
        if not periodic:
            # d[0] and d[1] ARE the Godunov backward and forward differences, so the fallback is
            # the other scheme exactly, not an approximation of it.
            interior = _away_from_hard_boundary(grid, axis)
            dm = jnp.where(interior, dm, d[0])
            dp = jnp.where(interior, dp, d[1])
        minus.append(dm)
        plus.append(dp)
    return minus, plus


def godunov_grad_norm(phi: jax.Array, grid: Grid, speed: jax.Array,
                      scheme: str = "godunov") -> jax.Array:
    """|∇φ| for the Hamiltonian, upwinded against the direction of motion (PRD §5.2).

    For φ_t + F|∇φ| = 0 the upwind choice depends on the sign of F: take the branch that looks in
    the direction information is coming from. M2 advects φ_t − R|∇φ| = 0 with R an etch rate
    (decision §1), so the caller passes F = −R.

    `max(x, 0)²` is C¹, so this is differentiable everywhere despite the max — §11 forbids kinks in
    the differentiated path, not the word `max`.

    `scheme` selects how D⁻ and D⁺ are reconstructed. The Godunov upwind selection below is the same
    either way, so the name still describes what this function does.
    """
    minus, plus = one_sided_differences(phi, grid, scheme)
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
