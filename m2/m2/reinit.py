"""Reinitialisation: re-impose the signed-distance property (PRD §5.3).

φ drifts away from a signed distance under advection, and — as M2.1's finding F1 showed — a
band-limited velocity extension makes that drift accumulate until the front stalls. This repairs it
by iterating

    ∂φ/∂τ + S(φ₀)(|∇φ| − 1) = 0

for a **fixed** number of iterations, every `reinit_every` steps.

Three rules from the PRD, none of them negotiable:
- `S(φ₀)` is the **smoothed** sign φ₀/√(φ₀² + dx²). The exact sign function is non-differentiable at
  the interface and would corrupt the adjoint.
- The iteration count is **fixed** from config. A convergence-based stop would make the computation
  graph depend on the parameters, which is the §5.2 problem all over again.
- **No `stop_gradient`** (§11). Reinitialisation is a real part of the forward map and its
  derivative is real. If you think one is needed, stop and ask.

The spatial scheme is the Godunov form for the reinitialisation Hamiltonian (Rouy–Tourin): per
axis, take the branch that looks upwind of the interface, then the larger of the two squared
one-sided differences. Every square root is double-where guarded, because d(√x)/dx is infinite at
zero and `0 * inf` is NaN in the backward pass — the bug M2.1's F4 caught.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import m2  # noqa: F401  (enables fp64)
from m2.constants import REINIT_DTAU_CELLS
from m2.schema import Grid
from m2.stencils import GRAD_FLOOR, one_sided_differences


def smoothed_sign(phi0: jax.Array, grid: Grid) -> jax.Array:
    """S(φ₀) = φ₀/√(φ₀² + dx²) (§5.3): smooth through the interface, and |S| ≤ 1."""
    dx = grid.spacing_nm
    return phi0 / jnp.sqrt(phi0**2 + dx**2)


def _godunov_grad_norm_for_reinit(phi: jax.Array, sign: jax.Array, grid: Grid) -> jax.Array:
    """|∇φ| upwinded by the sign of φ₀, as the reinitialisation equation requires."""
    minus, plus = one_sided_differences(phi, grid)
    total = jnp.zeros_like(phi)
    for dm, dp in zip(minus, plus):
        positive = jnp.maximum(jnp.maximum(dm, 0.0) ** 2, jnp.minimum(dp, 0.0) ** 2)
        negative = jnp.maximum(jnp.minimum(dm, 0.0) ** 2, jnp.maximum(dp, 0.0) ** 2)
        total = total + jnp.where(sign >= 0, positive, negative)
    safe = jnp.where(total > GRAD_FLOOR**2, total, 1.0)
    return jnp.where(total > GRAD_FLOOR**2, jnp.sqrt(safe), 0.0)


def reinit_iteration(phi: jax.Array, phi0: jax.Array, grid: Grid, dtau: float) -> jax.Array:
    """One pseudo-time step of the reinitialisation PDE."""
    sign = smoothed_sign(phi0, grid)
    grad_norm = _godunov_grad_norm_for_reinit(phi, sign, grid)
    return phi - dtau * sign * (grad_norm - 1.0)


def reinitialise(phi: jax.Array, grid: Grid, n_iterations: int,
                 dtau_cells: float = REINIT_DTAU_CELLS) -> jax.Array:
    """`n_iterations` fixed iterations, always from the same φ₀ (decision C2: dτ = 0.5·dx).

    The repair reaches about `n_iterations · dtau_cells` cells from the interface per cycle. That
    number is worth knowing: it must keep up with how far the interface moves between cycles, and
    V11 checks a 3-cell band. `repair_reach_cells` states it so the comparison is explicit.
    """
    if n_iterations < 0:
        raise ValueError(f"n_iterations must be >= 0, got {n_iterations}")
    dtau = dtau_cells * grid.spacing_nm
    phi0 = phi

    def body(current, _):
        return reinit_iteration(current, phi0, grid, dtau), None

    if n_iterations == 0:
        return phi
    out, _ = jax.lax.scan(body, phi, jnp.arange(n_iterations))
    return out


def repair_reach_cells(n_iterations: int, dtau_cells: float = REINIT_DTAU_CELLS) -> float:
    """How far from the interface one reinitialisation cycle can repair, in cells."""
    return n_iterations * dtau_cells


def interface_motion_cells(cfl: float, reinit_every: int) -> float:
    """How far the interface moves between reinitialisation cycles, in cells."""
    return cfl * reinit_every
