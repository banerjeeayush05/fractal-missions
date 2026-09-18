"""Reinitialisation: restore phi to a signed distance function. PRD §5.3.

Advection does not preserve `|grad phi| = 1`. After a few steps phi is still zero on the right
surface but is no longer a DISTANCE away from it, and three things built earlier depend on it being
one: the band is measured in cells of distance, `closest_points` walks `phi` along the normal and
lands on the surface only if `phi` is the true distance, and the free velocity extension the
capstone relied on holds only on a distance function.

Iterates, in pseudo-time tau,

    d(phi)/d(tau) + S(phi0) (|grad phi| - 1) = 0

for a FIXED number of iterations. Where `|grad phi| > 1` the field is too steep and relaxes; where
`< 1` it steepens. `S` is the sign of the ORIGINAL field, frozen for the whole cycle, so information
flows outward from the interface in both directions.

**Three rules, each one a way this goes wrong.**

1. **Smoothed sign, `phi0 / sqrt(phi0^2 + dx^2)`.** The exact sign function jumps at the interface,
   and its derivative there is not defined -- a kink in the differentiated path, which §11 forbids.
2. **Fixed iteration count.** A convergence-based stopping rule makes the NUMBER of iterations depend
   on phi, and so on the parameters. The graph shape then changes discontinuously with theta, and the
   gradient through that is wrong in a way that looks like noise (§5.2's argument, again).
3. **No stop_gradient.** Reinitialisation is a real part of the forward map. Its derivative is real.
   Wrapping it would make gradients cheaper and wrong.

**The failure it must not have** is moving the interface. V12 checks exactly that. Reinitialisation
that shifts the zero level set produces an etch-rate bias that looks like physics, calibrates into
the closure parameters, and then fails to transfer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array, lax

from geocore.constants import REINIT_DTAU_OVER_DX
from geocore.schema import Grid
from geocore.stencils import grad_mag_godunov


def smoothed_sign(phi0: Array, grid: Grid) -> Array:
    """`phi0 / sqrt(phi0^2 + dx^2)`. Smooth through zero, and saturates to +/-1 about a cell away."""
    return phi0 / jnp.sqrt(phi0**2 + grid.spacing_nm**2)


def _pseudo_rhs(phi: Array, sign: Array, grid: Grid, scheme: str) -> Array:
    """`-S (|grad phi| - 1)`: the right-hand side of the reinitialisation PDE in pseudo-time."""
    return -sign * (grad_mag_godunov(phi, sign, grid, scheme=scheme) - 1.0)


def reinit_iteration(phi: Array, sign: Array, grid: Grid, scheme: str = "godunov") -> Array:
    """One pseudo-time step. `dtau = 0.5 dx` (decision C2), so with `|sign| <= 1` the pseudo-CFL is
    0.5 by construction.

    The Godunov branch is selected by `sign`, which is the coefficient of `|grad phi|` in the
    standard form -- the same `grad_mag_godunov` advection uses, with a different speed.

    **The time integrator depends on the spatial scheme** (finding S14.1).

    * `godunov`: forward Euler. The first-order Godunov operator is monotone, and forward Euler is
      stable for it at this pseudo-CFL. Unchanged from stage 12, so every recorded Godunov number
      still holds.
    * `weno5`: three-stage SSP-RK3 (Shu-Osher). WENO5's linearised operator has eigenvalues close
      to the imaginary axis, which lie OUTSIDE forward Euler's stability region. Nothing in the
      band damps the far field, so those modes grow slowly and then explosively: with forward Euler
      a 150-cell disk at dx = 1 nm opened 1268 cells of phantom holes, and Zalesak's disk grew a
      phantom surface at the domain corner that tripped the CFL assertion. SSP-RK3's stability region
      contains that part of the imaginary axis; with it both cases are clean.

    Same iteration count and same `dtau` for both, so nothing is tuned. RK3 costs three operator
    evaluations per iteration.
    """
    dtau = REINIT_DTAU_OVER_DX * grid.spacing_nm
    if scheme == "godunov":
        return phi + dtau * _pseudo_rhs(phi, sign, grid, scheme)
    stage1 = phi + dtau * _pseudo_rhs(phi, sign, grid, scheme)
    stage2 = 0.75 * phi + 0.25 * (stage1 + dtau * _pseudo_rhs(stage1, sign, grid, scheme))
    return phi / 3.0 + (2.0 / 3.0) * (stage2 + dtau * _pseudo_rhs(stage2, sign, grid, scheme))


def reinitialize(phi: Array, grid: Grid, n_iterations: int, scheme: str = "godunov") -> Array:
    """One reinitialisation CYCLE: exactly `n_iterations` pseudo-time steps.

    `n_iterations` is a Python int and must stay one. It fixes the graph at trace time; a traced
    value here would be a data-dependent loop count, which is the thing rule 2 forbids.

    **The iterations run under `lax.scan`, and each one is rematerialised.** Both are implementation
    choices, not changes to the arithmetic: `length` is still the static Python int, `sign` is closed
    over and frozen exactly as before, and an iteration computes what it always did. The unrolled and
    scanned forms agree to 1.7e-13, and stored and rematerialised gradients to 2.9e-13 -- fp
    reassociation, well inside V17's 1e-12 -- with the objective's derivative unchanged to ten digits.

    Both exist because this sits inside the per-step scan body, where its cost is multiplied by every
    step of the run, and under `weno5` an iteration is three SSP-RK3 stages of WENO reconstructions.

    * `lax.scan` instead of a Python loop (S20.1) traces the body ONCE rather than `n_iterations`
      times. Compiling the gradient of a 200-step solve took 133 s unrolled against 23 s scanned,
      and ran 21.0 s against 16.3 s.
    * `jax.checkpoint` per iteration (S20.6) stops reverse mode holding every iteration's residuals
      at once. Measured in 3D, a five-iteration cycle's residual factor k is **2291.7 stored against
      11.0 rematerialised**. That is the number M2.4's 40 GB memory gate turns on, because those
      residuals were four fifths of the modelled peak.

    Rematerialisation usually trades time for memory. Here it buys both, which is worth stating
    plainly: on a 200-step 2D solve the gradient ran 3.79 s stored against 3.21 s rematerialised and
    the adjoint ratio went 9.23x -> 7.58x, because storing that many fields costs more in memory
    traffic than recomputing them.
    """
    if not isinstance(n_iterations, int) or isinstance(n_iterations, bool) or n_iterations < 1:
        raise ValueError(f"n_iterations must be a positive Python int, got {n_iterations!r}")
    sign = smoothed_sign(phi, grid)
    iteration = jax.checkpoint(lambda carry: reinit_iteration(carry, sign, grid, scheme))
    return lax.scan(lambda carry, _: (iteration(carry), None), phi, None, length=n_iterations)[0]
