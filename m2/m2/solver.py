"""The 2D/3D forward solve: fixed step count, TVD-RK2, Godunov upwind (PRD §5.2).

Structure of one step, twice per step because Heun has two stages:

    assemble the padded request on the evaluation band  ->  call the velocity model once
    ->  closest-point gather across the extension band  ->  Godunov |∇φ|  ->  φ_t = +R|∇φ|

Hard requirements this file exists to honour:
- **Fixed step count.** N comes from the config (derived from depth, dx and the CFL target) and
  never from the data, so the computation graph is static and the gradient is not taken through a
  step count that jumps with the parameters (§5.2).
- **One velocity call per RK stage** (§5.5), with `stage_index` passed so a stochastic model can
  seed independently per stage.
- **The CFL check is a scan output, asserted on the host** (decision D8), not `checkify`, which
  avoids any question about how it composes with `grad`.
- **Endpoint-only output** (decision §9): the differentiated solve returns the final φ. Diagnostics
  come back alongside it but are not part of the differentiated value.
- **Reinitialisation on a fixed schedule** (§5.3, M2.2): every `reinit_every` steps, `n_reinit`
  iterations. The schedule depends on the step index alone, never on the data, so the graph stays
  static; `lax.cond` only chooses *when*, not *how many*.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

import m2  # noqa: F401  (enables fp64)
from m2.band import assemble_request, estimate_capacity, gather_to_band
from m2.config import M2Config
from m2.constants import CFL_MAX, DPHI_DT_RATE_SIGN
from m2.reinit import reinitialise
from m2.schema import Grid, check_velocity_output
from m2.velocity import model_for


class CFLViolation(RuntimeError):
    """The CFL bound was exceeded: the run is not trustworthy and must not be reported."""


class CapacityOverflow(RuntimeError):
    """More interface-adjacent cells than the padded request can hold."""


@dataclass(frozen=True)
class SolveResult:
    phi: jax.Array  # final φ — the only differentiated output (decision §9)
    max_cfl: float
    peak_occupancy: int
    capacity: int
    n_steps: int


def rate_field(
    phi: jax.Array,
    material: jax.Array,
    params: Any,
    *,
    grid: Grid,
    bands,
    model: Callable,
    capacity: int,
    time: float,
    step_index,
    stage_index,
    run_seed,
) -> tuple[jax.Array, jax.Array]:
    """One velocity-model call for one RK stage, gathered onto the extension band.

    Returns the dense etch-rate field and the number of active band points.
    """
    selection = assemble_request(phi, material, grid, bands, capacity=capacity, time=time,
                                step_index=step_index, stage_index=stage_index, run_seed=run_seed)
    speed = model(selection.request, params)
    check_velocity_output(selection.request, speed)
    dense = gather_to_band(speed, selection.request, phi, grid, bands)
    return dense, selection.n_active


def _advect(phi: jax.Array, rate: jax.Array, grid: Grid) -> jax.Array:
    """dφ/dt = +R|∇φ| (decision §1: a positive rate removes material)."""
    from m2.stencils import godunov_grad_norm

    # φ_t + F|∇φ| = 0 with F = −R fixes the upwind direction.
    grad_norm = godunov_grad_norm(phi, grid, -rate)
    return DPHI_DT_RATE_SIGN * rate * grad_norm


def step(
    phi: jax.Array,
    material: jax.Array,
    params: Any,
    *,
    grid: Grid,
    bands,
    model: Callable,
    capacity: int,
    dt: float,
    step_index,
    run_seed,
    time: float,
    n_reinit: int = 0,
    reinit_every: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One TVD-RK2 (Heun) step, with reinitialisation on the fixed schedule.

    Returns φ, the step's CFL number and its peak occupancy.
    """
    kwargs = dict(grid=grid, bands=bands, model=model, capacity=capacity, run_seed=run_seed,
                  step_index=step_index)

    rate0, active0 = rate_field(phi, material, params, time=time, stage_index=0, **kwargs)
    phi_euler = phi + dt * _advect(phi, rate0, grid)

    rate1, active1 = rate_field(phi_euler, material, params, time=time + dt, stage_index=1, **kwargs)
    phi_next = 0.5 * (phi + phi_euler + dt * _advect(phi_euler, rate1, grid))

    if n_reinit > 0:
        # The schedule is a function of the step index only — data-independent, so `lax.cond`
        # selects when to repair, never how many iterations to run (§5.2, §5.3).
        due = (step_index + 1) % reinit_every == 0
        phi_next = jax.lax.cond(due, lambda p: reinitialise(p, grid, n_reinit), lambda p: p, phi_next)

    cfl = jnp.maximum(jnp.max(jnp.abs(rate0)), jnp.max(jnp.abs(rate1))) * dt / grid.spacing_nm
    return phi_next, cfl, jnp.maximum(active0, active1)


def solve(
    cfg: M2Config,
    phi0: jax.Array,
    material: jax.Array,
    params: Any,
    *,
    model: Callable | None = None,
    capacity: int | None = None,
    n_steps: int | None = None,
    dt: float | None = None,
) -> SolveResult:
    """Run the forward solve and assert the CFL bound on the host afterwards.

    `n_steps` and `dt` override the config only so that V20 can drive the error path; production
    runs take both from the config, where N is derived and cannot be set below the CFL minimum.
    """
    model = model_for(cfg.velocity.model) if model is None else model
    capacity = estimate_capacity(cfg.grid, cfg.bands) if capacity is None else capacity
    n = cfg.n_steps if n_steps is None else n_steps
    h = cfg.dt_s if dt is None else dt

    def body(carry, i):
        phi, t = carry
        phi_next, cfl, occupancy = step(phi, material, params, grid=cfg.grid, bands=cfg.bands,
                                        model=model, capacity=capacity, dt=h, step_index=i,
                                        run_seed=cfg.seed, time=t, n_reinit=cfg.n_reinit,
                                        reinit_every=cfg.reinit_every)
        return (phi_next, t + h), (cfl, occupancy)

    (phi_final, _), (cfls, occupancies) = jax.lax.scan(body, (phi0, 0.0), jnp.arange(n))

    max_cfl = float(jnp.max(cfls))
    peak = int(jnp.max(occupancies))
    if max_cfl > CFL_MAX:
        raise CFLViolation(
            f"CFL {max_cfl:.3f} exceeds {CFL_MAX} with N = {n} steps of dt = {h:.6g} s. "
            f"N is too small for this rate and grid: raise N (config derives it as "
            f"ceil(depth / (cfl_target * dx))), or lower the rate. The run is discarded."
        )
    if peak >= capacity:
        raise CapacityOverflow(
            f"the padded velocity request filled: peak occupancy {peak} of capacity {capacity}. "
            f"Interface-adjacent cells grew past the estimate, so some were dropped silently. "
            f"Raise bands.capacity; size it from the deepest expected geometry (decision C10)."
        )
    return SolveResult(phi=phi_final, max_cfl=max_cfl, peak_occupancy=peak, capacity=capacity,
                       n_steps=n)


def final_phi(cfg: M2Config, phi0: jax.Array, material: jax.Array, params: Any,
              *, model: Callable | None = None, capacity: int | None = None) -> jax.Array:
    """The differentiable entry point: parameters in, final φ out, nothing else in the path."""
    model = model_for(cfg.velocity.model) if model is None else model
    capacity = estimate_capacity(cfg.grid, cfg.bands) if capacity is None else capacity

    def body(carry, i):
        phi, t = carry
        phi_next, _, _ = step(phi, material, params, grid=cfg.grid, bands=cfg.bands, model=model,
                              capacity=capacity, dt=cfg.dt_s, step_index=i, run_seed=cfg.seed,
                              time=t, n_reinit=cfg.n_reinit, reinit_every=cfg.reinit_every)
        return (phi_next, t + cfg.dt_s), None

    (phi_out, _), _ = jax.lax.scan(body, (phi0, 0.0), jnp.arange(cfg.n_steps))
    return phi_out
