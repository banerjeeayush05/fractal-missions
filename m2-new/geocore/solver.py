"""Time integration of the level set. PRD §5.2.

Solves `phi_t - R|grad phi| = 0`, where R is an etch rate in nm/s and positive removes material
(decision §1). The sign flip lives here and nowhere else, so an M3 or M5 velocity model never has
to get it right.

**Fixed step count. A hard requirement, and the reason is the gradient.**

The natural scheme is `dt = CFL * dx / max|R|` integrated to a fixed final time. But `max|R|`
depends on the parameters, so `dt` does, so `N = ceil(T/dt)` changes DISCONTINUOUSLY with them.
A gradient through a discontinuous step count is wrong, and it is wrong in the worst way — it
looks like noise rather than like a bug. Instead N is fixed, `dt = T/N`, and CFL is ASSERTED
every step. Fixed N also gives JAX a static graph, which the whole dense-array approach rests on.

**CFL is checked on the host, not with `checkify`** (decision D8). The scan returns the per-step
CFL as an auxiliary output and `assert_cfl` raises afterwards. `checkify` would work, but how it
composes with `grad` is a question nobody needs to answer to ship this, and the aux-output route
has no such question.

**The velocity source is a callable.** `evolve` never learns where R comes from — at stage 11 it
will be the band, the closest-point gather and a velocity model; here it is whatever the caller
passes. That keeps the integrator testable against constant rates with exact answers, and means
the M3 contract can change without touching this file.

**Endpoint only.** The scan carries `phi` and emits one scalar per step. It does NOT emit
trajectory frames: at N = 625 on a 270x100x100 grid that would be 13.5 GB of `ys`, and the
differentiated solve needs none of it. Debug frames are dumped outside the differentiated path.

**Reinitialisation runs inside the scan**, every `reinit_every` steps, when the plan asks for it.
The decision to reinitialise depends only on the STEP INDEX, never on phi or the parameters, so it
adds no data-dependent branch to the differentiated path.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import Array

from geocore.config import CaseConfig
from geocore.constants import ADVECTION_RATE_SIGN, CFL_MAX, UPWIND_SELECTOR_SIGN
from geocore.checkpoint import checkpointed_scan
from geocore.reinit import reinitialize
from geocore.schema import Grid
from geocore.stencils import SPATIAL_SCHEMES, grad_mag_godunov

TEMPORAL_SCHEMES = ("rk2", "rk3")


class CFLViolation(RuntimeError):
    """The step was too large somewhere in the run. Names N, because N is what you change."""


@dataclasses.dataclass(frozen=True)
class StepContext:
    """Where in the run a rate evaluation is happening.

    `stage_index` is here because TVD-RK evaluates the rate more than once per step, and a
    stochastic velocity model (M3) must draw independently per stage — PRD §5.5 forbids assuming
    two calls with the same inputs return the same value.
    """

    time: Array
    step_index: Array
    stage_index: int


class StepDiagnostics(NamedTuple):
    """Per-step auxiliaries carried out of the scan and checked on the HOST (decision D8).

    Both are the same pattern: a quantity that must be asserted every step, returned as a scan
    output rather than raised from inside it. `occupancy` is how many cells were in the velocity
    evaluation band, which PRD §5.0 requires be reported every run -- K is sized from the worst
    step, so a peak that only appears at step 900 must still be seen.
    """

    cfl: Array
    occupancy: Array


class RateField(Protocol):
    """Returns `(etch rate R on the grid, a diagnostic scalar)`.

    R is in nm/s and positive removes material. The second return is band occupancy for a banded
    field, and zero for one that does not use a band.
    """

    def __call__(self, phi: Array, params: Any,
                 ctx: StepContext) -> tuple[Array, Array]: ...


@dataclasses.dataclass(frozen=True)
class SolvePlan:
    """What the integrator needs, with nothing about the case it came from.

    Constructed from a `CaseConfig` in the normal path, or by hand in a test that wants an exact
    analytic answer without inventing a whole case file.
    """

    grid: Grid
    dt_s: float
    n_steps: int
    temporal_scheme: str = "rk2"
    reinit_every: int | None = None     # None: no reinitialisation (analytic tests of advection)
    n_reinit: int = 5
    # PRD §5.2. WENO5 is the DEFAULT from 2026-09-17 (owner, finding J5): it cuts reinitialisation
    # drift from 2.15 % to 0.002 % and V1's radius error 93x, at roughly double the adjoint cost. The
    # flag reaches reinitialisation as well as advection, and under WENO5 reinitialisation steps with
    # SSP-RK3 rather than forward Euler (S14.1).
    spatial_scheme: str = "weno5"
    # V5 ONLY (PRD §8.2). A manufactured-solution source term S(grid, t) added to the right-hand
    # side. Never set in production: `from_case` leaves it None and a test asserts so. It is a
    # plan field rather than a global switch so it cannot leak from one run into another.
    mms_source: Callable[[Grid, Array], Array] | None = None
    # PRD §7.4. Segment length L for two-level checkpointing; None runs unchecked. Never write a
    # number here by hand for a production run: derive it with `checkpoint.segment_length_for` from
    # a MEASURED residual factor k (finding I4).
    checkpoint_segment: int | None = None

    def __post_init__(self) -> None:
        if self.reinit_every is not None and self.reinit_every < 1:
            raise ValueError(f"reinit_every must be at least 1 or None, got {self.reinit_every}")
        if self.checkpoint_segment is not None and self.checkpoint_segment < 1:
            raise ValueError(f"checkpoint_segment must be at least 1 or None, "
                             f"got {self.checkpoint_segment}")
        if self.n_reinit < 1:
            raise ValueError(f"n_reinit must be at least 1, got {self.n_reinit}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}")
        if not self.dt_s > 0.0:
            raise ValueError(f"dt_s must be positive, got {self.dt_s}")
        if self.spatial_scheme not in SPATIAL_SCHEMES:
            raise ValueError(f"spatial_scheme must be one of {SPATIAL_SCHEMES}")
        if self.temporal_scheme not in TEMPORAL_SCHEMES:
            raise ValueError(f"temporal_scheme must be one of {TEMPORAL_SCHEMES}")

    @classmethod
    def from_case(cls, case: CaseConfig) -> "SolvePlan":
        return cls(grid=case.grid, dt_s=case.dt_s, n_steps=case.n_steps,
                   temporal_scheme=case.temporal_scheme,
                   reinit_every=case.reinit.every, n_reinit=case.reinit.n_reinit,
                   spatial_scheme=case.spatial_scheme)

    @property
    def final_time_s(self) -> float:
        return self.dt_s * self.n_steps


@dataclasses.dataclass(frozen=True)
class SolveResult:
    phi: Array
    diagnostics: StepDiagnostics
    plan: SolvePlan

    @property
    def cfl_per_step(self) -> Array:
        return self.diagnostics.cfl

    @property
    def max_cfl(self) -> float:
        return float(jnp.max(self.diagnostics.cfl))

    @property
    def peak_occupancy(self) -> int:
        return int(jnp.max(self.diagnostics.occupancy))


# --------------------------------------------------------------------------------- the PDE


def _rhs(phi: Array, rate: Array, grid: Grid, scheme: str = "godunov") -> Array:
    """d(phi)/dt = +R |grad phi|.

    The Godunov branch is selected by the coefficient in the standard form
    `phi_t + c|grad phi| = 0`, which for this equation is `c = -R`. That relationship is
    `UPWIND_SELECTOR_SIGN`, asserted in `tests/test_constants.py` so the two cannot drift apart.
    """
    upwind_coefficient = UPWIND_SELECTOR_SIGN * rate
    return ADVECTION_RATE_SIGN * rate * grad_mag_godunov(phi, upwind_coefficient, grid,
                                                          scheme=scheme)


def _cfl(rate: Array, plan: SolvePlan) -> Array:
    return jnp.max(jnp.abs(rate)) * plan.dt_s / plan.grid.spacing_nm


def _step(phi: Array, params: Any, rate_fn: RateField, plan: SolvePlan,
          step_index: Array) -> tuple[Array, "StepDiagnostics"]:
    """One TVD-RK step. Returns the new phi and the largest CFL seen across its stages.

    The rate is RECOMPUTED at every stage, never reused from the previous one. Reusing it would
    silently demote the scheme to first order, and PRD §5.5 forbids assuming two calls agree
    anyway — under M3 they genuinely will not.
    """
    dt = plan.dt_s
    grid = plan.grid
    t0 = step_index.astype(jnp.float64) * dt

    def stage(field: Array, offset: float, index: int):
        ctx = StepContext(time=t0 + offset * dt, step_index=step_index, stage_index=index)
        rate, occupancy = rate_fn(field, params, ctx)
        rhs = _rhs(field, rate, grid, plan.spatial_scheme)
        if plan.mms_source is not None:
            rhs = rhs + plan.mms_source(grid, ctx.time)
        return field + dt * rhs, _cfl(rate, plan), occupancy

    phi1, cfl0, occ0 = stage(phi, 0.0, 0)
    if plan.temporal_scheme == "rk2":
        # Heun / TVD-RK2: phi_next = 1/2 phi + 1/2 (phi1 + dt L(phi1))
        phi2, cfl1, occ1 = stage(phi1, 1.0, 1)
        return 0.5 * phi + 0.5 * phi2, StepDiagnostics(
            cfl=jnp.maximum(cfl0, cfl1), occupancy=jnp.maximum(occ0, occ1))

    # Shu-Osher TVD-RK3
    phi2, cfl1, occ1 = stage(phi1, 1.0, 1)
    phi2 = 0.75 * phi + 0.25 * phi2
    phi3, cfl2, occ2 = stage(phi2, 0.5, 2)
    phi3 = (1.0 / 3.0) * phi + (2.0 / 3.0) * phi3
    return phi3, StepDiagnostics(
        cfl=jnp.maximum(jnp.maximum(cfl0, cfl1), cfl2),
        occupancy=jnp.maximum(jnp.maximum(occ0, occ1), occ2))


# ----------------------------------------------------------------------------- the solve


def evolve(phi0: Array, params: Any, rate_fn: RateField,
           plan: SolvePlan) -> tuple[Array, StepDiagnostics]:
    """Run exactly `plan.n_steps` steps. Pure, jittable, differentiable.

    Returns `(phi_final, diagnostics)`, where the diagnostics are the per-step auxiliaries
    decision D8 asks for; pass them to `assert_cfl` and `assert_capacity`. When differentiating,
    use `jax.grad(..., has_aux=True)` so they still reach the host and the run is CHECKED rather
    than assumed.
    """
    def body(carry: Array, step_index: Array):
        phi, diagnostics = _step(carry, params, rate_fn, plan, step_index)
        if plan.reinit_every is not None:
            # Branch on the step INDEX only. `lax.cond` runs one side at run time; both sides are
            # traced, which is fine -- neither can produce a NaN on a finite phi.
            due = (step_index + 1) % plan.reinit_every == 0
            phi = jax.lax.cond(
                due,
                lambda f: reinitialize(f, plan.grid, plan.n_reinit, plan.spatial_scheme),
                lambda f: f,
                phi,
            )
        return phi, diagnostics

    if plan.checkpoint_segment is None:
        return jax.lax.scan(body, phi0, jnp.arange(plan.n_steps, dtype=jnp.int32))
    return checkpointed_scan(body, phi0, plan.n_steps, plan.checkpoint_segment)


def assert_cfl(cfl_per_step: Array, plan: SolvePlan) -> float:
    """Host-side CFL check. Raises naming N, because N is the knob that fixes it."""
    worst = float(jnp.max(cfl_per_step))
    if worst > CFL_MAX:
        step = int(jnp.argmax(cfl_per_step))
        raise CFLViolation(
            f"CFL {worst:.4f} exceeds the hard ceiling {CFL_MAX} at step {step} of "
            f"N={plan.n_steps} (dt={plan.dt_s:g} s, dx={plan.grid.spacing_nm:g} nm). "
            f"Raise N: the derived minimum comes from the target depth and the CFL target, so an "
            f"N below it is rejected at config load rather than here."
        )
    return worst


def solve(phi0: Array, params: Any, rate_fn: RateField, plan: SolvePlan) -> SolveResult:
    """Forward run with the CFL check applied. The ordinary entry point.

    Not itself differentiable — `assert_cfl` branches on a concrete value. Differentiate `evolve`
    instead, and check its auxiliary output on the host.
    """
    phi, diagnostics = evolve(phi0, params, rate_fn, plan)
    assert_cfl(diagnostics.cfl, plan)
    return SolveResult(phi=phi, diagnostics=diagnostics, plan=plan)


# ------------------------------------------------------------------------- rate adapters


def constant_rate(value_key: str = "rate_nm_per_s") -> RateField:
    """A spatially uniform rate read from the params PyTree.

    Deliberately reads from `params` rather than closing over a number. PRD §7.3: a value
    captured in a closure is invisible to `jax.grad`, which returns a silently zero column for
    it — no error, no warning, just a gradient that is wrong in one place.
    """
    def rate_fn(phi: Array, params: Any, ctx: StepContext) -> tuple[Array, Array]:
        return jnp.full(phi.shape, params[value_key]), jnp.int32(0)

    return rate_fn
