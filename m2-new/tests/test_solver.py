"""The time integrator: exact on a case with an exact answer, fixed-N, CFL-checked, and
differentiable.

None of these claim a registry check ID. V1, V1a, V2 and V20 are gates on the FULL forward path
— advection plus the velocity band and the closest-point gather — which arrives at stage 11.
Claiming them here would record a check as passing against half the path it names.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geocore.config import load_case
from geocore.schema import Grid
from geocore.solver import (
    CFLViolation, SolvePlan, StepContext, _step, assert_cfl, constant_rate, evolve, solve,
)

GRID = Grid(shape=(64, 16), spacing_nm=1.0, periodic=(False, True))
PARAMS = {"rate_nm_per_s": jnp.float64(0.5)}


def _plane(interface_z: float = 40.0) -> jnp.ndarray:
    """phi = z - interface_z. Exact signed distance; solid below, plasma above."""
    z = jnp.arange(GRID.shape[0], dtype=jnp.float64)[:, None] * jnp.ones(GRID.shape)
    return z - interface_z


def _interface_z(phi: jnp.ndarray, column: int = 8) -> float:
    """Sub-cell zero crossing down one column."""
    col = np.asarray(phi[:, column])
    k = int(np.argmax(col >= 0.0))
    return k - 1 + (-col[k - 1]) / (col[k] - col[k - 1])


@pytest.mark.parametrize("scheme", ["rk2", "rk3"])
def test_a_plane_translates_at_exactly_the_etch_rate(scheme):
    """The case with an exact answer. |grad phi| = 1 for a plane, so phi_t = R and the interface
    moves at R with no distortion. If this is wrong, the upwind scheme or a boundary is wrong.

    It recedes DOWNWARD -- away from the plasma -- because axis 0 increases toward the plasma
    and a positive rate removes material.
    """
    plan = SolvePlan(GRID, dt_s=0.5, n_steps=40, temporal_scheme=scheme)
    result = solve(_plane(), PARAMS, constant_rate(), plan)
    expected = 40.0 - 0.5 * plan.final_time_s
    assert _interface_z(result.phi) == pytest.approx(expected, abs=1e-6)


def test_the_step_count_does_not_depend_on_the_parameters():
    """PRD §5.2's hard requirement. A step count that moved with the parameters would make the
    gradient wrong in a way that reads as noise rather than as a bug."""
    plan = SolvePlan(GRID, dt_s=0.1, n_steps=17)
    for rate in (0.1, 0.5, 1.0):
        _, diag = evolve(_plane(), {"rate_nm_per_s": jnp.float64(rate)}, constant_rate(), plan)
        assert diag.cfl.shape == (17,)


def test_the_scan_emits_one_scalar_per_step_and_no_frames():
    """Endpoint-only output. At N = 625 on the S03 3D grid, emitting trajectory frames would be
    13.5 GB of `ys` that the differentiated solve never reads."""
    plan = SolvePlan(GRID, dt_s=0.5, n_steps=40)
    phi, diag = evolve(_plane(), PARAMS, constant_rate(), plan)
    assert phi.shape == GRID.shape        # one field, not a trajectory
    assert diag.cfl.shape == (40,)


def test_cfl_is_reported_per_step_and_matches_the_hand_calculation():
    plan = SolvePlan(GRID, dt_s=0.5, n_steps=8)
    _, diag = evolve(_plane(), PARAMS, constant_rate(), plan)
    assert float(jnp.max(diag.cfl)) == pytest.approx(0.5 * 0.5 / 1.0)


def test_a_cfl_violation_aborts_naming_n():
    """Decision D8. The abort names N because N is the knob that fixes it -- an error that only
    reported the CFL number would leave the reader to work that out."""
    plan = SolvePlan(GRID, dt_s=3.0, n_steps=10)
    with pytest.raises(CFLViolation, match=r"N=10"):
        solve(_plane(), PARAMS, constant_rate(), plan)


def test_assert_cfl_accepts_exactly_the_ceiling():
    """0.5 is the ceiling, not the first forbidden value. An off-by-one here would reject every
    case that sits exactly on the limit."""
    plan = SolvePlan(GRID, dt_s=0.1, n_steps=3)
    assert assert_cfl(jnp.array([0.4, 0.5, 0.3]), plan) == 0.5
    with pytest.raises(CFLViolation):
        assert_cfl(jnp.array([0.4, 0.5 + 1e-9, 0.3]), plan)


def test_the_gradient_through_the_solve_is_finite_and_correct_in_sign():
    """A larger rate removes more material, so phi rises everywhere: d(sum phi)/d(rate) > 0."""
    plan = SolvePlan(GRID, dt_s=0.5, n_steps=40)

    def total(params):
        phi, _d = evolve(_plane(), params, constant_rate(), plan)
        return jnp.sum(phi)

    g = jax.grad(total)(PARAMS)["rate_nm_per_s"]
    assert bool(jnp.isfinite(g))
    assert float(g) > 0.0


def test_the_cfl_aux_survives_differentiation():
    """Decision D8 only works if the aux output still reaches the host under `grad`. With
    `has_aux=True` the run is checked rather than assumed."""
    plan = SolvePlan(GRID, dt_s=0.5, n_steps=12)

    def total_with_aux(params):
        phi, diagnostics = evolve(_plane(), params, constant_rate(), plan)
        return jnp.sum(phi), diagnostics

    _, diagnostics = jax.grad(total_with_aux, has_aux=True)(PARAMS)
    assert diagnostics.cfl.shape == (12,)
    assert diagnostics.occupancy.shape == (12,)
    assert assert_cfl(diagnostics.cfl, plan) == pytest.approx(0.25)


def test_a_rate_captured_in_a_closure_gets_a_silently_zero_gradient():
    """PRD §7.3, made executable rather than asserted in prose.

    This is the failure the rule exists to prevent: no exception, no warning, no NaN. Just a
    gradient column that is zero, on a run that otherwise looks completely healthy.
    """
    plan = SolvePlan(GRID, dt_s=0.5, n_steps=10)
    captured = 0.5

    def closure_rate(phi, params, ctx):
        return jnp.full(phi.shape, captured), jnp.int32(0)   # NOT from params -- the bug

    def total(params):
        phi, _d = evolve(_plane(), params, closure_rate, plan)
        return jnp.sum(phi)

    g = jax.grad(total)(PARAMS)["rate_nm_per_s"]
    assert float(g) == 0.0, "a captured parameter is invisible to jax.grad -- silently, not loudly"

    def honest(params):
        phi, _d = evolve(_plane(), params, constant_rate(), plan)
        return jnp.sum(phi)

    assert float(jax.grad(honest)(PARAMS)["rate_nm_per_s"]) != 0.0


def test_the_rate_is_recomputed_at_every_rk_stage():
    """Reusing a stage-0 rate would silently demote TVD-RK2 to first order, and PRD §5.5 forbids
    assuming two calls return the same value -- under M3's stochastic velocity they will not."""
    seen: list[int] = []

    def recording_rate(phi, params, ctx):
        seen.append(ctx.stage_index)
        return jnp.full(phi.shape, params["rate_nm_per_s"]), jnp.int32(0)

    plan2 = SolvePlan(GRID, dt_s=0.5, n_steps=1, temporal_scheme="rk2")
    _step(_plane(), PARAMS, recording_rate, plan2, jnp.int32(0))
    assert seen == [0, 1]

    seen.clear()
    plan3 = SolvePlan(GRID, dt_s=0.5, n_steps=1, temporal_scheme="rk3")
    _step(_plane(), PARAMS, recording_rate, plan3, jnp.int32(0))
    assert seen == [0, 1, 2]


def test_plan_from_case_derives_dt_and_n_from_the_config():
    """The config names a target depth; N and dt are derived. Nothing writes a final time."""
    case = load_case("configs/dev/S00.yaml")
    plan = SolvePlan.from_case(case)
    assert plan.n_steps == case.n_steps == 125
    assert plan.dt_s == pytest.approx(case.final_time_s / 125)
    assert plan.final_time_s == pytest.approx(case.target_depth_nm / case.rate.nm_per_s)


def test_plan_rejects_a_nonsense_configuration():
    with pytest.raises(ValueError, match="n_steps"):
        SolvePlan(GRID, dt_s=0.5, n_steps=0)
    with pytest.raises(ValueError, match="dt_s"):
        SolvePlan(GRID, dt_s=0.0, n_steps=10)
    with pytest.raises(ValueError, match="temporal_scheme"):
        SolvePlan(GRID, dt_s=0.5, n_steps=10, temporal_scheme="euler")
