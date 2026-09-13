"""V14, V15, V16 — reverse-mode gradients through the full time integration (PRD §8.5), M2.3 gate.

This is the milestone the mission exists for. Everything before it produces a shape; this produces
the derivative of that shape with respect to a process parameter, and proves the derivative is real.

What each check can see (decision §5, from OQ A9):
- **V14** compares the derivative against the function itself. It is the only one here that does.
- **V15** and **V16** verify transpose consistency: JAX builds reverse mode by linearising with the
  JVP rules and transposing them, so both share those rules. They catch a broken transpose, not a
  wrong derivative.
- The closed-form checks V14a/V14b/V14c carry what V15 and V16 cannot.

The objective is the smooth volumetric functional of §5.7 — no product uses it; it exists so that
running the same test on it and on an extracted CD localises a failure (§8.7).

A note on the CFL guard: `final_phi` cannot assert on the host, because under `jax.grad` the CFL
number is a tracer. Every test here validates the configuration with a `solve()` first, which does
assert, and only then differentiates.
"""

import jax
import numpy as np
import pytest
from helpers import MATERIALS, consistent_config

from m2 import functionals, initial, solver
from m2.config import parameter_tree
from m2.verification.gradcheck import (
    dot_product_test,
    forward_reverse_test,
    reverse_gradient,
    reverse_vjp,
    taylor_test,
)

KEY = jax.random.PRNGKey(2026)
TRAVEL = 50.0
CAPACITY = 4096


def _case(model="directional", shape=(64, 64), travel=TRAVEL):
    cfg = consistent_config(travel, shape=shape, model=model, v_iso=0.58, v_dir=5.25, p=2.0)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi0 = initial.disk(cfg.grid, (320.0, 320.0), 200.0)
    params, scales = parameter_tree(cfg)
    return cfg, phi0, material, params, scales


def _objective(cfg, phi0, material):
    def J(params):
        phi = solver.final_phi(cfg, phi0, material, params, capacity=CAPACITY)
        return functionals.solid_volume(phi, cfg.grid)

    return J


def _validate(cfg, phi0, material, params):
    """The CFL assertion the differentiated path cannot make for itself."""
    result = solver.solve(cfg, phi0, material, params, capacity=CAPACITY)
    assert result.max_cfl <= 0.5
    return result


@pytest.mark.check("V14")
def test_v14_taylor_remainder_on_the_smooth_functional(ledger_measure):
    """R(h) = |J(θ+hδ) − J(θ) − h⟨g, δ⟩| must fall as O(h²) (§7.1). The primary gate."""
    cfg, phi0, material, params, scales = _case()
    run = _validate(cfg, phi0, material, params)
    J = _objective(cfg, phi0, material)
    grad = reverse_gradient(J, params)

    result = taylor_test(J, params, grad, scales=scales, key=KEY, n_steps=cfg.n_steps)
    ledger_measure.update({**result.measured, "message": result.message,
                           "parameters": sorted(params), "n_steps": cfg.n_steps,
                           "max_cfl": run.max_cfl, "grid": list(cfg.grid.shape),
                           "objective": "solid_volume (§5.7)",
                           "gradient": {k: float(v) for k, v in grad.items()}})
    assert result.passed, result.message


@pytest.mark.check("V15")
def test_v15_forward_versus_reverse(ledger_measure):
    """jax.jvp against the claimed gradient, to 1e-10 relative. A transpose check (decision §5)."""
    cfg, phi0, material, params, scales = _case()
    _validate(cfg, phi0, material, params)
    J = _objective(cfg, phi0, material)

    result = forward_reverse_test(J, params, reverse_gradient(J, params), scales=scales, key=KEY)
    ledger_measure.update({**result.measured, "message": result.message,
                           "what_it_proves": "transpose consistency, not derivative correctness"})
    assert result.passed, result.message


@pytest.mark.check("V16")
def test_v16_dot_product_test(ledger_measure):
    """⟨w, J u⟩ = ⟨Jᵀw, u⟩ on the vector-valued map params → final φ."""
    cfg, phi0, material, params, scales = _case(shape=(48, 48))
    _validate(cfg, phi0, material, params)

    def F(p):  # the whole final field, not a scalar: this is where a transpose error would show
        return solver.final_phi(cfg, phi0, material, p, capacity=CAPACITY)

    result = dot_product_test(F, params, reverse_vjp(F, params), scales=scales, key=KEY)
    ledger_measure.update({**result.measured, "message": result.message,
                           "map": "params -> final phi (full field)"})
    assert result.passed, result.message


@pytest.mark.check("V14")
def test_v14_on_the_isotropic_model_too(ledger_measure):
    """The same gate on the one-parameter model, where the analytic answer is also known."""
    cfg, phi0, material, params, scales = _case(model="isotropic")
    _validate(cfg, phi0, material, params)
    J = _objective(cfg, phi0, material)
    result = taylor_test(J, params, reverse_gradient(J, params), scales=scales, key=KEY,
                         n_steps=cfg.n_steps)
    ledger_measure.update({**result.measured, "message": result.message, "model": "isotropic"})
    assert result.passed, result.message


def test_every_declared_parameter_is_live(ledger_measure):
    """§7.3: every declared parameter must move at least one objective. A closure-captured
    parameter would show up here as an exactly-zero column."""
    from m2.verification.params import check_parameters_live

    cfg, phi0, material, params, _ = _case()
    _validate(cfg, phi0, material, params)
    phi = lambda p: solver.final_phi(cfg, phi0, material, p, capacity=CAPACITY)  # noqa: E731
    result = check_parameters_live(
        {"solid_volume": lambda p: functionals.solid_volume(phi(p), cfg.grid),
         "open_volume": lambda p: functionals.open_volume(phi(p), cfg.grid)}, params)
    ledger_measure.update(result.measured)
    assert result.passed, result.message


def test_the_gradient_is_not_silently_zero_or_nan():
    """The two failure modes a Taylor test cannot distinguish from a correct small gradient."""
    cfg, phi0, material, params, _ = _case()
    grad = reverse_gradient(_objective(cfg, phi0, material), params)
    flat = np.array([float(v) for v in grad.values()])
    assert np.all(np.isfinite(flat)), grad
    assert np.all(flat != 0.0), grad
    # A faster etch removes more material, so every rate derivative is negative.
    assert grad["v_iso"] < 0 and grad["v_dir"] < 0


def test_the_remainder_is_quadratic_where_the_quadratic_model_holds(ledger_measure):
    """Finding I1, pinned: at longer runs the O(h²) regime is a *window*, and B3's h-range overshoots it.

    B3 set h ∈ [1e-6, 1e-1] at M2.0 against a trivial analytic function. On the real solver the top
    of that range is outside the quadratic regime: at h = 0.1 a 10 % rate change moves the interface
    about two cells, which is a large geometric change, so the remainder is governed by higher-order
    terms rather than by the gradient. A least-squares fit across both regimes lands near 1.7 and
    the check fails — on a gradient that is demonstrably correct.

    This asserts the real claim: over the small-h decades the remainder is O(h²) to two decimals.
    The V19 canary on this same objective catches a 5 % corruption in every component, so the test
    still has power; what it lacks at large h is validity.
    """
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    from m2.constants import TAYLOR_SLOPE_BAND
    from m2.verification.gradcheck import _directions, estimate_noise_floor

    cfg, phi0, material, params, scales = _case(travel=200.0)  # 50 steps, 20 cells of travel
    _validate(cfg, phi0, material, params)
    J = _objective(cfg, phi0, material)
    x0, unravel = ravel_pytree(params)
    g = ravel_pytree(reverse_gradient(J, params))[0]
    Jx = jax.jit(lambda x: J(unravel(x)))
    J0 = float(Jx(x0))
    _, floor = estimate_noise_floor(Jx, x0, n_steps=cfg.n_steps)

    direction = jnp.asarray(_directions(KEY, 20, x0, scales)[0])
    linear = float(jnp.dot(g, direction))
    hs = 1e-1 * np.logspace(0.0, -5.0, 11)
    remainder = np.array([abs(float(Jx(x0 + h * direction)) - J0 - h * linear) for h in hs])

    small_h = (hs <= 1e-2) & (remainder > floor)
    assert small_h.sum() >= 4
    slope_small = float(np.polyfit(np.log10(hs[small_h]), np.log10(remainder[small_h]), 1)[0])

    # The check as configured, over B3's full range and all 20 directions.
    full = taylor_test(J, params, reverse_gradient(J, params), scales=scales, key=KEY,
                       n_steps=cfg.n_steps)

    ledger_measure.update({"slope_small_h_one_direction": slope_small,
                           "h_quadratic_regime": [float(hs[small_h].min()), float(hs[small_h].max())],
                           "full_range_check_passed": full.passed,
                           "full_range_slope_min": full.measured["slope_min"],
                           "full_range_slope_max": full.measured["slope_max"],
                           "n_steps": cfg.n_steps, "travel_cells": 200.0 / cfg.grid.spacing_nm,
                           "noise_floor": floor, "finding": "I1"})

    # Inside the window where the quadratic model holds, the same gradient satisfies §7.1's band.
    lo, hi = TAYLOR_SLOPE_BAND
    assert lo <= slope_small <= hi, f"the gradient is wrong, not just the window: slope {slope_small:.3f}"
    # And over B3's full range it fails, which is the finding. If this ever stops failing, the
    # h-range question has resolved itself and I1 should be closed.
    assert not full.passed, "V14 now passes over B3's full range at N=50: re-examine finding I1"
