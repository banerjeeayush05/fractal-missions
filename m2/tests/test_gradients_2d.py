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


@pytest.mark.nightly  # a diagnostic of findings I1/I6/I7, not a gate check: §8.0's fast tier is a budget
def test_the_anchored_window_finds_the_quadratic_regime_at_a_long_run(ledger_measure):
    """Findings I1, I6 and I7, closed: the long run that B3's fixed h-range could not score.

    B3 set h ∈ [1e-6, 1e-1] at M2.0 against a trivial analytic function, and fitted everything in it
    that cleared the noise floor. On the real solver at 50 steps that fit lands near 1.7 and the
    check fails on a gradient that is demonstrably correct: at h = 0.1 a 10 % rate change moves the
    interface about two cells, which is a large geometric change, so the remainder there is governed
    by higher-order terms and by the kinks in a piecewise-smooth map — not by the gradient.

    Decision I7 anchors the fit at the measured floor instead. This is the case that motivated it,
    so this is where it has to be shown to work, and shown not to have been blinded in the process:

    1. the anchored scoring passes, on the same gradient and the same objective;
    2. the top two decades of B3's range, fitted on their own, score ~1.29 — not merely a degraded
       slope but outside the band on the *low* side, the signature the scoring reads as "first-order
       term present: gradient error". So the finding was real and it is the scoring that changed,
       not the solver;
    3. a 5 % corruption in a single component is still caught, at this hardest case.

    Point 3 is the obligation that travels with any narrowing of a test window (§11).

    The measured remainder curve for one direction, recorded in the ledger, shows all three regimes
    at once and is worth reading directly:

        h = 1e-1 … 1e-3   R = 1.6 … 2.6e-3     slope 1.29  kinks and higher-order geometry
        h = 1e-3 … 3e-7   R = 2.6e-3 … 2.7e-10 slope 2.00  the quadratic regime  <- scored here
        h < 1e-7          R ≈ 1e-11            slope 0     the fp64 floor
    """
    from m2.constants import TAYLOR_SLOPE_BAND, V19_CORRUPTION_FACTOR
    from m2.verification.canary import corrupt_component

    cfg, phi0, material, params, scales = _case(travel=200.0)  # 50 steps, 20 cells of travel
    _validate(cfg, phi0, material, params)
    J = _objective(cfg, phi0, material)
    grad = reverse_gradient(J, params)

    anchored = taylor_test(J, params, grad, scales=scales, key=KEY, n_steps=cfg.n_steps)

    # The same remainder data, fitted where B3 put the top of its range.
    hs = np.asarray(anchored.measured["h_swept"])
    R = np.asarray(anchored.measured["remainder_curve_delta0"])
    floor = anchored.measured["noise_floor_measured_max"]
    fit = lambda m: float(np.polyfit(np.log10(hs[m]), np.log10(R[m]), 1)[0])  # noqa: E731
    top_slope = fit(hs >= 1e-3)  # the two decades below h_max, on their own
    legacy_slope = fit((hs >= 1e-6) & (R > floor))  # B3's whole range, one least-squares fit

    # The canary, at this case: every component, one at a time.
    caught = {}
    for index, name in enumerate(sorted(params)):  # ravel_pytree flattens a dict in key order
        bad = corrupt_component(grad, index, V19_CORRUPTION_FACTOR)
        result = taylor_test(J, params, bad, scales=scales, key=KEY, n_steps=cfg.n_steps)
        caught[name] = {"passed": result.passed, "slope_min": result.measured["slope_min"]}

    lo, hi = TAYLOR_SLOPE_BAND
    ledger_measure.update({
        "anchored_passed": anchored.passed,
        "anchored_slope_min": anchored.measured["slope_min"],
        "anchored_slope_max": anchored.measured["slope_max"],
        "anchored_window": [anchored.measured["h_window_min"], anchored.measured["h_window_max"]],
        "top_two_decades_slope_delta0": top_slope,
        "legacy_b3_whole_range_slope_delta0": legacy_slope,
        "noise_floor_measured": floor,
        "noise_floor_b19_model": anchored.measured["noise_floor_b19_model"],
        "model_over_measured": anchored.measured["noise_floor_model_over_measured"],
        "corruption_caught": caught,
        "n_steps": cfg.n_steps, "travel_cells": 200.0 / cfg.grid.spacing_nm,
        "findings": "I1, I6, I7 (closed by decision I7)"})

    assert anchored.passed, anchored.message
    assert top_slope < lo, (
        f"the top two decades of B3's range now score {top_slope:.3f}, inside the band — finding I1 "
        f"no longer reproduces and the anchored window of decision I7 needs re-justifying")
    for name, outcome in caught.items():
        assert not outcome["passed"], (
            f"a 5 % corruption of d/d{name} passed V14 in the anchored window: the window has been "
            f"narrowed to the point of blindness (§11)")
