"""WENO5 — the fifth-order spatial scheme (PRD §5.2), landed after M2.3 by owner decision H1.

The scheme exists because of V3: at dx = 10 nm the first-order Godunov scheme costs ~1.4° of
sidewall angle against V3's 0.5° requirement. It was deliberately *not* built earlier, and the
reason is worth keeping: new numerics belong where V14/V15/V16 can verify them, and that harness did
not exist at M2.2. Building it here cost nothing extra, because the adjoint is automatic — changing
the forward scheme means re-running the gradient checks, not rewriting an adjoint.

What this file pins, in the order the claims depend on each other:

1. the reconstruction really is fifth order, on a field with a known derivative;
2. the fixed ε (a deliberate deviation from Jiang & Peng, for differentiability) does not change
   the answer, so the deviation is not load-bearing;
3. the boundary fallback is exactly the Godunov scheme, not an approximation of it;
4. the gradient still passes V14/V15/V16 **and the V19 canary still catches 5 % corruptions** —
   the obligation that travels with any change to the differentiated path.

`test_order.py` carries the scheme-level convergence claim for both schemes.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import MATERIALS, consistent_config, radius_at_angle, synthetic_config

from m2 import functionals, initial, solver
from m2.config import parameter_tree
from m2.schema import Grid
from m2.stencils import WENO_BOUNDARY_CELLS, one_sided_differences
from m2.verification.canary import corrupt_component
from m2.verification.gradcheck import (
    dot_product_test,
    forward_reverse_test,
    reverse_gradient,
    reverse_vjp,
    taylor_test,
)

KEY = jax.random.PRNGKey(2026)


def _smooth_case(n, wavelength=100.0):
    """sin(kx) along the periodic lateral axis, whose derivative is known exactly."""
    dx = wavelength / n
    grid = Grid((n, n), dx, (False, True))
    k = 2 * np.pi / wavelength
    x = jnp.arange(n) * dx
    phi = jnp.broadcast_to(jnp.sin(k * x)[None, :], (n, n))
    exact = jnp.broadcast_to((k * jnp.cos(k * x))[None, :], (n, n))
    return grid, phi, exact, dx


def _reconstruction_order(scheme, sizes=(32, 64, 128, 256)):
    errors, spacings = [], []
    for n in sizes:
        grid, phi, exact, dx = _smooth_case(n)
        minus, plus = one_sided_differences(phi, grid, scheme)
        # Axis 1 is the periodic one, so no boundary fallback is in play here.
        errors.append(max(float(jnp.max(jnp.abs(minus[1] - exact))),
                          float(jnp.max(jnp.abs(plus[1] - exact)))))
        spacings.append(dx)
    return float(np.polyfit(np.log(spacings), np.log(errors), 1)[0]), errors


def test_the_weno5_reconstruction_is_fifth_order(ledger_measure):
    """The claim the whole scheme rests on, measured before anything else uses it.

    A one-sided WENO5 derivative is fifth order where φ is smooth. If this is not ~5, nothing
    further in this file means anything, so it runs on a field with an analytic derivative rather
    than through the solver.
    """
    order_weno, errors_weno = _reconstruction_order("weno5")
    order_godunov, errors_godunov = _reconstruction_order("godunov")

    ledger_measure.update({"weno5_order": order_weno, "godunov_order": order_godunov,
                           "weno5_errors": errors_weno, "godunov_errors": errors_godunov,
                           "error_ratio_at_coarsest": errors_godunov[0] / errors_weno[0]})

    assert order_weno >= 4.5, f"WENO5 reconstruction is only order {order_weno:.3f}"
    assert 0.9 <= order_godunov <= 1.1, f"the first-order baseline moved: {order_godunov:.3f}"
    assert errors_weno[0] < errors_godunov[0] / 100.0


def test_the_fixed_epsilon_is_not_load_bearing(monkeypatch):
    """`WENO_EPS` is fixed where Jiang & Peng scale it by max(v₁²…v₅²) — a `max` over the stencil
    is a kink in the differentiated path, which §11 forbids.

    A deviation from a published scheme has to be shown to be inert, not asserted to be. Moving ε by
    ±100× must not move the observed order: if it did, the scheme's accuracy would depend on a
    constant chosen for a reason unrelated to accuracy, and the deviation would need revisiting.
    """
    from m2 import stencils

    orders = {}
    for eps in (1e-8, 1e-6, 1e-4):
        monkeypatch.setattr(stencils, "WENO_EPS", eps)
        orders[eps] = _reconstruction_order("weno5", sizes=(32, 64, 128))[0]

    assert min(orders.values()) >= 4.5, f"the order depends on epsilon: {orders}"


def test_the_boundary_fallback_is_exactly_godunov():
    """Within `WENO_BOUNDARY_CELLS` of a hard edge the wide stencil would read clamped values and
    treat them as smooth data. The fallback must be the other scheme *exactly* — an approximation
    of it would be a third scheme that nothing verifies.

    The mask is a function of the grid index alone, so it is also checked to be static: the same
    cells fall back regardless of φ.
    """
    grid = Grid((32, 32), 1.0, (False, True))
    rng = np.random.default_rng(0)
    for _ in range(2):
        phi = jnp.asarray(rng.normal(size=(32, 32)))
        g_minus, g_plus = one_sided_differences(phi, grid, "godunov")
        w_minus, w_plus = one_sided_differences(phi, grid, "weno5")
        edge = np.zeros(32, dtype=bool)
        edge[:WENO_BOUNDARY_CELLS] = True
        edge[-WENO_BOUNDARY_CELLS:] = True
        # Axis 0 is the non-periodic one, so the fallback applies to its edge rows.
        assert np.allclose(np.asarray(w_minus[0])[edge], np.asarray(g_minus[0])[edge])
        assert np.allclose(np.asarray(w_plus[0])[edge], np.asarray(g_plus[0])[edge])
        # ... and nowhere else: the interior must actually use the wide stencil.
        assert not np.allclose(np.asarray(w_minus[0])[~edge], np.asarray(g_minus[0])[~edge])
        # The periodic axis has no fallback anywhere.
        assert not np.allclose(np.asarray(w_minus[1]), np.asarray(g_minus[1]))


def test_an_unknown_scheme_is_refused():
    """A typo in a config must not silently fall back to first order."""
    grid = Grid((8, 8), 1.0, (False, True))
    with pytest.raises(ValueError, match="unknown spatial scheme"):
        one_sided_differences(jnp.zeros((8, 8)), grid, "weno")


# --- the scheme through the solver --------------------------------------------------------------


def _gradient_case(scheme, model="directional"):
    kw = dict(v_iso=0.58, v_dir=5.25, p=2.0) if model == "directional" else dict(v_iso=0.58)
    cfg = dataclasses.replace(consistent_config(50.0, shape=(64, 64), model=model, **kw),
                              spatial_scheme=scheme)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi0 = initial.disk(cfg.grid, (320.0, 320.0), 200.0)
    params, scales = parameter_tree(cfg)
    return cfg, material, phi0, params, scales


@pytest.mark.nightly
def test_v14_v15_v16_hold_under_weno5(ledger_measure):
    """The obligation that travels with any change to the differentiated path (§11).

    WENO5's weights are rational functions of the data, so it is a genuinely more nonlinear map than
    the two-point difference. That is exactly the kind of change that can break a gradient quietly,
    which is why the scheme was sequenced to land after this harness existed rather than before.
    """
    cfg, material, phi0, params, scales = _gradient_case("weno5")
    assert solver.solve(cfg, phi0, material, params, capacity=4096).max_cfl <= 0.5

    J = lambda p: functionals.solid_volume(  # noqa: E731
        solver.final_phi(cfg, phi0, material, p, capacity=4096), cfg.grid)
    F = lambda p: solver.final_phi(cfg, phi0, material, p, capacity=4096)  # noqa: E731
    grad = reverse_gradient(J, params)

    v14 = taylor_test(J, params, grad, scales=scales, key=KEY, n_steps=cfg.n_steps)
    v15 = forward_reverse_test(J, params, grad, scales=scales, key=KEY)
    v16 = dot_product_test(F, params, reverse_vjp(F, params), scales=scales, key=KEY)

    ledger_measure.update({
        "scheme": "weno5", "v14_passed": v14.passed, "v15_passed": v15.passed,
        "v16_passed": v16.passed, "v14_slope_min": v14.measured["slope_min"],
        "v14_slope_max": v14.measured["slope_max"],
        "v14_cancellation_capped": v14.measured["cancellation_capped"],
        "v15_max_rel_error": v15.measured["max_rel_error"],
        "v16_max_rel_error": v16.measured["max_rel_error"]})

    assert v14.passed, v14.message
    assert v15.passed, v15.message
    assert v16.passed, v16.message


@pytest.mark.nightly
def test_the_canary_still_catches_corruption_under_weno5(ledger_measure):
    """A 5 % error in any one gradient component must still fail V14 under the new scheme.

    This is the check that makes the previous one mean something: V14 passing proves nothing if the
    scheme has made the test insensitive. Both velocity models are run, because the isotropic case
    has a single parameter and so cannot hide an error in a direction orthogonal to it.
    """
    caught = {}
    for model in ("isotropic", "directional"):
        cfg, material, phi0, params, scales = _gradient_case("weno5", model=model)
        J = lambda p: functionals.solid_volume(  # noqa: E731
            solver.final_phi(cfg, phi0, material, p, capacity=4096), cfg.grid)
        grad = reverse_gradient(J, params)
        assert taylor_test(J, params, grad, scales=scales, key=KEY,
                           n_steps=cfg.n_steps).passed, "the control must pass first"
        for index, name in enumerate(sorted(params)):
            result = taylor_test(J, params, corrupt_component(grad, index, 1.05), scales=scales,
                                 key=KEY, n_steps=cfg.n_steps)
            caught[f"{model}.{name}"] = {"caught": not result.passed,
                                         "slope_min": result.measured["slope_min"]}

    ledger_measure.update({"scheme": "weno5", "corruption": 1.05, "per_component": caught})
    for key, outcome in caught.items():
        assert outcome["caught"], f"a 5 % corruption of {key} passed V14 under WENO5 (§7.2, §11)"


@pytest.mark.nightly
def test_weno5_reduces_grid_anisotropy_and_radius_error(ledger_measure):
    """V1 and V10 under both schemes — the product-shaped measurement, not a benchmark.

    V10's grid anisotropy is the artifact that would otherwise reappear as a spurious
    sidewall-angle dependence once M3 supplies angle-dependent velocity, so its size is the number
    that matters for the mission rather than for the literature.
    """
    measured = {}
    for scheme in ("godunov", "weno5"):
        cfg = dataclasses.replace(synthetic_config((192, 192), spacing_nm=5.0, n_steps=200),
                                  spatial_scheme=scheme)
        material = initial.uniform_material(cfg.grid, MATERIALS)
        result = solver.solve(cfg, initial.disk(cfg.grid, (480.0, 480.0), 300.0), material,
                              {"v_iso": 5.83}, dt=150.0 / 5.83 / 200, capacity=40000)
        angles = np.linspace(0.0, 2 * np.pi, 33)[:-1]
        radii = np.array([radius_at_angle(result.phi, 5.0, 96, a) for a in angles])
        measured[scheme] = {
            "anisotropy": float((radii.max() - radii.min()) / radii.mean()),
            "radius_nm": float(radii.mean()),
            "radius_rel_error": float(abs(radii.mean() - 150.0) / 150.0)}

    ledger_measure.update({**measured,
                           "anisotropy_improvement": measured["godunov"]["anisotropy"]
                           / measured["weno5"]["anisotropy"],
                           "v10_tolerance": 0.02})

    assert measured["weno5"]["anisotropy"] < measured["godunov"]["anisotropy"] / 10.0
    assert measured["weno5"]["radius_rel_error"] < measured["godunov"]["radius_rel_error"] / 10.0
    assert measured["weno5"]["anisotropy"] < 0.02, "V10's tolerance must still be met"
