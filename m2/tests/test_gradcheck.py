"""The gradient harness: correct gradients pass, the protocol cannot be loosened, and the
two-zone scoring of decision B15/B21 classifies rather than guesses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from m2.verification import analytic as A
from m2.verification.canary import corrupt_component
from m2.verification.gradcheck import (
    CLEAN,
    DEGENERATE,
    _directions,
    dot_product_test,
    estimate_noise_floor,
    forward_reverse_test,
    perturbation_scale,
    reverse_gradient,
    reverse_vjp,
    taylor_test,
)

KEY = jax.random.PRNGKey(7)
ONES = {"x": jnp.asarray(1.0)}


def test_correct_gradient_passes_all_three():
    th = A.theta0()
    g = reverse_gradient(A.objective, th)
    t = taylor_test(A.objective, th, g, scales=A.scales(), key=KEY)
    assert t.passed, t.message
    assert 1.95 < t.measured["slope_min"] <= t.measured["slope_max"] < 2.05
    assert t.measured["classifications"] == [CLEAN] * 20
    assert t.measured["degenerate_fraction"] == 0.0
    # The sweep spans the full eight decades; the scored window is two of them, found at the
    # bottom rather than assumed (decision I7).
    assert t.measured["h_range_swept"][1] / t.measured["h_range_swept"][0] == pytest.approx(1e8)
    assert t.measured["h_window_max"] / t.measured["h_window_min"] <= 1e2 + 1e-9
    assert t.measured["h_window_min"] > 0.0
    assert forward_reverse_test(A.objective, th, g, scales=A.scales(), key=KEY).passed
    assert dot_product_test(A.vector_map, th, reverse_vjp(A.vector_map, th), scales=A.scales(),
                            key=KEY).passed


def test_exact_quadratic_has_slope_two():
    """An exact quadratic has remainder ½h²δᵀHδ with no higher terms, so the slope is 2 by
    construction — which makes it the calibration case for the anchored window (decision I7).

    The tolerance is the bias the window itself admits, derived rather than fitted: the weakest
    scored point sits at TAYLOR_FLOOR_MARGIN times the noise floor, so it carries 1/M of noise and
    shifts the fit by about log10(1 + 1/M) / span. That is 0.003 at M = 100 over two decades. If
    this ever needs loosening, the window has drifted into the floor and the margin is wrong.
    """
    from m2.constants import TAYLOR_ANCHOR_DECADES, TAYLOR_FLOOR_MARGIN

    th = {"x": jnp.array([0.3, -1.2, 2.0])}
    J = lambda t: jnp.sum(t["x"] ** 2) + t["x"][0] * t["x"][1]  # noqa: E731
    r = taylor_test(J, th, reverse_gradient(J, th), scales=th, key=KEY)
    bias = np.log10(1.0 + 1.0 / TAYLOR_FLOOR_MARGIN) / (TAYLOR_ANCHOR_DECADES - 0.5)
    assert r.passed, r.message
    assert abs(r.measured["slope_min"] - 2.0) < bias, r.measured["slopes"]
    assert abs(r.measured["slope_max"] - 2.0) < bias, r.measured["slopes"]


@pytest.mark.parametrize(
    "kwargs",
    [{"n_directions": 19}, {"decades": 4}, {"n_h": 5}, {"slope_band": (1.5, 2.2)},
     {"slope_band": (1.8, 2.5)}, {"window_decades": 1.0}],
)
def test_taylor_protocol_cannot_be_loosened(kwargs):
    th = A.theta0()
    with pytest.raises(ValueError):
        taylor_test(A.objective, th, reverse_gradient(A.objective, th), scales=A.scales(), key=KEY,
                    **kwargs)


def test_tolerances_cannot_be_loosened():
    th = A.theta0()
    g = reverse_gradient(A.objective, th)
    with pytest.raises(ValueError):
        forward_reverse_test(A.objective, th, g, scales=A.scales(), key=KEY, rtol=1e-8)
    with pytest.raises(ValueError):
        forward_reverse_test(A.objective, th, g, scales=A.scales(), key=KEY, n_directions=5)
    with pytest.raises(ValueError):
        dot_product_test(A.vector_map, th, reverse_vjp(A.vector_map, th), scales=A.scales(), key=KEY,
                         rtol=1e-6)


def test_stricter_protocol_is_allowed():
    th = A.theta0()
    g = reverse_gradient(A.objective, th)
    assert taylor_test(A.objective, th, g, scales=A.scales(), key=KEY, n_directions=30,
                       slope_band=(1.9, 2.1)).passed


# --- the two-zone scoring rule (decision B15/B21) -------------------------------------------


def test_slope_above_the_band_passes_as_degenerate():
    """No incorrect gradient produces a slope above 2, so above the band is degeneracy, not a bug.
    Here J'' = 0 at the base point, so the remainder is O(h³)."""
    J = lambda t: (t["x"] - 1.0) ** 3 + t["x"]  # noqa: E731
    r = taylor_test(J, ONES, reverse_gradient(J, ONES), scales=ONES, key=KEY)
    assert r.passed, r.message
    assert r.measured["classifications"] == [DEGENERATE] * 20
    assert r.measured["degenerate_fraction"] == 1.0
    assert r.measured["slope_max"] > 2.2


def test_degenerate_direction_with_a_wrong_gradient_still_fails():
    """The new blind spot the loosening opens, closed: degeneracy is not a hiding place."""
    J = lambda t: (t["x"] - 1.0) ** 3 + t["x"]  # noqa: E731
    bad = corrupt_component(reverse_gradient(J, ONES), 0, 1.05)
    r = taylor_test(J, ONES, bad, scales=ONES, key=KEY)
    assert not r.passed and "gradient error" in r.message


def test_linear_objective_fails_as_insufficient_signal():
    """A linear functional's remainder is roundoff at every h, so fewer than 3 points survive the
    floor: FAIL. Decision §4 says keep linear functionals out of Taylor tests — V14a–V14c cover
    them — and the harness must say so rather than pass quietly."""
    th = {"x": jnp.array([0.3, -1.2, 2.0])}
    J = lambda t: jnp.dot(jnp.array([1.0, 2.0, -0.5]), t["x"])  # noqa: E731
    r = taylor_test(J, th, reverse_gradient(J, th), scales=th, key=KEY)
    assert not r.passed
    assert "insufficient signal" in r.message
    assert r.measured["classifications"] == ["insufficient_signal"] * 20


def test_insufficient_signal_when_the_floor_eats_the_sweep():
    th = A.theta0()
    g = reverse_gradient(A.objective, th)
    r = taylor_test(A.objective, th, g, scales=A.scales(), key=KEY, noise_floor=1e-3)
    assert not r.passed and "insufficient signal" in r.message


def test_first_order_error_is_named_as_such():
    th = A.theta0()
    bad = corrupt_component(reverse_gradient(A.objective, th), 2, 1.05)
    r = taylor_test(A.objective, th, bad, scales=A.scales(), key=KEY)
    assert not r.passed
    assert "gradient error" in r.message
    assert "first_order_error" in r.measured["classifications"]


# --- noise floor (decision B19) --------------------------------------------------------------


def test_noise_floor_falls_back_to_accumulated_roundoff_and_scales_with_sqrt_n():
    th = A.theta0()
    x0, unravel = ravel_pytree(th)
    Jx = jax.jit(lambda x: A.objective(unravel(x)))
    spread, floor_1 = estimate_noise_floor(Jx, x0, n_steps=1)
    _, floor_625 = estimate_noise_floor(Jx, x0, n_steps=625)
    assert spread == 0.0  # deterministic J
    assert floor_625 / floor_1 == pytest.approx(np.sqrt(625), rel=1e-9)
    assert 0 < floor_1 < 1e-12


def test_measured_noise_spread_wins_when_it_is_larger():
    counter = {"n": 0}

    def noisy(_x):
        counter["n"] += 1
        return 1.0 + 1e-6 * counter["n"]

    spread, floor = estimate_noise_floor(noisy, jnp.zeros(2), n_steps=1)
    assert floor == spread > 1e-7


# --- declared parameter scales (decision B20) ------------------------------------------------


def test_scales_are_required():
    th = A.theta0()
    with pytest.raises(ValueError, match="scales are required"):
        taylor_test(A.objective, th, reverse_gradient(A.objective, th), scales=None, key=KEY)


def test_a_tiny_parameter_is_still_probed_at_its_declared_scale():
    """Guarding only exact zero misses the case that never announces itself: a parameter sitting at
    1e-12 would get a 1e-12 perturbation, probing nothing while reporting a pass."""
    x0 = jnp.array([1e-12, 1.0])
    scale = perturbation_scale(x0, {"x": jnp.array([1.0, 1.0])})
    assert scale[0] == 1.0  # the declared scale, not |theta|
    d = _directions(jax.random.PRNGKey(0), 50, x0, {"x": jnp.array([1.0, 1.0])})
    assert np.abs(d[:, 0]).mean() > 0.1


def test_scales_must_be_positive_and_match_the_parameter_tree():
    x0 = jnp.array([1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        perturbation_scale(x0, {"x": jnp.array([1.0, 0.0])})
    with pytest.raises(ValueError, match="components"):
        perturbation_scale(x0, {"x": jnp.array([1.0])})


def test_directions_follow_the_larger_of_value_and_scale():
    d = _directions(jax.random.PRNGKey(0), 50, jnp.array([1.0, 1000.0]),
                    {"x": jnp.array([1.0, 1.0])})
    ratio = np.abs(d[:, 1]).mean() / np.abs(d[:, 0]).mean()
    assert 100 < ratio < 10000


# --- V15/V16 ---------------------------------------------------------------------------------


def test_v15_normalisation_is_by_gradient_norm():
    """A delta nearly orthogonal to the gradient must not inflate the relative error (B4)."""
    th = {"x": jnp.array([1.0, 1.0])}
    J = lambda t: t["x"][0] + 1e-12 * t["x"][1] ** 2  # noqa: E731
    r = forward_reverse_test(J, th, reverse_gradient(J, th), scales=th, key=KEY)
    assert r.passed, r.message


def test_gradient_shape_mismatch_is_an_error():
    th = A.theta0()
    with pytest.raises(ValueError, match="components"):
        taylor_test(A.objective, th, {"a": jnp.asarray(1.0)}, scales=A.scales(), key=KEY)


# --- the cancellation guard (finding J1) ------------------------------------------------------


def test_a_cancellation_notch_is_excluded_from_the_fit():
    """|R| must grow with h; where it does not, two terms are cancelling and the slope is meaningless.

    Built as a synthetic remainder rather than run through the solver, so the notch is placed on
    purpose and the guard is tested rather than observed. J(θ) = c₂·t² + c₃·t³ with c₃ of opposite
    sign gives a remainder whose h² and h³ terms cancel at a chosen h.
    """
    from m2.verification.gradcheck import _cancellation_ceiling

    hs = 1e-1 * np.logspace(0.0, -8.0, 17)          # descending, as taylor_test builds them
    R = hs ** 2                                      # clean quadratic
    above = np.ones_like(hs, dtype=bool)
    assert _cancellation_ceiling(hs, R, above) == pytest.approx(hs.max())

    notched = R.copy()
    notched[3] = R[3] / 40.0                         # a 40x dip, as measured under WENO5
    ceiling = _cancellation_ceiling(hs, notched, above)
    assert ceiling < hs[3], "the guard must cut BELOW the notch, not at it"
    assert ceiling > hs[4], "and must not throw away the clean decades under it"


def test_the_cancellation_guard_cannot_rescue_a_wrong_gradient():
    """The property that makes the guard safe rather than convenient.

    A first-order error gives R ∝ h, strictly monotone, so there is no notch and nothing is cut.
    And when a wrong gradient *does* notch — its h term can cancel against the h² term — the guard
    keeps everything BELOW the notch, which is exactly where the first-order term dominates. So the
    slope the fit sees still tends to 1.
    """
    from m2.verification.gradcheck import _cancellation_ceiling

    hs = 1e-1 * np.logspace(0.0, -8.0, 17)
    above = np.ones_like(hs, dtype=bool)
    assert _cancellation_ceiling(hs, hs.copy(), above) == pytest.approx(hs.max())

    # A wrong gradient whose first-order term cancels against a quadratic one at h ~ 1e-3.
    mixed = np.abs(-1e-3 * hs + hs ** 2)
    ceiling = _cancellation_ceiling(hs, mixed, above)
    kept = hs <= ceiling
    slope = float(np.polyfit(np.log10(hs[kept][-6:]), np.log10(mixed[kept][-6:]), 1)[0])
    assert slope < 1.5, f"the retained window must still read a first-order error, got {slope:.3f}"
