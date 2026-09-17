"""The gradient harness itself: a correct gradient passes, a wrong one does not, and the two
documented limitations stay documented rather than quietly becoming passes."""

from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geocore.verification import analytic as A
from geocore.verification.canary import corrupt_component
from geocore.verification.gradcheck import (
    CLEAN, DEGENERATE, INSUFFICIENT, SHALLOW,
    WINDOW_DECADES,
    dot_product_test, forward_reverse_test, perturbation_scale, taylor_test,
)

KEY = jax.random.PRNGKey(19)


def test_the_hand_derived_gradient_matches_jax():
    """Everything else here is circular if this fails: the fixture's closed-form gradient is the
    only thing in the suite that is not computed by the system under test."""
    th = A.theta0()
    hand, auto = A.objective_gradient(th), jax.grad(A.objective)(th)
    for k in hand:
        assert float(jnp.max(jnp.abs(hand[k] - auto[k]))) == 0.0


def test_correct_gradient_passes_all_three():
    th = A.theta0()
    t = taylor_test(A.objective, th, A.objective_gradient(th), A.scales(), KEY)
    assert t.passed, t.message
    assert 1.8 < t.measured["slope_min"] <= t.measured["slope_max"] < 2.2
    assert t.measured["classifications"] == [CLEAN] * 20
    assert t.measured["degenerate_fraction"] == 0.0
    assert forward_reverse_test(A.objective, th, KEY).passed
    assert dot_product_test(A.vector_map, th, KEY).passed


def test_the_scored_window_is_two_decades_inside_an_eight_decade_sweep():
    """Decision I7. The sweep is wide so the floor can be MEASURED; the scored window is narrow
    and anchored at that floor. Widening the window is the obvious way to 'fix' a failing V14,
    which is exactly why the shape is pinned here."""
    th = A.theta0()
    m = taylor_test(A.objective, th, A.objective_gradient(th), A.scales(), KEY).measured
    lo, hi = m["h_range_swept"]
    assert hi / lo == pytest.approx(1e8), "the sweep must span eight decades"
    # Per direction, not aggregated: h_window_min/max are the UNION across 20 directions, which
    # is legitimately wider than two decades because each direction anchors at its own floor.
    assert m["window_decades_max"] <= WINDOW_DECADES + 1e-9


@pytest.mark.parametrize("component", range(6))
def test_a_five_percent_error_in_any_component_fails(component):
    """Every component, not just the first: a harness that only watches one is a harness with
    five blind spots."""
    th = A.theta0()
    bad = corrupt_component(A.objective_gradient(th), component, 1.05)
    result = taylor_test(A.objective, th, bad, A.scales(), KEY)
    assert not result.passed
    slopes = [s for s in [result.measured["slope_min"], result.measured["slope_max"]]
              if s is not None]
    assert all(s < 1.8 for s in slopes), f"a wrong gradient should tend to slope 1, got {slopes}"


def test_parameter_scales_are_required_and_must_be_positive():
    """PRD §7.1. A parameter at 1e-12 given a 1e-12 perturbation probes nothing and reports a
    pass, so the scale is part of the protocol rather than a convenience."""
    th = A.theta0()
    s = perturbation_scale(th, A.scales())
    assert bool(jnp.all(s >= jnp.abs(jnp.concatenate([th["a"], th["b"]]))))
    with pytest.raises(ValueError, match="strictly positive"):
        perturbation_scale(th, {"a": jnp.zeros(3), "b": jnp.ones(3)})
    with pytest.raises(ValueError, match="shape"):
        perturbation_scale(th, {"a": jnp.ones(2), "b": jnp.ones(3)})


def test_zero_curvature_direction_fails_with_a_diagnostic_not_a_pass():
    """FINDING A10, pinned. `linear_objective` has an exactly zero Hessian, so a CORRECT gradient
    leaves a remainder sitting on the roundoff floor at every h. The [1.8, 2.2] band cannot pass
    that, and the honest report is a FAILURE carrying `insufficient_signal` -- not a pass.

    This is a real limitation of the §7.1 gate awaiting an owner rule, not a harness bug. The
    test exists so nobody 'fixes' it into a silent pass.
    """
    th = A.theta0()
    result = taylor_test(A.linear_objective, th, A.linear_gradient(th), A.scales(), KEY)
    assert not result.passed
    assert set(result.measured["classifications"]) == {INSUFFICIENT}
    assert result.measured["h_window_min"] is None


def test_the_fixture_hessian_is_definite():
    """Why the fixture uses all-positive `a`. An indefinite Hessian guarantees directions with
    v.Hv = 0, where a correct gradient has no quadratic term for V14 to find -- which made the
    V19 control fail on 2 of 3 seeds before the fixture was corrected."""
    from jax.flatten_util import ravel_pytree
    flat, unravel = ravel_pytree(A.theta0())
    hessian = np.asarray(jax.hessian(lambda x: A.objective(unravel(x)))(flat))
    assert np.linalg.eigvalsh(hessian).min() > 0.0
