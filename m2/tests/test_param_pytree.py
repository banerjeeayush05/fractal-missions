"""PRD §7.3: every declared parameter must produce a nonzero gradient on at least one objective.
Includes a canary: a parameter captured in a closure must be caught."""

import jax.numpy as jnp
import pytest

from m2.verification import analytic as A
from m2.verification.params import check_parameters_live, component_names


def test_all_parameters_live_on_analytic_objectives():
    r = check_parameters_live({"J": A.objective, "F0": lambda t: A.vector_map(t)[0]}, A.theta0())
    assert r.passed, r.message
    assert set(r.measured["max_abs_grad"]) == {"['a']", "['b'][0]", "['b'][1]", "['c'][0]", "['c'][1]"}


def test_closure_captured_parameter_is_caught():
    params = {"v0": jnp.asarray(1.5), "p": jnp.asarray(2.0)}
    p_captured = params["p"]  # the bug: read from the closure, not from the PyTree

    def objective(prm):
        return prm["v0"] ** 2 * p_captured

    r = check_parameters_live({"obj": objective}, params)
    assert not r.passed
    assert r.measured["dead"] == ["['p']"]
    assert "closure" in r.message


def test_liveness_on_any_objective_suffices():
    params = {"a": jnp.asarray(1.0), "b": jnp.asarray(2.0)}
    r = check_parameters_live({"only_a": lambda t: t["a"] ** 2, "only_b": lambda t: jnp.sin(t["b"])}, params)
    assert r.passed


def test_partially_dead_array_leaf_names_the_element():
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    r = check_parameters_live({"J": lambda t: t["w"][0] ** 2 + t["w"][2]}, params)
    assert r.measured["dead"] == ["['w'][1]"]


def test_component_names_order_matches_ravel():
    assert component_names({"b": jnp.zeros(2), "a": jnp.asarray(0.0)}) == ["['a']", "['b'][0]", "['b'][1]"]


def test_requires_an_objective():
    with pytest.raises(ValueError):
        check_parameters_live({}, {"a": jnp.asarray(1.0)})
