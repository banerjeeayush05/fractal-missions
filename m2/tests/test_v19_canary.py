"""V19 — corrupted-gradient canary. The M2.0 gate (PRD §6, §7.2, §8.5).

Built against a trivial analytic function (no solver exists yet). The ledger row records the
control results, per-component detection, and how small a corruption V14 can still catch.
"""

import jax
import jax.numpy as jnp
import pytest

from m2.constants import V19_CORRUPTION_FACTOR
from m2.verification import analytic as A
from m2.verification.canary import corrupt_component, run_canary, taylor_detection_threshold
from m2.verification.gradcheck import DEGENERATE, reverse_gradient, taylor_test


@pytest.fixture(scope="module")
def report():
    return run_canary(A.objective, A.vector_map, A.theta0(), scales=A.scales(),
                      key=jax.random.PRNGKey(19))


@pytest.mark.check("V19")
def test_control_passes_all_three(report, ledger_measure):
    """Without this, a harness that always fails would 'pass' the canary."""
    for check, r in report.control.items():
        assert r.passed, f"control {check} failed on a correct gradient: {r.message}"
    ledger_measure.update(report.measured()["control"])


@pytest.mark.check("V19")
@pytest.mark.parametrize("check", ["V14", "V15", "V16"])
def test_corrupted_component_is_caught(report, check, ledger_measure):
    missed = [k for k, per_k in report.corrupted.items() if per_k[check].passed]
    assert not missed, f"{check} PASSED a gradient with component(s) {missed} scaled by 1.05"
    if check == "V14":  # and it must name the cause correctly, not blame roundoff
        for per_k in report.corrupted.values():
            assert "gradient error" in per_k["V14"].message and "roundoff" not in per_k["V14"].message
    ledger_measure["components_caught"] = f"{len(report.corrupted)}/{len(report.corrupted)}"
    ledger_measure["results"] = {f"component_{k}": {"message": per_k[check].message, **per_k[check].measured}
                                 for k, per_k in report.corrupted.items()}


@pytest.mark.check("V19")
def test_canary_caught_overall(report, ledger_measure):
    assert report.caught
    m = report.measured()
    m["taylor_detection"] = taylor_detection_threshold(A.objective, A.theta0(), scales=A.scales(),
                                                       key=jax.random.PRNGKey(19))
    ledger_measure.update({k: m[k] for k in ("corruption_factor", "n_components", "corrupted_detected",
                                             "corrupted_V14_slope_min", "corrupted_V15_max_rel_error",
                                             "corrupted_V16_max_rel_error", "taylor_detection")})


@pytest.mark.check("V19")
def test_canary_refuses_undetectable_setup(ledger_measure):
    """A component whose gradient is ~0 cannot reveal a 5 % error; the canary must refuse the
    setup instead of reporting 'caught' (OPEN_QUESTIONS B13)."""
    import jax.numpy as jnp

    th = {"x": jnp.array([0.0, 1.0])}  # ∂J/∂x0 = 0 at x0 = 0
    with pytest.raises(ValueError, match="precondition"):
        run_canary(lambda t: jnp.sum(t["x"] ** 2), lambda t: t["x"] ** 2, th,
                   scales={"x": jnp.asarray([1e-9, 1.0])}, key=jax.random.PRNGKey(0))
    ledger_measure["precondition_enforced"] = True


@pytest.mark.check("V19")
def test_degeneracy_is_not_a_hiding_place(ledger_measure):
    """Decision B15 lets a slope above the band pass as degenerate, which opens one new blind spot:
    a harness returning garbage with a high slope would pass. So a genuinely degenerate objective
    with a corrupted gradient must still FAIL."""
    th = {"x": jnp.asarray(1.0)}
    J = lambda t: (t["x"] - 1.0) ** 3 + t["x"]  # J'' = 0 at x = 1: genuinely degenerate  # noqa: E731
    key = jax.random.PRNGKey(19)
    good = reverse_gradient(J, th)
    clean = taylor_test(J, th, good, scales=th, key=key)
    assert clean.passed and clean.measured["classifications"] == [DEGENERATE] * 20, clean.message

    corrupted = taylor_test(J, th, corrupt_component(good, 0, V19_CORRUPTION_FACTOR), scales=th, key=key)
    assert not corrupted.passed, "a corrupted gradient hid behind a degenerate direction"
    assert "gradient error" in corrupted.message
    ledger_measure["degenerate_control_slope"] = clean.measured["slope_min"]
    ledger_measure["corrupted_slope"] = corrupted.measured["slope_min"]
