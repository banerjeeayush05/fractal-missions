"""V19's own mutation guard.

V19 asserts that a corrupted gradient makes V14, V15 and V16 fail. But V19 is itself code, and a
broken V19 would report a green canary over a blind harness. So: sabotage the canary five ways
and require each sabotage to turn V19 RED.

This is the same argument one level up. `test_environment.py` did it for the import scanner at
stage 1; V19 does it for the gradient harness; this file does it for V19. The recursion stops
here, because a test that checks this test would be checked by nothing.
"""

import dataclasses

import jax
import pytest

from geocore.verification import analytic as A
from geocore.verification.canary import corrupt_component, run_canary
from geocore.verification.gradcheck import CheckResult

KEY = jax.random.PRNGKey(19)
ONE_COMPONENT = [0]  # one component is enough to expose a sabotage, and keeps the fast tier fast


def _always(passed: bool):
    def fake(*_args, **_kwargs) -> CheckResult:
        return CheckResult(passed, f"sabotaged: always {'passes' if passed else 'fails'}", {})
    return fake


def _no_op_corruption(gradient, index, factor):  # noqa: ARG001
    """The corruption that isn't. Mutant and control become identical."""
    return gradient


def _canary(**overrides):
    return run_canary(A.objective, A.vector_map, A.theta0(), A.scales(), KEY,
                      components=ONE_COMPONENT, **overrides)


@pytest.mark.parametrize("sabotage,label", [
    ({"taylor": _always(True)}, "V14 always passes"),
    ({"forward_reverse": _always(True)}, "V15 always passes"),
    ({"dot_product": _always(True)}, "V16 always passes"),
])
def test_a_check_that_always_passes_turns_v19_red(sabotage, label):
    """A blind check cannot detect a corrupted gradient, so V19 must not report all-clear."""
    verdict = _canary(**sabotage).verdict()
    assert not verdict.passed, f"{label}: V19 should have failed but reported {verdict.message}"
    assert "NOT caught" in verdict.message


def test_a_harness_that_always_fails_turns_v19_red():
    """Without the control half, this is the sabotage V19 would miss: every mutant 'fails', which
    looks exactly like success. The control is what distinguishes detection from breakage."""
    verdict = _canary(taylor=_always(False), forward_reverse=_always(False),
                      dot_product=_always(False)).verdict()
    assert not verdict.passed
    assert "control" in verdict.message.lower()


def test_a_no_op_corruption_turns_v19_red():
    """If the injection does nothing, mutant and control are the same run and everything passes.
    A canary that never actually corrupts anything is a canary that always sings."""
    verdict = _canary(corrupt=_no_op_corruption).verdict()
    assert not verdict.passed
    assert "NOT caught" in verdict.message


def test_the_unsabotaged_canary_passes():
    """The control for this file. Without it, every assertion above would be satisfied by a
    `verdict()` that always returned False."""
    assert _canary().verdict().passed


def test_corrupt_component_actually_changes_one_component():
    """The smallest possible unit of the same idea."""
    g = A.objective_gradient(A.theta0())
    from jax.flatten_util import ravel_pytree
    before, _ = ravel_pytree(g)
    after, _ = ravel_pytree(corrupt_component(g, 2, 1.05))
    changed = [i for i in range(before.shape[0]) if float(before[i]) != float(after[i])]
    assert changed == [2]
    assert float(after[2]) == pytest.approx(float(before[2]) * 1.05)
