"""V19 -- corrupted-gradient canary. The M2.0 gate (PRD §6, §7.2, §8.5).

Built against a trivial analytic function; no solver exists yet. The ledger row records the
control results, per-component detection, and how small a corruption V14 can still catch.
"""

import jax
import pytest

from geocore.constants import V19_CORRUPTION_FACTOR
from geocore.verification import analytic as A
from geocore.verification.canary import NEAR_ZERO, run_canary, taylor_detection_threshold

KEY = jax.random.PRNGKey(19)


@pytest.fixture(scope="module")
def report():
    return run_canary(A.objective, A.vector_map, A.theta0(), A.scales(), KEY)


@pytest.mark.check("V19")
def test_control_passes_and_every_corruption_is_caught(report, ledger_measure):
    """Both halves of V19 in one assertion, because either alone is worthless.

    Without a passing CONTROL, a harness that always fails would 'pass' the canary. Without the
    MUTANTS failing, a harness that always passes would too.
    """
    verdict = report.verdict()
    ledger_measure(**verdict.measured)
    ledger_measure(
        detection_threshold=taylor_detection_threshold(
            A.objective, A.theta0(), A.scales(), KEY
        ),
        corruption_factor=V19_CORRUPTION_FACTOR,
    )
    assert verdict.passed, verdict.message


def test_no_gradient_component_is_near_zero(report):
    """Decision B13, asserted as a PRECONDITION. A multiplicative corruption of a zero component
    is not a corruption at all, so such a case would report an impossible task as a failure."""
    assert all(abs(v) > NEAR_ZERO for v in report.gradient_values)


def test_all_three_checks_see_every_component(report):
    """The same corruption must reach V14, V15 and V16. If it only reached V14 -- which is what
    happens if V15 and V16 are allowed to recompute their own reverse mode -- then V19 would be
    verifying one check while claiming to verify three."""
    for i in range(len(report.mutants)):
        detected = report.mutant_detected_by(i)
        assert all(detected.values()), f"component {i}: {detected}"
