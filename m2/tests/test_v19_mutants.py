"""Test the test of the test: sabotage the harness and require V19 to go red.

V19 shows the gradient harness can fail. This shows V19 can fail. Each mutant breaks one piece
(a check that always passes, a harness that always fails, a corruption that does nothing), runs
the real V19 tests in a throwaway session with a temporary ledger, and requires both a failing
test and a ``fail`` row. If a future edit makes V19 insensitive to one of these, this test
catches it.
"""

import json
from pathlib import Path

import pytest

import m2.verification.canary as canary
from m2.verification.gradcheck import CheckResult

V19_TESTS = Path(__file__).with_name("test_v19_canary.py")


def _const(name, passed, measured=None):
    return lambda *a, **k: CheckResult(name, passed, "mutant", measured or {})


MUTANTS = {
    "taylor_always_passes": {"taylor_test": _const("V14", True)},
    "v15_always_passes": {"forward_reverse_test": _const("V15", True)},
    "v16_always_passes": {"dot_product_test": _const("V16", True)},
    "harness_always_fails": {"taylor_test": _const("V14", False), "forward_reverse_test": _const("V15", False),
                             "dot_product_test": _const("V16", False)},
    "corruption_is_noop": {"corrupt_component": lambda tree, index, factor=1.05: tree},
    # Decision B15: above-band slopes now pass as degenerate, so a harness that reports a high
    # slope for everything must not be able to certify anything.
    "taylor_reports_garbage_slope_six": {
        "taylor_test": _const("V14", True, {"slopes": [6.0] * 20,
                                            "classifications": ["degenerate_direction"] * 20,
                                            "degenerate_fraction": 1.0})},
}


def _run_v19(pytester, tmp_path):
    path = tmp_path / "ledger.json"
    result = pytester.runpytest(str(V19_TESTS), "-p", "m2.verification.pytest_plugin", "--ledger-path", str(path),
                                "--ledger-any-scope", "-p", "no:cacheprovider")
    return result.parseoutcomes(), json.loads(path.read_text())["checks"]["V19"]


@pytest.mark.check("V19")
def test_control_unmutated_v19_is_green_in_the_same_harness(pytester, tmp_path):
    """Without this, an environment problem in the inner session would make every mutant look caught."""
    outcomes, row = _run_v19(pytester, tmp_path)
    n_cases = len([ln for ln in V19_TESTS.read_text().splitlines() if ln.startswith("def test_")])
    assert outcomes.get("passed", 0) >= n_cases
    assert not outcomes.get("failed") and not outcomes.get("errors")
    assert row["result"] == "pass"


@pytest.mark.check("V19")
@pytest.mark.parametrize("mutant", list(MUTANTS))
def test_v19_goes_red_under_mutant(mutant, pytester, monkeypatch, tmp_path):
    for attr, replacement in MUTANTS[mutant].items():
        monkeypatch.setattr(canary, attr, replacement)
    outcomes, row = _run_v19(pytester, tmp_path)
    assert outcomes.get("failed", 0) >= 1, f"V19 stayed green under mutant {mutant}"
    assert row["result"] == "fail"
