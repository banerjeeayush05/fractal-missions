"""Verification ledger and check registration (PRD §8, §8.0; OPEN_QUESTIONS B7).

The end-to-end tests run a throwaway pytest session (pytester) against a temporary ledger, so
the real ledger is never touched by these tests.
"""

import json
import textwrap

import pytest

from m2.verification import ledger
from m2.verification.registry import CHECKS


def test_registry_holds_the_prd_checks_plus_the_decision_additions():
    numeric = [c for c in CHECKS if c[-1].isdigit()]
    assert numeric == [f"V{i}" for i in range(1, 23)]
    # Decision §1 and §5 added these; V23..V40 belong to M3 and must never appear here.
    assert {"V1a", "V14a", "V14b", "V14c"} <= set(CHECKS)
    assert all(CHECKS[c].cadence == ("fast",) for c in ("V1a", "V14a", "V14b", "V14c"))
    tiers = {cid: spec.cadence for cid, spec in CHECKS.items() if cid[-1].isdigit()}
    fast = {c for c, t in tiers.items() if "fast" in t}
    nightly = {c for c, t in tiers.items() if "nightly" in t}
    gate = {c for c, t in tiers.items() if "gate" in t}
    # PRD §8.0 table, verbatim.
    assert fast == {"V1", "V2", "V11", "V12", "V14", "V15", "V16", "V19", "V20", "V21"}
    assert nightly == {"V5", "V6", "V7", "V8", "V9", "V10", "V13", "V17", "V18"}
    assert gate == {"V3", "V4", "V22", "V14"}


def test_empty_ledger_preregisters_all_checks(tmp_path):
    data = ledger.load_ledger(tmp_path / "none.json")
    assert set(data["checks"]) == set(CHECKS)
    assert all(r["result"] == "not_implemented" for r in data["checks"].values())


def test_record_run_writes_row_with_required_fields(tmp_path):
    path = tmp_path / "ledger.json"
    row = ledger.record_run("V19", "fast", passed=True, measured={"x": 1.5, "bad": float("nan")},
                            tests=["t::a"], path=path)
    assert row["result"] == "pass"
    on_disk = json.loads(path.read_text())["checks"]["V19"]
    run = on_disk["runs"]["fast"]
    for key in ("result", "last_run", "git_sha", "git_dirty", "measured", "tests", "environment"):
        assert key in run
    for key in ("id", "description", "cadence"):
        assert key in on_disk
    assert run["measured"] == {"x": 1.5, "bad": "nan"}
    assert run["environment"]["x64"] is True
    assert len(json.loads(path.read_text())["checks"]) == len(CHECKS)


def test_overall_result_rules(tmp_path):
    path = tmp_path / "ledger.json"
    assert ledger.record_run("V14", "fast", passed=True, measured={}, tests=[], path=path)["result"] == "incomplete"
    assert ledger.record_run("V14", "gate", passed=True, measured={}, tests=[], path=path)["result"] == "pass"
    assert ledger.record_run("V14", "fast", passed=False, measured={}, tests=[], path=path)["result"] == "fail"


def test_record_run_rejects_unknown_id_or_tier(tmp_path):
    with pytest.raises(KeyError):
        ledger.record_run("V23", "fast", passed=True, measured={}, tests=[], path=tmp_path / "l.json")
    with pytest.raises(ValueError):
        ledger.record_run("V19", "nightly", passed=True, measured={}, tests=[], path=tmp_path / "l.json")


def test_load_rejects_foreign_ids(tmp_path):
    path = tmp_path / "l.json"
    data = ledger.empty_ledger()
    data["checks"]["V99"] = {}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="V99"):
        ledger.load_ledger(path)


# --- end-to-end through the pytest plugin -------------------------------------------------

_INNER = textwrap.dedent(
    """
    import pytest

    @pytest.mark.check("V19")
    def test_canary_a(ledger_measure):
        ledger_measure["slope"] = 2.0

    @pytest.mark.check("V19")
    def test_canary_b():
        pass

    @pytest.mark.check("V20")
    def test_cfl_fails():
        assert False, "deliberate"

    @pytest.mark.check("V21")
    def test_golden_skipped():
        pytest.skip("no golden file")

    def test_structural():
        pass
    """
)


def _run(pytester, tmp_path, *args):
    path = tmp_path / "inner_ledger.json"
    pytester.makepyfile(test_inner=_INNER)
    result = pytester.runpytest("-p", "m2.verification.pytest_plugin", "--ledger-path", str(path),
                                "--ledger-any-scope", *args)
    checks = json.loads(path.read_text())["checks"] if path.exists() else None
    return result, checks


def test_plugin_writes_pass_fail_and_skip_rows(pytester, tmp_path):
    result, checks = _run(pytester, tmp_path)
    result.assert_outcomes(passed=3, failed=1, skipped=1)
    assert checks["V19"]["runs"]["fast"]["result"] == "pass"
    assert checks["V19"]["runs"]["fast"]["measured"]["test_canary_a"] == {"slope": 2.0}
    assert len(checks["V19"]["runs"]["fast"]["tests"]) == 2
    assert checks["V20"]["result"] == "fail"
    assert checks["V21"]["result"] == "fail"  # a skipped check is not a passed check
    assert checks["V1"]["result"] == "not_implemented"


def test_plugin_does_not_certify_partial_selection(pytester, tmp_path):
    result, checks = _run(pytester, tmp_path, "-k", "canary_a")
    result.assert_outcomes(passed=1)
    assert checks is None or checks["V19"]["result"] == "not_implemented"


def test_plugin_does_not_certify_explicit_path_runs(pytester, tmp_path):
    """`pytest some_file.py` may omit some of a check's tests, so it must not write rows."""
    path = tmp_path / "inner_ledger.json"
    pytester.makepyfile(test_inner=_INNER)
    result = pytester.runpytest("-p", "m2.verification.pytest_plugin", "--ledger-path", str(path), "test_inner.py")
    result.stdout.fnmatch_lines(["*verification ledger NOT written*"])
    assert not path.exists()


def test_plugin_tier_selection(pytester, tmp_path):
    result, _ = _run(pytester, tmp_path, "-m", "nightly")
    result.assert_outcomes()  # nothing in the nightly tier here
    result, _ = _run(pytester, tmp_path, "-m", "fast")
    result.assert_outcomes(passed=3, failed=1, skipped=1)


def test_plugin_rejects_unknown_check_id_and_wrong_tier(pytester, tmp_path):
    pytester.makepyfile(test_bad=textwrap.dedent(
        """
        import pytest
        @pytest.mark.check("V99")
        def test_a(): pass
        @pytest.mark.check("V19", tier="nightly")
        def test_b(): pass
        """
    ))
    result = pytester.runpytest("-p", "m2.verification.pytest_plugin", "--ledger-path", str(tmp_path / "l.json"))
    result.stderr.fnmatch_lines(["*unknown check ID 'V99'*", "*V19 does not run in tier 'nightly'*"])
    assert result.ret != 0


# --- ledger of record vs local ledger (decision §13) ---------------------------------------


def test_default_path_is_the_untracked_local_ledger():
    assert ledger.DEFAULT_LEDGER_PATH == ledger.LOCAL_LEDGER_PATH
    assert ledger.LOCAL_LEDGER_PATH.parent.name == "local"
    assert ledger.RECORD_LEDGER_PATH.name == "verification_ledger.json"


def test_environment_records_the_device_that_produced_the_row():
    env = ledger.environment()
    assert env["device_kind"] and env["backend"]
    assert env["x64"] is True


def test_of_record_refuses_a_dirty_working_tree(pytester, tmp_path, monkeypatch):
    """CI on a clean checkout is the only writer of the ledger of record (decision §13)."""
    monkeypatch.setattr(ledger, "git_state", lambda *a, **k: ("deadbeef", True))
    path = tmp_path / "record.json"
    pytester.makepyfile(test_inner=_INNER)
    result = pytester.runpytest("-p", "m2.verification.pytest_plugin", "--ledger-path", str(path),
                                "--ledger-any-scope", "--ledger-of-record")
    result.stdout.fnmatch_lines(["*of record NOT written*dirty*"])
    assert not path.exists()


def test_of_record_writes_when_the_tree_is_clean(pytester, tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "git_state", lambda *a, **k: ("deadbeef", False))
    path = tmp_path / "record.json"
    pytester.makepyfile(test_inner=_INNER)
    pytester.runpytest("-p", "m2.verification.pytest_plugin", "--ledger-path", str(path),
                       "--ledger-any-scope", "--ledger-of-record")
    assert json.loads(path.read_text())["checks"]["V19"]["result"] == "pass"
