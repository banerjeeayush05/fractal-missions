"""The ledger: rows survive across runs, and the ledger of record refuses a dirty tree."""

import json

import pytest

from geocore.verification.ledger import Ledger, LedgerRefusal


def test_rows_from_earlier_runs_are_carried_forward(tmp_path):
    """Finding S13.4. A fast-tier run knows nothing about V8; writing only its own rows would delete
    V8's, and a record generated from that file would be partial while looking complete."""
    nightly = Ledger(tmp_path)
    nightly.record("V8", "xfailed", {"area_loss_fraction": 0.43}, "tests/test_invariants.py::v8")
    nightly.write()

    fast = Ledger(tmp_path)
    fast.record("V14", "passed", {"slope_min": 1.99}, "tests/test_gradients_2d.py::v14")
    path = fast.write()

    rows = {row["check_id"]: row for row in json.loads(path.read_text())["rows"]}
    assert set(rows) == {"V8", "V14"}
    assert rows["V8"]["measured"]["area_loss_fraction"] == 0.43
    assert json.loads(path.read_text())["checks_this_run"] == ["V14"]


def test_a_rerun_replaces_its_own_row(tmp_path):
    first = Ledger(tmp_path)
    first.record("V14", "failed", {"slope_min": 1.0}, "t")
    first.write()
    second = Ledger(tmp_path)
    second.record("V14", "passed", {"slope_min": 2.0}, "t")
    path = second.write()
    rows = json.loads(path.read_text())["rows"]
    assert len(rows) == 1 and rows[0]["result"] == "passed"


def test_the_worst_result_wins_within_one_run(tmp_path):
    """Several tests can claim one check. One failing means the check did not pass."""
    ledger = Ledger(tmp_path)
    ledger.record("V14", "passed", {"a": 1}, "t1")
    ledger.record("V14", "failed", {"b": 2}, "t2")
    assert ledger.rows[0].result == "failed"
    assert ledger.rows[0].measured == {"a": 1, "b": 2}, "measurements from both tests are kept"


def test_the_ledger_of_record_refuses_a_dirty_tree(tmp_path):
    """Repo rule: CI on a clean checkout is its only writer. A result whose provenance is 'someone's
    laptop, with uncommitted edits' is an anecdote with a timestamp."""
    (tmp_path / "untracked.txt").write_text("x")
    with pytest.raises(LedgerRefusal, match="not a git checkout|DIRTY"):
        Ledger(tmp_path, of_record=True)


def test_an_unknown_check_id_cannot_be_recorded(tmp_path):
    from geocore.verification.registry import UnknownCheck

    with pytest.raises(UnknownCheck):
        Ledger(tmp_path).record("V99", "passed")
