"""Pytest plugin: check registration, tier markers, and ledger rows (PRD §8.0, §8; OPEN_QUESTIONS B7).

- ``@pytest.mark.check("V19")`` ties a test to a check ID (optionally ``tier="gate"``). The tier
  marker (``fast`` / ``nightly`` / ``gate``) is added from the registry. Select tiers with
  ``-m fast`` etc. Untagged tests are structural and count as fast tier.
- A check's ledger row is written at session end for each (ID, tier) whose tests *all* ran in
  this session. If any of them was deselected (``-k``/``-m``/``--lf``) or never ran, nothing is
  written: a partial run must not certify a check. A skip counts as a failure.
- Rows are written only when collection covered the default ``testpaths`` (plain ``pytest``,
  optionally with ``-m <tier>``). ``pytest tests/one_file.py`` could leave out some of a check's
  tests, so it writes nothing. ``--ledger-any-scope`` lifts this, for the plugin's own tests
  against a temporary ledger.
- Tests attach measured quantities with the ``ledger_measure`` fixture (a dict).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from m2.verification import ledger
from m2.verification.registry import CHECKS, TIERS

_KEY = pytest.StashKey[dict]()
_MEASURE = pytest.StashKey[dict]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--ledger-path", default=str(ledger.DEFAULT_LEDGER_PATH),
                     help="verification ledger JSON to write (default: reports/verification_ledger.json)")
    parser.addoption("--ledger-any-scope", action="store_true",
                     help="write ledger rows even when collection was limited to explicit paths")
    parser.addoption("--ledger-of-record", action="store_true",
                     help="write the tracked ledger of record (CI only; requires a clean checkout)")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "check(id, tier=None): tie a test to a verification check ID")
    for tier in TIERS:
        config.addinivalue_line("markers", f"{tier}: {tier} tier (PRD §8.0)")
    config.stash[_KEY] = {"collected": defaultdict(list), "deselected": set(), "outcome": {}}


@pytest.hookimpl(tryfirst=True)  # before -m/-k deselection, so tier markers are selectable
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    state = config.stash[_KEY]
    errors = []
    for item in items:
        marks = list(item.iter_markers("check"))
        if not marks:
            item.add_marker(pytest.mark.fast)
            continue
        if len(marks) > 1:
            errors.append(f"{item.nodeid}: at most one @pytest.mark.check per test")
            continue
        mark = marks[0]
        cid = mark.args[0] if mark.args else None
        if cid not in CHECKS:
            errors.append(f"{item.nodeid}: unknown check ID {cid!r} (registry: V1..V22)")
            continue
        tier = mark.kwargs.get("tier", CHECKS[cid].cadence[0])
        if tier not in CHECKS[cid].cadence:
            errors.append(f"{item.nodeid}: {cid} does not run in tier {tier!r} "
                          f"(cadence {CHECKS[cid].cadence})")
            continue
        item.add_marker(getattr(pytest.mark, tier))
        state["collected"][(cid, tier)].append(item.nodeid)
    if errors:
        raise pytest.UsageError("check registration errors:\n  " + "\n  ".join(errors))


def pytest_deselected(items: list[pytest.Item]) -> None:
    if items:
        items[0].config.stash[_KEY]["deselected"].update(i.nodeid for i in items)


@pytest.fixture
def ledger_measure(request: pytest.FixtureRequest) -> dict:
    """Dict of measured quantities for this test's ledger row."""
    d: dict = {}
    request.node.stash[_MEASURE] = d
    return d


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    report = yield
    outcome = item.config.stash[_KEY]["outcome"]
    prev = outcome.get(item.nodeid)
    if report.failed or report.skipped:
        outcome[item.nodeid] = ("failed" if report.failed else "skipped", report.longreprtext[-600:])
    elif report.when == "call" and prev is None:
        outcome[item.nodeid] = ("passed", "")
    if report.when == "teardown":
        item.config.stash[_KEY].setdefault("measured", {})[item.nodeid] = item.stash.get(_MEASURE, {})
    return report


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    state = session.config.stash.get(_KEY, None)
    if state is None:
        return
    full_scope = session.config.args_source == pytest.Config.ArgsSource.TESTPATHS
    if not (full_scope or session.config.getoption("--ledger-any-scope")):
        state["written"] = "NOT written: collection limited to explicit paths; run plain `pytest` to update"
        return
    of_record = session.config.getoption("--ledger-of-record")
    path = Path(session.config.getoption("--ledger-path"))
    if of_record:
        # Decision §13: only a clean checkout may write the ledger of record, so every published
        # row identifies exactly the code that produced it.
        if path == ledger.LOCAL_LEDGER_PATH:
            path = ledger.RECORD_LEDGER_PATH
        _, dirty = ledger.git_state()
        if dirty:
            state["written"] = ("of record NOT written: the working tree is dirty; CI on a clean "
                                "checkout is the only writer (decision §13)")
            return
    written = []
    for (cid, tier), nodeids in sorted(state["collected"].items()):
        if any(n in state["deselected"] for n in nodeids):
            continue  # partial selection: do not certify
        results = [state["outcome"].get(n) for n in nodeids]
        if any(r is None for r in results):
            continue  # some test never ran (e.g. -x stopped the session)
        passed = all(r[0] == "passed" for r in results)
        failures = [f"{n}: {r[0]}" for n, r in zip(nodeids, results) if r[0] != "passed"]
        measured = {n.split("::")[-1]: state.get("measured", {}).get(n, {}) for n in nodeids}
        ledger.record_run(cid, tier, passed=passed, measured=measured, tests=nodeids, path=path,
                          detail="; ".join(failures))
        written.append(f"{cid}[{tier}]={'pass' if passed else 'FAIL'}")
    state["written"] = written


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    written = config.stash[_KEY].get("written") if _KEY in config.stash else None
    if isinstance(written, str):
        terminalreporter.write_line(f"verification ledger {written}")
    elif written:
        terminalreporter.write_line(f"verification ledger rows written: {', '.join(written)} "
                                    f"-> {config.getoption('--ledger-path')}")
