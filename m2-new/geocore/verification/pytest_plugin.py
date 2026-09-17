"""Wires the check registry and the ledger into pytest.

A test declares what it verifies:

    @pytest.mark.check("V14")
    def test_taylor_remainder(ledger_measure):
        ...
        ledger_measure(slope_min=1.98, degenerate_fraction=0.0)

and the plugin does the rest:

* **Validates the ID at collection time.** An undeclared ID is a collection error, not a test
  failure, so it cannot be mistaken for a physics problem.
* **Applies the tier marker from the registry**, so `-m "not nightly"` works without anyone
  remembering to decorate. A check cannot drift into the wrong tier.
* **Records the outcome**, including failures. A ledger that only records passes is a
  advertisement, not a record.
* **Writes the ledger once at the end**, to the local path by default and to the ledger of
  record only under `--ledger-of-record` on a clean tree.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from geocore.verification import registry
from geocore.verification.ledger import Ledger, LedgerRefusal


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--ledger-of-record", action="store_true", default=False,
        help="write reports/verification_ledger.json instead of the local ledger. "
             "Refuses on a dirty tree; intended for CI on a clean checkout.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "check(id): ties a test to a verification check ID; drives ledger rows and tier"
    )
    for cadence in registry.Cadence:
        config.addinivalue_line("markers", f"{cadence.value}: {cadence.value} tier (PRD §8.0)")

    root = pathlib.Path(str(config.rootpath))
    try:
        config._geocore_ledger = Ledger(root, of_record=config.getoption("--ledger-of-record"))
    except LedgerRefusal as exc:
        raise pytest.UsageError(str(exc)) from None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Validate every claimed ID and stamp the tier marker from the registry."""
    unknown: list[str] = []
    for item in items:
        marker = item.get_closest_marker("check")
        if marker is None:
            continue
        check_id = marker.args[0] if marker.args else None
        try:
            check = registry.get(check_id)
        except registry.UnknownCheck as exc:
            unknown.append(f"{item.nodeid}: {exc}")
            continue
        # PRD §8.0 assigns tiers by what a run costs, not only by check ID: the fast tier is
        # "2D only, coarse" and small 3D runs belong to nightly. A test that runs a check in a more
        # expensive configuration may declare `nightly` or `gate` itself, and the registry's tier is
        # then not added on top. A test can move a check to a SLOWER tier this way, never a faster
        # one: the fast marker is only ever applied from the registry.
        explicit = {m.name for m in item.iter_markers()} & {"nightly", "gate"}
        if not explicit:
            item.add_marker(getattr(pytest.mark, check.cadence.value))
        if check.status is registry.Status.ON_HOLD:
            item.add_marker(
                pytest.mark.skip(reason=f"{check.id} is ON HOLD: {check.note or 'see registry'}")
            )
    if unknown:
        raise pytest.UsageError(
            "test(s) claim an undeclared verification check:\n  " + "\n  ".join(unknown)
        )


@pytest.fixture
def ledger_measure(request: pytest.FixtureRequest):
    """Record the numbers a check measured.

    The measured VALUE is the part of a ledger row that has lasting worth. "V14 passed" ages
    badly; "V14 passed, worst slope 1.987, degenerate fraction 0.0" lets a later reader see a
    trend — and a jump in the degenerate fraction is itself a finding (m2/CLAUDE.md).
    """
    measured: dict[str, Any] = {}

    def record(**values: Any) -> None:
        measured.update(values)

    request.node._geocore_measured = measured
    yield record


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    marker = item.get_closest_marker("check")
    if marker is None or not marker.args:
        return
    ledger = getattr(item.config, "_geocore_ledger", None)
    if ledger is None:
        return

    if report.skipped:
        result = "xfailed" if hasattr(report, "wasxfail") else "skipped"
    elif report.passed:
        result = "xpassed" if hasattr(report, "wasxfail") else "passed"
    else:
        result = "failed"

    ledger.record(
        marker.args[0], result,
        measured=getattr(item, "_geocore_measured", {}),
        test_id=item.nodeid,
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    ledger = getattr(session.config, "_geocore_ledger", None)
    if ledger is None or not ledger.rows:
        return
    path = ledger.write()
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        missing = registry.unclaimed(ledger.claimed_ids())
        reporter.write_sep("-", f"ledger: {len(ledger.rows)} row(s) -> "
                                f"{path.relative_to(session.config.rootpath)}")
        if missing:
            reporter.write_line(f"checks declared but not implemented: {', '.join(missing)}")
