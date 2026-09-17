"""The verification ledger: a machine-readable record of what every check measured.

PRD deliverable 2. The customer-facing `verification.md` is GENERATED from this file, never hand-written,
because a hand-maintained verification record drifts from reality within a month and then reads
as authoritative while being wrong.

Two ledgers, and the difference is the point:

* **The ledger of record** (`verification_ledger.json`) is tracked, and CI on a clean
  checkout is its ONLY writer. Writing it requires `--ledger-of-record`, and that flag refuses on
  a dirty tree. A result whose provenance is "someone's laptop, with uncommitted edits" is not
  evidence; it is an anecdote with a timestamp.
* **The local ledger** (`.verification_ledger_local.json`) is untracked and is written by
  every ordinary run. It exists so a developer can see measured values while working, without
  being able to promote them.

Every row carries the git SHA and the dirty flag alongside the measurement, so a row can always
be traced back to the exact tree that produced it.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import pathlib
import subprocess
from typing import Any, Final

from geocore.verification import registry

# PRD §8 originally named a `reports/` path. This build keeps no `reports/` directory (see
# DECISIONS.md, 2026-09-17), so the ledger sits at the project root: the tracked one under its PRD
# name, the local one hidden and gitignored.
LEDGER_OF_RECORD: Final[str] = "verification_ledger.json"
LOCAL_LEDGER: Final[str] = ".verification_ledger_local.json"
SCHEMA_VERSION: Final[int] = 1


class LedgerRefusal(RuntimeError):
    """Raised when a write to the ledger of record is not permitted."""


# ------------------------------------------------------------------------------------ git


def _git(*args: str, root: pathlib.Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@dataclasses.dataclass(frozen=True)
class Provenance:
    """Where a measurement came from. Recorded per row, not per file, because a long run can
    span a commit."""

    git_sha: str | None
    dirty: bool
    timestamp: str

    @classmethod
    def capture(cls, root: pathlib.Path) -> "Provenance":
        sha = _git("rev-parse", "HEAD", root=root)
        status = _git("status", "--porcelain", root=root)
        return cls(
            git_sha=sha,
            dirty=bool(status) if status is not None else True,
            timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        )


# ------------------------------------------------------------------------------------ rows


@dataclasses.dataclass(frozen=True)
class LedgerRow:
    check_id: str
    description: str
    cadence: str
    result: str                 # passed | failed | xfailed | xpassed | skipped | errored
    measured: dict[str, Any]
    test_id: str
    provenance: Provenance

    def to_json(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        row["provenance"] = dataclasses.asdict(self.provenance)
        return row


# ---------------------------------------------------------------------------------- ledger


class Ledger:
    """Accumulates rows during a run and writes them once at the end.

    Written in one atomic replace rather than appended per check: a run killed halfway should
    leave the previous ledger intact rather than a half-updated one that still parses.
    """

    def __init__(self, root: pathlib.Path, of_record: bool = False) -> None:
        self.root = pathlib.Path(root)
        self.of_record = of_record
        self.provenance = Provenance.capture(self.root)
        self._rows: dict[str, LedgerRow] = {}
        if of_record:
            self._refuse_if_not_publishable()

    @property
    def path(self) -> pathlib.Path:
        return self.root / (LEDGER_OF_RECORD if self.of_record else LOCAL_LEDGER)

    def _refuse_if_not_publishable(self) -> None:
        if self.provenance.git_sha is None:
            raise LedgerRefusal(
                "refusing to write the ledger of record: not a git checkout, so a row could not "
                "name the tree that produced it"
            )
        if self.provenance.dirty:
            raise LedgerRefusal(
                f"refusing to write the ledger of record from a DIRTY tree "
                f"(sha {self.provenance.git_sha[:12]}). The ledger of record is written by CI on "
                f"a clean checkout; local runs write {LOCAL_LEDGER}. Commit, or drop "
                f"--ledger-of-record."
            )

    def record(self, check_id: str, result: str, measured: dict[str, Any] | None = None,
               test_id: str = "") -> LedgerRow:
        """Add or replace the row for a check. Unknown IDs raise — see registry.get."""
        check = registry.get(check_id)
        row = LedgerRow(
            check_id=check.id,
            description=check.description,
            cadence=check.cadence.value,
            result=result,
            measured=dict(measured or {}),
            test_id=test_id,
            provenance=self.provenance,
        )
        # A check claimed by several tests keeps the WORST outcome -- one failing test means the check
        # did not pass, whatever the others did -- and the UNION of what they measured, whichever way
        # round they ran. Keeping only the worst row's measurements would silently drop the numbers
        # from the passing tests, which are the ones a later reader compares against.
        existing = self._rows.get(check.id)
        if existing is not None:
            measured = dict(existing.measured)
            measured.update(row.measured)
            worst = existing if _severity(existing.result) >= _severity(result) else row
            row = dataclasses.replace(worst, measured=measured)
        self._rows[check.id] = row
        return row

    @property
    def rows(self) -> list[LedgerRow]:
        return [self._rows[cid] for cid in sorted(self._rows)]

    def claimed_ids(self) -> set[str]:
        return set(self._rows)

    def _existing_rows(self) -> dict[str, dict]:
        """Rows already on disk, keyed by check ID.

        Finding S13.4: a run only knows about the checks it ran. Writing just those would delete the
        nightly rows a full run wrote earlier, and a record generated from that file would be partial
        while looking complete. Rows from this run replace their namesakes; the rest are carried
        forward, each keeping the provenance of the run that produced it.
        """
        if not self.path.exists():
            return {}
        try:
            previous = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return {row["check_id"]: row for row in previous.get("rows", [])
                if isinstance(row, dict) and "check_id" in row}

    def write(self) -> pathlib.Path:
        merged = self._existing_rows()
        merged.update({row.check_id: row.to_json() for row in self.rows})
        rows = [merged[cid] for cid in sorted(merged)]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "of_record": self.of_record,
            "written": self.provenance.timestamp,
            "git_sha": self.provenance.git_sha,
            "dirty": self.provenance.dirty,
            "checks_this_run": sorted(self.claimed_ids()),
            "unclaimed_checks": registry.unclaimed(set(merged)),
            "rows": rows,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        os.replace(tmp, self.path)
        return self.path


_SEVERITY = {"passed": 0, "xpassed": 1, "skipped": 2, "xfailed": 3, "failed": 4, "errored": 5}


def _severity(result: str) -> int:
    return _SEVERITY.get(result, 5)
