"""The verification ledger: a machine-readable record of what every check measured.

PRD deliverable 2. `reports/verification.md` is GENERATED from this file and never hand-written,
because a hand-maintained verification record drifts from reality within a month and then reads
as authoritative while being wrong.

Two ledgers, and the difference is the point:

* **The ledger of record** (`reports/verification_ledger.json`) is tracked, and CI on a clean
  checkout is its ONLY writer. Writing it requires `--ledger-of-record`, and that flag refuses on
  a dirty tree. A result whose provenance is "someone's laptop, with uncommitted edits" is not
  evidence; it is an anecdote with a timestamp.
* **The local ledger** (`reports/local/verification_ledger.json`) is untracked and is written by
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

LEDGER_OF_RECORD: Final[str] = "reports/verification_ledger.json"
LOCAL_LEDGER: Final[str] = "reports/local/verification_ledger.json"
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
        # A check claimed by several tests keeps the worst outcome: one failing test means the
        # check did not pass, whatever the others did.
        existing = self._rows.get(check.id)
        if existing is not None and _severity(existing.result) >= _severity(result):
            merged = dict(existing.measured)
            merged.update(row.measured)
            row = dataclasses.replace(existing, measured=merged)
        self._rows[check.id] = row
        return row

    @property
    def rows(self) -> list[LedgerRow]:
        return [self._rows[cid] for cid in sorted(self._rows)]

    def claimed_ids(self) -> set[str]:
        return set(self._rows)

    def write(self) -> pathlib.Path:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "of_record": self.of_record,
            "written": self.provenance.timestamp,
            "git_sha": self.provenance.git_sha,
            "dirty": self.provenance.dirty,
            "unclaimed_checks": registry.unclaimed(self.claimed_ids()),
            "rows": [r.to_json() for r in self.rows],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        os.replace(tmp, self.path)
        return self.path


_SEVERITY = {"passed": 0, "xpassed": 1, "skipped": 2, "xfailed": 3, "failed": 4, "errored": 5}


def _severity(result: str) -> int:
    return _SEVERITY.get(result, 5)
