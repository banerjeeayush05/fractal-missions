"""Verification ledger: ``reports/verification_ledger.json`` (PRD §8, §9 item 2).

Written by the checks themselves, through the pytest plugin, never by hand. One row per check ID,
with one sub-record per cadence tier (OPEN_QUESTIONS B7). All 22 IDs are always present, so an
unimplemented check shows as ``not_implemented`` instead of silently missing.

Row result:
- ``not_implemented``: no tier has ever run;
- ``fail``: the latest run of any tier failed;
- ``incomplete``: every run so far passed, but some tier has never run;
- ``pass``: the latest run of every tier passed.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import json
import math
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from m2.verification.registry import CHECKS

SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Decision §13: CI, on a clean checkout, is the only writer of the ledger of record. Local runs
# write an untracked ledger, so a dirty working tree can never certify a check.
RECORD_LEDGER_PATH = PROJECT_ROOT / "reports" / "verification_ledger.json"
LOCAL_LEDGER_PATH = PROJECT_ROOT / "reports" / "local" / "verification_ledger.json"
DEFAULT_LEDGER_PATH = LOCAL_LEDGER_PATH


def empty_ledger() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "note": "Written by the checks (m2/verification/pytest_plugin.py). Do not hand-edit. PRD §8.",
        "checks": {cid: _empty_row(cid) for cid in CHECKS},
    }


def _empty_row(cid: str) -> dict[str, Any]:
    spec = CHECKS[cid]
    return {
        "id": cid,
        "mission": spec.mission,
        "description": spec.description,
        "section": spec.section,
        "cadence": list(spec.cadence),
        "milestones": list(spec.milestones),
        "claims_coupon_agreement": spec.claims_coupon_agreement,
        "result": "not_implemented",
        "runs": {tier: None for tier in spec.cadence},
    }


def load_ledger(path: Path = DEFAULT_LEDGER_PATH) -> dict[str, Any]:
    """Load the ledger, reconciling static fields with the registry. Unknown IDs are an error."""
    if not Path(path).exists():
        return empty_ledger()
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"ledger schema_version {data.get('schema_version')} != {SCHEMA_VERSION}")
    unknown = set(data.get("checks", {})) - set(CHECKS)
    if unknown:
        raise ValueError(f"ledger contains check IDs not in the registry: {sorted(unknown)}")
    fresh = empty_ledger()
    for cid, row in fresh["checks"].items():
        old_runs = data["checks"].get(cid, {}).get("runs", {})
        row["runs"] = {tier: old_runs.get(tier) for tier in row["cadence"]}
        row["result"] = overall_result(row["runs"])
    return fresh


def overall_result(runs: Mapping[str, Mapping[str, Any] | None]) -> str:
    results = [r["result"] for r in runs.values() if r is not None]
    if not results:
        return "not_implemented"
    if any(r != "pass" for r in results):
        return "fail"
    return "pass" if len(results) == len(runs) else "incomplete"


def record_run(
    check_id: str,
    tier: str,
    *,
    passed: bool,
    measured: Mapping[str, Any],
    tests: list[str],
    path: Path = DEFAULT_LEDGER_PATH,
    detail: str = "",
) -> dict[str, Any]:
    """Write one tier's run into a check's row (locked, atomic). Returns the updated row."""
    if check_id not in CHECKS:
        raise KeyError(f"unknown check ID {check_id!r}")
    if tier not in CHECKS[check_id].cadence:
        raise ValueError(f"{check_id} does not run in tier {tier!r} (cadence {CHECKS[check_id].cadence})")
    sha, dirty = git_state()
    run = {
        "result": "pass" if passed else "fail",
        "last_run": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "measured": _jsonable(measured),
        "tests": sorted(tests),
        "detail": detail,
        "environment": environment(),
    }
    path = Path(path)
    with _locked(path):
        ledger = load_ledger(path)
        row = ledger["checks"][check_id]
        row["runs"][tier] = run
        row["result"] = overall_result(row["runs"])
        _atomic_write(path, ledger)
    return row


def git_state(repo_dir: Path = PROJECT_ROOT) -> tuple[str, bool | None]:
    """HEAD SHA, and whether the tree (excluding reports/) differs from it."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True,
                             text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", ".", ":(exclude)reports"],
                                cwd=repo_dir, capture_output=True, text=True, check=True).stdout
        return sha, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def environment() -> dict[str, Any]:
    import jax
    import jaxlib

    device = jax.devices()[0]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "backend": jax.default_backend(),
        # Which accelerator produced the row: V18 and V21 can differ between GPU models and
        # driver versions, so a row without this cannot be compared across machines.
        "device_kind": getattr(device, "device_kind", str(device)),
        "device_count": jax.device_count(),
        "x64": bool(jax.config.jax_enable_x64),
    }


def _jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "tolist"):  # numpy / jax arrays and scalars
        return _jsonable(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return repr(x)  # JSON has no NaN/inf
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return repr(x)


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(path.suffix + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _atomic_write(path: Path, data: Mapping[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def write_empty_ledger(path: Path = DEFAULT_LEDGER_PATH) -> None:
    with _locked(Path(path)):
        _atomic_write(Path(path), load_ledger(path) if Path(path).exists() else empty_ledger())
