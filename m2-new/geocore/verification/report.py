"""Generated records. PRD deliverables 3, 4 and 7.

Three documents, three readers:

| document | reader | question it answers |
|---|---|---|
| verification | a sceptical customer, M11 | "show me the gradient is verified, not assumed" |
| gradient verification | the M8 owner | "which objectives were checked, when, at what slope" |
| simplifications | M3 and M5 owners | "what does a green suite NOT cover" |

**Generated, never hand-written** (PRD §8, §11). A hand-maintained verification record drifts from
reality inside a month and then reads as authoritative while being wrong. These functions return
markdown built from the ledger and the check registry; nothing is stored in the repo except the ledger
itself.

    uv run python -m geocore.verification.report                  # all three, to stdout
    uv run python -m geocore.verification.report verification     # one of them

**Provenance is stated, not implied.** A record built from a local ledger on a dirty tree says so in
its first lines. Only CI on a clean checkout may write the ledger of record, and until this project has
CI, every generated record carries that caveat (OPEN_QUESTIONS S13.4).
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys
from typing import Any

from geocore.verification import registry
from geocore.verification.ledger import LEDGER_OF_RECORD, LOCAL_LEDGER

GRADIENT_CHECKS = ("V14", "V14a", "V14b", "V14c", "V15", "V16", "V17", "V18", "V19")

# PRD deliverable 7. Each entry: what is simplified, why it is acceptable for M2, where it is recorded.
SIMPLIFICATIONS: tuple[tuple[str, str, str], ...] = (
    ("Velocity is prescribed and analytic",
     "M2 verifies that the geometry engine solves its equations correctly, not that they are the "
     "right equations. Flux, shadowing, re-emission and redeposition are M3.",
     "PRD §3; geocore/velocity.py"),
    ("The mask etches at exactly zero rate",
     "Infinite selectivity. The model therefore predicts zero mask loss, so any measured loss is "
     "direct evidence the multiplier is wrong and sets the real selectivity.",
     "PRD §5.6; geocore/materials.py"),
    ("Material boundaries are mollified over w_mat = 2 cells",
     "A sharp boundary makes the etch rate a step function of position and destroys the gradient "
     "with respect to arrival time. The smoothing is numerical: d(depth)/d(film rate) moves about "
     "5 % between 1-cell and 2-cell widths.",
     "OPEN_QUESTIONS S17.1; tests/test_w_mat_study.py"),
    ("Cumulative reinitialisation drift is accepted, not fixed",
     "Reinitialisation alone shifts an etched disk's radius about 2 % over 40 cycles under Godunov. "
     "Accepted as a known discrepancy pending a WENO5 re-measurement.",
     "OPEN_QUESTIONS S12.1, S12.3; DECISIONS.md"),
    ("Godunov, first order in space, is the default scheme",
     "WENO5 exists behind a flag and cuts V1's error about 90x, but triples the residual factor k "
     "and doubles the adjoint ratio. Making it the default is an open owner decision (J5).",
     "OPEN_QUESTIONS S14.3; geocore/stencils.py"),
    ("Extraction handles 2D profiles and 3D line features only",
     "Holes and other genuinely 3D features are not handled; a 3D line feature reduces exactly by "
     "averaging along its length.",
     "geocore/extraction.py"),
    ("Dense storage, masked compute",
     "Memory scales with volume and compute is wasted far from the interface. Static shapes are what "
     "let plain JAX reverse mode work; narrow-band sparsity would reintroduce an active set that "
     "jumps with the parameters.",
     "PRD §4"),
    ("Endpoint-only output from the differentiated solve",
     "No trajectory frames: at N = 625 on the S03 grid they would be 13.5 GB the adjoint never reads.",
     "geocore/solver.py"),
    ("The collimated-aperture limit is not reproduced",
     "A local cos^p law cannot hold a flat floor beside a mask corner at p = 64; a real collimated "
     "etch stays flat because the flux is uniform and non-local, which is M3.",
     "OPEN_QUESTIONS S18.1; tests/test_aperture.py"),
)


def load_ledger(root: pathlib.Path | str = ".") -> dict[str, Any]:
    root = pathlib.Path(root)
    for name, of_record in ((LEDGER_OF_RECORD, True), (LOCAL_LEDGER, False)):
        path = root / name
        if path.exists():
            data = json.loads(path.read_text())
            data["_path"], data["_of_record"] = str(path), of_record
            return data
    raise FileNotFoundError("no ledger found; run the suite first")


def _provenance(ledger: dict[str, Any]) -> list[str]:
    sha = (ledger.get("git_sha") or "unknown")[:12]
    dirty = ledger.get("dirty", True)
    caveat = ("**This record is NOT of record.** It was generated from a local ledger"
              + (" on a DIRTY tree" if dirty else "")
              + ". Only CI on a clean checkout may write the ledger of record, and this project has "
                "no CI yet (OPEN_QUESTIONS S13.4).") if not ledger["_of_record"] or dirty else \
             "Generated from the ledger of record."
    return [f"Generated {dt.date.today().isoformat()} from `{ledger['_path']}` at commit `{sha}`.",
            "", caveat, ""]


def verification_markdown(ledger: dict[str, Any]) -> str:
    rows = {row["check_id"]: row for row in ledger["rows"]}
    lines = ["# Verification record — M2", ""] + _provenance(ledger)
    lines += ["## Checks", "",
              "| check | tier | result | measured | description |", "|---|---|---|---|---|"]
    for check_id in sorted(registry.CHECKS):
        check = registry.CHECKS[check_id]
        row = rows.get(check_id)
        if check.status is registry.Status.ON_HOLD:
            result = "on hold"
        elif row is None:
            result = "**not implemented**"
        else:
            result = row["result"]
        measured = ""
        if row and row["measured"]:
            measured = "; ".join(f"{k} = {_fmt(v)}" for k, v in list(row["measured"].items())[:3])
        lines.append(f"| {check_id} | {check.cadence.value} | {result} | {measured} | "
                     f"{check.description} |")

    failing = [r for r in ledger["rows"] if r["result"] in ("failed", "xfailed", "errored")]
    lines += ["", "## Open discrepancies", ""]
    lines += (["Every check that did not pass, with what it measured. None is tolerated silently: each "
               "is `xfail(strict=True)`, so if one starts passing the suite fails and the discrepancy "
               "cannot become folklore.", ""]
              + [f"- **{r['check_id']}** ({r['result']}): "
                 + "; ".join(f"{k} = {_fmt(v)}" for k, v in list(r["measured"].items())[:4])
                 for r in failing]
              if failing else ["None."])

    lines += ["", "## What M2 does not verify", "",
              "PRD §8.9. M2 verifies that the geometry engine solves its equations correctly. It says "
              "nothing about whether those are the right equations, because M2's velocity is "
              "prescribed and analytic.", "",
              "- No comparison against experimental data.",
              "- No ARDE, bowing, microloading or faceting from real flux. All require M3.",
              "- Grid anisotropy is measured (V10), not eliminated.",
              "- The full list of simplifications is the third document generated here.", ""]
    missing = ledger.get("unclaimed_checks") or []
    if missing:
        lines += [f"Declared but not implemented: {', '.join(missing)}.", ""]
    return "\n".join(lines)


def gradient_markdown(ledger: dict[str, Any]) -> str:
    rows = {row["check_id"]: row for row in ledger["rows"]}
    lines = ["# Gradient verification — M2", ""] + _provenance(ledger)
    lines += ["A gradient that is wrong but plausible is worse than no gradient. These are the checks "
              "that stand between this solver and that failure, with what they last measured.", "",
              "| check | result | measured |", "|---|---|---|"]
    for check_id in GRADIENT_CHECKS:
        row = rows.get(check_id)
        if row is None:
            lines.append(f"| {check_id} | **not implemented** | |")
            continue
        measured = "; ".join(f"{k} = {_fmt(v)}" for k, v in row["measured"].items()
                             if not isinstance(v, (dict, list)))
        lines.append(f"| {check_id} | {row['result']} | {measured[:300]} |")
    lines += ["", "V15 and V16 are TRANSPOSE checks, not derivative checks: JAX builds reverse mode by "
              "transposing the forward-mode rules, so both share them. Only V14 compares a derivative "
              "against the function itself, which is why V14a-c carry closed-form answers alongside "
              "it (decision §5, finding A9).", ""]
    return "\n".join(lines)


def simplifications_markdown() -> str:
    lines = ["# Simplifications — M2", "",
             "Every simplification in this build, with why it is acceptable here and where it is "
             "recorded. PRD deliverable 7. A green suite means the solver does what it says; this "
             "list is what it does not attempt.", "",
             "| simplification | why it is acceptable for M2 | recorded in |", "|---|---|---|"]
    lines += [f"| {what} | {why} | {where} |" for what, why, where in SIMPLIFICATIONS]
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, dict)):
        return f"({len(value)} entries)"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    which = (argv or sys.argv[1:] or ["all"])[0]
    ledger = load_ledger()
    parts = {"verification": lambda: verification_markdown(ledger),
             "gradient": lambda: gradient_markdown(ledger),
             "simplifications": simplifications_markdown}
    if which == "all":
        print("\n\n---\n\n".join(build() for build in parts.values()))
    elif which in parts:
        print(parts[which]())
    else:
        print(f"usage: python -m geocore.verification.report [all|{'|'.join(parts)}]")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
