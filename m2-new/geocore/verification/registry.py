"""The catalogue of verification checks. PRD §8.

Every check has an ID, and every ID is declared HERE before any test may claim it. A test says
`@pytest.mark.check("V14")` and the plugin refuses at collection time if V14 is not in this file.

That indirection buys three things:

* **A check cannot be invented by writing a test.** New IDs are a deliberate edit to a catalogue,
  not a side effect of naming a function.
* **Cadence lives in one place.** PRD §8.0 assigns every check to fast / nightly / gate. The
  plugin reads the tier from here and applies the pytest marker automatically, so a check cannot
  drift into the wrong tier by someone forgetting a decorator.
* **Coverage is measurable.** `unclaimed()` lists declared checks that no test implements, which
  is the honest answer to "what does the suite actually verify" — and it is a number that should
  shrink monotonically.

Tolerances recorded here are DESCRIPTIVE, for the ledger and for a human reading the record. The
authoritative numbers live in the code that applies them (`gradcheck.py` for V14/V15/V16). A
tolerance duplicated as a live value in two files is a tolerance that will drift.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Final


class Cadence(enum.Enum):
    """PRD §8.0. The fast tier is non-negotiable: if it exceeds three minutes people stop running
    it, and a verification suite nobody runs is worse than none because it produces false
    confidence rather than no confidence."""

    FAST = "fast"        # every commit, in CI, < 3 min, 2D only, coarse
    NIGHTLY = "nightly"  # scheduled, on main, < 60 min, 2D full + small 3D
    GATE = "gate"        # at the milestone that cites it, hours


class Status(enum.Enum):
    ACTIVE = "active"
    ON_HOLD = "on_hold"  # declared, deliberately not implemented; see the note


class UnknownCheck(KeyError):
    """Raised when a test claims an ID that is not declared here."""


@dataclasses.dataclass(frozen=True)
class Check:
    id: str
    description: str
    cadence: Cadence
    prd: str
    tolerance: str
    status: Status = Status.ACTIVE
    note: str = ""


def _c(*args, **kwargs) -> tuple[str, Check]:
    check = Check(*args, **kwargs)
    return check.id, check


CHECKS: Final[dict[str, Check]] = dict([
    # ---------------------------------------------------- §8.1 analytic solutions
    _c("V1", "Isotropic etch: a solid disk/sphere under constant R shrinks as r(t) = r0 - R*t",
       Cadence.FAST, "§8.1", "relative radius error < 1% at 200 steps"),
    _c("V1a", "Sign and orientation: a trench floor recedes from the plasma; an overhang "
              "underside does not move",
       Cadence.FAST, "§5.1", "qualitative — direction of motion",
       note="The ONLY guard against a sign inversion. V14-V16 all pass on a sign-flipped world."),
    _c("V2", "Plane translation: a flat interface under constant R translates at exactly R with "
             "no distortion",
       Cadence.FAST, "§8.1", "error < 0.1%"),
    _c("V3", "Collimated aperture limit: a perfectly collimated etch (p = 64) reproduces the "
             "aperture shape, no bow, no faceting",
       Cadence.GATE, "§8.1", "sidewall angle within 0.5 deg of 90"),
    _c("V4", "Facet angle", Cadence.GATE, "§8.1", "n/a", status=Status.ON_HOLD,
       note="ON HOLD (decision §11). A monotone cos^p law peaks at normal incidence; classical "
            "facet-angle results assume a rate peaking off-normal, which needs M5 yield curves. "
            "V14c covers what V4 was really guarding: a wrong normal-vector convention."),
    # ---------------------------------------------------- §8.2 convergence order
    _c("V5", "Method of manufactured solutions: add a source term, verify the observed order",
       Cadence.NIGHTLY, "§8.2", "observed order matches the scheme"),
    _c("V6", "Spatial order: refine dx over >= 4 levels against V1's analytic solution",
       Cadence.NIGHTLY, "§8.2", "godunov >= 0.9; weno5 >= 4"),
    _c("V7", "Temporal order: fix dx, refine dt", Cadence.NIGHTLY, "§8.2", "TVD-RK2 >= 1.9"),
    # ---------------------------------------------------- §8.3 invariants
    _c("V8", "Zalesak's disk: notched disk in rigid rotation, one revolution",
       Cadence.NIGHTLY, "§8.3", "area preserved < 2%; the notch survives"),
    _c("V9", "Reversibility: advect N steps under V, then N under -V",
       Cadence.NIGHTLY, "§8.3", "symmetric-difference area < 3%"),
    _c("V10", "Grid-orientation isotropy: V1 with the initial condition rotated 45 deg",
       Cadence.NIGHTLY, "§8.3", "(r_max - r_min)/r_mean < 2%"),
    _c("V11", "Reinitialisation restores distance", Cadence.FAST, "§8.3",
       "max | |grad phi| - 1 | < 5% within 3 cells of the interface"),
    _c("V12", "Reinitialisation does not move the interface", Cadence.FAST, "§8.3",
       "enclosed area drift < 0.1% per cycle",
       note="The most commonly missed check in level-set codes. Reinit that quietly shifts the "
            "zero level set produces an etch-rate bias that looks like physics, calibrates away "
            "into the closure parameters, and then fails to transfer."),
    _c("V12a", "Accumulated reinitialisation drift over a full run: interface displacement "
               "attributable to reinitialisation alone",
       Cadence.FAST, "§8.3 (amended 2026-09-16)",
       "mean over directions < 1 % of the final radius on V1's geometry (S12.3, provisional)",
       note="Added because V12 bounds ONE cycle and cannot see drift that accumulates over the "
            "tens to hundreds of cycles in a run (finding S12.1)."),
    # ---------------------------------------------------- §8.4 extraction
    _c("V13", "Analytic geometry: an exact trapezoid of known CD, depth and sidewall angle, "
              "swept by sub-cell offsets",
       Cadence.NIGHTLY, "§8.4", "CD error < 0.1 nm; sidewall angle < 0.2 deg; CD varies smoothly"),
    # ---------------------------------------------------- §8.5 gradients
    _c("V14", "Taylor remainder. THE PRIMARY GATE", Cadence.FAST, "§7.1, §8.5",
       "per direction over >= 20 directions: < 3 points in the window FAILS; slope < 1.8 FAILS; "
       "[1.8, 2.2] passes clean_quadratic; > 2.2 passes degenerate_direction"),
    _c("V14a", "Analytic sensitivity, V1 isotropic etch: dr/dR = -T",
       Cadence.FAST, "§8.5", "relative error < 1e-2 (S13.1, owner accepted 2026-09-16)"),
    _c("V14b", "Analytic sensitivity, V2 plane translation: d(depth)/dR = +T",
       Cadence.FAST, "§8.5", "relative error < 1e-2 (S13.1, owner accepted 2026-09-16)",
       note="The magnitude is T either way; the SIGN follows from §5.1, which is the point."),
    _c("V14c", "Analytic sensitivity, tilted plane under the directional law: derivatives in "
               "v0 and p are exact",
       Cadence.FAST, "§8.5", "relative error < 1e-2 (S13.1, owner accepted 2026-09-16)",
       note="Also guards the normal-vector convention V4 was meant to catch."),
    _c("V15", "Forward mode versus reverse mode: jax.jvp against jax.vjp",
       Cadence.FAST, "§8.5", "agreement < 1e-10 relative in fp64",
       note="A TRANSPOSE check, not a derivative check (decision §5, finding A9). JAX builds "
            "reverse mode by linearising with the JVP rules and transposing, so both share those "
            "rules. Only V14 compares a derivative against the function."),
    _c("V16", "Dot-product (transpose) test: <w, Ju> = <J^T w, u>",
       Cadence.FAST, "§8.5", "agreement < 1e-10 relative in fp64",
       note="Same caveat as V15. Becomes essential the moment a custom VJP rule exists."),
    _c("V17", "Checkpointed gradient equals unchecked gradient",
       Cadence.NIGHTLY, "§8.5", "agreement < 1e-12"),
    _c("V18", "Checkpoint recompute: RNG keys and sampled values bitwise identical; trajectory "
              "to 1e-12 relative",
       Cadence.NIGHTLY, "§8.5", "bitwise on keys; 1e-12 relative on the trajectory"),
    _c("V19", "Corrupted-gradient canary: a 5% error in one component must make V14, V15 and "
              "V16 all FAIL, with an unmutated control passing",
       Cadence.FAST, "§7.2, §8.5", "all three fail on the mutant; all three pass on the control",
       note="A verification suite that has never been shown to fail has not been verified."),
    # ---------------------------------------------------- §8.6 structural
    _c("V20", "The CFL assertion fires: set N too small and the run must abort naming N",
       Cadence.FAST, "§8.6", "aborts with a message naming N"),
    _c("V21", "Golden profile regression: a frozen reference 2D profile",
       Cadence.FAST, "§8.6", "reproduced to 1e-10"),
    # ---------------------------------------------------- §8.8 cross-code
    _c("V22", "ViennaPS cross-code comparison by symmetric Hausdorff distance",
       Cadence.GATE, "§8.8", "< 2 nm; any excess explained or logged as an open discrepancy",
       note="Separate process, VTK exchange, GPL clean room. Do NOT run before V6 and V7: "
            "without convergence orders you cannot tell a bug from a grid difference."),
])

# PRD §8.5: "IDs V23 and above belong to M3; never use them here."
M3_ID_FLOOR: Final[int] = 23


def get(check_id: str) -> Check:
    """Look up a check, with an error that says what to do about it."""
    try:
        return CHECKS[check_id]
    except KeyError:
        number = "".join(ch for ch in check_id if ch.isdigit())
        if number and int(number) >= M3_ID_FLOOR:
            raise UnknownCheck(
                f"{check_id} is in M3's range (V{M3_ID_FLOOR}+). M2 must not use it."
            ) from None
        raise UnknownCheck(
            f"{check_id} is not a declared check. Declare it in registry.py before claiming it; "
            f"known IDs: {sorted(CHECKS)}"
        ) from None


def ids_for(cadence: Cadence) -> list[str]:
    return sorted(cid for cid, c in CHECKS.items() if c.cadence is cadence)


def active_ids() -> list[str]:
    return sorted(cid for cid, c in CHECKS.items() if c.status is Status.ACTIVE)


def unclaimed(claimed: set[str]) -> list[str]:
    """Declared, active, and implemented by no test. The honest coverage number."""
    return sorted(set(active_ids()) - claimed)
