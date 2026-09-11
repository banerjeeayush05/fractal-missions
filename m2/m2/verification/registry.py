"""The 22 verification checks (PRD §8), with cadence tiers (§8.0) and citing milestones (§6).

A check can run in more than one tier: V14 is "reduced" in the fast tier and full-resolution 3D
in the gate tier. The ledger keeps one row per check, with a sub-record per tier.
"""

from __future__ import annotations

from dataclasses import dataclass

TIERS = ("fast", "nightly", "gate")


@dataclass(frozen=True)
class CheckSpec:
    id: str
    description: str
    section: str
    cadence: tuple[str, ...]  # tiers, first is the default for a test tagged with this ID
    milestones: tuple[str, ...]  # milestones whose gate cites it (§6)
    # Decision B16: suffixes keep IDs from colliding today; the mission field keeps them from
    # colliding when a later mission invents its own numbering.
    mission: str = "M2"
    # Decision B18: a check that compares against the fabricated coupon may not be certified from
    # a development config carrying a nominal (guessed) etch rate. None do yet.
    claims_coupon_agreement: bool = False


_SPECS = [
    CheckSpec("V1", "Isotropic etch: r(t) = r0 - R t (positive rate removes), rel. error < 1% at 200 steps", "§8.1", ("fast",), ("M2.1",)),
    CheckSpec("V2", "Plane translation at exactly V, error < 0.1%", "§8.1", ("fast",), ("M2.1",)),
    CheckSpec("V3", "Collimated aperture limit: sidewall within 0.5° of 90°", "§8.1", ("gate",), ("M2.7",)),
    CheckSpec("V4", "ON HOLD (decision §11): needs a steady facet angle derived for a monotone cos^p law", "§8.1", ("gate",), ("M2.7",)),
    CheckSpec("V5", "Method of manufactured solutions: observed order", "§8.2", ("nightly",), ("M2.3",)),
    CheckSpec("V6", "Spatial order (Godunov >= 0.9; WENO5 >= 4), L1 and Linf", "§8.2", ("nightly",), ("M2.3",)),
    CheckSpec("V7", "Temporal order (TVD-RK2 >= 1.9)", "§8.2", ("nightly",), ("M2.3",)),
    CheckSpec("V8", "Zalesak's disk: area preserved < 2%, notch survives", "§8.3", ("nightly",), ("M2.2",)),
    CheckSpec("V9", "Reversibility: symmetric-difference area < 3%", "§8.3", ("nightly",), ("M2.2",)),
    CheckSpec("V10", "Grid-orientation isotropy: (rmax - rmin)/rmean < 2%", "§8.3", ("nightly",), ("M2.2",)),
    CheckSpec("V11", "Reinit restores distance: max||grad phi| - 1| < 5% within 3 cells", "§8.3", ("fast",), ("M2.2",)),
    CheckSpec("V12", "Reinit does not move the interface: drift < 0.1% per cycle", "§8.3", ("fast",), ("M2.2",)),
    CheckSpec("V13", "Analytic trapezoid extraction: CD < 0.1 nm, SWA < 0.2°, sub-cell smooth", "§8.4", ("nightly",), ("M2.5",)),
    CheckSpec("V14", "Taylor remainder O(h^2): slope in [1.8, 2.2], >=5 decades, >=20 directions", "§8.5/§7.1",
              ("fast", "gate"), ("M2.3", "M2.4", "M2.5", "M2.7")),
    CheckSpec("V15", "Forward (jvp) vs reverse (vjp) directional derivative, 1e-10 relative", "§8.5", ("fast",), ("M2.3",)),
    CheckSpec("V16", "Dot-product test <w, J u> = <J^T w, u>", "§8.5", ("fast",), ("M2.3",)),
    CheckSpec("V17", "Checkpointed gradient equals unchecked to 1e-12", "§8.5", ("nightly",), ("M2.4",)),
    CheckSpec("V18", "Checkpoint recompute: RNG keys/samples bitwise identical, trajectory to 1e-12 relative", "§8.5/§7.6", ("nightly",), ("M2.4",)),
    CheckSpec("V19", "Corrupted-gradient canary: x1.05 on one component makes V14, V15, V16 fail",
              "§8.5/§7.2", ("fast",), ("M2.0",)),
    CheckSpec("V20", "CFL assertion fires with a clear message naming N", "§8.6", ("fast",), ("M2.1",)),
    CheckSpec("V21", "Golden 2D profile reproduced to 1e-10", "§8.6", ("fast",), ()),
    CheckSpec("V22", "ViennaPS cross-code: symmetric Hausdorff < 2 nm (separate process)", "§8.8", ("gate",), ("M2.7",)),
    CheckSpec("V1a", "Directional-law sign check: trench floor recedes from the plasma, overhang does not move",
              "decision §1", ("fast",), ("M2.1",)),
    CheckSpec("V14a", "Analytic sensitivity, isotropic etch: dr/dR = -T", "decision §5", ("fast",), ("M2.3",)),
    CheckSpec("V14b", "Analytic sensitivity, plane translation: dz/dR = -T (depth: +T)", "decision §5", ("fast",), ("M2.3",)),
    CheckSpec("V14c", "Analytic sensitivity, tilted plane under the directional law: exact d/dv0 and d/dp",
              "decision §5", ("fast",), ("M2.3",)),
]

CHECKS: dict[str, CheckSpec] = {s.id: s for s in _SPECS}

# V1..V22 from the PRD, plus the suffixed IDs added by the 2026-09-11 decisions.
# V23..V40 belong to M3 (decision §5) and must never be used here.
assert [c for c in CHECKS if c[-1].isdigit()] == [f"V{i}" for i in range(1, 23)]
assert {"V1a", "V14a", "V14b", "V14c"} <= set(CHECKS)
assert not any(c[1:].split("a")[0].isdigit() and int(c[1:].rstrip("abc")) > 22 for c in CHECKS)
assert all(t in TIERS for s in _SPECS for t in s.cadence)
