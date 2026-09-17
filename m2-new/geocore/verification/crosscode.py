"""V22 — cross-code comparison against ViennaPS. PRD §8.8.

**Protocol, frozen here so a disagreement can be attributed.** Two structures, one directional and one
isotropic, run in both codes with identical geometry, law and final time; each writes VTK; the contours
are compared offline by symmetric Hausdorff distance. Target under 2 nm, and any excess is explained or
logged as an open discrepancy -- never silently accepted.

**Constraints, non-negotiable (PRD §8.8, repo CLAUDE.md):**

* Separate process only. ViennaPS runs from its own repository under its own licence.
* ViennaPS is GPL-3.0. Do not link, import, vendor or copy it. Nothing in `geocore` imports it.
* Clean room: if you read its source, write a natural-language note; never transcribe.
* V6 and V7 run FIRST. Without convergence orders a discrepancy cannot be told from a grid difference.
  Both have run in this build: observed order 1.02 (spatial) and 1.97 (temporal).

**Conventions the other side must match**, because a mismatch here looks exactly like a solver
disagreement:

| quantity | this build |
|---|---|
| sign of phi | negative inside solid |
| units | nanometres, seconds |
| vertical axis | axis 0, increasing toward the plasma |
| rate | nm/s, positive removes material |
| directional law | `R = v0 * max(0, n . z_hat)^p`, p finite and stated |
| grid | cell-centred; VTK origin is the first cell CENTRE |
| final time | `T = depth / rate`, stated explicitly in the case |
| output | phi on its grid, and the zero contour as points |

If ViennaPS cannot express the same law, the mapping is stated in the case before the run, or a real
disagreement and a definitional one cannot be told apart.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class CrossCodeCase:
    """One frozen comparison case. Both sides run exactly this."""

    name: str
    law: str                      # "isotropic" or "directional"
    grid_shape: tuple[int, ...]
    spacing_nm: float
    film_top_nm: float
    mask_thickness_nm: float
    opening_nm: float
    start_depth_nm: float
    v0_nm_per_s: float
    p: float | None
    final_time_s: float
    tolerance_nm: float = 2.0


CASES = (
    CrossCodeCase("X1-directional", "directional", (120, 48), 5.0, 350.0, 50.0, 100.0, 20.0,
                  2.0, 2.0, 100.0),
    CrossCodeCase("X2-isotropic", "isotropic", (120, 48), 5.0, 350.0, 50.0, 100.0, 20.0,
                  2.0, None, 50.0),
)


def symmetric_hausdorff_nm(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """max over each set of the distance to the nearest point of the other set, in nm.

    Symmetric on purpose: the one-sided distance misses a feature present in one contour and absent
    from the other, which is exactly the disagreement worth catching.
    """
    from scipy.spatial import cKDTree

    if len(points_a) == 0 or len(points_b) == 0:
        raise ValueError("both contours must be non-empty")
    forward = cKDTree(points_b).query(points_a)[0].max()
    backward = cKDTree(points_a).query(points_b)[0].max()
    return float(max(forward, backward))


def compare_contours(ours: np.ndarray, theirs: np.ndarray, case: CrossCodeCase) -> dict:
    """The V22 verdict for one case, with the numbers behind it."""
    distance = symmetric_hausdorff_nm(ours, theirs)
    return {"case": case.name, "hausdorff_nm": distance, "tolerance_nm": case.tolerance_nm,
            "passed": distance < case.tolerance_nm, "points_ours": int(len(ours)),
            "points_theirs": int(len(theirs))}
