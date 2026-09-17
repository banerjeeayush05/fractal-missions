"""The V22 harness: VTK exchange and the comparison tool, verified without ViennaPS.

V22 itself cannot run here -- it needs ViennaPS on a Linux box, in its own repository, under its own
licence. What CAN be verified here is everything on our side of the exchange: that a level set survives
a round trip through VTK, that the contour extraction is sub-cell, and that the comparison tool measures
what it claims. Doing that first means a real disagreement later is about the two solvers.
"""

import numpy as np
import pytest

from geocore.initial import plane, sphere
from geocore.schema import Grid
from geocore.verification.crosscode import CASES, compare_contours, symmetric_hausdorff_nm
from geocore.vtk_io import contour_points, read_level_set, write_contour, write_level_set

GRID = Grid((48, 32), 5.0, (False, True))


def test_a_level_set_survives_a_round_trip_through_vtk(tmp_path):
    phi = sphere(GRID, (120.0, 80.0), 50.0)
    path = write_level_set(tmp_path / "phi.vti", phi, GRID)
    restored, spacing, origin = read_level_set(path)
    assert spacing == GRID.spacing_nm
    assert origin[0] == pytest.approx(0.5 * GRID.spacing_nm), "origin is the first cell CENTRE"
    assert np.max(np.abs(restored - np.asarray(phi))) == 0.0


@pytest.mark.parametrize("height", [117.5, 119.0, 121.3])
def test_contour_points_are_sub_cell_and_on_the_surface(height):
    """Every contour point sits on the plane, not on the nearer grid line. 117.5 nm is a cell centre,
    where phi is exactly zero and a strict sign change finds nothing."""
    points = contour_points(plane(GRID, height), GRID)
    assert len(points) > 20
    assert np.max(np.abs(points[:, 0] - height)) < 1e-9


def test_the_comparison_tool_measures_a_known_offset():
    """Two planes 3 nm apart: the symmetric Hausdorff distance must read exactly 3 nm.

    Planes, not spheres. Contour points sit on grid edges, so on a curved surface the nearest point of
    the other set is up to half a cell off laterally: two spheres 3 nm apart in radius measure 4.4 nm at
    dx = 5 nm, and that gap is the sampling, not the tool.
    """
    a = contour_points(plane(GRID, 100.0), GRID)
    b = contour_points(plane(GRID, 103.0), GRID)
    assert symmetric_hausdorff_nm(a, b) == pytest.approx(3.0, abs=1e-9)


def test_point_sampling_inflates_the_distance_on_a_curved_surface():
    """Pinned, because it sets what a V22 number means: at dx = 5 nm two spheres 3 nm apart read 4.4 nm.
    A cross-code comparison on curved geometry carries that floor, and the 2 nm tolerance has to be read
    against it."""
    a = contour_points(sphere(GRID, (120.0, 80.0), 50.0), GRID)
    b = contour_points(sphere(GRID, (120.0, 80.0), 53.0), GRID)
    assert symmetric_hausdorff_nm(a, b) > 3.5


def test_identical_contours_compare_as_zero():
    a = contour_points(sphere(GRID, (120.0, 80.0), 50.0), GRID)
    assert symmetric_hausdorff_nm(a, a.copy()) == 0.0


def test_the_symmetric_distance_catches_a_feature_missing_from_one_side():
    """A one-sided distance would miss this: every point of the smaller set is close to the larger one,
    so only the reverse direction sees the extra feature."""
    base = contour_points(plane(GRID, 100.0), GRID)
    extra = np.vstack([base, base + np.array([40.0, 0.0])])
    one_sided = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(extra).query(base)[0].max()
    assert one_sided < 1e-9
    assert symmetric_hausdorff_nm(base, extra) == pytest.approx(40.0, abs=1e-6)


def test_the_verdict_reports_the_numbers_behind_it():
    case = CASES[0]
    a = contour_points(plane(GRID, 100.0), GRID)
    verdict = compare_contours(a, contour_points(plane(GRID, 101.0), GRID), case)
    assert verdict["passed"] is True and verdict["hausdorff_nm"] == pytest.approx(1.0)
    far = contour_points(plane(GRID, 110.0), GRID)
    assert compare_contours(a, far, case)["passed"] is False


def test_the_frozen_cases_state_everything_both_sides_need():
    for case in CASES:
        assert case.law in ("directional", "isotropic")
        assert (case.p is None) == (case.law == "isotropic"), "a directional case must state p"
        assert case.final_time_s > 0 and case.tolerance_nm == 2.0
