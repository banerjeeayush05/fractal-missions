"""M2.5: differentiable extraction. V13 against exact geometry; V14 on CD and sidewall angle.

V13's reference is `initial.trapezoid`, an EXACT signed distance function of known CD, depth and
angle -- the reason stage 10 refused to accept a composed approximation -- and, independently,
scikit-image's marching-squares contour (decision A11: reference only, never in the package).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geocore import extraction as X
from geocore.band import BandedRateField, single_material
from geocore.config import BandConfig
from geocore.functionals import solid_volume
from geocore.initial import extrude, sidewall_angle_deg, trapezoid
from geocore.schema import Grid
from geocore.solver import SolvePlan, evolve
from geocore.velocity import directional
from geocore.verification.gradcheck import reverse_gradient, taylor_test
from geocore.verification.params import scales_for

GRID = Grid((300, 100), 10.0, (False, True))
SURFACE, DEPTH, TOP_CD, BOTTOM_CD, CENTRE = 2600.0, 1500.0, 500.0, 380.0, 500.0
HEIGHTS = (2400.0, 1800.0, 1300.0)          # each more than a cell from both corners (S16.2)
KEY = jax.random.PRNGKey(14)


def exact_cd(z, top=TOP_CD, bottom=BOTTOM_CD):
    return bottom + (top - bottom) * (z - (SURFACE - DEPTH)) / DEPTH


def skimage_cd(phi, z):
    """CD from scikit-image's zero contour: an implementation that shares nothing with ours."""
    from skimage import measure

    xs = []
    for contour in measure.find_contours(np.asarray(phi), 0.0):
        cz = (contour[:, 0] + 0.5) * GRID.spacing_nm
        cx = (contour[:, 1] + 0.5) * GRID.spacing_nm
        for i in range(len(contour) - 1):
            if (cz[i] - z) * (cz[i + 1] - z) < 0:
                xs.append(cx[i] + (cx[i + 1] - cx[i]) * (z - cz[i]) / (cz[i + 1] - cz[i]))
    xs = np.array(xs)
    return xs[xs > CENTRE].min() - xs[xs < CENTRE].max()


# ---------------------------------------------------------------------------------------- V13


@pytest.mark.check("V13")
def test_v13_analytic_trapezoid_and_subcell_smoothness(ledger_measure):
    """CD error < 0.1 nm, sidewall angle < 0.2 deg, and CD varying SMOOTHLY under sub-cell motion."""
    phi = trapezoid(GRID, SURFACE, DEPTH, TOP_CD, BOTTOM_CD, CENTRE)
    cd_errors = [abs(float(X.cd_at(phi, GRID, CENTRE, z)) - exact_cd(z)) for z in HEIGHTS]
    reference_gap = [abs(float(X.cd_at(phi, GRID, CENTRE, z)) - skimage_cd(phi, z)) for z in HEIGHTS]
    angle = float(X.sidewall_angle(phi, GRID, CENTRE, HEIGHTS[2], HEIGHTS[0]))
    angle_error = abs(angle - sidewall_angle_deg(DEPTH, TOP_CD, BOTTOM_CD))
    depth_error = abs(float(X.depth(phi, GRID, CENTRE, SURFACE)) - DEPTH)

    offsets = np.linspace(0.0, GRID.spacing_nm, 21)
    lateral = [float(X.cd_at(trapezoid(GRID, SURFACE, DEPTH, TOP_CD, BOTTOM_CD, CENTRE + o), GRID,
                             CENTRE + o, HEIGHTS[1])) for o in offsets]
    swept_tops = TOP_CD + offsets
    swept = np.array([float(X.cd_at(trapezoid(GRID, SURFACE, DEPTH, t, BOTTOM_CD, CENTRE), GRID,
                                    CENTRE, HEIGHTS[1])) for t in swept_tops])
    sweep_error = float(np.abs(swept - exact_cd(HEIGHTS[1], swept_tops)).max())
    steps = np.diff(swept)
    step_spread = float(steps.max() - steps.min())

    ledger_measure(cd_errors_nm=cd_errors, skimage_gap_nm=reference_gap,
                   sidewall_angle_error_deg=angle_error, depth_error_nm=depth_error,
                   lateral_subcell_cd_range_nm=float(max(lateral) - min(lateral)),
                   cd_sweep_max_error_nm=sweep_error, cd_sweep_step_spread_nm=step_spread,
                   tolerance_cd_nm=0.1, tolerance_angle_deg=0.2)
    assert max(cd_errors) < 0.1
    assert max(reference_gap) < 0.1
    assert angle_error < 0.2
    assert depth_error < 0.1
    assert max(lateral) - min(lateral) < 0.1, "a lateral shift must not change CD"
    assert sweep_error < 0.1
    assert step_spread < 0.01, "equal input steps must give equal output steps: no staircase"


def test_exactness_stops_within_a_cell_of_a_corner():
    """Finding S16.2, pinned. Linear interpolation is exact across a straight wall but not across a
    corner. 5 nm above the floor corner at dx = 10 nm the CD reads 0.19 nm low -- outside V13's
    0.1 nm. Extraction heights must stay more than a cell from corners."""
    phi = trapezoid(GRID, SURFACE, DEPTH, TOP_CD, BOTTOM_CD, CENTRE)
    z = SURFACE - DEPTH + 5.0
    assert abs(float(X.cd_at(phi, GRID, CENTRE, z)) - exact_cd(z)) > 0.1


def test_invalid_extraction_is_nan_not_plausible():
    phi = trapezoid(GRID, SURFACE, DEPTH, TOP_CD, BOTTOM_CD, CENTRE)
    assert np.isnan(float(X.cd_at(phi, GRID, CENTRE, SURFACE + 100.0))), "above the surface"
    assert np.isnan(float(X.cd_at(phi, GRID, CENTRE, SURFACE - DEPTH - 100.0))), "below the floor"


def test_line_profile_reduces_an_extruded_trench_exactly():
    grid3 = Grid((300, 6, 100), 10.0, (False, True, True))
    phi3 = extrude(trapezoid(GRID, SURFACE, DEPTH, TOP_CD, BOTTOM_CD, CENTRE), grid3)
    profile = X.line_profile(phi3)
    assert float(X.cd_at(profile, GRID, CENTRE, HEIGHTS[1])) == pytest.approx(exact_cd(HEIGHTS[1]))
    with pytest.raises(ValueError, match="2D profile"):
        X.cd_at(phi3, grid3, CENTRE, HEIGHTS[1])


def test_bow_is_zero_for_a_straight_taper():
    phi = trapezoid(GRID, SURFACE, DEPTH, TOP_CD, BOTTOM_CD, CENTRE)
    assert abs(float(X.bow(phi, GRID, CENTRE, HEIGHTS[2], HEIGHTS[0]))) < 1e-9


# --------------------------------------------------------------------- V14 on extracted quantities


@pytest.fixture(scope="module")
def etched():
    """A trench etched 20 steps under the directional law. Extraction heights stay inside the trench
    AFTER the etch: without a mask the field also etches (finding L2), so the top surface drops from
    520 to 480 nm and a height near the original surface would return NaN."""
    grid = Grid((64, 32), 10.0, (False, True))
    phi0 = trapezoid(grid, 520.0, 200.0, 140.0, 100.0, 160.0)
    field = BandedRateField(directional, grid, BandConfig(), 600, single_material(grid))
    plan = SolvePlan(grid, 1.0, 20, reinit_every=5)
    theta = {"v0_nm_per_s": jnp.float64(2.0), "p": jnp.float64(2.0)}
    return grid, phi0, field, plan, theta


def _v14(objective, theta):
    j = jax.jit(objective)
    gradient = reverse_gradient(j, theta)
    return taylor_test(j, theta, gradient, scales_for(theta), KEY), gradient


@pytest.mark.check("V14")
def test_v14_on_cd_mid(etched, ledger_measure):
    grid, phi0, field, plan, theta = etched
    result, gradient = _v14(lambda t: X.cd_at(evolve(phi0, t, field, plan)[0], grid, 160.0, 390.0),
                            theta)
    ledger_measure(**{f"cd_mid_{k}": v for k, v in result.measured.items()})
    assert result.passed, result.message
    assert float(gradient["v0_nm_per_s"]) > 0.0, "a faster etch widens the trench"
    assert float(gradient["p"]) < 0.0, "a more directional etch removes less from the walls"


@pytest.mark.nightly  # S20.4: moved out of the fast tier when WENO5 became the default
@pytest.mark.check("V14")
def test_v14_on_sidewall_angle(etched, ledger_measure):
    grid, phi0, field, plan, theta = etched
    result, gradient = _v14(lambda t: X.sidewall_angle(evolve(phi0, t, field, plan)[0], grid, 160.0,
                                                       340.0, 440.0), theta)
    ledger_measure(**{f"sidewall_angle_{k}": v for k, v in result.measured.items()})
    assert result.passed, result.message
    assert float(gradient["p"]) > 0.0, "a more directional etch makes walls more vertical"


@pytest.mark.nightly  # S20.4: moved out of the fast tier when WENO5 became the default
@pytest.mark.xfail(strict=True, reason=(
    "S20.3 OPEN (raised 2026-09-17 by the WENO5 default): 19 of 20 directions are clean at slope "
    "1.998-2.014; direction 18 reports insufficient_signal, meaning its Taylor remainder never "
    "rises out of the noise floor anywhere in the window. That is the check declining to measure, "
    "not a wrong gradient -- one shared gradient vector cannot be right in 19 directions and wrong "
    "in the 20th, and a wrong gradient tends to slope 1, not to silence. The window is NOT widened."))
def test_diagnostic_decomposition_volume_and_cd_both_pass(etched):
    """PRD §8.7: V14 on the smooth functional AND on extracted CD. Both passing means the chain is
    sound; volume passing with CD failing would put the bug in extraction.

    Under Godunov both passed, volume on 20/20 clean directions at slopes [1.987, 2.014]. Under the
    WENO5 default the CD half still passes and the volume half loses one direction to the noise
    floor: WENO5's solve is more accurate, so this fixture's second-order response in that one
    direction is now smaller than fp64 can resolve through 20 steps. See S20.3."""
    grid, phi0, field, plan, theta = etched
    volume, _ = _v14(lambda t: solid_volume(evolve(phi0, t, field, plan)[0], grid), theta)
    cd, _ = _v14(lambda t: X.cd_at(evolve(phi0, t, field, plan)[0], grid, 160.0, 390.0), theta)
    assert (volume.passed, cd.passed) == (True, True)
