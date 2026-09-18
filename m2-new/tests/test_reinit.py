"""The M2.2 reinitialisation checks -- V1 (full travel), V11, V12 -- and the findings behind them.

**Finding S12.1 -- reinitialisation drift is cumulative, and V12 cannot see it. CLOSED 2026-09-17**
by the move to WENO5 as the default (DECISIONS.md). It was accepted as a known discrepancy on
2026-09-16 under option c -- re-measure after WENO5 -- and the re-measurement is what closed it:
V1 -3.169 % -> -0.031 %, V12a 2.149 % -> 0.000 %. The diagnosis below is kept because it is what
identified reinitialisation as the term to fix, and the numbers in it are Godunov numbers.

Precise: V12 bounds the interface motion of ONE cycle on an exact distance function (measured
0.066 % area). Over a run the solver applies 40 cycles, and the drift accumulates. Decomposing V1 at
full travel on the P5 geometry:

    dx      advection only    + reinit     band vs everywhere
    10 nm       -1.02 %        -3.17 %        identical
     5 nm       -0.50 %        -1.56 %        identical
   2.5 nm       -0.25 %        -0.77 %        identical

Both terms are first order and converge. Reinitialisation is the larger share, and it is anisotropic:
at dx = 10 nm the radius along a grid axis is 148.9 nm and along the diagonal 143.2 nm, exact 150.2.

Plain: each reinitialisation cycle nudges the surface by an amount too small for V12 to fail. Forty
of them shrink the disk by 3 %, mostly along the diagonals. The band is not involved -- it matches
evaluate-everywhere to the third decimal.

Why it matters: PRD §8.3 names this exact failure -- reinitialisation that shifts the interface
"produces a systematic etch-rate bias that looks like physics, calibrates away into the closure
parameters, and then fails to transfer".

V1's 1 % tolerance was met under Godunov only if the radius was measured along a grid axis
(-0.84 %) and missed on the mean over directions (-3.17 %). The PRD does not say which. The mean is
asserted, because choosing the one direction the grid favours would select the flattering answer.
Under the WENO5 default (owner decision, 2026-09-17) the mean is -0.031 %, so the question no longer
decides the outcome and the discrepancy is closed. The tolerance is unchanged throughout.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from measures import (
    grad_mag_error_near_interface, radii_along_rays, solid_area, symmetric_difference_area,
)
from geocore.band import BandedRateField, single_material
from geocore.config import BandConfig
from geocore.initial import plane, sphere
from geocore.reinit import reinit_iteration, reinitialize, smoothed_sign
from geocore.schema import Grid
from geocore.solver import SolvePlan, constant_rate, evolve, solve
from geocore.velocity import isotropic

RATE = 5.83
BANDS = BandConfig()


def _disk_case(dx: float, cells: int, radius_nm: float = 300.0):
    grid = Grid((cells, cells), dx, (False, True))
    centre = cells * dx / 2.0
    return grid, (centre, centre), sphere(grid, (centre, centre), radius_nm)


def _field(grid):
    return BandedRateField(isotropic, grid, BANDS, grid.n_cells // 4, single_material(grid))


# ----------------------------------------------------------------------------------- V1 full


@pytest.mark.check("V1")
def test_v1_full_travel_with_reinitialisation(ledger_measure):
    """P5 geometry: r0 = 300 nm, dx = 10 nm, R = 5.83 nm/s, T = 25.7 s, 200 steps -> r = 150.2 nm.

    Run on 96x96 rather than P5's 64x64: P5 leaves the disk 2 cells from the edge, inside PRD §5.1's
    10-cell buffer. Measured to make no difference here (145.455 vs 145.410 nm), but a verification
    run should not sit inside a buffer its own specification forbids.
    """
    grid, centre, phi0 = _disk_case(10.0, 96)
    result = solve(phi0, {"v0_nm_per_s": jnp.float64(RATE)}, _field(grid),
                   SolvePlan(grid, 25.7 / 200, 200, reinit_every=5, n_reinit=5))
    radii = radii_along_rays(result.phi, 10.0, centre)
    exact = 300.0 - RATE * 25.7
    mean_err = abs(radii.mean() - exact) / exact

    ledger_measure(radius_exact_nm=exact, radius_mean_nm=float(radii.mean()),
                   radius_axis_nm=float(radii[0]), radius_diagonal_nm=float(radii[9]),
                   relative_error_mean=float(mean_err),
                   relative_error_axis=float(abs(radii[0] - exact) / exact),
                   reduced_form=False, peak_band_occupancy=result.peak_occupancy)
    assert mean_err < 0.01


def test_reinitialisation_is_what_removes_the_band_limit():
    """Finding G0, pinned. Without reinit, 150 nm of travel (15 cells, twice the extension band)
    stalls: phi outside the band is frozen while phi inside moves, until the front runs out of
    distance function. With it, the front travels."""
    grid, centre, phi0 = _disk_case(10.0, 96)
    exact = 300.0 - RATE * 25.7
    radii = {}
    for every in (None, 5):
        r = solve(phi0, {"v0_nm_per_s": jnp.float64(RATE)}, _field(grid),
                  SolvePlan(grid, 25.7 / 200, 200, reinit_every=every))
        radii[every] = radii_along_rays(r.phi, 10.0, centre).mean()
    assert abs(radii[None] - exact) / exact > 0.2, "without reinit the front should stall"
    assert abs(radii[5] - exact) / exact < 0.05, "with reinit the front should reach the target"


def test_the_band_adds_no_error_over_evaluating_everywhere():
    """Stage 11's machinery, checked against the capstone's approach on the run where it matters.
    The closest-point gather is exact for a law that depends only on the normal, so any difference
    would be a bug in the band -- and the measured difference is below the third decimal."""
    grid, centre, phi0 = _disk_case(10.0, 96)
    plan = SolvePlan(grid, 25.7 / 200, 200, reinit_every=5)
    params = {"v0_nm_per_s": jnp.float64(RATE)}
    banded = solve(phi0, params, _field(grid), plan).phi
    everywhere = solve(phi0, params, constant_rate("v0_nm_per_s"), plan).phi
    diff = np.abs(radii_along_rays(banded, 10.0, centre) - radii_along_rays(everywhere, 10.0, centre))
    assert diff.max() < 1e-2


def test_cumulative_reinit_drift_is_first_order_in_dx():
    """S12.1 was a discretisation error, not a bug: under Godunov, halving dx halves it.

    Pinned at `spatial_scheme="godunov"` EXPLICITLY. This test documents the first-order Godunov
    behaviour that motivated the move to WENO5; it is not a statement about the default. Reading it
    on the WENO5 default is meaningless -- the drift there is 0.000 %, so the ratio is a ratio of
    two numbers at the noise floor (measured 3.03, which is why this failed the moment the default
    flipped). V12a below is the check that governs the default.""" 
    errors = []
    for dx, cells in ((10.0, 96), (5.0, 192)):
        grid, centre, phi0 = _disk_case(dx, cells)
        steps = int(200 * 10.0 / dx)
        r = solve(phi0, {"v0_nm_per_s": jnp.float64(RATE)}, constant_rate("v0_nm_per_s"),
                  SolvePlan(grid, 25.7 / steps, steps, reinit_every=5, spatial_scheme="godunov"))
        errors.append(abs(radii_along_rays(r.phi, dx, centre).mean() - (300.0 - RATE * 25.7)))
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.25)


# ---------------------------------------------------------------------------------------- V11


@pytest.mark.check("V11")
def test_v11_reinitialisation_restores_the_distance_property(ledger_measure):
    """After a cycle, max ||grad phi| - 1| within 3 cells of the interface < 5 %.

    Run in the operational regime -- advect `reinit_every` steps at the CFL target, then one cycle --
    because that is the field the solver actually hands reinitialisation. A disk of 200 nm, clear of
    every boundary: an earlier fixture touching the edge read 0.515, all of it from the Neumann and
    periodic seams, none from reinitialisation.
    """
    grid = Grid((64, 64), 10.0, (False, True))
    phi0 = sphere(grid, (320.0, 320.0), 200.0)
    advected = solve(phi0, {"v0_nm_per_s": jnp.float64(RATE)}, _field(grid),
                     SolvePlan(grid, 0.4 * 10.0 / RATE, 5)).phi
    before = grad_mag_error_near_interface(advected, grid)
    after = grad_mag_error_near_interface(reinitialize(advected, grid, 5), grid)
    stretched = grad_mag_error_near_interface(reinitialize(1.5 * phi0, grid, 5), grid)

    ledger_measure(grad_error_before=before, grad_error_after=after, tolerance=0.05,
                   grad_error_after_on_1p5x_stretch=stretched,
                   note="reinit raises the error slightly on a near-distance field "
                        "(first-order scheme); still well inside tolerance")
    assert after < 0.05


# ---------------------------------------------------------------------------------------- V12


@pytest.mark.check("V12")
def test_v12_reinitialisation_does_not_move_the_interface(ledger_measure):
    """Area drift of ONE cycle on an exact distance function < 0.1 %.

    Area is the sub-cell zero-contour area (finding G2). Passing this does NOT bound cumulative
    drift over a run -- see S12.1 at the top of this file.
    """
    grid = Grid((64, 64), 10.0, (False, True))
    phi0 = sphere(grid, (320.0, 320.0), 200.0)
    after = reinitialize(phi0, grid, 5)
    area = solid_area(phi0, 10.0)
    drift = symmetric_difference_area(phi0, after, 10.0) / area

    ledger_measure(area_nm2=area, symmetric_difference_fraction=drift, tolerance=1e-3,
                   measure="sub-cell zero contour (G2)")
    assert drift < 1e-3


# --------------------------------------------------------------------------------------- V12a


@pytest.mark.check("V12a")
def test_v12a_accumulated_reinitialisation_drift_over_a_run(ledger_measure):
    """The interface displacement over a WHOLE RUN that reinitialisation alone is responsible for.

    Two runs, identical except that one reinitialises every 5 steps and one never does. Everything
    else -- geometry, rate, steps, advection error -- is common to both and cancels in the
    difference, so what remains is attributable to reinitialisation.

    The rate is applied everywhere rather than through the band. That is what lets the
    no-reinitialisation run travel the full distance without stalling (finding G0); the band has
    been shown to add no error over evaluating everywhere
    (`test_the_band_adds_no_error_over_evaluating_everywhere`), so the comparison is not weakened.

    Tolerance (S12.3, provisional): the mean shift must stay below V1's own forward tolerance, 1 %.
    If reinitialisation alone may move the interface further than the check it runs inside allows,
    then that check's verdict is decided by reinitialisation rather than by the physics.
    """
    grid, centre, phi0 = _disk_case(10.0, 96)
    params = {"v0_nm_per_s": jnp.float64(RATE)}
    exact = 300.0 - RATE * 25.7

    def final_radii(every):
        plan = SolvePlan(grid, 25.7 / 200, 200, reinit_every=every, n_reinit=5)
        phi = solve(phi0, params, constant_rate("v0_nm_per_s"), plan).phi
        return radii_along_rays(phi, 10.0, centre)

    without, with_reinit = final_radii(None), final_radii(5)
    shift = np.abs(with_reinit - without) / exact
    ledger_measure(mean_shift_fraction=float(shift.mean()), max_shift_fraction=float(shift.max()),
                   signed_mean_shift_nm=float((with_reinit - without).mean()), n_cycles=40,
                   tolerance=0.01, tolerance_basis="S12.3 provisional: V1 forward tolerance",
                   finding=None)
    assert shift.mean() < 0.01


# ------------------------------------------------------------------------------- the rules


def test_smoothed_sign_is_smooth_and_saturates():
    grid = Grid((8, 8), 10.0, (False, True))
    x = jnp.linspace(-50.0, 50.0, 101)
    s = smoothed_sign(x, grid)
    assert float(s[50]) == 0.0
    assert float(s[0]) < -0.97 and float(s[-1]) > 0.97
    assert bool(jnp.all(jnp.isfinite(jax.vmap(jax.grad(lambda v: smoothed_sign(v, grid)))(x))))


def test_an_exact_plane_is_left_unchanged():
    """|grad phi| = 1 already, so every iteration's update is zero away from the Neumann rows."""
    grid = Grid((40, 8), 1.0, (False, True))
    phi = plane(grid, 20.0)
    out = reinitialize(phi, grid, 5)
    assert float(jnp.max(jnp.abs((out - phi)[3:-3]))) < 1e-12


def test_the_iteration_count_must_be_a_fixed_python_int():
    """PRD §5.3: a data-dependent count reintroduces §5.2's discontinuous-graph problem."""
    grid = Grid((8, 8), 1.0, (False, True))
    phi = plane(grid, 4.0)
    for bad in (0, -1, 2.0, True, jnp.int32(3)):
        with pytest.raises(ValueError, match="positive Python int"):
            reinitialize(phi, grid, bad)


def test_one_iteration_matches_the_pde_by_hand():
    grid = Grid((12, 12), 1.0, (False, True))
    phi = 2.0 * plane(grid, 6.0)                    # |grad phi| = 2, too steep
    s = smoothed_sign(phi, grid)
    out = reinit_iteration(phi, s, grid)
    interior = (slice(2, -2), slice(None))
    # d(phi) = -dtau * S * (2 - 1) with dtau = 0.5 dx
    assert bool(jnp.allclose(out[interior], (phi - 0.5 * s * 1.0)[interior]))


def test_the_gradient_survives_reinitialisation_inside_the_solve():
    """No stop_gradient (PRD §5.3, §11). Reinit is part of the forward map and its derivative is
    real; a NaN here would come from the smoothed sign or the Godunov sqrt."""
    grid = Grid((48, 48), 10.0, (False, True))
    phi0 = sphere(grid, (240.0, 240.0), 150.0)
    plan = SolvePlan(grid, 0.4, 20, reinit_every=5)

    def objective(params):
        phi, _ = evolve(phi0, params, _field(grid), plan)
        return jnp.sum(phi)

    g = jax.grad(objective)({"v0_nm_per_s": jnp.float64(RATE)})["v0_nm_per_s"]
    assert bool(jnp.isfinite(g)) and float(g) > 0.0
