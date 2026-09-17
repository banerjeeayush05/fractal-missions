"""The M2.1 gate: V1 (reduced), V1a, V2, V20.

These run the FULL forward path -- request assembly, closest-point projection, a velocity model
called on the thin evaluation band, and the gather outward across the extension band -- not just
advection. That is what makes them worth a check ID.

**V1 runs in REDUCED form here** (PRD §8.1, owner decision 2026-09-11 option A): the travel
distance is held inside the extension band. Without reinitialisation, a band-limited velocity
extension distorts phi until the front stalls, because phi outside the band is frozen while phi
inside it moves. Full travel runs at stage 12, with reinitialisation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geocore.band import BandedRateField, BandOverflow, assert_capacity, single_material
from geocore.config import BandConfig
from geocore.initial import plane, sphere, trapezoid
from geocore.schema import Grid
from geocore.solver import CFLViolation, SolvePlan, evolve, solve
from geocore.velocity import directional, isotropic

BANDS = BandConfig()


def _field(model, grid, capacity):
    return BandedRateField(model=model, grid=grid, bands=BANDS, capacity=capacity,
                           material=single_material(grid))


def _radius(phi, grid, centre):
    """Sub-cell radius of the zero level set along +x from the centre."""
    row = np.asarray(phi[int(centre[0] / grid.spacing_nm)])
    start = int(centre[1] / grid.spacing_nm)
    for i in range(start, len(row) - 1):
        if row[i] < 0.0 <= row[i + 1]:
            frac = -row[i] / (row[i + 1] - row[i])
            return (i + 0.5 + frac) * grid.spacing_nm - centre[1]
    raise AssertionError("no crossing found")


def _interface_z(phi, column=4):
    col = np.asarray(phi[:, column])
    k = int(np.argmax(col >= 0.0))
    return (k - 1 + (-col[k - 1]) / (col[k] - col[k - 1]) + 0.5)


# --------------------------------------------------------------------------------------- V1


def test_v1_reduced_form_shrinks_a_disk_at_the_prescribed_rate(ledger_measure):
    """V1 in REDUCED form (travel inside the band, no reinitialisation). No longer claims the V1 ID:
    PRD §8.1 moves V1 to full travel at M2.2, which `test_reinit.py` runs. Kept as a regression test
    of the banded path alone.

    A solid disk under a constant rate must satisfy r(t) = r0 - R*t.

    Amended per decision §1: a POSITIVE rate removes material, so the disk SHRINKS. The PRD's
    original growth form assumed the pre-decision sign convention.
    """
    grid = Grid(shape=(80, 80), spacing_nm=1.0, periodic=(False, True))
    centre, r0 = (40.0, 40.0), 24.0
    phi0 = sphere(grid, centre, r0)

    rate, travel = 0.5, 4.0                      # held inside the 8-cell extension band
    plan = SolvePlan(grid, dt_s=0.25, n_steps=int(travel / rate / 0.25))
    params = {"v0_nm_per_s": jnp.float64(rate)}

    result = solve(phi0, params, _field(isotropic, grid, 1600), plan)
    measured = _radius(result.phi, grid, centre)
    expected = r0 - rate * plan.final_time_s
    relative = abs(measured - expected) / expected

    ledger_measure(radius_expected_nm=expected, radius_measured_nm=float(measured),
                   relative_error=float(relative), travel_nm=travel, reduced_form=True,
                   peak_band_occupancy=result.peak_occupancy, max_cfl=result.max_cfl)
    assert relative < 0.01, f"radius {measured:.4f} nm against an expected {expected:.4f} nm"


# -------------------------------------------------------------------------------------- V1a


@pytest.mark.check("V1a")
def test_v1a_a_trench_floor_recedes_and_an_overhang_underside_does_not(ledger_measure):
    """The ONLY check that can catch a sign or orientation inversion.

    V14, V15 and V16 all pass happily on a sign-flipped world: the gradient of a wrong forward
    map is still a correct gradient OF THAT MAP. Only a forward statement about which way
    material goes can catch it.

    Under the directional law, an up-facing trench floor recedes AWAY from the plasma, while the
    underside of an overhang -- whose normal points away from the plasma -- does not move at all.
    """
    grid = Grid(shape=(64, 48), spacing_nm=1.0, periodic=(False, True))
    surface, depth = 40.0, 8.0
    phi0 = trapezoid(grid, surface, depth, 20.0, 20.0, 24.0)
    plan = SolvePlan(grid, dt_s=0.25, n_steps=16)
    params = {"v0_nm_per_s": jnp.float64(0.5), "p": jnp.float64(2.0)}

    result = solve(phi0, params, _field(directional, grid, 1024), plan)
    floor_before, floor_after = _interface_z(np.asarray(phi0), 24), _interface_z(result.phi, 24)

    # An overhang: solid above, open below. Its normal points AWAY from the plasma.
    overhang = -(plane(grid, 30.0))
    over_result = solve(overhang, params, _field(directional, grid, 1024), plan)
    moved = float(jnp.max(jnp.abs(over_result.phi - overhang)))

    ledger_measure(floor_before_nm=float(floor_before), floor_after_nm=float(floor_after),
                   floor_recession_nm=float(floor_before - floor_after),
                   overhang_max_change_nm=moved)
    assert floor_after < floor_before, "a trench floor must recede AWAY from the plasma"
    assert moved < 1e-9, "a downward-facing surface must not move under a directional law"


# --------------------------------------------------------------------------------------- V2


@pytest.mark.check("V2")
def test_v2_a_plane_translates_at_exactly_the_rate(ledger_measure):
    """Nearly exact by construction. If it is not, the upwind scheme or a boundary is wrong."""
    grid = Grid(shape=(64, 16), spacing_nm=1.0, periodic=(False, True))
    surface, rate = 40.0, 0.5
    phi0 = plane(grid, surface)
    plan = SolvePlan(grid, dt_s=0.25, n_steps=32)
    params = {"v0_nm_per_s": jnp.float64(rate)}

    result = solve(phi0, params, _field(isotropic, grid, 1024), plan)
    measured = _interface_z(result.phi)
    expected = surface - rate * plan.final_time_s
    relative = abs(measured - expected) / expected

    ledger_measure(interface_expected_nm=expected, interface_measured_nm=float(measured),
                   relative_error=float(relative), peak_band_occupancy=result.peak_occupancy)
    assert relative < 1e-3


# -------------------------------------------------------------------------------------- V20


@pytest.mark.check("V20")
def test_v20_the_cfl_assertion_fires_and_names_n(ledger_measure):
    """PRD §8.6: deliberately set N too small; the run must abort with a message naming N.

    Tests the error path, which is otherwise never exercised -- and an error path that has never
    run is an error path nobody has checked.
    """
    grid = Grid(shape=(64, 16), spacing_nm=1.0, periodic=(False, True))
    phi0 = plane(grid, 40.0)
    params = {"v0_nm_per_s": jnp.float64(0.5)}
    too_few = SolvePlan(grid, dt_s=4.0, n_steps=4)      # CFL = 0.5 * 4 / 1 = 2.0

    with pytest.raises(CFLViolation, match=r"N=4") as excinfo:
        solve(phi0, params, _field(isotropic, grid, 1024), too_few)

    ledger_measure(message=str(excinfo.value)[:160], n_steps=4, names_n=True)
    assert "2.0" in str(excinfo.value) or "2.00" in str(excinfo.value)


def test_band_overflow_aborts_like_the_cfl_assertion():
    """PRD §5.0. K is sized from the WORST step, so the abort must name the peak, not the first."""
    grid = Grid(shape=(64, 16), spacing_nm=1.0, periodic=(False, True))
    phi0 = plane(grid, 40.0)
    plan = SolvePlan(grid, dt_s=0.25, n_steps=4)
    params = {"v0_nm_per_s": jnp.float64(0.5)}
    _, diagnostics = evolve(phi0, params, _field(isotropic, grid, 8), plan)
    with pytest.raises(BandOverflow, match="WORST step"):
        assert_capacity(diagnostics.occupancy, capacity=8)


def test_peak_occupancy_is_reported_every_run():
    grid = Grid(shape=(64, 16), spacing_nm=1.0, periodic=(False, True))
    plan = SolvePlan(grid, dt_s=0.25, n_steps=8)
    result = solve(plane(grid, 40.0), {"v0_nm_per_s": jnp.float64(0.5)},
                   _field(isotropic, grid, 1024), plan)
    assert result.diagnostics.occupancy.shape == (8,)
    assert 0 < result.peak_occupancy <= 1024


def test_the_gradient_survives_the_whole_banded_path():
    """Not a check ID -- V14 arrives at stage 13. But a NaN here would come from the band, the
    normals or the gather, and finding that out now is cheaper than finding it out then."""
    grid = Grid(shape=(48, 32), spacing_nm=1.0, periodic=(False, True))
    phi0 = trapezoid(grid, 30.0, 6.0, 14.0, 14.0, 16.0)
    plan = SolvePlan(grid, dt_s=0.25, n_steps=8)
    field = _field(directional, grid, 768)

    def objective(params):
        phi, _ = evolve(phi0, params, field, plan)
        return jnp.sum(phi)

    g = jax.grad(objective)({"v0_nm_per_s": jnp.float64(0.5), "p": jnp.float64(2.0)})
    assert bool(jnp.isfinite(g["v0_nm_per_s"])), "v0 gradient is NaN"
    assert bool(jnp.isfinite(g["p"])), "p gradient is NaN -- check the 0**p double-where"
    assert float(g["v0_nm_per_s"]) > 0.0
