"""Nightly invariants with no analytic solution: V8, V9, V10.

**Finding S12.2 -- V9 fails at dx = 10 nm, and the cause is S12.1. CLOSED 2026-09-17** with S12.1,
by the WENO5 default: V9 now measures 0.002 % against its unchanged 3 % tolerance. The table below
is the Godunov diagnosis that identified reinitialisation as the cause. Etch a 200 nm disk by 100 nm,
then deposit 100 nm, 100 steps each:

    dx      advection only    + reinit every 5
    10 nm        3.11 %           7.76 %
     5 nm        1.51 %           3.89 %
   2.5 nm        0.74 %           1.95 %

Both columns converge at first order; reinitialisation multiplies the error by about 2.5. Against
the 3 % tolerance V9 passed only at dx <= 2.5 nm under Godunov. The PRD does not specify V9's geometry; this one
uses V1's grid vocabulary with the disk clear of every edge, chosen before it was measured.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from measures import radii_along_rays, solid_area, symmetric_difference_area
from geocore.band import BandedRateField, single_material
from geocore.config import BandConfig
from geocore.initial import polygon_2d, sphere
from geocore.schema import Grid
from geocore.solver import SolvePlan, solve
from geocore.velocity import isotropic

RATE = 5.83
BANDS = BandConfig()


def rotation_model(centre_nm):
    """A rigid rotation as an etch rate: transport `phi_t + u.grad phi = 0` is `R = -u.n` in M2's
    `phi_t - R|grad phi| = 0`. A TEST fixture -- M2 ships only isotropic and directional (§5.5)."""
    cz, cx = centre_nm

    def model(request, params):
        w = params["omega"]
        z, x = request.positions[:, 0], request.positions[:, 1]
        uz, ux = -w * (x - cx), w * (z - cz)
        return -(uz * request.normals[:, 0] + ux * request.normals[:, 1])

    return model


def zalesak_disk(grid):
    """The canonical setup: radius 15, slot 5 wide and 25 long, centred (75, 50) on a 100^2 grid,
    as an exact polygon distance function."""
    edge = np.arcsin(2.5 / 15.0)
    arc = np.linspace(np.pi - edge, np.pi + edge - 2 * np.pi, 400)
    vertices = [(75 + 15 * np.cos(a), 50 + 15 * np.sin(a)) for a in arc]
    vertices += [(85.0, 47.5), (85.0, 52.5)]
    return polygon_2d(grid, jnp.array(vertices))


# ---------------------------------------------------------------------------------------- V8


@pytest.mark.check("V8")
def test_v8_zalesak_disk(ledger_measure):
    """One revolution. Area preserved to < 2 %; the notch survives (< 25 % filled, from J4).

G1 was the standing failure here: under the first-order Godunov scheme this lost 43.0 % of the
    disk's area at 100^2 (the rebuild measured 42.95 %), because first-order upwinding diffuses a
    sharp notch. WENO5 is the default from 2026-09-17 and the check passes on its own tolerance,
    which was never changed. The Godunov number is kept in this docstring because it is the reason
    the default moved.
    """
    grid = Grid((100, 100), 1.0, (False, True))
    phi0 = zalesak_disk(grid)
    field = BandedRateField(rotation_model((50.0, 50.0)), grid, BANDS, 1500, single_material(grid))
    period = 2 * np.pi / 0.01
    result = solve(phi0, {"omega": jnp.float64(0.01)}, field,
                   SolvePlan(grid, period / 628, 628, reinit_every=5))

    a0, a1 = solid_area(phi0, 1.0), solid_area(result.phi, 1.0)
    slot = np.zeros(grid.shape, bool)
    slot[61:85, 48:52] = True
    filled_before = float(np.mean(np.asarray(phi0)[slot] < 0.0))
    filled_after = float(np.mean(np.asarray(result.phi)[slot] < 0.0))

    ledger_measure(area_loss_fraction=(a0 - a1) / a0, notch_filled_fraction=filled_after,
                   notch_filled_before=filled_before, max_cfl=result.max_cfl,
                   peak_band_occupancy=result.peak_occupancy, finding="G1")
    assert filled_before == 0.0, "the fixture's slot must start empty"
    assert abs(a0 - a1) / a0 < 0.02
    assert filled_after < 0.25


# ---------------------------------------------------------------------------------------- V9


@pytest.mark.check("V9")
def test_v9_reversibility(ledger_measure):
    """Advect N steps under R, then N under -R. Symmetric-difference area < 3 %.

    S12.2: 7.76 % against 3 % under Godunov, because the drift that reinitialisation adds does not
    reverse with the rate. 0.002 % under the WENO5 default (2026-09-17). Tolerance unchanged."""
    grid = Grid((64, 64), 10.0, (False, True))
    phi0 = sphere(grid, (320.0, 320.0), 200.0)
    field = BandedRateField(isotropic, grid, BANDS, 1024, single_material(grid))
    plan = SolvePlan(grid, 100.0 / RATE / 100, 100, reinit_every=5)
    etched = solve(phi0, {"v0_nm_per_s": jnp.float64(RATE)}, field, plan).phi
    restored = solve(etched, {"v0_nm_per_s": jnp.float64(-RATE)}, field, plan).phi
    fraction = symmetric_difference_area(phi0, restored, 10.0) / solid_area(phi0, 10.0)

    ledger_measure(symmetric_difference_fraction=fraction, tolerance=0.03, travel_nm=100.0,
                   measure="sub-cell zero contour (G2)")
    assert fraction < 0.03


# --------------------------------------------------------------------------------------- V10


@pytest.mark.check("V10")
def test_v10_grid_orientation_isotropy(ledger_measure):
    """Isotropic etch must stay isotropic: (r_max - r_min)/r_mean < 2 %.

    Run at dx = 5 nm (finding G3: the anisotropy is first order in dx -- 4.2 % at 10 nm, 1.6 % at
    5 nm -- and 5 nm is the production-representative resolution). 192^2 keeps the disk 36 cells
    from every edge; a 128^2 run with 4 cells of clearance read 2.19 %, the excess from the seams.
    For a disk the "rotated 45 deg" initial condition is the same field, so anisotropy is measured
    directly over 72 directions.
    """
    grid = Grid((192, 192), 5.0, (False, True))
    phi0 = sphere(grid, (480.0, 480.0), 300.0)
    field = BandedRateField(isotropic, grid, BANDS, 6000, single_material(grid))
    result = solve(phi0, {"v0_nm_per_s": jnp.float64(RATE)}, field,
                   SolvePlan(grid, 150.0 / RATE / 200, 200, reinit_every=5))
    radii = radii_along_rays(result.phi, 5.0, (480.0, 480.0))
    anisotropy = (radii.max() - radii.min()) / radii.mean()

    ledger_measure(anisotropy=float(anisotropy), r_min_nm=float(radii.min()),
                   r_max_nm=float(radii.max()), r_mean_nm=float(radii.mean()), dx_nm=5.0,
                   n_directions=72, tolerance=0.02)
    assert anisotropy < 0.02
