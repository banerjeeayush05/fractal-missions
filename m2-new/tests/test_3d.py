"""M2.4: the same code path in 3D.

Nothing in `geocore` branches on dimension: stencils loop over axes, the vertical axis is 0 in both,
and the band, gather and reinitialisation are written for any ndim. These tests are what make that
claim a measurement.

Full-resolution 3D V14 on case S03 is a gate-tier check on an H100 (`scripts/gate_m2_4.py`); this is
the small-grid version that runs everywhere.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# PRD §8.0: small 3D runs are nightly-tier. The fast tier is "2D only, coarse".
pytestmark = pytest.mark.nightly

from geocore.band import BandedRateField, single_material
from geocore.checkpoint import segment_length_for
from geocore.config import BandConfig
from geocore.functionals import solid_volume
from geocore.initial import extrude, sphere, trapezoid
from geocore.reinit import reinitialize
from geocore.schema import Grid
from geocore.solver import SolvePlan, _step, evolve, solve
from geocore.velocity import directional, isotropic
from geocore.verification.cost import residual_factor
from geocore.verification.gradcheck import (
    dot_product_test, forward_reverse_test, reverse_gradient, taylor_test,
)
from geocore.verification.params import scales_for

BANDS = BandConfig()
KEY = jax.random.PRNGKey(14)
THETA = {"v0_nm_per_s": jnp.float64(2.0), "p": jnp.float64(2.0)}


def _trench(nz, nl, dx=10.0):
    surface = (nz - 11) * dx
    depth = (surface - 100.0) / 2.0
    profile = trapezoid(Grid((nz, nl), dx, (False, True)), surface, depth,
                        nl * dx * 0.5, nl * dx * 0.35, nl * dx * 0.5)
    grid = Grid((nz, nl, nl), dx, (False, True, True))
    return grid, extrude(profile, grid)


@pytest.fixture(scope="module")
def problem_3d():
    grid, phi0 = _trench(24, 12)
    field = BandedRateField(directional, grid, BANDS, 600, single_material(grid))
    plan = SolvePlan(grid, 1.0, 10, reinit_every=5, checkpoint_segment=segment_length_for(10, 344.0))
    objective = jax.jit(lambda t: solid_volume(evolve(phi0, t, field, plan)[0], grid))
    field_map = jax.jit(lambda t: evolve(phi0, t, field, plan)[0].reshape(-1))
    return objective, field_map


@pytest.mark.check("V14")
def test_3d_v14_taylor_remainder(problem_3d, ledger_measure):
    objective, _ = problem_3d
    result = taylor_test(objective, THETA, reverse_gradient(objective, THETA), scales_for(THETA), KEY)
    ledger_measure(**{f"3D_{k}": v for k, v in result.measured.items()}, grid_3d=[24, 12, 12],
                   checkpointed=True)
    assert result.passed, result.message


@pytest.mark.check("V15")
def test_3d_v15_forward_reverse(problem_3d, ledger_measure):
    objective, _ = problem_3d
    result = forward_reverse_test(objective, THETA, KEY)
    ledger_measure(**{f"3D_{k}": v for k, v in result.measured.items()})
    assert result.passed, result.message


@pytest.mark.check("V16")
def test_3d_v16_dot_product(problem_3d, ledger_measure):
    _, field_map = problem_3d
    result = dot_product_test(field_map, THETA, KEY)
    ledger_measure(**{f"3D_{k}": v for k, v in result.measured.items()})
    assert result.passed, result.message


def test_a_sphere_shrinks_at_the_prescribed_rate_in_3d():
    """V1's physics in 3D, through the banded path, in REDUCED form: travel held inside the band and
    no reinitialisation. That isolates the 3D code path from S12.1's reinitialisation drift, which
    is already recorded in 2D and would otherwise dominate.

    22 cells of radius, 4 cells of travel. Measured 0.320 % in 3D against 0.257 % for the identical
    2D disk, so the extra dimension adds essentially nothing. Tolerance is V1's 1 %.
    """
    from measures import radii_along_rays

    grid = Grid((64, 64, 64), 5.0, (False, True, True))
    centre, r0, rate, steps, dt = 160.0, 110.0, 5.0, 10, 0.4
    result = solve(sphere(grid, (centre,) * 3, r0), {"v0_nm_per_s": jnp.float64(rate)},
                   BandedRateField(isotropic, grid, BANDS, grid.n_cells // 4, single_material(grid)),
                   SolvePlan(grid, dt, steps))
    plane = np.asarray(result.phi)[:, 32, :]          # the z-x plane through the centre
    radii = radii_along_rays(plane, 5.0, (centre, centre), r_max_nm=150.0)
    expected = r0 - rate * dt * steps
    assert abs(radii.mean() - expected) / expected < 0.01


def test_residual_factor_does_not_depend_on_grid_size_in_3d():
    """What lets k measured on a laptop size an H100 run. Measured 341-344 from 2,700 to 172,800
    cells; the capacity K is sized from band occupancy at each size."""
    from geocore.band import occupancy

    ks = []
    for nz, nl in ((27, 10), (54, 20)):
        grid, phi = _trench(nz, nl)
        capacity = min(grid.n_cells, max(64, 4 * int(occupancy(phi, grid, BANDS))))
        field = BandedRateField(directional, grid, BANDS, capacity, single_material(grid))
        plan = SolvePlan(grid, 1.0, 1)
        ks.append(residual_factor(lambda q: _step(q, THETA, field, plan, jnp.int32(0))[0], phi))
    assert ks[0] == pytest.approx(ks[1], rel=0.05)
    assert 250.0 < ks[0] < 450.0
