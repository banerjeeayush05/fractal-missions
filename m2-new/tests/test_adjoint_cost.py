"""Residual factor k and the adjoint ratio: MEASURED at M2.3, not gated (decision I5(a)).

The numbers size M2.4's checkpoint schedule. They are recorded in OPEN_QUESTIONS.md S13.3; this file
keeps the measurement executable so it cannot become a number nobody can reproduce (finding J6).
"""

import jax.numpy as jnp

from geocore.band import BandedRateField, single_material
from geocore.config import BandConfig
from geocore.initial import trapezoid
from geocore.reinit import reinitialize
from geocore.schema import Grid
from geocore.solver import SolvePlan, _step
from geocore.velocity import directional
from geocore.verification.cost import optimal_segment_length, residual_factor


def test_residual_factor_is_measurable_and_positive():
    grid = Grid((40, 32), 10.0, (False, True))
    phi = trapezoid(grid, 280.0, 60.0, 140.0, 100.0, 160.0)
    field = BandedRateField(directional, grid, BandConfig(), 400, single_material(grid))
    plan = SolvePlan(grid, 1.0, 1)
    params = {"v0_nm_per_s": jnp.float64(2.0), "p": jnp.float64(2.0)}

    k_step = residual_factor(lambda f: _step(f, params, field, plan, jnp.int32(0))[0], phi)
    k_reinit = residual_factor(lambda f: reinitialize(f, grid, 5), phi)
    assert k_step > 1.0 and k_reinit > 1.0


def test_optimal_segment_length_formula():
    assert optimal_segment_length(625, 206.0) == (625 / 206.0) ** 0.5
