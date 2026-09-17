"""M2.6: multi-material, including the mask as geometry.

Gate (PRD §6): the gradient survives an interface crossing a material boundary; the w_mat sensitivity
study is reported (`tests/test_w_mat_study.py`, gate tier); and the mask, a solid body
in phi with a zero rate, gives pattern transfer, undercut beneath the mask edge, and a real mask corner.
"""

from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geocore import extraction as X
from geocore.band import BandedRateField, _fractions_at, single_material
from geocore.config import BandConfig
from geocore.initial import trapezoid
from geocore.materials import Layer, layer_fractions, masked_trench, with_selectivity
from geocore.schema import Grid, Material
from geocore.solver import SolvePlan, evolve, solve
from geocore.velocity import directional, isotropic
from geocore.verification.gradcheck import reverse_gradient, taylor_test
from geocore.verification.params import scales_for

FILM, MASK, SIGE = Material("film", 0), Material("mask", 1, is_mask=True), Material("sige", 2)
BANDS = BandConfig()
KEY = jax.random.PRNGKey(14)

# A masked trench: film top 300 nm, 50 nm mask, 100 nm opening, 50 nm starting depth, dx = 5 nm.
GRID = Grid((100, 120), 5.0, (False, True))
FILM_TOP, MASK_THICKNESS, CENTRE = 300.0, 50.0, 300.0
STACK = [Layer(FILM, None, FILM_TOP), Layer(MASK, FILM_TOP, None)]


def _masked_run(model, params, steps):
    fractions = layer_fractions(GRID, STACK, 2, 2.0)
    phi0 = masked_trench(GRID, FILM_TOP, MASK_THICKNESS, 50.0, 100.0, 100.0, CENTRE)
    field = BandedRateField(with_selectivity(model, [FILM, MASK]), GRID, BANDS, 5000, fractions)
    return phi0, solve(phi0, params, field, SolvePlan(GRID, 1.0, steps, reinit_every=5))


# ------------------------------------------------------------------------------ fractions


def test_layer_fractions_sum_to_one_everywhere_including_the_ramps():
    fractions = layer_fractions(GRID, [Layer(FILM, None, 100.0), Layer(SIGE, 100.0, 130.0),
                                       Layer(FILM, 130.0, FILM_TOP), Layer(MASK, FILM_TOP, None)],
                                3, 2.0)
    total = fractions.sum(axis=-1)
    assert float(jnp.max(jnp.abs(total - 1.0))) < 1e-12
    assert float(jnp.min(fractions)) >= 0.0


def test_layers_must_be_contiguous_and_unbounded_at_the_ends():
    with pytest.raises(ValueError, match="contiguous"):
        layer_fractions(GRID, [Layer(FILM, None, 100.0), Layer(MASK, 120.0, None)], 2, 2.0)
    with pytest.raises(ValueError, match="unbounded"):
        layer_fractions(GRID, [Layer(FILM, 0.0, 100.0), Layer(MASK, 100.0, None)], 2, 2.0)


def test_fractions_are_read_at_the_surface_point_not_the_cell():
    """A band cell sits up to a band-width off the surface. Reading its own cell would put the rate
    change a band-width early; the request must carry the material AT the closest point."""
    fractions = layer_fractions(GRID, STACK, 2, 2.0)
    at_surface = _fractions_at(fractions, jnp.array([[FILM_TOP - 30.0], [50.0]]), GRID)
    at_contact = _fractions_at(fractions, jnp.array([[FILM_TOP], [50.0]]), GRID)
    assert float(at_surface[0, 0]) == pytest.approx(1.0)
    assert float(at_contact[0, 1]) == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------------- the mask


def test_pattern_transfer_the_trench_deepens_and_the_mask_does_not_move():
    """Finding H2 is what this fixes. Directional etch, 100 nm of floor travel."""
    params = {"v0_nm_per_s": jnp.float64(2.0), "p": jnp.float64(2.0),
              "material_rate": jnp.array([1.0, 1.0])}
    phi0, result = _masked_run(directional, params, 50)
    assert float(X.mask_remaining(result.phi, GRID, 40.0, FILM_TOP)) == pytest.approx(MASK_THICKNESS,
                                                                                      abs=1e-6)
    drop = float(X.floor_height(phi0, GRID, CENTRE) - X.floor_height(result.phi, GRID, CENTRE))
    assert drop == pytest.approx(100.0, rel=0.01)


def test_without_a_mask_the_trench_translates():
    """Finding H2, pinned so the reason the mask exists stays visible: unmasked, the field etches at the
    floor's rate and the whole profile slides down."""
    phi0 = trapezoid(GRID, FILM_TOP, 50.0, 100.0, 100.0, CENTRE)
    params = {"v0_nm_per_s": jnp.float64(2.0), "p": jnp.float64(2.0)}
    result = solve(phi0, params, BandedRateField(directional, GRID, BANDS, 5000,
                                                 single_material(GRID)),
                   SolvePlan(GRID, 1.0, 50, reinit_every=5))
    field_drop = float(X.mask_remaining(phi0, GRID, 40.0, 0.0) - X.mask_remaining(result.phi, GRID,
                                                                                  40.0, 0.0))
    assert field_drop == pytest.approx(100.0, rel=0.01)


def test_undercut_beneath_the_mask_edge():
    """Isotropic etch, 50 nm. The opening in the mask stays 100 nm; the film beneath widens to the full
    100 + 2 * 50 = 200 nm away from the mask contact."""
    params = {"v0_nm_per_s": jnp.float64(2.0), "material_rate": jnp.array([1.0, 1.0])}
    _, result = _masked_run(isotropic, params, 25)
    assert float(X.cd_at(result.phi, GRID, CENTRE, FILM_TOP + 25.0)) == pytest.approx(100.0, abs=0.01)
    assert float(X.cd_at(result.phi, GRID, CENTRE, FILM_TOP - 25.0)) == pytest.approx(200.0, abs=0.5)
    assert float(X.mask_remaining(result.phi, GRID, 40.0, FILM_TOP)) == pytest.approx(MASK_THICKNESS,
                                                                                      abs=1e-6)


def test_the_contact_ramp_slows_the_film_right_under_the_mask():
    """Finding S17.2, pinned. Film within half a ramp-width of the mask contact is blended with
    zero-rate mask and etches slower: 2 nm under the mask the undercut CD is 186 nm, not 200. A
    numerical artefact of w_mat, reported rather than tuned."""
    params = {"v0_nm_per_s": jnp.float64(2.0), "material_rate": jnp.array([1.0, 1.0])}
    _, result = _masked_run(isotropic, params, 25)
    near_contact = float(X.cd_at(result.phi, GRID, CENTRE, FILM_TOP - 2.0))
    assert 150.0 < near_contact < 199.0


def test_the_mask_multiplier_is_fixed_at_zero_and_receives_no_gradient():
    """Infinite selectivity is a modelling decision, not a fitted parameter: a params value for the mask
    cannot undo it, and its gradient is exactly zero."""
    params = {"v0_nm_per_s": jnp.float64(2.0), "p": jnp.float64(2.0),
              "material_rate": jnp.array([1.0, 7.0])}
    fractions = layer_fractions(GRID, STACK, 2, 2.0)
    phi0 = masked_trench(GRID, FILM_TOP, MASK_THICKNESS, 50.0, 100.0, 100.0, CENTRE)
    field = BandedRateField(with_selectivity(directional, [FILM, MASK]), GRID, BANDS, 5000, fractions)
    plan = SolvePlan(GRID, 1.0, 10, reinit_every=5)
    objective = lambda t: X.mask_remaining(evolve(phi0, t, field, plan)[0], GRID, 40.0, FILM_TOP)
    assert float(objective(params)) == pytest.approx(MASK_THICKNESS, abs=1e-6)
    assert float(jax.grad(objective)(params)["material_rate"][1]) == 0.0


# ---------------------------------------------------------------- crossing a material boundary


def _crossing(width_cells):
    grid = Grid((120, 40), 2.0, (False, True))
    fractions = layer_fractions(grid, [Layer(FILM, None, 100.0), Layer(SIGE, 100.0, 130.0),
                                       Layer(FILM, 130.0, 180.0), Layer(MASK, 180.0, None)],
                                3, width_cells)
    phi0 = masked_trench(grid, 180.0, 20.0, 40.0, 40.0, 40.0, 40.0)
    field = BandedRateField(with_selectivity(directional, [FILM, MASK, SIGE]), grid, BANDS, 2000,
                            fractions)
    plan = SolvePlan(grid, 0.4, 100, reinit_every=5)
    params = {"v0_nm_per_s": jnp.float64(2.0), "p": jnp.float64(2.0),
              "material_rate": jnp.array([1.0, 1.0, 0.3])}
    return jax.jit(lambda t: X.depth(evolve(phi0, t, field, plan)[0], grid, 40.0, 180.0)), params


@pytest.mark.check("V14")
def test_v14_gradient_survives_crossing_into_a_material_layer(ledger_measure):
    """The M2.6 gate. The floor starts 10 nm above a 30 nm SiGe marker (rate 0.3x) and crosses into it.
    With an integer material ID the rate would be a step function of position and the gradient with
    respect to anything setting the arrival time would be destroyed."""
    objective, params = _crossing(2.0)
    gradient = reverse_gradient(objective, params)
    result = taylor_test(objective, params, gradient, scales_for(params), KEY)
    ledger_measure(**{f"material_crossing_{k}": v for k, v in result.measured.items()},
                   crossing_depth_nm=float(objective(params)), w_mat_cells=2.0)
    assert result.passed, result.message
    assert float(objective(params)) > 50.0, "the floor must actually reach the marker"


def test_widths_below_one_cell_change_nothing():
    """Finding S17.3. The fraction field is read at surface points by linear interpolation between grid
    nodes, which already smooths over one cell, so w_mat under a cell reproduces the 1-cell result."""
    fine, params = _crossing(0.05)
    one, _ = _crossing(1.0)
    g_fine, g_one = jax.grad(fine)(params), jax.grad(one)(params)
    assert float(fine(params)) == pytest.approx(float(one(params)), abs=1e-9)
    assert float(g_fine["material_rate"][0]) == pytest.approx(float(g_one["material_rate"][0]), abs=1e-9)
