"""The w_mat sensitivity study, PRD §5.6's required diagnostic, as a test rather than a report file.

The numbers it produces are recorded in OPEN_QUESTIONS.md S17.1 and DECISIONS.md; this keeps them
reproducible, which is the point of finding J6 (a diagnostic that justified a decision was lost because
it lived in an untracked scratch directory).

Reference case, per PRD §5.6 and the owner decision of 2026-09-17: the phase-2 SiGe marker, 30 nm
(provisional), at dx = 2 nm, with the default w_mat of 2 cells.
"""

import jax
import jax.numpy as jnp
import pytest

from geocore import extraction as X
from geocore.band import BandedRateField
from geocore.config import BandConfig
from geocore.materials import Layer, layer_fractions, masked_trench, with_selectivity
from geocore.schema import Grid, Material
from geocore.solver import SolvePlan, evolve
from geocore.velocity import directional

pytestmark = pytest.mark.gate

FILM, MASK, SIGE = Material("film", 0), Material("mask", 1, is_mask=True), Material("sige", 2)
MARKER_NM = 30.0


def _gradient(width_cells):
    grid = Grid((120, 40), 2.0, (False, True))
    fractions = layer_fractions(grid, [Layer(FILM, None, 100.0), Layer(SIGE, 100.0, 130.0),
                                       Layer(FILM, 130.0, 180.0), Layer(MASK, 180.0, None)],
                                3, width_cells)
    phi0 = masked_trench(grid, 180.0, 20.0, 40.0, 40.0, 40.0, 40.0)
    field = BandedRateField(with_selectivity(directional, [FILM, MASK, SIGE]), grid, BandConfig(),
                            2000, fractions)
    plan = SolvePlan(grid, 0.4, 100, reinit_every=5)
    params = {"v0_nm_per_s": jnp.float64(2.0), "p": jnp.float64(2.0),
              "material_rate": jnp.array([1.0, 1.0, 0.3])}
    objective = jax.jit(lambda t: X.depth(evolve(phi0, t, field, plan)[0], grid, 40.0, 180.0))
    return float(objective(params)), jax.grad(objective)(params)


def test_w_mat_sensitivity_study():
    """Reported whatever it shows (PRD §5.6). Asserted: the marker's own rate sensitivity and the
    overall rate sensitivity are stable across widths; the FILM rate sensitivity is not, and that is
    the finding, recorded in S17.1 rather than tuned away."""
    results = {w: _gradient(w) for w in (1.0, 2.0, 4.0, 8.0)}
    base_depth, base_grad = results[2.0]
    print("\n w_mat  share of marker   depth     d/dv0   d/d(SiGe)  d/d(film)  d/d(mask)")
    for w, (depth, g) in results.items():
        print(f" {w:5.0f}  {w*2/MARKER_NM*100:13.1f}%  {depth:7.3f}  {float(g['v0_nm_per_s']):7.3f}  "
              f"{float(g['material_rate'][2]):9.3f}  {float(g['material_rate'][0]):9.3f}  "
              f"{float(g['material_rate'][1]):9.3f}")

    relative = lambda value, base: abs(value - base) / abs(base)
    for w, (depth, g) in results.items():
        assert float(g["material_rate"][1]) == 0.0, "the mask multiplier is fixed at zero"
        assert relative(float(g["v0_nm_per_s"]), float(base_grad["v0_nm_per_s"])) < 0.02
        assert relative(depth, base_depth) < 0.02
    film = [float(results[w][1]["material_rate"][0]) for w in (1.0, 8.0)]
    assert relative(film[1], film[0]) > 0.2, (
        "S17.1: the film-rate sensitivity DOES move with w_mat (about 38 % from 1 to 8 cells). If this "
        "assertion starts failing, the finding has changed and S17.1 needs revisiting.")
