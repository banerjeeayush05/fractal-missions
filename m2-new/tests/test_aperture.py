"""V3 — collimated aperture limit. PRD §8.1, gate tier.

**V3 is ON HOLD** (owner, 2026-09-17), because the cause is the velocity law, not the solver.

Under a perfectly collimated etch (p = 64) the trench should reproduce the mask opening: vertical
sidewalls, no bow. Measured sidewall angle error is 3.7 deg against a 0.5 deg requirement, and the floor
travels 10 % of its target.

The evidence chain, each step ruling something out:

| test | result | rules out |
|---|---|---|
| flat plane, no corners, p = 64 | travel 199.977 of 200 nm, normals exactly 1.0 | the law and the code |
| masked trench, p = 2 / 8 / 16 / 64 | 86 % / 34 % / 22 % / 10 % of target travel | a p-independent bug |
| floor normals after 5 / 20 / 50 steps | 0.976 / 0.939 / 0.923, from 1.0 | a static error |
| band gather vs rate on every cell | 20.72 vs 20.58 nm | the band and the gather |
| central vs 4th-order normals | 20.72 vs 20.14 nm | stencil order |
| godunov vs weno5 | identical to 4 decimals | the advection scheme |
| dx = 10 / 5 / 2.5 nm | 10.2 % / 10.4 % / 25.8 % | convergence rescuing it |

Mechanism: `R = v0 cos^p(theta)` is LOCAL, so a tilted patch etches slower than a flat one. The mask
corner seeds a rounded edge; the rounded part then etches slower than the flat floor beside it, so the
rounding spreads inward. At p = 64 a 6 % normal error is a 100x rate error, and the feedback runs away.
A real collimated etch keeps a flat floor because the ION FLUX is uniform and non-local, which is M3's
visibility calculation, not something a local law can reproduce.

This is the same family as V4, which decision §11 put ON HOLD for the same reason: a monotone local
`cos^p` law cannot produce the classical result. **Owner decision needed** — see OPEN_QUESTIONS S18.1.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from geocore import extraction as X
from geocore.band import BandedRateField, single_material
from geocore.config import BandConfig
from geocore.initial import plane
from geocore.materials import Layer, layer_fractions, masked_trench, with_selectivity
from geocore.schema import Grid, Material
from geocore.solver import SolvePlan, solve
from geocore.velocity import directional

pytestmark = pytest.mark.gate

FILM, MASK = Material("film", 0), Material("mask", 1, is_mask=True)
GRID = Grid((120, 48), 5.0, (False, True))
FILM_TOP, CENTRE, P_COLLIMATED = 350.0, 120.0, 64.0


def _run(start_depth, steps=100, p=P_COLLIMATED):  # noqa: D103
    fractions = layer_fractions(GRID, [Layer(FILM, None, FILM_TOP), Layer(MASK, FILM_TOP, None)],
                                2, 2.0)
    phi0 = masked_trench(GRID, FILM_TOP, 50.0, start_depth, 100.0, 100.0, CENTRE)
    field = BandedRateField(with_selectivity(directional, [FILM, MASK]), GRID, BandConfig(), 4000,
                            fractions)
    params = {"v0_nm_per_s": jnp.float64(2.0), "p": jnp.float64(p),
              "material_rate": jnp.array([1.0, 1.0])}
    return solve(phi0, params, field, SolvePlan(GRID, 1.0, steps, reinit_every=5))


@pytest.mark.check("V3")
def test_v3_collimated_aperture(ledger_measure):
    """ON HOLD (owner, 2026-09-17). The registry marks V3 on hold, so the plugin skips this; the
    measurement below is kept so the decision can be revisited against real numbers when M3 arrives.
    """
    result = _run(start_depth=150.0)
    floor = float(X.floor_height(result.phi, GRID, CENTRE))
    low, high = floor + 15.0, FILM_TOP - 15.0
    angle = float(X.sidewall_angle(result.phi, GRID, CENTRE, low, high))
    ledger_measure(sidewall_angle_deg=angle, angle_error_deg=abs(angle - 90.0), tolerance_deg=0.5,
                   p=P_COLLIMATED, floor_travel_nm=FILM_TOP - 150.0 - floor,
                   expected_travel_nm=200.0,
                   cd_mid_nm=float(X.cd_at(result.phi, GRID, CENTRE, 0.5 * (low + high))),
                   bow_nm=float(X.bow(result.phi, GRID, CENTRE, low, high)), finding="S18.1")
    assert abs(angle - 90.0) < 0.5


def test_a_corner_free_surface_is_exact_at_p_64():
    """The control for S18.1: without a corner to seed the feedback, p = 64 is exact. So the law, the
    band, the 0**p guard and the solver are all fine, and the failure is the corner interaction."""
    params = {"v0_nm_per_s": jnp.float64(2.0), "p": jnp.float64(P_COLLIMATED)}
    field = BandedRateField(directional, GRID, BandConfig(), 2000, single_material(GRID))
    result = solve(plane(GRID, 400.0), params, field, SolvePlan(GRID, 1.0, 100, reinit_every=5))
    column = np.asarray(result.phi[:, 24])
    k = int(np.argmax(column >= 0.0))
    z = (k - 1 + 0.5 + (-column[k - 1]) / (column[k] - column[k - 1])) * 5.0
    assert abs((400.0 - z) - 200.0) < 0.05


def test_the_aperture_is_reproduced_near_p_equals_one():
    """The measurement that identifies WHICH p is the collimated limit of this law.

    For a surface z = h(x), the level set gives the VERTICAL descent speed as

        h_t = -v0 * cos^(p-1)(theta)

    so at p = 1 every part of the surface descends at v0 whatever its tilt: the mask shape translates
    downward, which is exactly what "perfectly collimated" means. For p > 1 a tilted patch descends
    more slowly, tilt grows, and the feedback of S18.1 starts. p is not "how collimated"; it is how
    strongly the model punishes tilt.

    Measured, 100 nm of travel: angle error 0.195 deg at p = 1, 0.110 deg at p = 1.05, 1.177 deg at
    p = 1.25, 4.717 deg at p = 2, 3.051 deg at p = 64. V3's 0.5 deg is met near p = 1 only.
    """
    result = _run(start_depth=150.0, steps=50, p=1.05)
    floor = float(X.floor_height(result.phi, GRID, CENTRE))
    low, high = floor + 15.0, FILM_TOP - 15.0
    assert abs(float(X.sidewall_angle(result.phi, GRID, CENTRE, low, high)) - 90.0) < 0.5
    assert abs((200.0 - floor) - 100.0) < 1.0, "the floor must travel the full distance"
    assert float(X.cd_at(result.phi, GRID, CENTRE, 0.5 * (low + high))) == pytest.approx(100.0,
                                                                                          abs=0.5)


def test_the_trench_never_widens_at_any_p():
    """Not isotropic etching. A vertical sidewall has n . z_hat = 0, so its rate is 0 at every p and
    the trench cannot widen: CD reads 100.00 nm at p = 2, 8 and 64. What changes with p is the FLOOR
    shape, not the width."""
    for p in (2.0, 8.0, 64.0):
        result = _run(start_depth=150.0, steps=50, p=p)
        floor = float(X.floor_height(result.phi, GRID, CENTRE))
        mid = 0.5 * (floor + 15.0 + FILM_TOP - 15.0)
        assert float(X.cd_at(result.phi, GRID, CENTRE, mid)) == pytest.approx(100.0, abs=0.5)


def test_the_stall_worsens_with_p():
    """Pins the mechanism: the higher the exponent, the more a small normal error costs, so the
    less the floor travels. If this ordering ever breaks, S18.1's explanation is wrong."""
    travel = {p: 330.0 - float(X.floor_height(_run(20.0, p=p).phi, GRID, CENTRE))
              for p in (2.0, 16.0, 64.0)}
    assert travel[2.0] > travel[16.0] > travel[64.0]
    assert travel[2.0] > 150.0, "at p = 2 the floor should travel most of the way"
    assert travel[64.0] < 50.0, "at p = 64 it stalls"
