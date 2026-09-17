"""WENO5 at the scheme level: order, the fixed-epsilon deviation, the boundary fallback, and the
flag reaching every place it must.

These run on fields with exact derivatives, so a failure here is in the stencil, not in the solver.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geocore.constants import WENO_BOUNDARY_FALLBACK_CELLS, WENO_EPS
from geocore.initial import sphere
from geocore.reinit import reinitialize
from geocore.schema import Grid
from geocore.solver import SolvePlan
from geocore.stencils import grad_mag_godunov, one_sided_differences


def _sine_errors(scheme, eps=WENO_EPS, sizes=(16, 32, 64, 128)):
    """Max error of D- and D+ against the exact derivative of sin(2 pi x) on a periodic axis.

    Jitted: run eagerly, WENO5's dozens of small array ops dispatch one at a time and this test
    alone took 20 s of the fast tier's 3-minute budget.
    """
    minus, plus = [], []
    for n in sizes:
        grid = Grid((8, n), 1.0 / n, (False, True))
        x = (jnp.arange(n) + 0.5) / n
        phi = jnp.broadcast_to(jnp.sin(2 * jnp.pi * x), (8, n))
        exact = 2 * jnp.pi * jnp.cos(2 * jnp.pi * x)
        dm, dp = jax.jit(lambda f: one_sided_differences(f, 1, grid, scheme, weno_eps=eps))(phi)
        minus.append(float(jnp.max(jnp.abs(dm[4] - exact))))
        plus.append(float(jnp.max(jnp.abs(dp[4] - exact))))
    return minus, plus


def _orders(errors):
    return [float(np.log(errors[i] / errors[i + 1]) / np.log(2)) for i in range(len(errors) - 1)]


def test_weno5_reconstruction_is_fifth_order_in_both_directions():
    """Both D- and D+: the Godunov selection uses both, so a wrong D+ would hide behind a correct
    D- in any test that only checks one."""
    minus, plus = _sine_errors("weno5")
    assert min(_orders(minus)) > 4.5
    assert min(_orders(plus)) > 4.5


def test_godunov_differences_are_first_order():
    """The control: without it, the test above would pass against a harness that measures order
    wrongly in a way that always reports five."""
    minus, plus = _sine_errors("godunov")
    assert all(0.9 < o < 1.1 for o in _orders(minus) + _orders(plus))


@pytest.mark.parametrize("factor", [1e-2, 1e2])
def test_the_fixed_epsilon_is_inert(factor):
    """Finding J0. The published regulariser is scaled by a stencil max, which is a kink; this one
    is fixed. Moving it 100x either way must not move the observed order -- shown, not asserted."""
    base = _orders(_sine_errors("weno5", WENO_EPS, (32, 64))[0])[0]
    moved = _orders(_sine_errors("weno5", WENO_EPS * factor, (32, 64))[0])[0]
    assert abs(moved - base) < 0.05


def test_weno5_falls_back_to_godunov_near_a_neumann_boundary():
    """Finding J3. Within the fallback band of a NON-periodic edge, WENO5's differences must equal
    Godunov's exactly; on the periodic axis there is no fallback."""
    grid = Grid((24, 24), 1.0, (False, True))
    phi = sphere(grid, (12.0, 12.0), 7.0)
    band = WENO_BOUNDARY_FALLBACK_CELLS
    for axis in (0, 1):
        g_m, g_p = one_sided_differences(phi, axis, grid, "godunov")
        w_m, w_p = one_sided_differences(phi, axis, grid, "weno5")
        if axis == 0:
            edge = np.r_[0:band, 24 - band:24]
            assert bool(jnp.array_equal(w_m[edge], g_m[edge]))
            assert bool(jnp.array_equal(w_p[edge], g_p[edge]))
            assert not bool(jnp.array_equal(w_m[band:-band], g_m[band:-band]))
        else:
            assert not bool(jnp.array_equal(w_m[:, :band], g_m[:, :band])), \
                "a periodic axis has no boundary, so no fallback"


def test_the_scheme_flag_reaches_reinitialisation():
    """PRD §5.2: the flag is threaded into reinitialisation as well as advection. A run that
    advected with WENO5 but reinitialised with Godunov would pay for WENO5 and keep Godunov's drift."""
    grid = Grid((48, 48), 1.0, (False, True))
    stretched = 1.5 * sphere(grid, (24.0, 24.0), 12.0)
    g = reinitialize(stretched, grid, 5, "godunov")
    w = reinitialize(stretched, grid, 5, "weno5")
    assert not bool(jnp.allclose(g, w))


def test_unknown_schemes_are_rejected():
    grid = Grid((8, 8), 1.0, (False, True))
    phi = jnp.zeros(grid.shape)
    with pytest.raises(ValueError, match="spatial scheme"):
        one_sided_differences(phi, 0, grid, "weno3")
    with pytest.raises(ValueError, match="spatial_scheme"):
        SolvePlan(grid, 1.0, 1, spatial_scheme="eno2")


def test_godunov_remains_the_default():
    """Finding J5 in the original tree: making WENO5 the default is an owner decision, not made."""
    grid = Grid((8, 8), 1.0, (False, True))
    assert SolvePlan(grid, 1.0, 1).spatial_scheme == "godunov"
    phi = sphere(grid, (4.0, 4.0), 2.0)
    assert bool(jnp.array_equal(grad_mag_godunov(phi, jnp.ones(grid.shape), grid),
                                grad_mag_godunov(phi, jnp.ones(grid.shape), grid, scheme="godunov")))
