"""Stencil arithmetic: exact where it should be exact, upwind in the right direction, and
differentiable at the one place it is tempting not to be."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geocore.constants import GRAD_MAG_EPS
from geocore.schema import Grid
from geocore.stencils import (
    backward_difference, central_gradient, forward_difference, grad_mag_godunov,
)

G2 = Grid(shape=(32, 32), spacing_nm=1.0, periodic=(False, True))
G3 = Grid(shape=(16, 16, 16), spacing_nm=1.0, periodic=(False, True, True))


def _plane(grid: Grid, slope: float = 1.0) -> jnp.ndarray:
    """phi = slope * z. An exact signed distance function when slope = 1."""
    z = jnp.arange(grid.shape[0], dtype=jnp.float64) - grid.shape[0] / 2
    return slope * z.reshape((-1,) + (1,) * (grid.ndim - 1)) * jnp.ones(grid.shape)


@pytest.mark.parametrize("grid", [G2, G3], ids=["2D", "3D"])
def test_grad_mag_is_one_on_an_exact_distance_function(grid):
    """|grad phi| = 1 is the defining property of a signed distance function, and it is exact
    for a plane under either upwind branch."""
    phi = _plane(grid)
    interior = (slice(1, -1),) + (slice(None),) * (grid.ndim - 1)
    for speed in (+1.0, -1.0):
        mag = grad_mag_godunov(phi, jnp.full(grid.shape, speed), grid)
        assert float(jnp.max(jnp.abs(mag[interior] - 1.0))) < 1e-8


def test_neumann_boundary_gives_zero_difference_across_the_top_and_bottom():
    """PRD §5.1. Zero normal derivative, imposed by replicating the edge value."""
    phi = _plane(G2)
    assert float(jnp.max(jnp.abs(forward_difference(phi, 0, G2)[-1]))) == 0.0
    assert float(jnp.max(jnp.abs(backward_difference(phi, 0, G2)[0]))) == 0.0


def test_periodic_lateral_boundary_wraps():
    """A field that is genuinely periodic laterally must have no seam at the wrap."""
    x = jnp.arange(G2.shape[1], dtype=jnp.float64)
    phi = jnp.sin(2 * jnp.pi * x / G2.shape[1]) * jnp.ones(G2.shape)
    d = forward_difference(phi, 1, G2)
    assert float(jnp.abs(d[0, -1] - d[0, 0])) < 0.5   # smooth across the seam, no jump
    assert float(jnp.max(jnp.abs(d))) < 1.0


def test_the_upwind_branch_follows_the_sign_of_the_speed():
    """The whole point of Godunov: take the difference from the side the front is coming FROM.

    phi here rises to the right, so D- and D+ are both positive. With c > 0 the scheme selects
    max(D-, 0) and discards D+; with c < 0 it selects max(D+, 0) and discards D-.
    """
    grid = Grid(shape=(8, 8), spacing_nm=1.0, periodic=(False, True))
    x = jnp.arange(8, dtype=jnp.float64)
    phi = jnp.where(x < 4, x, 4.0 + 3.0 * (x - 4))[None, :] * jnp.ones((8, 8))
    at_kink = (4, 4)   # slope 1 on the left, slope 3 on the right
    up = grad_mag_godunov(phi, jnp.ones(grid.shape), grid)
    down = grad_mag_godunov(phi, -jnp.ones(grid.shape), grid)
    assert float(up[at_kink]) == pytest.approx(1.0, abs=1e-6)    # from the left: slope 1
    assert float(down[at_kink]) == pytest.approx(3.0, abs=1e-6)  # from the right: slope 3


def test_grad_mag_is_differentiable_at_a_godunov_valley():
    """THE trap this module exists to survive.

    At a valley cell every selected branch is exactly zero, so the sum under the sqrt is zero,
    and d/dx sqrt(x) is infinite there. Without eps INSIDE the sqrt, one such cell returns NaN
    for every parameter in the run -- not a wrong number in one place, a dead gradient everywhere.
    """
    grid = Grid(shape=(9, 9), spacing_nm=1.0, periodic=(False, True))
    x = jnp.arange(9, dtype=jnp.float64) - 4.0
    phi = jnp.abs(x)[None, :] * jnp.ones((9, 9))      # a V: valley at the centre column

    speed = jnp.ones(grid.shape)
    mag = grad_mag_godunov(phi, speed, grid)
    assert float(mag[4, 4]) == pytest.approx(np.sqrt(GRAD_MAG_EPS), abs=1e-12), \
        "the valley cell must land exactly on the eps floor -- that is the case being guarded"

    def loss(scale):
        return jnp.sum(grad_mag_godunov(phi * scale, speed, grid))

    g = jax.grad(loss)(1.0)
    assert bool(jnp.isfinite(g)), "eps inside the sqrt is what keeps this finite"


def test_removing_eps_would_produce_nan():
    """The control for the test above. Without it, that test would pass against a stencil that
    never encountered a valley cell at all, and the guard would be untested."""
    grid = Grid(shape=(9, 9), spacing_nm=1.0, periodic=(False, True))
    x = jnp.arange(9, dtype=jnp.float64) - 4.0
    phi = jnp.abs(x)[None, :] * jnp.ones((9, 9))
    speed = jnp.ones(grid.shape)

    def loss_without_eps(scale):
        return jnp.sum(grad_mag_godunov(phi * scale, speed, grid, eps=0.0))

    assert bool(jnp.isnan(jax.grad(loss_without_eps)(1.0)))


@pytest.mark.parametrize("grid", [G2, G3], ids=["2D", "3D"])
def test_central_gradient_stacks_the_vertical_component_first(grid):
    """`gradient[0]` must be the vertical component in BOTH dimensions, or a shared 2D/3D code
    path would index a different axis depending on which one it is in."""
    phi = _plane(grid)
    g = central_gradient(phi, grid)
    assert g.shape == (grid.ndim, *grid.shape)
    interior = (slice(1, -1),) + (slice(None),) * (grid.ndim - 1)
    assert float(jnp.max(jnp.abs(g[0][interior] - 1.0))) < 1e-8
    for lateral in range(1, grid.ndim):
        assert float(jnp.max(jnp.abs(g[lateral]))) < 1e-12
