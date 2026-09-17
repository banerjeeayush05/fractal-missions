"""The M2.3 gate: V14, V15, V16 on the smooth functional, through the whole solver; and the analytic
sensitivities V14a, V14b, V14c.

**Finding S13.1 -- the PRD gives V14a-c no tolerance.** Rule below ACCEPTED by the owner 2026-09-16.
Decision §5 added them "with closed-form answers" and stops there. The registry previously carried
"relative error < 1e-6", which was written at stage 5 with no basis and is withdrawn.

Mechanical rule applied: **1 %**, from three constraints.
1. These checks exist to catch what V14-V16 cannot -- a wrong sign, a wrong normal convention, or a
   derivative wrong at the size V19 injects (5 %). The tolerance must sit clearly below 5 %.
2. A discrete derivative carries the same discretisation error as the discrete forward map, so it
   cannot be held tighter than the forward tolerance of its own configuration. V1's is 1 %.
3. One number for all three, so no check gets a tolerance fitted to its own result.
This rule was written AFTER the numbers below were measured. That is recorded rather than hidden:
under it V14b and V14c pass and V14a fails, so it was not chosen to make everything pass.

Measured: V14b 3.5e-5; V14c 1.6e-3 (v0) and 2.1e-3 (p); V14a 1.9e-2. V14a's failure is S12.1: with
reinitialisation off and the travel held inside the band, the same derivative is correct (see
`test_v14a_derivative_is_correct_without_cumulative_reinit_drift`).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.ndimage import map_coordinates

from geocore.band import BandedRateField, single_material
from geocore.config import BandConfig
from geocore.functionals import solid_volume
from geocore.initial import plane, polygon_2d, sphere, trapezoid
from geocore.schema import Grid
from geocore.solver import SolvePlan, evolve
from geocore.velocity import directional, isotropic
from geocore.verification.gradcheck import (
    dot_product_test, forward_reverse_test, reverse_gradient, taylor_test,
)
from geocore.verification.params import scales_for

BANDS = BandConfig()
ANALYTIC_TOLERANCE = 1e-2       # S13.1, owner accepted 2026-09-16
KEY = jax.random.PRNGKey(14)


# ------------------------------------------------------------------ differentiable measurements


def column_crossing(phi, column, dx):
    """Sub-cell height of the zero crossing up one column. Solid below. Differentiable in phi: the
    index comes from argmax and carries no gradient, the interpolation does."""
    c = phi[:, column]
    k = jnp.argmax(c >= 0.0)
    return (k - 1 + 0.5 + (-c[k - 1]) / (c[k] - c[k - 1])) * dx


def mean_radius(phi, dx, centre, r_max, n_angles=72, samples=2000):
    """Mean over directions of the first inside-to-outside crossing. Differentiable."""
    s = jnp.linspace(0.0, r_max, samples)

    def along(angle):
        z = (centre[0] + s * jnp.cos(angle)) / dx - 0.5
        x = (centre[1] + s * jnp.sin(angle)) / dx - 0.5
        v = map_coordinates(phi, [z, x], order=1, mode="nearest")
        k = jnp.argmax(v >= 0.0)
        return s[k - 1] + (s[k] - s[k - 1]) * (-v[k - 1]) / (v[k] - v[k - 1])

    return jnp.mean(jax.vmap(along)(jnp.linspace(0.0, 2 * jnp.pi, n_angles, endpoint=False)))


def _relative(measured, expected):
    return abs(float(measured) - expected) / abs(expected)


# -------------------------------------------------------------------- V14, V15, V16 on the solver


@pytest.fixture(scope="module")
def trench_problem():
    """A trapezoidal trench under the directional law, reinitialised, through the banded path.
    p = 2: the law's kink sits on vertical sidewalls, and p >= 2 keeps it out of the gradient."""
    grid = Grid((40, 32), 10.0, (False, True))
    phi0 = trapezoid(grid, 280.0, 60.0, 140.0, 100.0, 160.0)
    field = BandedRateField(directional, grid, BANDS, 400, single_material(grid))
    plan = SolvePlan(grid, dt_s=1.0, n_steps=20, reinit_every=5)
    theta = {"v0_nm_per_s": jnp.float64(2.0), "p": jnp.float64(2.0)}

    objective = jax.jit(lambda t: solid_volume(evolve(phi0, t, field, plan)[0], grid))
    field_map = jax.jit(lambda t: evolve(phi0, t, field, plan)[0].reshape(-1))
    return objective, field_map, theta


@pytest.mark.check("V14")
def test_v14_taylor_remainder_through_the_solver(trench_problem, ledger_measure):
    objective, _, theta = trench_problem
    gradient = reverse_gradient(objective, theta)
    result = taylor_test(objective, theta, gradient, scales_for(theta), KEY)
    ledger_measure(**result.measured, functional="solid_volume", grid=[40, 32], n_steps=20,
                   gradient={k: float(v) for k, v in gradient.items()})
    assert result.passed, result.message


@pytest.mark.check("V15")
def test_v15_forward_and_reverse_mode_agree_through_the_solver(trench_problem, ledger_measure):
    objective, _, theta = trench_problem
    result = forward_reverse_test(objective, theta, KEY)
    ledger_measure(**result.measured)
    assert result.passed, result.message


@pytest.mark.check("V16")
def test_v16_dot_product_through_the_solver(trench_problem, ledger_measure):
    _, field_map, theta = trench_problem
    result = dot_product_test(field_map, theta, KEY)
    ledger_measure(**result.measured, output="final phi field")
    assert result.passed, result.message


def test_gradient_signs_are_physical(trench_problem):
    """A correct gradient of a sign-flipped model still passes V14. These are forward statements."""
    objective, _, theta = trench_problem
    g = reverse_gradient(objective, theta)
    assert float(g["v0_nm_per_s"]) < 0.0, "a faster etch must leave less solid"
    assert float(g["p"]) > 0.0, "a more directional etch removes less from tilted walls"


# ----------------------------------------------------------------------------------- V14b plane


@pytest.mark.check("V14b")
def test_v14b_plane_depth_sensitivity_is_plus_t(ledger_measure):
    """d(depth)/dR = +T. The magnitude is T either way; the SIGN follows from §5.1."""
    grid = Grid((64, 16), 10.0, (False, True))
    field = BandedRateField(isotropic, grid, BANDS, 256, single_material(grid))
    duration = 20.0
    plan = SolvePlan(grid, duration / 40, 40, reinit_every=5)
    phi0 = plane(grid, 400.0)

    def depth(t):
        return 400.0 - column_crossing(evolve(phi0, t, field, plan)[0], 8, grid.spacing_nm)

    d = jax.grad(depth)({"v0_nm_per_s": jnp.float64(5.0)})["v0_nm_per_s"]
    error = _relative(d, duration)
    ledger_measure(derivative=float(d), expected=duration, relative_error=error,
                   tolerance=ANALYTIC_TOLERANCE, tolerance_basis="S13.1, owner accepted 2026-09-16")
    assert float(d) > 0.0
    assert error < ANALYTIC_TOLERANCE


# ----------------------------------------------------------------------------------- V14a disk


@pytest.mark.check("V14a")
@pytest.mark.xfail(strict=True, reason="S12.1 KNOWN DISCREPANCY (owner accepted 2026-09-16): derivative inherits "
                                       "cumulative reinit drift, 1.9 % against 1 %. Re-measure after WENO5.")
def test_v14a_disk_radius_sensitivity_is_minus_t(ledger_measure):
    """dr/dR = -T on V1's full-travel geometry. Radius is the mean over 72 directions, for the same
    reason as V1: the axis is the direction the grid favours."""
    grid = Grid((96, 96), 10.0, (False, True))
    field = BandedRateField(isotropic, grid, BANDS, 2400, single_material(grid))
    duration = 25.7
    plan = SolvePlan(grid, duration / 200, 200, reinit_every=5)
    phi0 = sphere(grid, (480.0, 480.0), 300.0)

    def radius(t):
        return mean_radius(evolve(phi0, t, field, plan)[0], 10.0, (480.0, 480.0), 400.0)

    d = jax.grad(radius)({"v0_nm_per_s": jnp.float64(5.83)})["v0_nm_per_s"]
    error = _relative(d, -duration)
    ledger_measure(derivative=float(d), expected=-duration, relative_error=error,
                   tolerance=ANALYTIC_TOLERANCE, finding="S12.1")
    assert float(d) < 0.0
    assert error < ANALYTIC_TOLERANCE


def test_v14a_derivative_is_correct_without_cumulative_reinit_drift():
    """Localises V14a's failure to S12.1. Same law, travel inside the band, no reinitialisation:
    the derivative is right. So the gradient machinery is sound and the error is the forward map's."""
    grid = Grid((96, 96), 10.0, (False, True))
    field = BandedRateField(isotropic, grid, BANDS, 2400, single_material(grid))
    duration = 5.0
    plan = SolvePlan(grid, duration / 40, 40)
    phi0 = sphere(grid, (480.0, 480.0), 300.0)

    def radius(t):
        return mean_radius(evolve(phi0, t, field, plan)[0], 10.0, (480.0, 480.0), 400.0)

    d = jax.grad(radius)({"v0_nm_per_s": jnp.float64(5.83)})["v0_nm_per_s"]
    assert _relative(d, -duration) < ANALYTIC_TOLERANCE


# ------------------------------------------------------------------------------ V14c tilted plane


def _roof(grid, period_nm, valley_z_nm, angle_rad):
    """A periodic V-profile of tilted facets, as an exact distance function. A single tilted plane is
    not periodic (finding G4: its phi jumps at the seam), so facets alternate. Built over three
    periods so every cell's nearest boundary is a real facet."""
    tan = np.tan(angle_rad)
    xs = [-period_nm + i * period_nm / 2 for i in range(7)]
    zs = [valley_z_nm + tan * period_nm / 2 if i % 2 == 0 else valley_z_nm for i in range(7)]
    skirt = 10.0 * max(grid.shape) * grid.spacing_nm
    vertices = [(-skirt, -skirt), (zs[0], -skirt)] + list(zip(zs, xs)) + \
               [(zs[-1], xs[-1] + skirt), (-skirt, xs[-1] + skirt)]
    return polygon_2d(grid, jnp.array(vertices))


@pytest.mark.check("V14c")
def test_v14c_tilted_facet_sensitivities_in_v0_and_p(ledger_measure):
    """A facet tilted by theta moves along its normal at v0 cos^p(theta). Measured vertically at
    mid-facet, z(T) = z0 - v0 cos^(p-1)(theta) T, so

        dz/dv0 = -cos^(p-1)(theta) T          dz/dp = -v0 cos^(p-1)(theta) ln(cos theta) T

    Guards the normal-vector convention that V4 was meant to catch."""
    grid = Grid((56, 32), 10.0, (False, True))
    theta = np.deg2rad(30.0)
    phi0 = _roof(grid, 320.0, 200.0, theta)
    field = BandedRateField(directional, grid, BANDS, 900, single_material(grid))
    duration = 20.0
    plan = SolvePlan(grid, duration / 40, 40, reinit_every=5)
    params = {"v0_nm_per_s": jnp.float64(2.0), "p": jnp.float64(2.0)}

    def height(t):
        return column_crossing(evolve(phi0, t, field, plan)[0], 8, grid.spacing_nm)

    g = jax.grad(height)(params)
    c = np.cos(theta)
    expected_v0 = -c ** (2.0 - 1.0) * duration
    expected_p = -2.0 * c ** (2.0 - 1.0) * np.log(c) * duration
    err_v0 = _relative(g["v0_nm_per_s"], expected_v0)
    err_p = _relative(g["p"], expected_p)

    ledger_measure(dz_dv0=float(g["v0_nm_per_s"]), expected_dz_dv0=expected_v0,
                   relative_error_v0=err_v0, dz_dp=float(g["p"]), expected_dz_dp=expected_p,
                   relative_error_p=err_p, facet_angle_deg=30.0, tolerance=ANALYTIC_TOLERANCE)
    assert np.sign(float(g["v0_nm_per_s"])) == np.sign(expected_v0)
    assert np.sign(float(g["p"])) == np.sign(expected_p)
    assert err_v0 < ANALYTIC_TOLERANCE
    assert err_p < ANALYTIC_TOLERANCE
