"""Convergence orders: V5 (manufactured solution), V6 (spatial), V7 (temporal).

Reported at M2.3, gated by their own tolerances. PRD §8.2: "Report observed order, not just 'error
decreases.' An order that is right by eye and 0.6 by measurement is a bug."

**Observed order is taken from the FINEST refinement pair**, the asymptotic regime; every pair is
recorded. This matters for V7: the pairs read 1.80, 1.91, 1.97, and a least-squares fit across all
four levels reads 1.895 -- just under the 1.9 requirement. The coarsest pair is pre-asymptotic. The
finest-pair convention is the standard one and was not chosen for V7, but it is the difference between
pass and fail here, so it was stated rather than left implicit. **Owner accepted it 2026-09-17**
(DECISIONS.md, S13.2).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from measures import radii_along_rays
from geocore.band import BandedRateField, single_material
from geocore.config import BandConfig, load_case
from geocore.initial import coordinate_fields, sphere
from geocore.schema import Grid
from geocore.solver import SolvePlan, constant_rate, solve
from geocore.velocity import isotropic

RATE = 5.83


def _orders(errors, spacings):
    return [float(np.log(errors[i] / errors[i + 1]) / np.log(spacings[i] / spacings[i + 1]))
            for i in range(len(errors) - 1)]


@pytest.mark.check("V6")
@pytest.mark.parametrize("scheme,required", [
    ("godunov", 0.9),
    pytest.param("weno5", 4.0, marks=pytest.mark.xfail(strict=True, reason=(
        "S20.2 OPEN (raised 2026-09-17, when weno5 became the default): the registry requires "
        "order >= 4 of weno5, and the full product path measures about 2.1. Finding J2 says why -- "
        "the velocity extension's multilinear gather is second order, so it caps the order of the "
        "whole path no matter how accurate the Hamiltonian is. The requirement is probably the "
        "thing that is wrong, but that is the owner's call, so the tolerance is UNCHANGED and the "
        "ledger records this as not passing."))),
])
def test_v6_spatial_order(ledger_measure, scheme, required):
    """Refine dx over 4 levels at constant CFL; error against V1's analytic radius.
    Full product path: band and reinitialisation.

    Run for BOTH schemes, explicitly. The scheme is the thing under test here, so inheriting the
    default would make the recorded number silently change meaning the day the default moved --
    which is exactly what happened on 2026-09-17.
    """
    l1, linf, spacings = [], [], []
    for dx in (8.0, 4.0, 2.0, 1.0):
        n = int(round(400.0 / dx))
        grid = Grid((n, n), dx, (False, True))
        centre = n * dx / 2.0
        steps = int(round(60.0 / (0.4 * dx)))
        field = BandedRateField(isotropic, grid, BandConfig(), grid.n_cells // 4,
                                single_material(grid))
        result = solve(sphere(grid, (centre, centre), 150.0), {"v0_nm_per_s": jnp.float64(RATE)},
                       field, SolvePlan(grid, 60.0 / RATE / steps, steps, reinit_every=5,
                                        spatial_scheme=scheme))
        error = np.abs(radii_along_rays(result.phi, dx, (centre, centre), r_max_nm=180.0) - 90.0)
        l1.append(float(error.mean()))
        linf.append(float(error.max()))
        spacings.append(dx)

    o1, oinf = _orders(l1, spacings), _orders(linf, spacings)
    ledger_measure(dx_nm=spacings, l1_error_nm=l1, linf_error_nm=linf, l1_orders=o1,
                   linf_orders=oinf, order_l1=o1[-1], order_linf=oinf[-1], required=required,
                   scheme=scheme, finding=None if scheme == "godunov" else "J2")
    assert o1[-1] >= required and oinf[-1] >= required


@pytest.mark.check("V7")
def test_v7_temporal_order(ledger_measure):
    """Fix dx, refine dt; self-convergence against a run 8x finer than the finest level. TVD-RK2 >= 1.9.

    The initial field is deliberately NOT a distance function. On an exact distance function under a
    constant rate the right-hand side does not change in time, the integrator is exact, and there is
    no temporal error to measure -- the check would pass having tested nothing. No reinitialisation,
    because its schedule is in steps, so refining dt would change how often it runs.
    """
    grid = Grid((64, 64), 10.0, (False, True))
    z, x = coordinate_fields(grid)
    phi0 = ((z - 320.0) ** 2 + (x - 320.0) ** 2 - 200.0 ** 2) / 400.0
    duration = 10.0

    runs = {n: np.asarray(solve(phi0, {"v0_nm_per_s": jnp.float64(RATE)},
                                constant_rate("v0_nm_per_s"),
                                SolvePlan(grid, duration / n, n)).phi)
            for n in (20, 40, 80, 160, 1280)}
    near = np.abs(runs[1280]) < 30.0
    levels = (20, 40, 80, 160)
    errors = [float(np.max(np.abs(runs[n] - runs[1280])[near])) for n in levels]
    dts = [duration / n for n in levels]
    orders = _orders(errors, dts)
    fit = float(np.polyfit(np.log(dts), np.log(errors), 1)[0])

    ledger_measure(dt_s=dts, linf_error=errors, orders=orders, order=orders[-1],
                   least_squares_order_all_levels=fit, required=1.9, finding="S13.2")
    assert orders[-1] >= 1.9


@pytest.mark.check("V5")
def test_v5_manufactured_solution(ledger_measure):
    """phi_ex = z - 160 + R t + A sin(kx). Substituting into phi_t - R|grad phi| = S gives
    S = R - R sqrt(1 + (A k cos kx)^2). Refine dx at constant CFL; expect the scheme's order, 1.

    Error is measured over the middle 40 % of the column, away from the Neumann rows, where a linear
    field meets a zero-gradient boundary condition that the manufactured solution does not satisfy.
    """
    amplitude = 20.0
    l1, linf, spacings = [], [], []
    for dx in (8.0, 4.0, 2.0, 1.0):
        n = int(round(320.0 / dx))
        grid = Grid((n, n), dx, (False, True))
        k = 2 * np.pi / (n * dx)
        z, x = coordinate_fields(grid)

        def exact(t):
            return z - 160.0 + RATE * t + amplitude * jnp.sin(k * x)

        def source(g, t, k=k, x=x):
            return RATE - RATE * jnp.sqrt(1.0 + (amplitude * k * jnp.cos(k * x)) ** 2)

        steps = int(round(10.0 / (0.2 * dx) * RATE))
        result = solve(exact(0.0), {"v0_nm_per_s": jnp.float64(RATE)},
                       constant_rate("v0_nm_per_s"),
                       SolvePlan(grid, 10.0 / steps, steps, mms_source=source))
        error = np.abs(np.asarray(result.phi) - np.asarray(exact(10.0)))[int(0.3 * n):int(0.7 * n)]
        l1.append(float(error.mean()))
        linf.append(float(error.max()))
        spacings.append(dx)

    o1, oinf = _orders(l1, spacings), _orders(linf, spacings)
    ledger_measure(dx_nm=spacings, l1_error=l1, linf_error=linf, l1_orders=o1, linf_orders=oinf,
                   order_l1=o1[-1], order_linf=oinf[-1], expected_order=1.0)
    assert o1[-1] >= 0.9 and oinf[-1] >= 0.9


def test_the_mms_source_is_off_in_production_runs():
    """PRD §8.2: the MMS path must be behind a flag, and a test must show the flag is off."""
    case = load_case("configs/dev/S00.yaml")
    assert SolvePlan.from_case(case).mms_source is None
