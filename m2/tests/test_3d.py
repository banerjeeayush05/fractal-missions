"""3D: the same solver, the same gradient checks, one more axis (PRD §6 M2.4).

There is almost nothing to build here, and that is the architectural decision of §4 being repaid.
Dense storage with static shapes means the third axis is a shape change, not a code path: `shift`,
the Godunov Hamiltonian, the band assembly and the closest-point gather are all written over
`grid.ndim`. So this file is mostly *verification* that the 2D results carry over, rather than new
machinery — which is exactly what should happen if the 2D work was done properly.

**Size.** The M2.4 gate names case S03 in 3D: 270 × 100 × 100 at dx = 10 nm, N = 625, 21.6 MB per
fp64 field. Its *gradient* needs about 20.6 GB under the best schedule (finding I4), so it does not
run on this 8 GB machine and nothing here claims it does. These checks run the same geometry at a
size that fits, and the gate case is projected from the measured residual factor. **The S03 3D
gradient and the 40 GB peak-memory figure both need the H100 run** (decision X13).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import MATERIALS, consistent_config, synthetic_config

from m2 import functionals, initial, solver
from m2.checkpoint import peak_field_equivalents, persistent_residuals
from m2.config import parameter_tree
from m2.stencils import cell_centres
from m2.verification.gradcheck import (
    dot_product_test,
    forward_reverse_test,
    reverse_gradient,
    reverse_vjp,
    taylor_test,
)

KEY = jax.random.PRNGKey(2026)


def _sphere(grid, radius):
    centre = [n * grid.spacing_nm / 2 for n in grid.shape]
    return jnp.sqrt(sum((cell_centres(grid)[i] - centre[i]) ** 2
                        for i in range(grid.ndim))) - radius


def _case(shape=(32, 32, 32), dx=10.0, travel=60.0, scheme="godunov"):
    # `consistent_config`, not `synthetic_config`: N must be derived from the travel, or dt comes
    # from an unrelated depth and the CFL bound is violated on the first step.
    cfg = dataclasses.replace(
        consistent_config(travel, spacing_nm=dx, shape=shape, model="directional",
                          v_iso=0.58, v_dir=5.25, p=2.0),
        spatial_scheme=scheme)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    return cfg, material, _sphere(cfg.grid, 100.0), parameter_tree(cfg)


def _objective(cfg, material, phi0, *, levels=0, segment=None, capacity=20000):
    def J(params):
        return functionals.solid_volume(
            solver.final_phi(cfg, phi0, material, params, capacity=capacity,
                             levels=levels, segment=segment), cfg.grid)

    return J


def test_the_solver_runs_in_3d():
    """The smoke test that has to pass before any 3D claim means anything."""
    cfg, material, phi0, (params, _) = _case()
    assert cfg.grid.ndim == 3
    assert cfg.grid.periodic == (False, True, True), "axis 0 is vertical and must not wrap"
    result = solver.solve(cfg, phi0, material, params, capacity=20000)
    assert result.max_cfl <= 0.5
    assert np.all(np.isfinite(np.asarray(result.phi)))
    assert result.peak_occupancy < result.capacity


@pytest.mark.check("V14")
@pytest.mark.nightly
def test_v14_in_3d(ledger_measure):
    """The M2.4 gate's headline: the Taylor test through a 3D solve.

    Run on a sphere rather than the coupon trench: a sphere exercises every octant of the stencil
    at once and has no flat faces where a broken axis could hide behind a zero derivative. The
    coupon geometry is checked separately below for the *shape*, where the analytic answer is known.
    """
    cfg, material, phi0, (params, scales) = _case()
    run = solver.solve(cfg, phi0, material, params, dt=cfg.dt_s, capacity=20000)
    assert run.max_cfl <= 0.5
    J = _objective(cfg, material, phi0)

    result = taylor_test(J, params, reverse_gradient(J, params), scales=scales, key=KEY,
                         n_steps=cfg.n_steps)
    ledger_measure.update({**result.measured, "message": result.message, "dimension": 3,
                           "grid": list(cfg.grid.shape), "n_steps": cfg.n_steps,
                           "max_cfl": run.max_cfl, "geometry": "sphere",
                           "note": "case S03 in 3D needs ~20.6 GB for its gradient and runs on the "
                                   "H100, not here (decision X13)"})
    assert result.passed, result.message


@pytest.mark.check("V15")
@pytest.mark.nightly
def test_v15_and_v16_in_3d(ledger_measure):
    """Transpose consistency with three axes: a Jacobian transpose that is right in 2D can still be
    wrong in 3D if an axis is dropped or transposed in the reduction."""
    cfg, material, phi0, (params, scales) = _case(shape=(24, 24, 24))
    J = _objective(cfg, material, phi0, capacity=12000)
    F = lambda p: solver.final_phi(cfg, phi0, material, p, capacity=12000)  # noqa: E731

    v15 = forward_reverse_test(J, params, reverse_gradient(J, params), scales=scales, key=KEY)
    v16 = dot_product_test(F, params, reverse_vjp(F, params), scales=scales, key=KEY)
    ledger_measure.update({"dimension": 3, "v15_max_rel_error": v15.measured["max_rel_error"],
                           "v16_max_rel_error": v16.measured["max_rel_error"]})
    assert v15.passed, v15.message
    assert v16.passed, v16.message


@pytest.mark.check("V17")
@pytest.mark.nightly
def test_v17_in_3d(ledger_measure):
    """Checkpointing must be identical in 3D too — it is where it will actually be used, since 2D
    fits in memory unchecked and 3D does not."""
    cfg, material, phi0, (params, _) = _case()
    base = jax.grad(_objective(cfg, material, phi0))(params)
    errors = {}
    for levels, segment in ((2, 1), (3, 5)):
        checked = jax.grad(_objective(cfg, material, phi0, levels=levels, segment=segment))(params)
        errors[f"levels{levels}_seg{segment}"] = max(
            abs(float(checked[n]) - float(base[n])) / abs(float(base[n])) for n in base)

    ledger_measure.update({"dimension": 3, "relative_errors": errors, "tolerance": 1e-12})
    for name, error in errors.items():
        assert error < 1e-12, f"3D checkpointed gradient differs by {error:.3e} ({name})"


@pytest.mark.nightly
def test_the_isotropic_sphere_shrinks_at_the_prescribed_rate(ledger_measure):
    """V1's analytic solution, in 3D: r(t) = r₀ − R·t holds for a sphere exactly as for a disk.

    A curvature-dependent error would show here and not in 2D, because a sphere has two principal
    curvatures where a disk has one — so this is not a redundant copy of V1.
    """
    cfg = synthetic_config((48, 48, 48), spacing_nm=10.0, n_steps=60)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    r0, rate, travel = 150.0, 5.83, 60.0
    phi0 = _sphere(cfg.grid, r0)
    result = solver.solve(cfg, phi0, material, {"v_iso": rate},
                          dt=travel / rate / cfg.n_steps, capacity=60000)
    assert result.max_cfl <= 0.5

    # Radius from the sub-cell crossing along +x through the centre, as V1 does.
    centre = 24
    ray = np.asarray(result.phi)[centre, centre, centre:]
    above = np.flatnonzero(ray > 0)
    j = int(above[0])
    frac = -ray[j - 1] / (ray[j] - ray[j - 1])
    measured = 10.0 * (j - 1 + frac)
    expected = r0 - travel

    error = abs(measured - expected) / expected
    ledger_measure.update({"dimension": 3, "radius_nm": measured, "expected_nm": expected,
                           "relative_error": error, "tolerance": 0.02,
                           "cells_across_final_radius": expected / 10.0})
    assert error < 0.02, f"3D radius error {error:.4f} (r = {measured:.2f} vs {expected:.2f})"


@pytest.mark.nightly
def test_the_gate_case_memory_is_projected_from_the_measured_residual_factor(ledger_measure):
    """What case S03 in 3D would need, projected from k measured on a 3D grid here.

    **This is a projection, and the distinction matters for the gate.** k is measured; the peak it
    implies is a model, because the transient replay storage never becomes a jaxpr constant and
    cannot be counted on this machine (see `checkpoint.py`). The 40 GB gate figure must come from a
    profiler on the H100, not from here. What this check *does* establish is that k measured in 3D
    matches k measured in 2D, which is what makes the projection legitimate at all.
    """
    cfg, material, phi0, (params, _) = _case(shape=(24, 24, 24), travel=30.0)
    field_bytes = int(np.prod(cfg.grid.shape)) * 8
    total, _ = persistent_residuals(_objective(cfg, material, phi0, capacity=12000),
                                    params, field_bytes)
    k = total / cfg.n_steps

    gate_field_mb, gate_steps, budget_gb = 21.6, 625, 40.0
    projections = {
        name: peak_field_equivalents(gate_steps, k, levels=levels, segment=segment)
        * gate_field_mb / 1000.0
        for name, (levels, segment) in {"none": (0, None), "two_level_L1": (2, 1),
                                        "three_level_L25": (3, 25)}.items()}

    ledger_measure.update({
        "k_measured_3d": k, "k_measured_2d": 349.2, "grid": list(cfg.grid.shape),
        "gate_case": "S03 3D, 270x100x100, N=625, 21.6 MB/field",
        "projected_peak_gb": projections, "budget_gb": budget_gb,
        "status": "PROJECTION — the 40 GB gate needs a device-memory profile on the H100 (X13)"})

    assert 100.0 < k < 2000.0, f"3D k = {k:.1f} is not in the range 2D measured (349)"
    assert projections["two_level_L1"] < budget_gb, (
        f"the best two-level schedule projects to {projections['two_level_L1']:.1f} GB against a "
        f"{budget_gb} GB budget")


def test_the_capacity_estimate_covers_the_gate_geometry_in_3d(ledger_measure):
    """Finding L1: `estimate_capacity` under-sized the padded request in 3D by about 1.5x.

    The old bound was `2·(nz + prod(shape[1:]))` — an interface *length*, which is right in 2D and
    wrong in 3D, where the interface is a surface. On the gate case (S03 3D, a 2500 nm trench) it
    returned 123,240 against an evaluation band holding 180,000 cells, so a gate run would have
    raised CapacityOverflow about 60 % of the way down the etch — on rented hardware, after paying
    for setup and compile.

    Asserted against the band occupancy counted directly from φ, not against the estimator's own
    arithmetic, so this cannot pass by agreeing with itself.
    """
    from m2.band import estimate_capacity
    from m2.config import BandConfig
    from m2.schema import Grid

    bands = BandConfig(8.0, 2.0, 1.5, None)
    grid = Grid((270, 100, 100), 10.0, (False, True, True))
    capacity = estimate_capacity(grid, bands)

    # The deepest trench the gate case reaches, counted straight from the field.
    phi = initial.trench(grid, 2600.0, 500.0, 2500.0)
    occupied = int((np.abs(np.asarray(phi)) < bands.evaluation_cells * grid.spacing_nm).sum())

    ledger_measure.update({"grid": list(grid.shape), "capacity": capacity,
                           "evaluation_band_cells": occupied,
                           "headroom": 1.0 - occupied / capacity,
                           "old_estimate_before_L1": 123240, "finding": "L1"})
    assert capacity > occupied, (
        f"capacity {capacity:,} is below the {occupied:,} cells the gate geometry puts in the "
        f"evaluation band: a gate run would overflow part-way through")
    assert capacity < 20 * occupied, f"capacity {capacity:,} is wastefully oversized"

    # And the 2D bound must be unchanged, or every existing 2D result moves.
    assert estimate_capacity(Grid((64, 64), 10.0, (False, True)), bands) == 1536

