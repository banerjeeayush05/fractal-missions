"""V11 and V12 — reinitialisation (PRD §8.3), M2.2 gate.

V12 is the one the PRD calls the most commonly missed check in level-set codes: a reinitialisation
that quietly shifts the zero level set produces a systematic etch-rate bias that looks like physics,
calibrates away into the closure parameters, and then fails to transfer.

**How the enclosed area is measured is a live question — see the V12 test.** The PRD says "measure
the enclosed area/volume" without saying how, and the two obvious measures disagree here.
"""

import numpy as np
import pytest
from helpers import MATERIALS, disk_radius, synthetic_config

from m2 import initial, reinit, solver, velocity
from m2.constants import HEAVISIDE_WIDTH_CELLS
from m2.stencils import normals

RATE = 5.83
RADIUS = 200.0
CENTRE = 320.0


def _setup():
    cfg = synthetic_config((64, 64), n_steps=200)
    return cfg, initial.disk(cfg.grid, (CENTRE, CENTRE), RADIUS), initial.uniform_material(
        cfg.grid, MATERIALS)


def _grad_error(phi, cfg, within_cells=3.0):
    _, norm = normals(phi, cfg.grid)
    band = np.abs(np.asarray(phi)) < within_cells * cfg.grid.spacing_nm
    return float(np.max(np.abs(np.asarray(norm)[band] - 1.0)))


def _mollified_area(phi, cfg, width_cells=HEAVISIDE_WIDTH_CELLS):
    eps = width_cells * cfg.grid.spacing_nm
    heaviside = 0.5 * (1.0 + np.tanh(np.asarray(phi) / eps))
    return float(np.sum(1.0 - heaviside) * cfg.grid.spacing_nm ** cfg.grid.ndim)


@pytest.mark.check("V11")
def test_reinitialisation_restores_the_distance_property(ledger_measure):
    """After a reinit cycle, max ||∇φ| − 1| within 3 cells of the interface < 5 % (§8.3).

    Run in the operational regime: advect `reinit_every` steps at the CFL target, then one cycle of
    `n_reinit` iterations — which is exactly what the solver's fixed schedule does.
    """
    cfg, phi, material = _setup()
    dx = cfg.grid.spacing_nm
    dt = cfg.cfl_target * dx / RATE

    for k in range(cfg.reinit_every):
        phi, _, _ = solver.step(phi, material, {"v_iso": RATE}, grid=cfg.grid, bands=cfg.bands,
                                model=velocity.isotropic, capacity=4096, dt=dt, step_index=k,
                                run_seed=0, time=k * dt, n_reinit=0)
    before = _grad_error(phi, cfg)
    after = _grad_error(reinit.reinitialise(phi, cfg.grid, cfg.n_reinit), cfg)

    reach = reinit.repair_reach_cells(cfg.n_reinit)
    motion = reinit.interface_motion_cells(cfg.cfl_target, cfg.reinit_every)
    ledger_measure.update({"grad_error_before": before, "grad_error_after": after,
                           "tolerance": 0.05, "n_reinit": cfg.n_reinit,
                           "reinit_every": cfg.reinit_every, "repair_reach_cells": reach,
                           "interface_motion_cells_per_cycle": motion,
                           "reach_margin": reach - motion})
    assert after < 0.05, f"max ||∇φ|−1| = {after:.4f} within 3 cells"
    # The repair must reach at least as far as the interface moves between cycles, or distortion
    # accumulates exactly as M2.1's finding F1 described.
    assert reach >= motion, f"repair reaches {reach} cells but the interface moves {motion}"


@pytest.mark.check("V12")
def test_reinitialisation_does_not_move_the_interface(ledger_measure):
    """Drift per cycle < 0.1 %, with no advection (§8.3).

    **Measurement note, flagged for review.** The PRD does not say how to measure the enclosed area,
    and the two natural measures disagree:

    - the **zero contour** (sub-cell crossings): drift 0.000 % — the interface does not move at all;
    - a **mollified Heaviside** integral: 0.115 % at the 1.5·dx width, and it grows with the
      smoothing width (0.068 % at 0.5·dx, 0.157 % at 3·dx).

    The mollified integral responds to changes in φ anywhere inside its smoothing band, so it moves
    when reinitialisation reshapes φ *near* the interface even though the zero level set is fixed.
    Since the check's stated purpose is that reinitialisation "does not move the interface", this
    asserts the contour measure and records both. Awaiting the owner's confirmation.
    """
    cfg, phi, _ = _setup()
    after = reinit.reinitialise(phi, cfg.grid, cfg.n_reinit)

    r_before = disk_radius(np.asarray(phi), cfg.grid.spacing_nm, centre_index=32)
    r_after = disk_radius(np.asarray(after), cfg.grid.spacing_nm, centre_index=32)
    contour_drift = abs(r_after**2 - r_before**2) / r_before**2

    mollified = {f"width_{w}_dx": abs(_mollified_area(after, cfg, w) - _mollified_area(phi, cfg, w))
                 / _mollified_area(phi, cfg, w) for w in (0.5, 1.0, 1.5, 2.0, 3.0)}

    ledger_measure.update({"contour_area_drift": contour_drift,
                           "contour_radius_before_nm": r_before, "contour_radius_after_nm": r_after,
                           "radius_shift_nm": abs(r_after - r_before),
                           "mollified_area_drift": mollified, "tolerance": 1e-3,
                           "measure_asserted": "zero contour (sub-cell crossings)",
                           "note": "mollified-Heaviside measure reads 0.115% at 1.5dx: see docstring"})
    assert contour_drift < 1e-3, f"the zero contour moved: area drift {contour_drift:.5f}"


def test_the_two_area_measures_disagree_and_the_difference_is_the_smoothing(ledger_measure=None):
    """Pins the reason the measures differ, so the V12 choice is not re-litigated from scratch."""
    cfg, phi, _ = _setup()
    after = reinit.reinitialise(phi, cfg.grid, cfg.n_reinit)
    drifts = [abs(_mollified_area(after, cfg, w) - _mollified_area(phi, cfg, w))
              / _mollified_area(phi, cfg, w) for w in (0.5, 1.0, 1.5, 2.0, 3.0)]
    assert drifts == sorted(drifts), "drift should grow with the smoothing width"
    assert drifts[0] < drifts[-1] / 2, "the measured drift is dominated by the smoothing width"


def test_reinitialisation_uses_a_smoothed_sign_not_jnp_sign():
    """§5.3: the exact sign is non-differentiable at the interface and corrupts the adjoint."""
    import jax

    cfg, phi, _ = _setup()
    s = reinit.smoothed_sign(phi, cfg.grid)
    assert float(np.max(np.abs(np.asarray(s)))) < 1.0  # never saturates to ±1
    d = jax.grad(lambda p: reinit.smoothed_sign(p, cfg.grid).sum())(phi)
    assert np.all(np.isfinite(np.asarray(d)))


def test_iteration_count_is_fixed_and_zero_is_a_no_op():
    cfg, phi, _ = _setup()
    assert np.array_equal(np.asarray(reinit.reinitialise(phi, cfg.grid, 0)), np.asarray(phi))
    once = reinit.reinitialise(phi, cfg.grid, 1)
    twice = reinit.reinitialise(phi, cfg.grid, 2)
    assert not np.array_equal(np.asarray(once), np.asarray(twice))
    with pytest.raises(ValueError):
        reinit.reinitialise(phi, cfg.grid, -1)
