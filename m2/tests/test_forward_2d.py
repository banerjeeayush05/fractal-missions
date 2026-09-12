"""V1 and V2 — the forward solve against analytic answers (PRD §8.1).

**Full travel, restored at M2.2.** At M2.1 these ran with the travel held inside the extension band,
because without reinitialisation a band-limited velocity extension distorts φ until the front stalls
(finding F1). Reinitialisation removes that limit, so the checks now run over a travel distance
roughly twice the band width, and the tripwire that guarded the reduced form has become the positive
statement below: with reinitialisation on the front does not stall, with it off it still does.
"""

import dataclasses

import numpy as np
import pytest
from helpers import MATERIALS, crossing_along, disk_radius, synthetic_config

from m2 import initial, solver, velocity
from m2.config import BandConfig

RATE = 5.83  # nm/s, the nominal rate of decision B18
STEPS = 200  # §8.1 specifies 200 steps for V1; V2 shares the setup
TRAVEL_NM = 150.0  # ~2x the 80 nm extension band: the regime that stalled before reinitialisation
BAND = BandConfig(8.0, 2.0, 1.5, None)


def _run(phi0, cfg, travel_nm=TRAVEL_NM):
    dt = travel_nm / RATE / cfg.n_steps
    material = initial.uniform_material(cfg.grid, MATERIALS)
    return solver.solve(cfg, phi0, material, {"v_iso": RATE}, dt=dt)


@pytest.mark.check("V2")
def test_v2_plane_translates_at_exactly_the_etch_rate(ledger_measure):
    """A flat interface under constant R translates at exactly R, with no distortion.
    §8.1 tolerance: error < 0.1 % of the distance travelled (decision C1)."""
    cfg = synthetic_config(n_steps=STEPS, bands=BAND)
    dx, surface = cfg.grid.spacing_nm, 480.0
    result = _run(initial.half_space(cfg.grid, surface), cfg)

    z = crossing_along(np.asarray(result.phi)[:, 32], dx)
    expected = surface - TRAVEL_NM  # a positive rate removes material: the surface recedes (§1)
    error_nm = abs(z - expected)
    relative = error_nm / TRAVEL_NM

    ledger_measure.update({"travel_nm": TRAVEL_NM, "expected_nm": expected, "measured_nm": z,
                           "error_nm": error_nm, "relative_error": relative, "n_steps": STEPS,
                           "max_cfl": result.max_cfl, "peak_occupancy": result.peak_occupancy,
                           "capacity": result.capacity, "n_reinit": cfg.n_reinit,
                           "reinit_every": cfg.reinit_every, "form": "full travel (M2.2)"})
    assert relative < 1e-3, f"plane moved to {z:.6f} nm, expected {expected:.6f} nm"
    assert result.max_cfl <= 0.5


@pytest.mark.check("V1")
def test_v1_isotropic_etch_shrinks_a_disk_at_the_etch_rate(ledger_measure):
    """r(t) = r₀ − R·t (decision §1: a positive rate removes material, so the disk shrinks).
    §8.1 tolerance: relative error < 1 % at 200 steps."""
    cfg = synthetic_config(n_steps=STEPS, bands=BAND)
    dx, r0, centre = cfg.grid.spacing_nm, 300.0, 320.0
    result = _run(initial.disk(cfg.grid, (centre, centre), r0), cfg)

    measured = disk_radius(np.asarray(result.phi), dx, centre_index=32)
    expected = r0 - TRAVEL_NM
    relative = abs(measured - expected) / expected

    ledger_measure.update({"r0_nm": r0, "expected_nm": expected, "measured_nm": measured,
                           "relative_error": relative, "travel_nm": TRAVEL_NM, "n_steps": STEPS,
                           "max_cfl": result.max_cfl, "peak_occupancy": result.peak_occupancy,
                           "n_reinit": cfg.n_reinit, "form": "full travel (M2.2)"})
    assert relative < 1e-2, f"radius {measured:.4f} nm, expected {expected:.4f} nm"


def test_a_single_step_moves_the_interface_by_exactly_rate_times_dt():
    """The sharpest statement available: one step, no reinitialisation, exact to 1e-12."""
    cfg = synthetic_config(n_steps=1, bands=BAND)
    dx, surface, dt = cfg.grid.spacing_nm, 320.0, 0.1285
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi, _, _ = solver.step(initial.half_space(cfg.grid, surface), material, {"v_iso": RATE},
                            grid=cfg.grid, bands=cfg.bands, model=velocity.isotropic,
                            capacity=2048, dt=dt, step_index=0, run_seed=0, time=0.0)
    moved = surface - crossing_along(np.asarray(phi)[:, 32], dx)
    assert moved == pytest.approx(RATE * dt, rel=1e-12)


def test_reinitialisation_is_what_removes_the_band_limit(ledger_measure):
    """M2.1's finding F1, now resolved — and pinned from both sides.

    With reinitialisation the front travels the full distance; with it switched off, the same run
    still stalls once the travel exceeds the extension band. Keeping both halves means the reason
    reinitialisation exists cannot quietly become folklore.
    """
    cfg = synthetic_config(n_steps=STEPS, bands=BAND)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    dt, surface = TRAVEL_NM / RATE / STEPS, 480.0

    on = solver.solve(cfg, initial.half_space(cfg.grid, surface), material, {"v_iso": RATE}, dt=dt)
    moved_on = surface - crossing_along(np.asarray(on.phi)[:, 32], cfg.grid.spacing_nm)

    off_cfg = dataclasses.replace(cfg, n_reinit=0)
    off = solver.solve(off_cfg, initial.half_space(cfg.grid, surface), material, {"v_iso": RATE}, dt=dt)
    moved_off = surface - crossing_along(np.asarray(off.phi)[:, 32], cfg.grid.spacing_nm)

    ledger_measure.update({"travel_nm": TRAVEL_NM, "moved_with_reinit_nm": moved_on,
                           "moved_without_reinit_nm": moved_off,
                           "extension_band_nm": BAND.extension_cells * cfg.grid.spacing_nm})
    assert moved_on == pytest.approx(TRAVEL_NM, rel=1e-6)
    assert moved_off < 0.6 * TRAVEL_NM, "the stall is gone for some other reason: investigate"
