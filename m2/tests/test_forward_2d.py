"""V1 and V2 — the forward solve against analytic answers (PRD §8.1), M2.1 gate.

**Reduced form, and why** (owner decision 2026-09-11, option A). Without reinitialisation, which is
M2.2, the band-limited velocity extension distorts φ a little more each step: φ advances inside the
extension band and not outside it. While the interface has travelled less than the band is wide the
distortion never reaches it and the scheme is exact. Once the travel exceeds the band width, |∇φ| at
the interface departs from 1, the closest-point projection stops landing on the surface, and the
front stalls — demonstrated in `test_band_limit_stalls_the_front_without_reinitialisation` below.

So V1 and V2 run here with the **protocol intact** (200 steps, §8.1's tolerances) and the
**problem reduced** (travel kept inside the band), per OPEN_QUESTIONS B6. The full-travel versions
run at M2.2, where the PRD's own description of V1 — "tests advection, reinitialisation and
extension together" — is finally satisfiable.
"""

import numpy as np
import pytest
from helpers import MATERIALS, crossing_along, disk_radius, synthetic_config

from m2 import initial, solver, velocity
from m2.config import BandConfig

RATE = 5.83  # nm/s, the nominal rate of decision B18
STEPS = 200  # §8.1 specifies 200 steps for V1; V2 shares the setup

# Travel is held to half the untapered extension band, so the distortion of §5.4's band-limited
# extension cannot reach the interface within the run. Band 8 cells, taper 2 -> untapered 6 cells.
TRAVEL_NM = 30.0
BAND = BandConfig(8.0, 2.0, 1.5, None)
UNTAPERED_NM = (BAND.extension_cells - BAND.extension_taper_cells) * 10.0


def _run(phi0, cfg, travel_nm=TRAVEL_NM):
    dt = travel_nm / RATE / cfg.n_steps
    material = initial.uniform_material(cfg.grid, MATERIALS)
    return solver.solve(cfg, phi0, material, {"v_iso": RATE}, dt=dt)


@pytest.mark.check("V2")
def test_v2_plane_translates_at_exactly_the_etch_rate(ledger_measure):
    """A flat interface under constant R translates at exactly R, with no distortion.
    §8.1 tolerance: error < 0.1 % of the distance travelled (decision C1)."""
    cfg = synthetic_config(n_steps=STEPS, bands=BAND)
    dx, surface = cfg.grid.spacing_nm, 320.0
    result = _run(initial.half_space(cfg.grid, surface), cfg)

    z = crossing_along(np.asarray(result.phi)[:, 32], dx)
    expected = surface - TRAVEL_NM  # positive rate removes material: the surface recedes (decision §1)
    error_nm = abs(z - expected)
    relative = error_nm / TRAVEL_NM

    ledger_measure.update({"travel_nm": TRAVEL_NM, "band_untapered_nm": UNTAPERED_NM,
                           "expected_nm": expected, "measured_nm": z, "error_nm": error_nm,
                           "relative_error": relative, "n_steps": STEPS, "max_cfl": result.max_cfl,
                           "peak_occupancy": result.peak_occupancy, "capacity": result.capacity,
                           "form": "reduced: travel within the extension band, reinit is M2.2"})
    assert relative < 1e-3, f"plane moved {z:.6f} nm, expected {expected:.6f} nm"
    assert result.max_cfl <= 0.5


@pytest.mark.check("V1")
def test_v1_isotropic_etch_shrinks_a_disk_at_the_etch_rate(ledger_measure):
    """r(t) = r₀ − R·t (decision §1: a positive rate removes material, so the disk shrinks).
    §8.1 tolerance: relative error < 1 % at 200 steps."""
    cfg = synthetic_config(n_steps=STEPS, bands=BAND)
    dx, r0, centre = cfg.grid.spacing_nm, 300.0, 320.0
    result = _run(initial.disk(cfg.grid, (centre, centre), r0), cfg)

    # The rays start at the centre cell, so a crossing coordinate is already a radius.
    measured = disk_radius(np.asarray(result.phi), dx, centre_index=32)
    expected = r0 - TRAVEL_NM
    relative = abs(measured - expected) / expected

    ledger_measure.update({"r0_nm": r0, "expected_nm": expected, "measured_nm": measured,
                           "relative_error": relative, "travel_nm": TRAVEL_NM, "n_steps": STEPS,
                           "max_cfl": result.max_cfl, "peak_occupancy": result.peak_occupancy,
                           "form": "reduced: travel within the extension band, reinit is M2.2"})
    assert relative < 1e-2, f"radius {measured:.4f} nm, expected {expected:.4f} nm"


def test_a_single_step_moves_the_interface_by_exactly_rate_times_dt():
    """The sharpest statement of correctness available before reinitialisation exists."""
    cfg = synthetic_config(n_steps=1, bands=BAND)
    dx, surface, dt = cfg.grid.spacing_nm, 320.0, 0.1285
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi, _, _ = solver.step(initial.half_space(cfg.grid, surface), material, {"v_iso": RATE},
                            grid=cfg.grid, bands=cfg.bands, model=velocity.isotropic,
                            capacity=2048, dt=dt, step_index=0, run_seed=0, time=0.0)
    moved = surface - crossing_along(np.asarray(phi)[:, 32], dx)
    assert moved == pytest.approx(RATE * dt, rel=1e-12)


def test_band_limit_stalls_the_front_without_reinitialisation():
    """Executable record of the M2.1 limitation, so it cannot be forgotten or rediscovered.

    Travel well beyond the extension band: φ's distortion reaches the interface, |∇φ| leaves 1, and
    the front stops. This is what reinitialisation (M2.2) repairs, and why V1/V2 run in reduced form
    here. If this test ever starts passing at full travel, reinitialisation has landed and the full
    V1/V2 belong back in the gate.
    """
    cfg = synthetic_config(n_steps=STEPS, bands=BAND)
    far = 150.0  # nm, ~2x the 80 nm extension band
    result = _run(initial.half_space(cfg.grid, 320.0), cfg, travel_nm=far)
    moved = 320.0 - crossing_along(np.asarray(result.phi)[:, 32], cfg.grid.spacing_nm)
    assert moved < 0.6 * far, ("the front no longer stalls beyond the band: reinitialisation may "
                               "have landed, so restore full-travel V1/V2 to the gate")


def test_wider_band_restores_exactness_diagnostic_only():
    """The mechanism, pinned: with the band wider than the travel, the same run is exact.

    Diagnostic only. §11 forbids tuning a numerical parameter to make a check pass, so the shipped
    band stays at decision C10's 8 cells and V1/V2 reduce their travel instead.

    The residual here is ~7e-7 relative, from the bilinear sampling in the closest-point gather, not
    from advection: a single step is exact to 1e-12 (test above). Contrast with the 47 % shortfall
    when the travel exceeds the band.
    """
    wide = synthetic_config(n_steps=STEPS, bands=BandConfig(20.0, 2.0, 1.5, None))
    far = 150.0
    result = _run(initial.half_space(wide.grid, 320.0), wide, travel_nm=far)
    moved = 320.0 - crossing_along(np.asarray(result.phi)[:, 32], wide.grid.spacing_nm)
    assert moved == pytest.approx(far, rel=1e-5), f"moved {moved!r}"
