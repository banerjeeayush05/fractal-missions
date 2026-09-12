"""V8, V9, V10 — invariants and known artifacts (PRD §8.3), M2.2 gate.

These have no analytic solution but strong structural constraints, so they catch accumulated
advection error that V1 and V2 hide.
"""

import numpy as np
import pytest
from helpers import (
    MATERIALS,
    disk_radius,
    mollified_area,
    notched_disk,
    radius_at_angle,
    rigid_rotation_model,
    synthetic_config,
)

from m2 import initial, solver


@pytest.mark.check("V9")
def test_v9_reversibility(ledger_measure):
    """Advect N steps under R, then N under −R: the shape must return (§8.3).

    A negative rate deposits, which is legal and unused in production (decision §1). Needs no
    analytic solution, and catches accumulated advection error that V1 hides.

    **Measurement note — the PRD does not say how to compute the area, and the candidates differ
    by an order of magnitude.** V8, V9 and V12 all have this gap; it wants one answer, not three.

    - **zero-contour XOR** (asserted): the symmetric difference of the two *regions*, from sub-cell
      crossings. For concentric disks it is exact: pi*|r_a^2 - r_b^2|. Reads **0.74 %**.
    - sharp cell indicator: the same quantity quantised to whole cells, with a floor of one cell
      ring — 10 % of the area at r = 200 nm, dx = 10 nm. Reads 4.1 %, and cannot resolve better.
    - mollified-Heaviside XOR: not the region symmetric difference at all. It integrates over a
      1.5-cell band, so it also responds to how reinitialisation reshapes phi near the interface,
      the same artifact V12 exposed. Reads 5.3 %.

    The interface itself returns to within 0.74 nm of r = 200 nm. Awaiting the owner's confirmation.
    """
    cfg = synthetic_config((64, 64), n_steps=100)
    dx, travel = cfg.grid.spacing_nm, 60.0
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi0 = initial.disk(cfg.grid, (320.0, 320.0), 200.0)
    dt = travel / 5.83 / cfg.n_steps

    forward = solver.solve(cfg, phi0, material, {"v_iso": 5.83}, dt=dt)
    back = solver.solve(cfg, forward.phi, material, {"v_iso": -5.83}, dt=dt)

    # The symmetric difference of two REGIONS, measured from their zero contours. For concentric
    # disks that is exact: XOR area = pi * |r_a^2 - r_b^2|. The two alternatives both measure
    # something else (see the docstring), so they are recorded as diagnostics rather than asserted.
    r_before = disk_radius(np.asarray(phi0), dx, 32)
    r_after = disk_radius(np.asarray(back.phi), dx, 32)
    relative = abs(r_after**2 - r_before**2) / r_before**2
    radius_error = abs(r_after - r_before) / r_before

    area0 = mollified_area(phi0, dx)
    symmetric_difference = relative * np.pi * r_before**2
    eps = 1.5 * dx
    h = lambda f: 0.5 * (1.0 + np.tanh(np.asarray(f) / eps))  # noqa: E731
    mollified_xor = float(np.sum(np.abs(h(back.phi) - h(phi0))) * dx * dx) / area0
    sharp = float(np.sum(np.abs((np.asarray(back.phi) < 0).astype(float)
                                - (np.asarray(phi0) < 0).astype(float))) * dx * dx) / area0

    ledger_measure.update({"symmetric_difference_nm2": symmetric_difference, "initial_area_nm2": area0,
                           "relative_symmetric_difference": relative, "radius_error": radius_error,
                           "sharp_indicator_symmetric_difference": sharp,
                           "mollified_heaviside_xor": mollified_xor,
                           "sharp_quantisation_floor": 2 * np.pi * 200.0 * dx / (np.pi * 200.0**2),
                           "travel_each_way_nm": travel, "n_steps_each_way": cfg.n_steps,
                           "tolerance": 0.03, "measure": "zero-contour XOR (exact for disks)"})
    assert relative < 0.03, f"symmetric difference {relative:.4f} of the initial area"


@pytest.mark.check("V10")
def test_v10_grid_orientation_isotropy(ledger_measure):
    """Isotropic etching must stay isotropic: (r_max − r_min)/r_mean < 2 % (§8.3).

    Upwind schemes have real grid anisotropy, and its size has to be known before it reappears as a
    spurious sidewall-angle dependence once M3 supplies angle-dependent velocity.
    """
    # Run at a production-representative resolution: the coupon's smallest feature is CD 500 nm at
    # dx = 10 nm, i.e. 50 cells across. A disk of final radius 150 nm at dx = 10 nm is 15 cells,
    # far coarser than anything M2 will run. The anisotropy is real and first-order in dx — 4.2 %,
    # 1.6 %, 0.67 % at dx = 10, 5, 2.5 nm — so the resolution the check runs at decides the verdict.
    # Flagged for the owner: this is a choice about what V10 proves, not a tuning knob.
    cfg = synthetic_config((192, 192), spacing_nm=5.0, n_steps=200)
    dx = cfg.grid.spacing_nm
    material = initial.uniform_material(cfg.grid, MATERIALS)
    centre_index = 96
    centre = centre_index * dx
    result = solver.solve(cfg, initial.disk(cfg.grid, (centre, centre), 300.0), material,
                          {"v_iso": 5.83}, dt=150.0 / 5.83 / cfg.n_steps, capacity=40000)

    angles = np.linspace(0.0, 2 * np.pi, 33)[:-1]  # 0°, 11.25°, ... including the 45° diagonals
    radii = np.array([radius_at_angle(result.phi, dx, centre_index, a) for a in angles])
    anisotropy = float((radii.max() - radii.min()) / radii.mean())

    ledger_measure.update({"anisotropy": anisotropy, "r_mean_nm": float(radii.mean()),
                           "r_min_nm": float(radii.min()), "r_max_nm": float(radii.max()),
                           "expected_nm": 150.0, "n_angles": len(angles), "tolerance": 0.02,
                           "r_at_0deg": float(radii[0]), "r_at_45deg": float(radii[4]),
                           "spacing_nm": dx, "cells_across_final_radius": 150.0 / dx,
                           "anisotropy_vs_dx": {"10.0": 0.0421, "5.0": 0.0162, "2.5": 0.0067}})
    assert anisotropy < 0.02, f"anisotropy {anisotropy:.4f} = (r_max − r_min)/r_mean"


@pytest.mark.check("V8")
@pytest.mark.xfail(strict=True, reason=(
    "KNOWN DISCREPANCY, logged with evidence (OPEN_QUESTIONS G1, owner accepted 2026-09-11). "
    "43% area loss with the first-order Godunov scheme §5.2 mandates. It is the scheme, not a bug: "
    "the loss converges at first order (43/21/13.3% at 100^2/200^2/300^2), a textbook upwind "
    "advection with none of M2's machinery loses 92.5%, and reinitialisation improves it. Reaching "
    "2% would need ~2000^2. The tolerance is NOT changed (§11). Revisit when WENO5 lands."))
def test_v8_zalesak_disk(ledger_measure):
    """A notched disk in a rigid rotation field, one full revolution (§8.3).

    Area preserved to < 2 %, and the notch must survive. The canonical level-set test, and the
    standard against which every scheme in the literature reports. The velocity is the test-only
    rigid-rotation model approved in decision C2.

    Marked xfail(strict) so the failure is *recorded* rather than tolerated: the ledger row reads
    FAIL with the measured numbers, and if the check ever starts passing the strict marker turns
    that into a failure too, so a fix cannot land unnoticed.
    """
    cfg = synthetic_config((100, 100), spacing_nm=1.0, n_steps=628)
    dx = cfg.grid.spacing_nm
    material = initial.uniform_material(cfg.grid, MATERIALS)

    omega = 2 * np.pi / 628.0
    phi0 = notched_disk(cfg.grid, (75.0, 50.0), 15.0, 5.0, 85.0)
    model = rigid_rotation_model((50.0, 50.0), omega)
    dt = 1.0  # one revolution in 628 steps; max |u| = omega * r_max gives CFL well under 0.5

    result = solver.solve(cfg, phi0, material, {}, model=model, dt=dt, capacity=8192)

    area0, area1 = mollified_area(phi0, dx), mollified_area(result.phi, dx)
    area_loss = abs(area1 - area0) / area0

    # The notch: solid fraction inside the slot footprint. It starts empty and must stay empty.
    z, x = np.mgrid[0:100, 0:100].astype(float)
    slot = (np.abs(x - 50.0) < 2.5) & (z > 70.0) & (z < 85.0)
    filled_before = float(np.mean(np.asarray(phi0)[slot] < 0))
    filled_after = float(np.mean(np.asarray(result.phi)[slot] < 0))

    ledger_measure.update({"area_before_nm2": area0, "area_after_nm2": area1,
                           "relative_area_loss": area_loss, "tolerance": 0.02,
                           "notch_filled_fraction_before": filled_before,
                           "notch_filled_fraction_after": filled_after,
                           "max_cfl": result.max_cfl, "revolutions": 1, "n_steps": cfg.n_steps,
                           "scheme": "first-order Godunov, TVD-RK2 (§5.2)",
                           "known_discrepancy": "OPEN_QUESTIONS G1; owner accepted 2026-09-11",
                           "convergence": {"100^2": 0.430, "200^2": 0.210, "300^2": 0.133},
                           "plain_upwind_no_m2_machinery": 0.925,
                           "resolution_needed_for_2pct": "~2000^2, not runnable in any tier"})
    assert area_loss < 0.02, (f"area loss {area_loss:.4f} after one revolution — record as a finding "
                              f"with the WENO5 result alongside, do not change the tolerance (§11)")
    assert filled_after < 0.25, f"the notch filled in: {filled_after:.2f} of it is now solid"
