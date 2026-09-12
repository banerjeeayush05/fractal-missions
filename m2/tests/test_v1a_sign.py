"""V1a — the forward sign check (decision §1), M2.1 gate.

V14–V16 are blind to a sign inversion: they verify the derivative of the function you wrote, not of
the function you meant. A globally flipped convention produces a perfectly consistent gradient of a
model that deposits when it should etch. This check is the only guard.

Under the directional law R = v_iso + v_dir·max(0, n·ẑ)^p with ẑ toward the plasma and v_iso = 0:
- a trench floor (open volume above it) faces the plasma and must recede **away** from the plasma;
- a downward-facing overhang faces away and must not move at all;
- a vertical sidewall has n·ẑ = 0 and must not move either.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from helpers import MATERIALS, crossing_along, synthetic_config

from m2 import initial, solver
from m2.constants import PHI_OPEN_SIGN, PHI_SOLID_SIGN, Z_AXIS, Z_HAT_SIGN

V_DIR, P, TRAVEL_NM, STEPS = 5.25, 2.0, 30.0, 50


def _run(phi0, cfg):
    material = initial.uniform_material(cfg.grid, MATERIALS)
    dt = TRAVEL_NM / V_DIR / cfg.n_steps
    return solver.solve(cfg, phi0, material, {"v_iso": 0.0, "v_dir": V_DIR, "p": P}, dt=dt)


def _cfg():
    return synthetic_config(n_steps=STEPS, model="directional", v_iso=0.0, v_dir=V_DIR, p=P)


@pytest.mark.check("V1a")
def test_floor_recedes_away_from_the_plasma(ledger_measure):
    cfg = _cfg()
    surface = 320.0
    result = _run(initial.half_space(cfg.grid, surface, solid_below=True), cfg)
    z = crossing_along(np.asarray(result.phi)[:, 32], cfg.grid.spacing_nm)
    moved = surface - z

    assert z < surface, "an up-facing surface must recede away from the plasma, not toward it"
    assert moved == pytest.approx(TRAVEL_NM, rel=2e-3), f"moved {moved:.4f} nm of {TRAVEL_NM}"
    ledger_measure.update({"surface_nm": surface, "final_nm": z, "moved_nm": moved,
                           "expected_nm": TRAVEL_NM, "axis0_increases_toward": "plasma",
                           "z_hat_axis": Z_AXIS, "z_hat_sign": Z_HAT_SIGN})


@pytest.mark.check("V1a")
def test_downward_facing_surface_does_not_move(ledger_measure):
    """An overhang underside: n·ẑ < 0, so the directional law gives exactly zero rate."""
    cfg = _cfg()
    phi0 = initial.half_space(cfg.grid, 320.0, solid_below=False)  # solid above, open below
    result = _run(phi0, cfg)
    drift = float(np.max(np.abs(np.asarray(result.phi) - np.asarray(phi0))))
    assert drift == 0.0, f"a downward-facing surface moved by {drift:g} nm"
    ledger_measure["overhang_drift_nm"] = drift


@pytest.mark.check("V1a")
def test_vertical_sidewall_does_not_move(ledger_measure):
    """n·ẑ = 0 on a vertical wall — the most common surface in a trench, and the kink in the law.

    The geometry is a **slab**, not a single wall. Lateral boundaries are periodic (§5.1), and a φ
    that increases monotonically across x is not periodic: it jumps by the domain width at the seam,
    which reinitialisation then correctly repairs. A slab's signed distance — to the nearer periodic
    image — is genuinely periodic, and comes back bit-for-bit unchanged.
    """
    cfg = _cfg()
    dx = cfg.grid.spacing_nm
    x = np.arange(cfg.grid.shape[1]) * dx
    extent = cfg.grid.shape[1] * dx
    distance = np.minimum(np.abs(x - 320.0), extent - np.abs(x - 320.0)) - 100.0
    phi0 = np.broadcast_to(distance, cfg.grid.shape).astype(np.float64)

    result = _run(jnp.asarray(phi0), cfg)
    drift = float(np.max(np.abs(np.asarray(result.phi) - phi0)))
    assert drift == 0.0, f"a vertical sidewall moved by {drift:g} nm"
    ledger_measure.update({"sidewall_drift_nm": drift, "geometry": "periodic slab",
                           "note": "a monotone-in-x wall is not periodic; the seam is not a sidewall"})


@pytest.mark.check("V1a")
def test_the_convention_itself_is_what_is_being_checked(ledger_measure):
    """If the sign of the advection term were flipped, the floor would advance toward the plasma.
    Stated here so the check's purpose survives a future reader."""
    cfg = _cfg()
    surface = 320.0
    phi0 = initial.half_space(cfg.grid, surface, solid_below=True)
    below, above = surface - 50.0, surface + 50.0
    z_index = lambda z: int(z / cfg.grid.spacing_nm)  # noqa: E731
    assert np.sign(np.asarray(phi0)[z_index(below), 32]) == PHI_SOLID_SIGN
    assert np.sign(np.asarray(phi0)[z_index(above), 32]) == PHI_OPEN_SIGN
    result = _run(phi0, cfg)
    after = np.asarray(result.phi)[z_index(surface - 0.5 * TRAVEL_NM), 32]
    assert np.sign(after) == PHI_OPEN_SIGN, "material that should have been removed is still solid"
    ledger_measure["solid_became_open"] = True
