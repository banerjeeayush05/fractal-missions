"""V14a, V14b, V14c — sensitivities with closed-form answers (decision §5), M2.3 gate.

V15 and V16 verify transpose consistency, not derivative correctness: JAX builds reverse mode by
linearising with the JVP rules and transposing them, so both share those rules. That leaves V14 as
the only check comparing a derivative against the function — and V14 answers "is this consistent",
not "is this the right number". These three answer the second question, against exact analysis.

**How the tolerances were set, since that is the place cherry-picking would hide.** Each bound is
derived from the scheme's order and the feature size *before* looking at the result, and the
measured value is recorded either way:

- V14a, a curved interface under a first-order scheme: relative error is O(dx / r), so the bound is
  dx / r. Refinement must also reduce the error, which is a statement no single threshold can fake.
- V14b and V14c, planar interfaces: motion is exact for a plane (V2 measures 5e-8), so the residual
  comes from the mollified Heaviside's width against the measured extent — the bound is
  1.5·dx / extent.

A bound loose enough to admit the truth is also loose enough to admit some error; that is why V14a
carries the refinement condition, and why the V19 canary runs against these same objectives.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import MATERIALS, consistent_config

from m2 import functionals, initial, solver
from m2.constants import HEAVISIDE_WIDTH_CELLS
from m2.stencils import cell_centres

RATE, R0 = 5.83, 200.0
V_ISO, V_DIR, P = 0.58, 5.25, 2.0
ALPHA = np.deg2rad(30.0)  # angle between the tilted surface's normal and ẑ


def _disk_sensitivity(dx, n):
    """dr/dR for an isotropically etched disk, via the smooth volumetric functional."""
    cfg = consistent_config(50.0, spacing_nm=dx, shape=(n, n), model="isotropic", v_iso=RATE)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    centre = n * dx / 2
    phi0 = initial.disk(cfg.grid, (centre, centre), R0)
    assert solver.solve(cfg, phi0, material, {"v_iso": RATE}, capacity=16384).max_cfl <= 0.5

    def J(p):
        return functionals.solid_volume(solver.final_phi(cfg, phi0, material, p, capacity=16384),
                                        cfg.grid)

    grad = jax.grad(J)({"v_iso": RATE})["v_iso"]
    T = cfg.final_time_s
    r_end = R0 - RATE * T
    return float(grad / (2 * np.pi * r_end)), -T, dx / r_end


@pytest.mark.check("V14a")
def test_v14a_isotropic_disk_dr_dR_equals_minus_T(ledger_measure):
    """r(t) = r₀ − R·t, so dr/dR = −T exactly (decision §1: a positive rate removes material)."""
    coarse, exact, bound_coarse = _disk_sensitivity(10.0, 64)
    fine, _, bound_fine = _disk_sensitivity(5.0, 128)
    err_coarse = abs(coarse - exact) / abs(exact)
    err_fine = abs(fine - exact) / abs(exact)

    ledger_measure.update({"dr_dR_dx10": coarse, "dr_dR_dx5": fine, "exact_minus_T": exact,
                           "rel_error_dx10": err_coarse, "rel_error_dx5": err_fine,
                           "a_priori_bound_dx10": bound_coarse, "a_priori_bound_dx5": bound_fine,
                           "refinement_ratio": err_coarse / err_fine,
                           "bound_rule": "dx / r: first-order scheme on a curved interface"})
    assert err_coarse < bound_coarse, f"{err_coarse:.3e} exceeds the a priori bound {bound_coarse:.3e}"
    assert err_fine < bound_fine
    assert err_fine < err_coarse, "refining the grid did not reduce the error: this is not discretisation"
    assert coarse < 0, "a faster etch must shrink the disk"


@pytest.mark.check("V14b")
def test_v14b_plane_translation_dz_dR_equals_minus_T(ledger_measure):
    """A flat interface translates at exactly R, so dz/dR = −T."""
    dx, n = 10.0, 64
    cfg = consistent_config(50.0, spacing_nm=dx, shape=(n, n), model="isotropic", v_iso=RATE)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi0 = initial.half_space(cfg.grid, n * dx / 2)
    assert solver.solve(cfg, phi0, material, {"v_iso": RATE}, capacity=16384).max_cfl <= 0.5

    def J(p):
        return functionals.solid_volume(solver.final_phi(cfg, phi0, material, p, capacity=16384),
                                        cfg.grid)

    width, T = n * dx, cfg.final_time_s
    dz_dR = float(jax.grad(J)({"v_iso": RATE})["v_iso"] / width)
    error = abs(dz_dR - (-T)) / T
    bound = HEAVISIDE_WIDTH_CELLS * dx / (n * dx)

    ledger_measure.update({"dz_dR": dz_dR, "exact_minus_T": -T, "rel_error": error,
                           "a_priori_bound": bound,
                           "bound_rule": "1.5*dx / domain height: plane motion is exact, so the "
                                         "residual is the mollification width against the extent"})
    assert error < bound, f"{error:.3e} exceeds the a priori bound {bound:.3e}"
    assert dz_dR < 0


@pytest.mark.check("V14c")
def test_v14c_tilted_plane_under_the_directional_law(ledger_measure):
    """A tilted plane moves along its normal at R = v_iso + v_dir·cos^p α, so all three
    derivatives are exact — including ∂/∂p, which is the one that exercises the power and log path
    and the double-where guard that keeps 0^p out of the derivative.

    The functional is windowed away from the lateral seam: a tilted plane is not periodic, so φ
    jumps by the domain width where the lateral boundary wraps (finding G4). The window is static,
    so it narrows where the functional looks without adding any parameter dependence, and the seam's
    influence cannot reach it within this run.
    """
    dx, nz, nx = 10.0, 96, 128
    cfg = consistent_config(50.0, spacing_nm=dx, shape=(nz, nx), model="directional",
                            v_iso=V_ISO, v_dir=V_DIR, p=P)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    coords = cell_centres(cfg.grid)
    phi0 = jnp.cos(ALPHA) * (coords[0] - nz * dx / 2) + jnp.sin(ALPHA) * (coords[1] - nx * dx / 2)

    lo, hi = int(0.3 * nx), int(0.7 * nx)  # 30 % clear of each seam
    window = jnp.asarray(np.pad(np.ones((nz, hi - lo)), ((0, 0), (lo, nx - hi))))
    width = (hi - lo) * dx

    params = {"v_iso": V_ISO, "v_dir": V_DIR, "p": P}
    assert solver.solve(cfg, phi0, material, params, capacity=40000).max_cfl <= 0.5

    def J(p):
        return functionals.solid_volume_in_window(
            solver.final_phi(cfg, phi0, material, p, capacity=40000), cfg.grid, window)

    grad = jax.grad(J)(params)
    T, ca = cfg.final_time_s, float(np.cos(ALPHA))
    exact = {"v_iso": -T * width / ca,
             "v_dir": -T * width * ca ** (P - 1),
             "p": -T * width * V_DIR * ca ** (P - 1) * np.log(ca)}
    errors = {k: abs(float(grad[k]) - exact[k]) / abs(exact[k]) for k in exact}
    bound = HEAVISIDE_WIDTH_CELLS * dx / width

    ledger_measure.update({"numerical": {k: float(v) for k, v in grad.items()}, "exact": exact,
                           "rel_errors": errors, "a_priori_bound": bound,
                           "tilt_deg": float(np.degrees(ALPHA)), "window_nm": width,
                           "bound_rule": "1.5*dx / window width: plane motion is exact"})
    for name, error in errors.items():
        assert error < bound, f"d/d{name}: {error:.3e} exceeds the a priori bound {bound:.3e}"
    assert grad["p"] > 0, "a larger exponent slows the etch, so more solid remains"
    assert grad["v_iso"] < 0 and grad["v_dir"] < 0


def test_the_p_derivative_survives_a_surface_facing_away_from_the_plasma():
    """0^p is the trap: its p-derivative is −∞·0 in the naive form, and a NaN in an untaken branch
    still poisons the backward pass. A downward-facing surface has n·ẑ < 0 everywhere."""
    dx, n = 10.0, 48
    cfg = consistent_config(20.0, spacing_nm=dx, shape=(n, n), model="directional",
                            v_iso=V_ISO, v_dir=V_DIR, p=P)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi0 = initial.half_space(cfg.grid, n * dx / 2, solid_below=False)  # a ceiling

    def J(p):
        return functionals.solid_volume(solver.final_phi(cfg, phi0, material, p, capacity=8192),
                                        cfg.grid)

    grad = jax.grad(J)({"v_iso": V_ISO, "v_dir": V_DIR, "p": P})
    assert np.isfinite(float(grad["p"])), "0^p produced a NaN in the p-derivative"
    assert float(grad["v_dir"]) == 0.0, "a surface facing away from the plasma has no directional term"
    assert float(grad["v_iso"]) < 0.0, "the isotropic term still etches it"
