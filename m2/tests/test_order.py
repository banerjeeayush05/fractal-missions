"""V5, V6, V7 — observed convergence order (PRD §8.2), M2.3 gate.

§8.2's point is that "order studies are what distinguish 'wrong' from 'under-resolved'". Without
them, a discrepancy against ViennaPS at M2.7 cannot be attributed. It also sets the standard these
report to: **observed order, not "the error decreases"** — an order that is right by eye and 0.6 by
measurement is a bug.

**Reinitialisation is ON for V5 and V6 and OFF for V7, and the asymmetry is the point.**

The extension band is 8 *cells* wide, so halving dx halves it in nanometres while the travel stays
fixed. Without reinitialisation the interface then walks out of the band it started in — the rate is
tapered to zero outside, so those cells simply do not move, and φ near the interface is built from
stale values. Measured: the observed order under dx-refinement with reinitialisation off is
**−1.96**, i.e. the error grows by about 4× at every refinement. That is not a bug in the scheme;
it is the M2.1 decision showing up as a number. Reinitialisation is what makes travel beyond the
band legal, so a dx-refinement study of this solver has to run with it on, exactly as production
does. `test_the_order_study_needs_reinitialisation` pins that −1.96 so the choice stays visible
instead of looking like a convenient default.

V7 is the opposite case and has to switch it off: the reinit schedule is fixed in *steps*, so
refining dt fires it a different number of times per unit of physical time, and V7 would measure the
schedule rather than the time integrator. It therefore runs a short enough travel to stay inside the
band without repair.

The measurement region is `|φ_exact| ≤ 4·dx`, inside the band at every level. It shrinks with dx —
which is correct here: outside the band the solver deliberately does not solve the PDE, so including
those cells would measure a region the scheme makes no claim about. The pointwise error is O(dx)
uniformly inside it, so both the mean (L¹) and the maximum (L∞) are O(dx) whatever its size.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from helpers import MATERIALS, disk_radius, synthetic_config

from m2 import initial, solver
from m2.band import extension_weight
from m2.stencils import cell_centres

DOMAIN_NM = 640.0
CENTRE = DOMAIN_NM / 2
R0 = 200.0
RATE = 5.83
TRAVEL = 40.0
T_END = TRAVEL / RATE


def _config(dx, *, model="isotropic", n_steps=1, n_reinit=5):
    """Production defaults (n_reinit = 5 every 5 steps) unless a check has a stated reason."""
    n = int(round(DOMAIN_NM / dx))
    cfg = synthetic_config((n, n), spacing_nm=dx, n_steps=n_steps, model=model)
    return dataclasses.replace(cfg, n_reinit=n_reinit)


def _exact_sdf(grid, radius):
    """|x − c| − r: an exact signed distance for any r, which is what keeps the band honest."""
    coords = cell_centres(grid)
    return jnp.sqrt(sum((coords[i] - CENTRE) ** 2 for i in range(grid.ndim))) - radius


def _band_errors(phi_num, phi_exact, grid, *, cells=4.0):
    inside = np.abs(np.asarray(phi_exact)) <= cells * grid.spacing_nm
    err = np.abs(np.asarray(phi_num) - np.asarray(phi_exact))[inside]
    return float(err.mean()), float(err.max()), int(inside.sum())


def _observed_order(spacings, errors):
    """Least-squares slope of log(error) against log(dx), plus the pairwise orders.

    Both are reported: a single fitted number can hide a level that is not converging at all, and a
    pairwise sequence that drifts is the signature of a scheme falling out of its asymptotic range.
    """
    spacings, errors = np.asarray(spacings, float), np.asarray(errors, float)
    fitted = float(np.polyfit(np.log(spacings), np.log(errors), 1)[0])
    pairwise = [float(np.log(errors[i] / errors[i + 1]) / np.log(spacings[i] / spacings[i + 1]))
                for i in range(len(errors) - 1)]
    return fitted, pairwise


# --- V6: spatial order -------------------------------------------------------------------------


def _spatial_run(dx, dt, n_steps, *, n_reinit=5):
    cfg = dataclasses.replace(_config(dx), n_reinit=n_reinit)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi0 = _exact_sdf(cfg.grid, R0)
    result = solver.solve(cfg, phi0, material, {"v_iso": RATE}, n_steps=n_steps, dt=dt,
                          capacity=16384)
    assert result.max_cfl <= 0.5
    exact = _exact_sdf(cfg.grid, R0 - RATE * dt * n_steps)
    return cfg, result, exact


@pytest.mark.check("V6")
@pytest.mark.nightly
def test_v6_spatial_order(ledger_measure):
    """dt fixed and small, dx refined over 4 levels, against V1's analytic solution (§8.2).

    Godunov upwind must show observed order ≥ 0.9. The time step is held at the value that gives
    CFL ≈ 0.2 on the *finest* grid, so the temporal error is the same constant at every level and
    below the finest spatial error — otherwise this measures the time integrator.

    **Measured 1.60, and that is not a claim that the scheme is better than first order.** The
    pairwise orders drift downward — 1.69, 1.63, 1.48 — which is the signature of superconvergence
    decaying toward the asymptotic rate, not of a scheme that is uniformly 1.6. A circle is the
    friendliest possible interface for an upwind scheme: it is smooth, the solution stays an exact
    signed distance, and reinitialisation keeps pushing it back onto one. The honest reading is
    "≥ 0.9 is met with room to spare, and the margin will shrink on harder geometry". The pairwise
    sequence is asserted and recorded so that shrinkage is visible rather than surprising, and V3's
    sidewall-angle finding at dx = 10 is the reminder that the margin is not uniform.
    """
    spacings = [20.0, 10.0, 5.0, 2.5]
    dt = 0.2 * min(spacings) / RATE
    n_steps = int(round(T_END / dt))

    l1s, linfs, radii, cells = [], [], [], []
    for dx in spacings:
        cfg, result, exact = _spatial_run(dx, dt, n_steps)
        l1, linf, n_in = _band_errors(result.phi, exact, cfg.grid)
        l1s.append(l1)
        linfs.append(linf)
        cells.append(n_in)
        radii.append(disk_radius(np.asarray(result.phi), dx, int(round(CENTRE / dx))))

    order_l1, pairwise_l1 = _observed_order(spacings, l1s)
    order_linf, pairwise_linf = _observed_order(spacings, linfs)
    r_exact = R0 - RATE * dt * n_steps
    radius_errors = [abs(r - r_exact) for r in radii]
    order_radius, _ = _observed_order(spacings, radius_errors)

    ledger_measure.update({
        "spacings_nm": spacings, "dt_s": dt, "n_steps": n_steps,
        "cfl_finest": RATE * dt / min(spacings), "cfl_coarsest": RATE * dt / max(spacings),
        "l1_error_nm": l1s, "linf_error_nm": linfs, "cells_in_band": cells,
        "observed_order_l1": order_l1, "observed_order_linf": order_linf,
        "pairwise_order_l1": pairwise_l1, "pairwise_order_linf": pairwise_linf,
        "interface_radius_nm": radii, "exact_radius_nm": r_exact,
        "radius_error_nm": radius_errors, "observed_order_radius": order_radius,
        "requirement": "Godunov upwind >= 0.9 (§8.2)", "scheme": "first-order Godunov, TVD-RK2",
        "reinitialisation": "on (5 iterations every 5 steps, the production default) — the band is "
                            "8 cells, so without it the travel leaves the band as dx refines",
        "region": "|phi_exact| <= 4*dx (the extension band)"})

    assert order_l1 >= 0.9, f"observed L1 order {order_l1:.3f} < 0.9 (§8.2)"
    assert order_linf >= 0.9, f"observed Linf order {order_linf:.3f} < 0.9 (§8.2)"
    assert min(pairwise_l1) >= 0.9, f"a refinement level is not converging: {pairwise_l1}"


@pytest.mark.nightly
def test_the_order_study_needs_reinitialisation(ledger_measure):
    """Why V6 runs with reinitialisation on — pinned, so the choice cannot quietly become a default.

    The extension band is 8 *cells*. Halving dx halves it in nanometres while the travel stays fixed,
    so past some refinement the interface leaves the band it started in; outside the band the rate is
    tapered to zero, so those cells never moved and φ near the interface is assembled from stale
    values. The error then *grows* under refinement.

    This is the M2.1 decision (owner, option A) reappearing as a number: travel beyond the extension
    band requires reinitialisation. If this test ever stops showing a collapse, either the band
    became physical rather than per-cell or the travel changed, and V6's configuration should be
    revisited rather than trusted.
    """
    spacings = [20.0, 10.0, 5.0]
    dt = 0.2 * min(spacings) / RATE
    n_steps = int(round(T_END / dt))

    orders = {}
    for n_reinit in (0, 5):
        errs = []
        for dx in spacings:
            cfg, result, exact = _spatial_run(dx, dt, n_steps, n_reinit=n_reinit)
            errs.append(_band_errors(result.phi, exact, cfg.grid)[0])
        orders[n_reinit] = (_observed_order(spacings, errs)[0], errs)

    ledger_measure.update({
        "order_reinit_off": orders[0][0], "order_reinit_on": orders[5][0],
        "l1_reinit_off_nm": orders[0][1], "l1_reinit_on_nm": orders[5][1],
        "travel_nm": TRAVEL, "band_nm_by_dx": {str(dx): 8.0 * dx for dx in spacings},
        "why": "8-cell band shrinks with dx while travel is fixed; without repair the interface "
               "leaves the band and the error grows under refinement (M2.1 decision, option A)"})

    assert orders[0][0] < 0.0, (
        f"with reinitialisation off the order is {orders[0][0]:.3f}, no longer a collapse — "
        f"re-examine why V6 needs it")
    assert orders[5][0] >= 0.9, f"with reinitialisation on the order is {orders[5][0]:.3f}"


# --- V7: temporal order ------------------------------------------------------------------------


@pytest.mark.check("V7")
@pytest.mark.nightly
def test_v7_temporal_order(ledger_measure):
    """dx fixed, dt refined. TVD-RK2 must show observed order ≥ 1.9 (§8.2).

    Measured by **self-convergence**, against a reference run at dt/8 on the same grid, not against
    the analytic solution. At fixed dx the spatial error is a constant far larger than the temporal
    one, so an analytic comparison would saturate immediately and report order ≈ 0 — it would be
    measuring the space discretisation with the clock refined. Differencing two runs that share a dx
    cancels that constant exactly, which is the only way to see the time integrator at all.
    """
    dx = 5.0
    # Reinitialisation off (see the module docstring), so the travel is held to 15 nm — 3 cells,
    # comfortably inside the 8-cell extension band, which is what reinitialisation would otherwise
    # be needed for. This is a bound from the band width, not a number chosen after seeing a result.
    travel, cfg = 15.0, _config(dx, n_reinit=0)
    t_end = travel / RATE
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi0 = _exact_sdf(cfg.grid, R0)

    def run(n_steps):
        dt = t_end / n_steps
        result = solver.solve(cfg, phi0, material, {"v_iso": RATE}, n_steps=n_steps, dt=dt,
                              capacity=16384)
        assert result.max_cfl <= 0.5
        return np.asarray(result.phi), dt

    step_counts = [40, 80, 160, 320]
    reference, dt_ref = run(step_counts[-1] * 8)
    exact = _exact_sdf(cfg.grid, R0 - RATE * t_end)

    dts, errors = [], []
    for n in step_counts:
        phi, dt = run(n)
        dts.append(dt)
        errors.append(_band_errors(phi, reference, cfg.grid)[0])

    order, pairwise = _observed_order(dts, errors)
    spatial = _band_errors(reference, exact, cfg.grid)[0]

    ledger_measure.update({
        "spacing_nm": dx, "travel_nm": travel, "dt_s": dts, "n_steps": step_counts,
        "reference_dt_s": dt_ref, "reference_n_steps": step_counts[-1] * 8,
        "l1_self_convergence_nm": errors, "observed_order": order, "pairwise_order": pairwise,
        "spatial_error_of_reference_nm": spatial,
        "temporal_over_spatial_at_coarsest_dt": errors[0] / spatial,
        "requirement": "TVD-RK2 >= 1.9 (§8.2)",
        "method": "self-convergence against dt/8: at fixed dx the spatial error is a constant that "
                  "swamps the temporal one and cancels only in a difference of two runs",
        "reinitialisation": "off — its schedule is fixed in steps, so it would vary with dt; the "
                            "travel is held to 3 cells so the band alone suffices"})

    assert errors[0] < spatial, (
        "the coarsest temporal error already exceeds the spatial error, so the reference is not "
        "resolved enough to be a reference")
    assert order >= 1.9, f"observed temporal order {order:.3f} < 1.9 (§8.2)"


# --- V5: method of manufactured solutions -------------------------------------------------------


def _mms_radius(t):
    """r(t) = r₀ − R̄t + a·sin(2πt/T): a prescribed motion no velocity model would produce."""
    return R0 - 3.0 * t + 4.0 * jnp.sin(2 * jnp.pi * t / T_END)


def _mms_radius_rate(t):
    return -3.0 + 4.0 * (2 * jnp.pi / T_END) * jnp.cos(2 * jnp.pi * t / T_END)


def _mms_source(cfg, params):
    """S = ∂φ_exact/∂t − w(φ_exact)·R_exact, which makes φ_exact an exact solution of §5.2's PDE.

    Two details decide whether this is a valid manufactured solution rather than a decorated one:

    - **φ_exact is an exact signed distance** (|x − c| − r(t) for any r), so |∇φ_exact| = 1 and the
      closest-point projection the gather performs lands on the true interface. A manufactured φ
      with |∇φ| ≠ 1 would be projected somewhere else and the solver's rate would not be the
      R_exact this source assumes.
    - **The source carries the band taper.** The solver's rate field is `R(projection) ·
      extension_weight(φ)`, which falls to zero outside the band — outside it the solver is not
      solving this PDE at all. Using an untapered R here would leave the far field drifting away
      from φ_exact at ∫R dt, and that error would propagate inward one cell per step and contaminate
      the band. With the taper included, the far field satisfies φ_t = −r'(t) exactly.

    R_exact is time-independent: the normals of concentric circles do not move, so the directional
    law gives the same angular rate profile at every t. That is what makes S writable in closed form
    while still varying in space.
    """
    coords = cell_centres(cfg.grid)
    offset = [coords[i] - CENTRE for i in range(cfg.grid.ndim)]
    radius = jnp.sqrt(sum(o**2 for o in offset))
    safe = jnp.where(radius > 0, radius, 1.0)
    n_z = jnp.where(radius > 0, offset[0] / safe, 0.0)  # axis 0 is vertical, ẑ = +e₀
    rate = params["v_iso"] + params["v_dir"] * jnp.maximum(n_z, 0.0) ** params["p"]

    def source(t):
        phi_exact = radius - _mms_radius(t)
        weight = extension_weight(phi_exact, cfg.grid, cfg.bands)
        return -_mms_radius_rate(t) - weight * rate

    return source


@pytest.mark.check("V5")
@pytest.mark.nightly
def test_v5_method_of_manufactured_solutions(ledger_measure):
    """§8.2: substitute a chosen φ_exact into φ_t − R|∇φ| = 0, add the residual as a source, and
    verify the observed order against a solution that is known everywhere and at every time.

    This is the only forward check whose exact answer is not a shape the solver could reach on its
    own: the prescribed radius oscillates, which no positive etch rate produces. So it tests the
    discretisation rather than the physics, and it exercises the **directional** law, whose rate
    varies over the interface — V1, V6 and V7 all run the constant-rate model.

    Measured 1.73 (L¹) and 1.62 (L∞), with the same downward drift in the pairwise sequence that
    V6 shows and for the same reason; see V6's note. The requirement is ≥ 0.9.
    """
    params = {"v_iso": 0.58, "v_dir": 5.25, "p": 2.0}
    spacings = [20.0, 10.0, 5.0, 2.5]
    dt = 0.2 * min(spacings) / RATE
    n_steps = int(round(T_END / dt))

    l1s, linfs = [], []
    for dx in spacings:
        cfg = _config(dx, model="directional")
        material = initial.uniform_material(cfg.grid, MATERIALS)
        phi0 = _exact_sdf(cfg.grid, float(_mms_radius(0.0)))
        result = solver.solve(cfg, phi0, material, params, n_steps=n_steps, dt=dt, capacity=16384,
                              source=_mms_source(cfg, params))
        assert result.max_cfl <= 0.5
        exact = _exact_sdf(cfg.grid, float(_mms_radius(dt * n_steps)))
        l1, linf, _ = _band_errors(result.phi, exact, cfg.grid)
        l1s.append(l1)
        linfs.append(linf)

    order_l1, pairwise_l1 = _observed_order(spacings, l1s)
    order_linf, _ = _observed_order(spacings, linfs)

    ledger_measure.update({
        "spacings_nm": spacings, "dt_s": dt, "n_steps": n_steps,
        "l1_error_nm": l1s, "linf_error_nm": linfs,
        "observed_order_l1": order_l1, "observed_order_linf": order_linf,
        "pairwise_order_l1": pairwise_l1,
        "manufactured_radius": "r(t) = 200 - 3t + 4 sin(2 pi t / T), which no etch rate produces",
        "velocity_model": "directional (rate varies over the interface)", "params": params,
        "source": "S = -r'(t) - extension_weight(phi_exact) * R_exact(x)",
        "requirement": "observed order >= 0.9, matching the first-order scheme (§8.2)"})

    assert order_l1 >= 0.9, f"observed MMS L1 order {order_l1:.3f} < 0.9 (§8.2)"
    assert order_linf >= 0.9, f"observed MMS Linf order {order_linf:.3f} < 0.9 (§8.2)"
    assert min(pairwise_l1) >= 0.9, f"a refinement level is not converging: {pairwise_l1}"


def test_the_mms_source_path_is_off_in_production():
    """§8.2: "keep it behind a flag and test that the flag is off in production runs."

    The flag is the `source` argument. Three things make it unreachable from production rather than
    merely defaulted off, and all three are asserted here because a default is one edit from being
    changed:

    1. `final_phi` — the differentiated entry point, the one an optimiser calls — has no `source`
       parameter at all, so no configuration of it can turn the path on.
    2. `M2Config` has no source field, so nothing in `configs/` can reach it either.
    3. `solve` without `source=` is bitwise identical to `step` with `source=None`.
    """
    import inspect

    from m2.config import M2Config

    assert "source" not in inspect.signature(solver.final_phi).parameters, (
        "the MMS source reached the differentiated production path (§8.2, §11)")
    assert not any(f.name == "source" for f in dataclasses.fields(M2Config)), (
        "the MMS source became configurable, so a production run could switch it on")

    cfg = _config(10.0)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi0 = _exact_sdf(cfg.grid, R0)
    plain = solver.solve(cfg, phi0, material, {"v_iso": RATE}, n_steps=5, dt=0.05, capacity=16384)
    explicit_none = solver.solve(cfg, phi0, material, {"v_iso": RATE}, n_steps=5, dt=0.05,
                                 capacity=16384, source=None)
    assert np.array_equal(np.asarray(plain.phi), np.asarray(explicit_none.phi))

    # And a non-zero source must actually change the answer, or the path is dead code and the
    # order studies above are measuring nothing.
    with_source = solver.solve(cfg, phi0, material, {"v_iso": RATE}, n_steps=5, dt=0.05,
                               capacity=16384, source=lambda t: jnp.full(cfg.grid.shape, 1.0))
    assert not np.allclose(np.asarray(plain.phi), np.asarray(with_source.phi))


@pytest.mark.check("V6")
@pytest.mark.nightly
@pytest.mark.xfail(strict=True, reason=(
    "KNOWN DISCREPANCY, logged with evidence (OPEN_QUESTIONS J2). The registry requires observed "
    "order >= 4 for WENO5 and the measurement is ~2.08. The reconstruction itself is verified at "
    "order 5.13 in test_weno.py, so this is NOT a broken WENO5: the scheme-level order is capped by "
    "the velocity-extension path, which is second order by construction (central-difference normals "
    "and a bilinear closest-point gather). The L1 error still falls 155x at dx=20. The requirement "
    "is NOT changed (S11); it needs an owner decision. See J2 for the experiment that would confirm "
    "the cap."))
def test_v6_spatial_order_under_weno5(ledger_measure):
    """The same refinement study as V6, with the fifth-order scheme.

    Marked xfail(strict) so the shortfall is *recorded* rather than tolerated, and so that a future
    change which does reach 4 turns this red and forces the finding to be revisited — the same
    pattern V8 uses.
    """
    spacings = [20.0, 10.0, 5.0, 2.5]
    dt = 0.2 * min(spacings) / RATE
    n_steps = int(round(T_END / dt))

    errors = {}
    for scheme in ("godunov", "weno5"):
        l1s = []
        for dx in spacings:
            cfg = dataclasses.replace(_config(dx), spatial_scheme=scheme)
            material = initial.uniform_material(cfg.grid, MATERIALS)
            result = solver.solve(cfg, _exact_sdf(cfg.grid, R0), material, {"v_iso": RATE},
                                  n_steps=n_steps, dt=dt, capacity=16384)
            assert result.max_cfl <= 0.5
            exact = _exact_sdf(cfg.grid, R0 - RATE * dt * n_steps)
            l1s.append(_band_errors(result.phi, exact, cfg.grid)[0])
        errors[scheme] = (l1s, *_observed_order(spacings, l1s))

    weno_l1, weno_order, weno_pairwise = errors["weno5"]
    godunov_l1, godunov_order, _ = errors["godunov"]

    ledger_measure.update({
        "spacings_nm": spacings, "dt_s": dt, "n_steps": n_steps,
        "weno5_l1_nm": weno_l1, "godunov_l1_nm": godunov_l1,
        "weno5_observed_order": weno_order, "godunov_observed_order": godunov_order,
        "weno5_pairwise": weno_pairwise,
        "error_ratio_at_dx20": godunov_l1[0] / weno_l1[0],
        "requirement": "WENO5 >= 4 (§8.2)", "reconstruction_order_measured": 5.13,
        "hypothesis": "capped by the 2nd-order velocity-extension path: central-difference normals "
                      "and bilinear closest-point gather",
        "finding": "J2 (needs an owner decision)"})

    assert weno_l1[0] < godunov_l1[0] / 100.0, "WENO5 must at least cut the error constant"
    assert weno_order >= 4.0, (
        f"observed WENO5 order {weno_order:.3f} < 4 (§8.2) — record as a finding with the "
        f"reconstruction-order evidence alongside, do not change the requirement (§11)")

