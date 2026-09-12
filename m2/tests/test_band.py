"""The velocity-request assembly and closest-point gather (decisions §3, C10, A16, B14).

Moved into M2.1 by the 2026-09-11 scope decision so that V1 and V2 — which have exact answers —
are what first exercise the contract-shaped path.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import MATERIALS, synthetic_config

from m2 import band, initial, solver, velocity
from m2.config import BandConfig
from m2.schema import VelocityRequest


def _setup(shape=(64, 64), radius=200.0):
    cfg = synthetic_config(shape)
    phi = initial.disk(cfg.grid, (320.0, 320.0), radius)  # a medial axis at the centre: the NaN trap
    material = initial.uniform_material(cfg.grid, MATERIALS)
    return cfg, phi, material


def _request(cfg, phi, material, capacity=4096):
    return band.assemble_request(phi, material, cfg.grid, cfg.bands, capacity=capacity, time=0.0,
                                 step_index=0, stage_index=0, run_seed=7)


# --- band weights ---------------------------------------------------------------------------


def test_evaluation_weight_is_one_at_the_interface_and_zero_at_the_band_edge():
    cfg, phi, _ = _setup()
    b = cfg.bands.evaluation_cells * cfg.grid.spacing_nm
    w = band.evaluation_weight(jnp.asarray([0.0, 0.5 * b, b, 2 * b, -b, -0.5 * b]), cfg.grid, cfg.bands)
    assert float(w[0]) == 1.0
    assert float(w[2]) == 0.0 and float(w[3]) == 0.0  # exactly zero at and beyond the edge
    assert 0.0 < float(w[1]) < 1.0
    assert float(w[1]) == pytest.approx(float(w[5]))  # symmetric in φ


def test_evaluation_weight_is_smooth_through_the_interface():
    """Written in φ² rather than |φ|: a kink at φ = 0 would sit in the differentiated path (§11)."""
    cfg, _, _ = _setup()
    d = jax.grad(lambda p: band.evaluation_weight(p, cfg.grid, cfg.bands))
    assert float(d(jnp.asarray(0.0))) == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(float(d(jnp.asarray(3.0))))


def test_extension_band_reaches_further_than_the_evaluation_band():
    cfg, phi, _ = _setup()
    ev = np.asarray(band.evaluation_weight(phi, cfg.grid, cfg.bands))
    ext = np.asarray(band.extension_weight(phi, cfg.grid, cfg.bands))
    assert (ext > 0).sum() > (ev > 0).sum()
    assert np.all(ext[ev > 0] > 0)  # everything evaluated is inside the extension band


# --- the padded request ---------------------------------------------------------------------


def test_request_is_padded_and_padding_carries_zero_weight():
    cfg, phi, material = _setup()
    sel = _request(cfg, phi, material)
    req, n_active = sel.request, int(sel.n_active)
    assert isinstance(req, VelocityRequest)
    assert req.weights.shape == (sel.capacity,)
    assert 0 < n_active < sel.capacity
    assert float(jnp.max(req.weights[n_active:])) == 0.0  # padded slots contribute nothing
    assert float(jnp.min(req.weights[:n_active])) > 0.0
    assert int(req.n_active) == n_active


def test_padded_entries_are_finite_because_zero_times_nan_is_nan():
    """Decision B14: a zero weight does not protect against a NaN position."""
    cfg, phi, material = _setup()
    req = _request(cfg, phi, material).request
    for name in ("positions", "normals", "material_fractions", "weights"):
        assert bool(jnp.all(jnp.isfinite(getattr(req, name)))), name


def test_positions_are_closest_points_on_the_interface_not_cell_centres():
    cfg, phi, material = _setup(radius=200.0)
    req = _request(cfg, phi, material).request
    live = np.asarray(req.weights) > 0
    pts = np.asarray(req.positions)[live]
    radii = np.linalg.norm(pts - np.array([320.0, 320.0]), axis=1)
    assert np.allclose(radii, 200.0, atol=0.2), f"max deviation {np.max(np.abs(radii - 200.0)):.3f} nm"


def test_cell_id_is_the_flattened_grid_index_and_is_stable():
    """Decision A16: the id identifies the cell, not the row, so draws survive band membership changes."""
    cfg, phi, material = _setup(radius=200.0)
    sel_a = _request(cfg, phi, material)
    ids_a = np.asarray(sel_a.request.cell_id)[: int(sel_a.n_active)]
    assert ids_a.max() < np.prod(cfg.grid.shape)
    assert np.all(np.diff(ids_a) > 0)  # ascending flattened indices

    # A slightly larger disk: the band shifts, rows renumber, but a cell keeps its id.
    sel_b = _request(cfg, initial.disk(cfg.grid, (320.0, 320.0), 205.0), material)
    ids_b = np.asarray(sel_b.request.cell_id)[: int(sel_b.n_active)]
    shared = np.intersect1d(ids_a, ids_b)
    assert shared.size > 0
    for cell in shared[:5]:
        row_a, row_b = int(np.where(ids_a == cell)[0][0]), int(np.where(ids_b == cell)[0][0])
        assert np.asarray(sel_a.request.cell_id)[row_a] == np.asarray(sel_b.request.cell_id)[row_b]


# --- the gather -------------------------------------------------------------------------------


def test_gather_reproduces_a_constant_rate_across_the_band():
    """The gathered field is the rate times the band taper: exactly R where the taper is 1, and
    smoothly to zero at the band edge, which is what keeps the extended velocity continuous."""
    cfg, phi, material = _setup()
    req = _request(cfg, phi, material).request
    speed = velocity.isotropic(req, {"v_iso": 5.83})
    dense = np.asarray(band.gather_to_band(speed, req, phi, cfg.grid, cfg.bands))
    taper = np.asarray(band.extension_weight(phi, cfg.grid, cfg.bands))

    untapered = taper == 1.0
    assert untapered.sum() > 0
    assert np.allclose(dense[untapered], 5.83, rtol=1e-12)  # exact where the taper does nothing
    tapered = (taper > 0) & ~untapered
    assert np.allclose(dense[tapered], 5.83 * taper[tapered], rtol=1e-12)
    assert np.all(np.isfinite(dense))


def test_gather_is_zero_outside_the_extension_band():
    cfg, phi, material = _setup()
    req = _request(cfg, phi, material).request
    dense = np.asarray(band.gather_to_band(velocity.isotropic(req, {"v_iso": 1.0}), req, phi,
                                           cfg.grid, cfg.bands))
    outside = np.abs(np.asarray(phi)) > cfg.bands.extension_cells * cfg.grid.spacing_nm
    assert np.all(dense[outside] == 0.0)


def test_capacity_estimate_covers_a_trench_geometry():
    cfg = synthetic_config((270, 100))
    capacity = band.estimate_capacity(cfg.grid, cfg.bands)
    phi = initial.trench(cfg.grid, surface_nm=2200.0, cd_nm=500.0, depth_nm=1500.0)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    sel = band.assemble_request(phi, material, cfg.grid, cfg.bands, capacity=capacity, time=0.0,
                                step_index=0, stage_index=0, run_seed=0)
    assert int(sel.n_active) < capacity


# --- the NaN trap -----------------------------------------------------------------------------


def test_no_nan_reaches_the_gradient_through_the_medial_axis():
    """∇φ/|∇φ| is singular at a disk's centre. The double-where pattern must keep the *gradient*
    finite, not just the forward value.

    This asserts NaN-freeness only. Whether the gradient is *correct* is V14/V15/V16 at M2.3.
    """
    cfg, phi, material = _setup()
    cfg = synthetic_config((64, 64), n_steps=3)

    def objective(v_iso):
        out = solver.final_phi(cfg, phi, material, {"v_iso": v_iso}, capacity=4096)
        return jnp.sum(out**2)

    value, grad = jax.value_and_grad(objective)(5.83)
    assert np.isfinite(float(value))
    assert np.isfinite(float(grad)), "a NaN from the medial axis reached the gradient"
    assert float(grad) != 0.0, "the rate has no influence on the result: the parameter is dead"
