"""Adjoint cost and the residual factor k — M2.3's two measurements for the M2.4 gate.

There is no V-number for these; they are the numbers the M2.4 memory and cost gates are set from,
and M2.3's job is to measure rather than assume them. They are recorded in the ledger through the
V14 row's run so the M2.4 proposal can cite a measurement instead of the 1–50 bracket guessed at
M2.0.

Nothing here asserts a cost target. Originally that was because the measured ratio missed the M2.3
gate's ≤3× (finding I5) and inventing a looser assertion would be exactly the move §11 forbids. The
owner has since resolved I5 as option (a), 2026-09-13: **the M2.3 ratio is measured and reported,
not gated**, and M2.4's ≤4× becomes the first gated ratio, taken on a checkpointed run on the gate
hardware rather than on an unchecked 2D run on a laptop.

So this file keeps measuring and keeps asserting nothing about the ratio — which is the point. A
number that is reported but not gated still shows a regression in the ledger; deleting the
measurement would make M2.4 the first time anyone looks. The assertions below are only the facts
that must hold for the measurement to mean anything at all.
"""

import time

import jax
import numpy as np
import pytest
from helpers import MATERIALS, consistent_config

from m2 import functionals, initial, solver
from m2.config import parameter_tree


def _case(n=64, dx=10.0, travel=200.0):
    cfg = consistent_config(travel, spacing_nm=dx, shape=(n, n), model="directional",
                            v_iso=0.58, v_dir=5.25, p=2.0)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi0 = initial.disk(cfg.grid, (n * dx / 2, n * dx / 2), 200.0)

    def J(params):
        return functionals.solid_volume(
            solver.final_phi(cfg, phi0, material, params, capacity=8192), cfg.grid)

    return cfg, J, parameter_tree(cfg)[0]


def _fastest(fn, params, reps=3):
    fn(params)  # warm: compile time is excluded and reported separately (decision §7)
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(fn(params))
        best = min(best, time.perf_counter() - start)
    return best


def residual_factor(cfg, J, params) -> tuple[float, float]:
    """k: field-sized arrays JAX keeps per step for the backward pass.

    Measured, not estimated: `jax.linearize` partially evaluates the function, and the residuals are
    exactly the constants closed over by the linearised part.
    """
    _, linear = jax.linearize(J, params)
    consts = jax.make_jaxpr(linear)({k: 0.0 for k in params}).consts
    total_bytes = sum(int(np.prod(np.shape(c))) * np.asarray(c).dtype.itemsize for c in consts)
    field_bytes = int(np.prod(cfg.grid.shape)) * 8
    return total_bytes / field_bytes / cfg.n_steps, total_bytes


@pytest.mark.check("V14")
def test_adjoint_cost_and_residual_factor_are_measured(ledger_measure):
    cfg, J, params = _case()
    forward = _fastest(jax.jit(J), params)
    adjoint = _fastest(jax.jit(jax.value_and_grad(J)), params)
    k, total_bytes = residual_factor(cfg, J, params)

    # What the measured k implies for the M2.4 gate case (S03 in 3D: 21.6 MB per fp64 field, N=625).
    gate_field_mb, gate_steps = 21.6, 625
    two_level_gb = 2 * np.sqrt(gate_steps * k) * gate_field_mb / 1000
    naive_tb = gate_steps * k * gate_field_mb / 1e6

    ledger_measure.update({
        "forward_s": forward, "adjoint_s": adjoint, "adjoint_ratio": adjoint / forward,
        "m2_3_target_ratio": "withdrawn (decision I5(a), owner 2026-09-13): measured, not gated",
        "m2_4_gated_target_ratio": 4.0,
        "m2_4_target_met_on_this_2d_unchecked_run": bool(adjoint / forward <= 4.0),
        "note": "M2.4's 4x is gated on a CHECKPOINTED run on the gate hardware; this 2D unchecked "
                "laptop figure is not that measurement and does not stand in for it",
        "k_residual_fields_per_step": k, "residual_bytes": total_bytes,
        "n_steps": cfg.n_steps, "grid": list(cfg.grid.shape),
        "m2_0_estimate_of_k": "1-50 (bracket, superseded)",
        "implied_m2_4_two_level_gb": two_level_gb, "implied_m2_4_naive_tb": naive_tb,
        "finding": "I4 (k measured), I5 (resolved: ratio measured and reported, not gated)"})

    assert adjoint > forward, "a gradient cannot be cheaper than the value it differentiates"
    assert 1.0 < k < 10000.0, f"k = {k} is not a plausible residual factor"
    assert np.isfinite(total_bytes) and total_bytes > 0


@pytest.mark.nightly
def test_k_is_independent_of_the_step_count():
    """k counts residuals *per step*, so it must not drift with N — otherwise it is not a factor
    and the M2.4 memory projection cannot use it."""
    factors = []
    for travel in (100.0, 200.0, 400.0):
        cfg, J, params = _case(travel=travel)
        factors.append(residual_factor(cfg, J, params)[0])
    assert max(factors) / min(factors) < 1.05, f"k drifts with N: {factors}"


@pytest.mark.nightly
def test_reinitialisation_accounts_for_a_large_share_of_the_residuals():
    """Finding I2 has a price: the smoothing reinitialisation buys is not free in memory."""
    import dataclasses

    cfg, J, params = _case()
    with_reinit = residual_factor(cfg, J, params)[0]

    cfg_off = dataclasses.replace(cfg, n_reinit=0)
    material = initial.uniform_material(cfg_off.grid, MATERIALS)
    phi0 = initial.disk(cfg_off.grid, (320.0, 320.0), 200.0)
    J_off = lambda p: functionals.solid_volume(  # noqa: E731
        solver.final_phi(cfg_off, phi0, material, p, capacity=8192), cfg_off.grid)
    without = residual_factor(cfg_off, J_off, params)[0]

    assert without < with_reinit, "reinitialisation should add residuals, not remove them"
    assert with_reinit / without > 1.2, f"expected a material share: {with_reinit} vs {without}"
