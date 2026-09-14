"""V17, V18 and the checkpoint schedule — bounded adjoint memory (PRD §7.4, §8.5), M2.4 gate.

Checkpointing is the one change in M2 that is *supposed* to alter nothing observable. It trades
recompute for memory, and if it also alters the gradient it does so by a small amount that only
appears at large N — which is precisely the "wrong but plausible" failure §2 exists to prevent.
So V17 is an equality check, not a tolerance check in spirit: 1e-12, on a quantity that should
differ only by the order floating-point addition happens in.

**What is measured here and what is only modelled**, because conflating them would understate the
memory gate by an order of magnitude:

- *persistent* storage — the segment-boundary carries kept for the whole backward pass — is
  measured directly, and must equal ceil(N/L);
- *transient* storage — the residuals live while one segment is replayed — is recomputed and freed,
  so it never becomes a jaxpr constant and cannot be measured this way. It is modelled.

True peak device memory needs a profiler on the gate hardware. **M2.4's H100 run is where the 40 GB
number must come from**; nothing in this file establishes it.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import MATERIALS, consistent_config

from m2 import functionals, initial, rng, solver
from m2.checkpoint import optimal_segment, peak_field_equivalents, persistent_residuals, scan_steps
from m2.config import parameter_tree
from m2.schema import VelocityRequest

TRAVEL = 200.0


def _case(shape=(64, 64), scheme="godunov"):
    cfg = dataclasses.replace(
        consistent_config(TRAVEL, spacing_nm=10.0, shape=shape, model="directional",
                          v_iso=0.58, v_dir=5.25, p=2.0),
        spatial_scheme=scheme)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    centre = tuple(n * cfg.grid.spacing_nm / 2 for n in cfg.grid.shape)
    phi0 = initial.disk(cfg.grid, centre, 200.0)
    return cfg, material, phi0, parameter_tree(cfg)[0]


def _objective(cfg, material, phi0, *, levels=0, segment=None, capacity=8192):
    def J(params):
        phi = solver.final_phi(cfg, phi0, material, params, capacity=capacity,
                               levels=levels, segment=segment)
        return functionals.solid_volume(phi, cfg.grid)

    return J


# --- V17 ----------------------------------------------------------------------------------------


@pytest.mark.check("V17")
@pytest.mark.nightly
@pytest.mark.parametrize("levels,segment", [(2, 1), (2, 2), (2, 5), (3, 5), (3, 10)])
def test_v17_checkpointed_gradient_equals_unchecked(levels, segment, ledger_measure):
    """§8.5: at small N where both fit, the checkpointed gradient must equal the unchecked one.

    Every schedule is checked, not just the one we intend to ship. A bug in a segment length we do
    not currently use is a bug that appears the first time k changes and the optimum moves — and k
    already moved by 3× when WENO5 landed (finding J5).
    """
    cfg, material, phi0, params = _case()
    base = jax.grad(_objective(cfg, material, phi0))(params)
    checkpointed = jax.grad(_objective(cfg, material, phi0, levels=levels, segment=segment))(params)

    errors = {name: abs(float(checkpointed[name]) - float(base[name])) / abs(float(base[name]))
              for name in base}
    ledger_measure.update({
        "levels": levels, "segment": segment, "n_steps": cfg.n_steps,
        "unchecked_gradient": {k: float(v) for k, v in base.items()},
        "checkpointed_gradient": {k: float(v) for k, v in checkpointed.items()},
        "relative_errors": errors, "tolerance": 1e-12})

    for name, error in errors.items():
        assert error < 1e-12, f"d/d{name} differs by {error:.3e} under levels={levels} (§8.5)"
    assert all(float(v) != 0.0 for v in base.values()), "a zero gradient would pass this vacuously"


@pytest.mark.nightly
def test_the_checkpointed_forward_value_also_matches():
    """V17 checks the gradient; a forward solve that also changed would be a different bug and
    would make the gradient agreement meaningless."""
    cfg, material, phi0, params = _case()
    plain = float(_objective(cfg, material, phi0)(params))
    checkpointed = float(_objective(cfg, material, phi0, levels=2, segment=5)(params))
    assert abs(checkpointed - plain) / abs(plain) < 1e-12


# --- the schedule -------------------------------------------------------------------------------


def test_the_optimal_segment_follows_the_measured_k():
    """L* = √(N/k), and at the k this solver actually has it is 1 — see `checkpoint.py` on why the
    PRD's warning about remat-on-the-step-function is inverted at measured k."""
    assert optimal_segment(625, 349.2) == 1      # Godunov
    assert optimal_segment(625, 1049.5) == 1     # WENO5
    assert optimal_segment(10000, 1.0) == 100    # the k = 1 regime the PRD was written for
    assert optimal_segment(100, 1000.0) == 1     # k > N clamps rather than going below 1
    assert optimal_segment(4, 0.01) == 4         # and never exceeds N
    with pytest.raises(ValueError):
        optimal_segment(0, 10.0)
    with pytest.raises(ValueError):
        optimal_segment(10, 0.0)


def test_a_schedule_cannot_be_requested_without_a_basis_for_it():
    """The segment length sets how much recompute the gradient pays. It may be stated, or derived
    from a measured k — never defaulted silently to something that happens to work."""
    cfg, material, phi0, params = _case()
    with pytest.raises(ValueError, match="residual_factor"):
        scan_steps(lambda c, i: c, phi0, cfg.n_steps, levels=2)
    with pytest.raises(ValueError, match="levels"):
        scan_steps(lambda c, i: c, phi0, cfg.n_steps, levels=1, segment=2)
    with pytest.raises(ValueError, match="segment"):
        scan_steps(lambda c, i: c, phi0, cfg.n_steps, levels=2, segment=cfg.n_steps + 1)


@pytest.mark.nightly
def test_persistent_storage_matches_the_schedule(ledger_measure):
    """The structural check that the schedule is doing what it claims.

    Persistent residuals must equal ceil(N/L) — the segment boundaries and nothing else. If
    checkpointing silently failed to apply, this would read N·k instead, which is a 300× difference
    rather than a subtle one.
    """
    cfg, material, phi0, params = _case()
    field_bytes = int(np.prod(cfg.grid.shape)) * 8
    unchecked, _ = persistent_residuals(_objective(cfg, material, phi0), params, field_bytes)
    k = unchecked / cfg.n_steps

    rows = {}
    for levels, segment in ((2, 1), (2, 2), (2, 5), (3, 5), (3, 10)):
        measured, _ = persistent_residuals(
            _objective(cfg, material, phi0, levels=levels, segment=segment), params, field_bytes)
        expected = int(np.ceil(cfg.n_steps / segment))
        rows[f"levels{levels}_seg{segment}"] = {
            "persistent_measured": measured, "persistent_expected": expected,
            "peak_modelled": peak_field_equivalents(cfg.n_steps, k, levels=levels, segment=segment)}
        # A couple of field-equivalents of slack for the scalar carries alongside φ.
        assert expected <= measured <= expected + 4, (
            f"levels={levels} seg={segment}: persistent storage {measured:.1f} is not the "
            f"{expected} segment boundaries — the schedule is not being applied as claimed")
        assert measured < unchecked / 10.0

    ledger_measure.update({"n_steps": cfg.n_steps, "measured_k": k,
                           "unchecked_persistent": unchecked, "schedules": rows,
                           "note": "persistent only; transient replay storage is modelled, and "
                                   "true peak needs a profiler on the gate hardware"})


# --- V18 ----------------------------------------------------------------------------------------


def _request(n=16, *, run_seed=7, step_index=3, stage_index=0, cell_id=None):
    cells = jnp.arange(n, dtype=jnp.int32) * 37 if cell_id is None else cell_id
    return VelocityRequest(
        positions=jnp.zeros((n, 2)), normals=jnp.tile(jnp.array([1.0, 0.0]), (n, 1)),
        material_fractions=jnp.ones((n, 1)), weights=jnp.ones(n), cell_id=cells,
        n_active=jnp.asarray(n), time=0.0, step_index=step_index, stage_index=stage_index,
        run_seed=run_seed)


@pytest.mark.check("V18")
def test_v18_rng_keys_are_bitwise_reproducible_from_the_request(ledger_measure):
    """§7.6, decision §8: the keys and samples a replayed segment derives must be **bitwise**
    identical, and the mechanism that guarantees it is that they are a pure function of the four
    contract fields.

    Bitwise, not close: a sampled velocity that differs in the last bit on recompute makes the
    backward pass differentiate a slightly different function than the forward pass evaluated, and
    the resulting gradient error is invisible to every check that compares a gradient against
    itself.
    """
    first = rng.request_keys(_request())
    second = rng.request_keys(_request())
    assert np.array_equal(np.asarray(first), np.asarray(second)), "keys are not reproducible"
    assert np.array_equal(np.asarray(rng.normal_samples(_request())),
                          np.asarray(rng.normal_samples(_request())))

    # ... and genuinely depend on each of the four fields, or reproducibility is vacuous.
    base = np.asarray(rng.request_keys(_request()))
    changed = {
        "run_seed": np.asarray(rng.request_keys(_request(run_seed=8))),
        "step_index": np.asarray(rng.request_keys(_request(step_index=4))),
        "stage_index": np.asarray(rng.request_keys(_request(stage_index=1))),
        "cell_id": np.asarray(rng.request_keys(_request(cell_id=jnp.arange(16, dtype=jnp.int32)))),
    }
    ledger_measure.update({"n_entries": 16, "fields": sorted(changed),
                          "keying": "fold_in(fold_in(PRNGKey(run_seed), step), stage) then cell_id"})
    for name, keys in changed.items():
        assert not np.array_equal(base, keys), f"the key does not depend on {name} (§7.6)"

    # A cell keeps its own draw regardless of where it sits in the padded array (decision A16):
    # otherwise one cell entering the band shifts every later entry along, resampling the whole
    # domain, and J jumps discontinuously in θ. Reversing the order must permute the keys, not
    # change them.
    reversed_ids = jnp.asarray(np.arange(16)[::-1].copy() * 37, dtype=jnp.int32)
    reversed_keys = np.asarray(rng.request_keys(_request(cell_id=reversed_ids)))
    assert np.array_equal(reversed_keys, base[::-1]), (
        "a cell's key followed its position in the request rather than its cell_id (decision A16)")


@pytest.mark.check("V18")
@pytest.mark.nightly
def test_v18_a_stochastic_model_survives_checkpoint_replay(ledger_measure):
    """The end-to-end half of V18: a stochastic velocity in the loop, checkpointed against not.

    §8.5 requires the trajectory to agree to 1e-12 relative — bitwise equality of a whole trajectory
    is not something XLA guarantees across two different scan structures, and decision §8 restated
    the check accordingly. If the replay derived *different keys*, the disagreement would be of
    order the noise amplitude here (10 %), not 1e-12, so this separates the two failure modes by
    ten orders of magnitude rather than relying on a tight tolerance.
    """
    cfg, material, phi0, params = _case()

    def stochastic(request, p):
        from m2.velocity import directional
        return directional(request, p) * (1.0 + 0.1 * rng.normal_samples(request))

    plain = solver.final_phi(cfg, phi0, material, params, model=stochastic, capacity=8192)
    for levels, segment in ((2, 1), (2, 5), (3, 5)):
        replayed = solver.final_phi(cfg, phi0, material, params, model=stochastic, capacity=8192,
                                    levels=levels, segment=segment)
        scale = float(jnp.max(jnp.abs(plain)))
        error = float(jnp.max(jnp.abs(replayed - plain))) / scale
        ledger_measure[f"levels{levels}_seg{segment}_rel_error"] = error
        assert error < 1e-12, (
            f"levels={levels} seg={segment}: trajectory differs by {error:.3e}. At 1e-1 this would "
            f"be re-drawn noise; at 1e-12 it is summation order (§8.5, decision §8)")

    ledger_measure.update({"noise_amplitude": 0.1, "tolerance": 1e-12, "n_steps": cfg.n_steps,
                           "separation": "a re-keyed replay would differ by ~1e-1, not 1e-12"})


@pytest.mark.nightly
def test_checkpointing_makes_the_adjoint_faster_not_slower(ledger_measure):
    """The time-for-memory trade runs **backwards** here, and decision I5(a) rests on it.

    The usual reading of checkpointing is that it buys memory by paying recompute. That holds only
    while the unchecked tape fits. At k ≈ 349 it does not: the tape is 4.3 GB at 128²·N=100 on an
    8 GB machine, so the unchecked adjoint swaps, and replaying a segment from cache is far cheaper
    than faulting its residuals back in. Measured warm, compile excluded:

        64^2  N=50    unchecked  4.75x   two-level L=1  3.19x   three-level L=25  4.15x
        128^2 N=100   unchecked 22.22x   two-level L=1  4.91x   three-level L=25  6.11x
        192^2 N=100   unchecked 34.30x   two-level L=1  5.77x   three-level L=25  6.74x

    This is the evidence behind I5(a) — that the M2.3 ratio of 4.96× was a measurement of swapping
    rather than of the adjoint — so it is asserted rather than left as a remark. **It does not
    establish M2.4's ≤4× gate**, which is a checkpointed 3D run on the H100; these are 2D on CPU.

    Two-level beating three-level is the expected ordering (one replay against two) and is asserted
    so that an accidental reversal shows up as a failure rather than as a slightly worse number.
    """
    import time

    cfg, material, phi0, params = _case(shape=(128, 128))
    cfg = dataclasses.replace(cfg, n_steps=100)

    def timed(fn):
        fn(params)
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            jax.block_until_ready(fn(params))
            best = min(best, time.perf_counter() - start)
        return best

    forward = timed(jax.jit(_objective(cfg, material, phi0, capacity=40000)))
    ratios = {}
    for name, (levels, segment) in {"unchecked": (0, None), "two_level_L1": (2, 1),
                                    "three_level_L25": (3, 25)}.items():
        adjoint = timed(jax.jit(jax.value_and_grad(
            _objective(cfg, material, phi0, levels=levels, segment=segment, capacity=40000))))
        ratios[name] = adjoint / forward

    ledger_measure.update({
        "grid": list(cfg.grid.shape), "n_steps": cfg.n_steps, "forward_s": forward,
        "adjoint_ratios": ratios, "m2_4_gate_target": 4.0,
        "status": "2D on CPU — the M2.4 <=4x gate is a checkpointed 3D run on the H100 (X13)",
        "finding": "K1 (decision I5(a)'s premise, confirmed)"})

    assert ratios["two_level_L1"] < ratios["unchecked"], (
        "checkpointing was slower than not checkpointing — the premise of decision I5(a) is wrong "
        "on this machine and I5 should be reopened")
    assert ratios["two_level_L1"] < ratios["three_level_L25"], (
        "three-level beat two-level: it does strictly more recompute, so this is backwards")

