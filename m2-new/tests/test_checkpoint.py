"""M2.4: checkpointing is a pure rearrangement of the same computation.

V17 -- the checkpointed gradient equals the unchecked gradient to 1e-12, at small N where both fit.
V18 -- replaying a segment reproduces its random keys and draws BITWISE, and the trajectory to 1e-12.

Checkpointing bugs are the kind that show up only at large N, as a gradient that is slightly wrong.
These catch them at small N, where there is an unchecked answer to compare against.
"""

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geocore.band import BandedRateField, single_material
from geocore.checkpoint import checkpointed_scan, peak_fields, segment_length_for
from geocore.config import BandConfig
from geocore.functionals import solid_volume
from geocore.initial import extrude, sphere, trapezoid
from geocore.rng import request_keys
from geocore.schema import Grid
from geocore.solver import SolvePlan, evolve
from geocore.velocity import directional

# M2.4 checks (V17, V18) are nightly-tier in PRD §8.0; the schedule tests here run with them.
pytestmark = pytest.mark.nightly

BANDS = BandConfig()
THETA = {"v0_nm_per_s": jnp.float64(2.0), "p": jnp.float64(2.0)}
N_STEPS = 23         # prime, so L = 2, 3 and 7 all leave a remainder segment


def _trench_2d():
    grid = Grid((40, 32), 10.0, (False, True))
    return grid, trapezoid(grid, 280.0, 60.0, 140.0, 100.0, 160.0), 400


def _trench_3d():
    profile_grid = Grid((24, 12), 10.0, (False, True))
    grid = Grid((24, 12, 12), 10.0, (False, True, True))
    return grid, extrude(trapezoid(profile_grid, 130.0, 15.0, 60.0, 42.0, 60.0), grid), 600


def _gradient(grid, phi0, capacity, segment, model=directional, seed=0, params=THETA,
              n_steps=N_STEPS):
    field = BandedRateField(model, grid, BANDS, capacity, single_material(grid), run_seed=seed)
    plan = SolvePlan(grid, 1.0, n_steps, reinit_every=5, checkpoint_segment=segment)

    def objective(t):
        phi, diagnostics = evolve(phi0, t, field, plan)
        return solid_volume(phi, grid), (phi, diagnostics)

    return jax.value_and_grad(objective, has_aux=True)(params)


def _max_relative(a: dict, b: dict) -> float:
    return max(abs(float(a[k]) - float(b[k])) / abs(float(b[k])) for k in b)


# ---------------------------------------------------------------------------------------- V17


@pytest.mark.check("V17")
@pytest.mark.parametrize("dimension", ["2D", "3D"])
def test_v17_checkpointed_gradient_equals_unchecked(dimension, ledger_measure):
    grid, phi0, capacity = _trench_2d() if dimension == "2D" else _trench_3d()
    (_, (phi_ref, diag_ref)), grad_ref = _gradient(grid, phi0, capacity, None)

    worst = 0.0
    worst_cfl = 0.0
    per_segment = {}
    for segment in (1, 2, 3, 7, N_STEPS, 4 * N_STEPS):
        (_, (phi, diag)), grad = _gradient(grid, phi0, capacity, segment)
        rel = _max_relative(grad, grad_ref)
        per_segment[segment] = rel
        worst = max(worst, rel)
        # The trajectory is specified to 1e-12 RELATIVE, not bitwise (registry V18: "bitwise on
        # keys; 1e-12 relative on the trajectory"). It was asserted bitwise here, which is stricter
        # than anything V17 or V18 asks for, and under the WENO5 default that is no longer true:
        # the checkpointed path fuses differently from the unchecked one on a larger graph, so CFL
        # moves by 2.776e-17 absolute, 1.388e-16 relative -- ONE ULP at 0.2, identical at every
        # segment length including 1, and unchanged under Godunov. The gradient itself, which is
        # what V17 is about, agrees to 3.0e-15 against its 1e-12 tolerance. See S20.5.
        cfl_rel = float(jnp.max(jnp.abs(diag.cfl - diag_ref.cfl) / jnp.abs(diag_ref.cfl)))
        worst_cfl = max(worst_cfl, cfl_rel)
        assert cfl_rel < 1e-12, f"trajectory must line up to 1e-12, got {cfl_rel:.3e}"
        # Occupancy stays EXACT. It counts cells in the band, so any difference at all would mean
        # the replay took a different trajectory rather than the same one rounded differently.
        assert bool(jnp.array_equal(diag.occupancy, diag_ref.occupancy))

    ledger_measure(**{f"{dimension}_worst_relative_difference": worst,
                      f"{dimension}_by_segment": {str(k): v for k, v in per_segment.items()},
                      f"{dimension}_worst_cfl_relative": worst_cfl,
                      "n_steps": N_STEPS, "tolerance": 1e-12})
    assert worst < 1e-12


# ---------------------------------------------------------------------------------------- V18


def _stochastic_model(log):
    """A TEST velocity model with random draws, standing in for M3's Monte Carlo velocity. Every
    draw is logged through a debug callback, which fires again when a checkpointed segment is
    replayed on the backward pass -- so forward and replay can be compared directly."""

    def record(step, stage, keys, draws):
        log[(int(step), int(stage))].append((np.asarray(keys).tobytes(), np.asarray(draws).tobytes()))

    def model(request, params):
        keys = request_keys(request.run_seed, request.step_index, request.stage_index,
                            request.cell_id)
        draws = jax.vmap(lambda k: jax.random.uniform(k, dtype=jnp.float64))(keys)
        jax.debug.callback(record, request.step_index, request.stage_index, keys, draws)
        return params["v0_nm_per_s"] * (1.0 + 0.1 * (draws - 0.5))

    return model


@pytest.mark.check("V18")
def test_v18_replay_reproduces_random_draws_bitwise(ledger_measure):
    """PRD §8.5, restated by decision §8: keys and sampled values bitwise identical on recompute;
    trajectory to 1e-12 relative. Guards the stochastic-velocity trap before M3 can spring it: a replay
    that drew different numbers would differentiate a mixture of two different random runs."""
    grid = Grid((32, 32), 10.0, (False, True))
    phi0 = sphere(grid, (160.0, 160.0), 100.0)
    params = {"v0_nm_per_s": jnp.float64(5.0)}

    unchecked_log, checked_log = defaultdict(list), defaultdict(list)
    # 7 steps with segments of 3 leaves a remainder segment. At 5 nm/s the 100 nm disk loses 35 nm,
    # so it survives -- a disk etched away entirely has a zero volume gradient and proves nothing.
    (_, (phi_ref, _)), grad_ref = _gradient(grid, phi0, 512, None, _stochastic_model(unchecked_log),
                                            seed=7, params=params, n_steps=7)
    (_, (phi_ckpt, _)), grad_ckpt = _gradient(grid, phi0, 512, 3, _stochastic_model(checked_log),
                                              seed=7, params=params, n_steps=7)

    replayed = [key for key, copies in checked_log.items() if len(copies) == 2]
    bitwise = all(len(set(copies)) == 1 for copies in checked_log.values())
    matches_unchecked = all(checked_log[k][0] == unchecked_log[k][0] for k in unchecked_log)
    trajectory = float(jnp.max(jnp.abs(phi_ckpt - phi_ref)) / jnp.max(jnp.abs(phi_ref)))

    ledger_measure(stage_evaluations=len(checked_log), replayed_evaluations=len(replayed),
                   draws_bitwise_identical_on_replay=bitwise,
                   draws_identical_to_unchecked_run=matches_unchecked,
                   trajectory_relative_difference=trajectory,
                   gradient_relative_difference=_max_relative(grad_ckpt, grad_ref))

    assert all(len(c) == 1 for c in unchecked_log.values()), "unchecked: each stage sampled once"
    assert len(replayed) == len(checked_log), \
        "every stage must be replayed once -- otherwise nothing was recomputed and this proves nothing"
    assert bitwise, "a replay must draw exactly the numbers the forward pass drew"
    assert matches_unchecked
    assert trajectory < 1e-12
    assert _max_relative(grad_ckpt, grad_ref) < 1e-12


# -------------------------------------------------------------------------------- the schedule


def test_segment_length_comes_from_measured_k_not_sqrt_n():
    """Finding I4. At the measured 3D k and S03's N, sqrt(N) is several times worse than L*."""
    n, k = 625, 344.0
    chosen = segment_length_for(n, k)
    assert chosen == 1
    assert peak_fields(n, chosen, k) < peak_fields(n, 25, k) / 5
    for other in range(1, 60):
        assert peak_fields(n, chosen, k) <= peak_fields(n, other, k)


def test_checkpointed_scan_matches_plain_scan_for_every_remainder():
    def body(c, i):
        return c * 1.01 + jnp.sin(i.astype(jnp.float64)), c

    for n in range(1, 12):
        plain = jax.lax.scan(body, jnp.float64(1.0), jnp.arange(n, dtype=jnp.int32))
        for segment in range(1, n + 2):
            carry, ys = checkpointed_scan(body, jnp.float64(1.0), n, segment)
            assert float(carry) == float(plain[0])
            assert bool(jnp.array_equal(ys, plain[1]))


def test_invalid_schedules_are_rejected():
    with pytest.raises(ValueError):
        segment_length_for(0, 10.0)
    with pytest.raises(ValueError):
        segment_length_for(10, 0.0)
    with pytest.raises(ValueError):
        SolvePlan(Grid((8, 8), 1.0, (False, True)), 1.0, 4, checkpoint_segment=0)
