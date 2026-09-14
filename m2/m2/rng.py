"""Deterministic RNG keying for stochastic velocity models (PRD §7.6), M2.4.

M2 ships only analytic velocity models, so nothing here is used in production yet. It exists
because **V18 has to guard the trap before M3 can spring it**, and because the keying is part of the
velocity contract M3 consumes (CROSS_MISSION X2, X3) rather than a detail of whoever implements the
first stochastic model.

The rule, from `interface.py`: seed from `(run_seed, step_index, stage_index, cell_id)` — never
global, never stateful, never hashed from wall-clock `time`. Every one of those four is carried in
`VelocityRequest`, which is the point: the key is a pure function of the request, so a checkpointed
segment replayed during the backward pass derives **bitwise identical** keys without having to
remember anything.

**Why `cell_id` rather than the position in the padded array.** A cell's arrival in the evaluation
band shifts every later entry along. If the key came from the array index, one cell entering the
band would resample every other cell in the domain, and J would jump discontinuously in θ. The
`cell_id` is a stable flattened grid index, so a cell keeps its draw for as long as it is in the
band. It is opaque: seed with it, never index geometry with it.

That is only half the mechanism (decision A16). The other half is the smooth band weight going to
zero at the band edge, so the *arriving* cell's own draw enters continuously rather than as a step.
Remove either and J is discontinuous in θ; neither one alone is a safeguard.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import m2  # noqa: F401  (enables fp64)
from m2.schema import VelocityRequest


def base_key(run_seed, step_index, stage_index) -> jax.Array:
    """The per-stage key, before any cell is folded in.

    `fold_in` rather than arithmetic on the seed: mixing indices by addition makes
    (seed+1, step) collide with (seed, step+1), so two runs that should be independent share a
    stream. Order is fixed here so that M2 and M3 cannot disagree about it.
    """
    return jax.random.fold_in(
        jax.random.fold_in(jax.random.PRNGKey(run_seed), step_index), stage_index)


def request_keys(request: VelocityRequest) -> jax.Array:
    """One key per band entry, keyed by `(run_seed, step_index, stage_index, cell_id)`.

    Padded entries get keys too, and that is deliberate: they carry weight exactly zero, so their
    samples cannot reach the result, and giving them keys keeps the shape static and the mapping
    from entry to key independent of how many entries happen to be live.
    """
    key = base_key(request.run_seed, request.step_index, request.stage_index)
    return jax.vmap(lambda cell: jax.random.fold_in(key, cell))(request.cell_id)


def normal_samples(request: VelocityRequest) -> jax.Array:
    """Standard normal draws, one per band entry, reproducible from the request alone."""
    return jax.vmap(lambda k: jax.random.normal(k, dtype=jnp.float64))(request_keys(request))
