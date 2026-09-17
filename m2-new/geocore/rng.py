"""Deterministic random keys for stochastic velocity models. PRD §5.5, §7.6, decision A16.

M2's own velocity models are deterministic. This exists anyway, and is exercised by the contract
from the start, because M3 will supply a stochastic Monte Carlo velocity and the reproducibility
rules have to be in place BEFORE anything depends on them.

The key is a pure function of `(run_seed, step_index, stage_index, cell_id)`. Never global, never
stateful, never hashed from the clock.

**Why `cell_id` and not the row index.** Request rows are a padded set whose membership changes as
the interface moves. Seed from the row and a single cell entering the band shifts every other
cell's draw -- the objective then jitters for a bookkeeping reason with no physical content, and
worse, it jitters DISCONTINUOUSLY in the parameters. `cell_id` is the flattened grid index, which
belongs to the cell rather than to its position in an array.

**Why per stage.** TVD-RK evaluates the rate two or three times per step. Reusing one draw across
stages correlates them, which biases the average the integrator computes.

Common random numbers do NOT survive a change of grid spacing: ids mean different things at
different dx. Harmless in M2, a real constraint on M3's convergence studies (CROSS_MISSION X3).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def run_key(run_seed: int | Array) -> Array:
    """The root key for a run."""
    return jax.random.PRNGKey(jnp.asarray(run_seed, dtype=jnp.uint32))


def stage_key(run_seed: int | Array, step_index: Array, stage_index: Array) -> Array:
    """One key per (run, step, stage). Folding is order-dependent, so the order is fixed here
    and nowhere else."""
    key = run_key(run_seed)
    key = jax.random.fold_in(key, jnp.asarray(step_index, dtype=jnp.int32))
    return jax.random.fold_in(key, jnp.asarray(stage_index, dtype=jnp.int32))


def request_keys(run_seed: int | Array, step_index: Array, stage_index: Array,
                 cell_id: Array) -> Array:
    """One key per request entry, keyed by the cell's identity rather than its row.

    Returns shape `(K, 2)`. A velocity model indexes this in parallel with the request's other
    fields; it must never use `cell_id` for anything else (PRD §5.0: the id is OPAQUE, or M3
    acquires a dependency on M2's grid layout).
    """
    base = stage_key(run_seed, step_index, stage_index)
    return jax.vmap(lambda cid: jax.random.fold_in(base, cid))(
        jnp.asarray(cell_id, dtype=jnp.int32)
    )
