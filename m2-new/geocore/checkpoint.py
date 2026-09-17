"""Two-level checkpointing of the time loop. PRD §7.4, findings I4, K1, K2.

**The problem.** Reverse mode keeps every intermediate array the backward pass will need. One solver
step leaves about k phi-sized fields of these "residuals" behind (k = 206 for Godunov, measured at
stage 13). Unchecked, N steps hold about N*k fields: on the S03 case in 3D that is 625 * 206 *
21.6 MB = 2.8 TB, against a 40 GB budget.

**The trade.** Split the N steps into segments of L. Keep only the state at the START of each segment
(N/L fields, kept for the whole backward pass). When the backward pass reaches a segment, replay it
forward to regenerate its residuals (L*k fields, alive only while that segment is processed). Peak:

    peak(L) = N/L + L*k          minimised at   L* = sqrt(N/k)

**L comes from measured k. Never from sqrt(N), never hand-picked** (finding I4). At k = 206 and
N = 625, L* = 1.7: sqrt(N) = 25 would give a peak seven times worse. And a hand-picked L is a knob that
can be turned until a memory gate passes, which is exactly what this project's rules forbid.

**Built as a nested scan.** The outer `lax.scan` runs over segments and carries only phi. The inner
scan runs L steps and is wrapped in `jax.checkpoint`, so its residuals are discarded after the forward
pass and recomputed on the way back. If N is not a multiple of L, a shorter final segment runs the
remainder, statically -- the number and length of segments depend on N and L only.

Everything here is a pure rearrangement of the same computation. The checkpointed gradient must equal
the unchecked one to 1e-12 (V17), and a replayed segment must reproduce its random draws bitwise (V18).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array


def segment_length_for(n_steps: int, k: float) -> int:
    """The integer segment length minimising `N/L + L*k`, from a MEASURED k.

    Checks the floor and ceiling of `sqrt(N/k)` rather than rounding, because the cost is not
    symmetric about its minimum.
    """
    if n_steps < 1 or k <= 0:
        raise ValueError(f"need n_steps >= 1 and k > 0, got {n_steps}, {k}")
    ideal = math.sqrt(n_steps / k)
    candidates = {max(1, math.floor(ideal)), max(1, math.ceil(ideal))}
    return min(candidates, key=lambda L: (peak_fields(n_steps, L, k), L))


def peak_fields(n_steps: int, segment: int, k: float) -> float:
    """Modelled peak, in phi-sized fields: persistent segment starts + one segment's residuals.

    Only the first term is directly measurable without a GPU (it is the stored carry). The second is
    a MODEL built from measured k, and it dominates; report it labelled as such.
    """
    return math.ceil(n_steps / segment) + segment * k


def checkpointed_scan(body: Callable[[Array, Array], tuple[Array, Any]], init: Array,
                      n_steps: int, segment: int) -> tuple[Array, Any]:
    """`lax.scan(body, init, arange(n_steps))`, recomputing each segment of `segment` steps on the
    backward pass instead of storing its residuals.

    Same inputs, same outputs, same per-step `ys` in the same order. Only memory and time differ.
    """
    if segment < 1:
        raise ValueError(f"segment length must be at least 1, got {segment}")
    segment = min(segment, n_steps)
    n_full, remainder = divmod(n_steps, segment)

    def run_segment(carry: Array, start: Array, length: int):
        indices = start + jnp.arange(length, dtype=jnp.int32)
        return jax.lax.scan(body, carry, indices)

    full = jax.checkpoint(lambda c, s: run_segment(c, s, segment))
    starts = jnp.arange(n_full, dtype=jnp.int32) * segment
    carry, ys = jax.lax.scan(full, init, starts)
    ys = jax.tree_util.tree_map(lambda y: y.reshape((n_full * segment,) + y.shape[2:]), ys)

    if remainder:
        tail = jax.checkpoint(lambda c, s: run_segment(c, s, remainder))
        carry, tail_ys = tail(carry, jnp.int32(n_full * segment))
        ys = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b], axis=0), ys, tail_ys)
    return carry, ys
