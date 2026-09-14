"""Checkpointed time integration: bounded adjoint memory (PRD §7.4), M2.4.

Reverse mode over N steps stores, for every step, every intermediate array the backward pass will
need. That is `N·k` field-sized arrays, where **k is the residual factor** — measured, not assumed:
349 with the Godunov scheme and 1050 with WENO5 (finding I4, and J5 for the WENO5 figure). At the
M2.4 gate case that is 4.5 TB and 14.2 TB respectively, so some form of recompute is not optional.

**The schedule family.** All three options below are one family with one parameter, the segment
length L, and it is worth seeing them that way rather than as three tricks:

    peak stored field-equivalents  =  N/L  +  L·k        (two-level)
                                   =  N/L  +  L  +  k    (three-level, one extra remat inside)

`N/L` is the segment boundaries the outer scan keeps; `L·k` is the residuals live while one segment
is being replayed. Minimising the first over L gives **L* = √(N/k)**.

**Which means the PRD's warning about `jax.checkpoint` is now inverted, and this is the correction
that matters most here.** §7.4 says remat-on-the-step-function "does not do this — it still stores
all N carries, costing as much as the naive figure". That was written under the implicit k = 1
accounting, where N carries really is the naive cost. At the measured k, `L* = √(625/349) = 1.3`, so
**the optimum is L = 1, which is exactly remat-on-the-step-function**: it stores N carries plus one
step's residuals, N + k = 955 field-equivalents against the naive 218,000. The thing §7.4 warns
against is the thing the arithmetic now recommends.

Three-level is the genuinely different option: remat *inside* the segment as well, so only one
step's residuals are ever live, at the cost of a second forward replay. It is 2.5× smaller again
(380 field-equivalents at the gate case) for 2× recompute.

Nothing here decides which to use. `optimal_segment` states the arithmetic and the caller chooses,
because the choice trades memory against the adjoint-ratio gate and only M2.4's measurements on the
real hardware can settle that.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

import m2  # noqa: F401  (enables fp64)

Carry = Any


def optimal_segment(n_steps: int, residual_factor: float) -> int:
    """L* = √(N/k), clamped to [1, N]: the segment length minimising two-level peak memory.

    Returns 1 whenever k ≥ N, which is the regime this solver is actually in — see the module
    docstring. The value is derived from the *measured* k and must never be hand-tuned: it sets how
    much recompute the gradient pays, and a number chosen to make a memory gate pass would be
    exactly the move §11 forbids.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if residual_factor <= 0:
        raise ValueError(f"residual_factor must be positive, got {residual_factor}")
    return int(min(n_steps, max(1, round(math.sqrt(n_steps / residual_factor)))))


def peak_field_equivalents(n_steps: int, residual_factor: float, *, levels: int,
                           segment: int | None = None) -> float:
    """Model of PEAK field-equivalents: persistent checkpoints **plus** the transient replay set.

    **The two parts are not both measurable the same way, and conflating them would understate the
    memory gate by an order of magnitude.**

    - *Persistent*: the segment-boundary carries the outer scan keeps for the whole backward pass,
      ceil(N/L) of them. `persistent_residuals` measures this directly.
    - *Transient*: the residuals live while one segment is being replayed — L·k at two levels, or
      L + k at three. These are recomputed and freed, so they never appear as jaxpr constants and
      `persistent_residuals` cannot see them. They nonetheless have to fit in memory at once, so
      they are what the 40 GB gate is mostly about: at N = 625, L = 1, k = 349 the split is 625
      persistent against 349 transient.

    So this function models the sum, `persistent_residuals` measures the first term, and the second
    is **modelled, not measured, on this machine**. Measuring true peak device memory needs a
    profiler on the gate hardware; M2.4's H100 run is where that number should come from, and until
    then any peak figure quoted from here carries that caveat.

    Excludes static fields (material fractions), the cotangent carry and XLA scratch, so it is a
    lower bound even as a model.
    """
    if levels not in (0, 2, 3):
        raise ValueError(f"levels must be 0 (none), 2 or 3, got {levels}")
    if levels == 0:
        return n_steps * residual_factor
    length = optimal_segment(n_steps, residual_factor) if segment is None else segment
    if not 1 <= length <= n_steps:
        raise ValueError(f"segment {length} outside [1, {n_steps}]")
    outer = math.ceil(n_steps / length)
    if levels == 2:
        return outer + length * residual_factor
    return outer + length + residual_factor


def scan_steps(step: Callable[[Carry, jax.Array], Carry], carry: Carry, n_steps: int, *,
               levels: int = 0, segment: int | None = None,
               residual_factor: float | None = None) -> Carry:
    """Run `step` for `n_steps`, with the requested checkpoint schedule.

    `step(carry, index) -> carry` takes the **global** step index, so the reinitialisation schedule
    and the RNG keying — both functions of `step_index` — are identical whatever the segmentation.
    That is the property V18 checks, and it is why the index is threaded rather than recomputed
    from a position within a segment.

    `levels=0` is the unchecked path, kept because V17 compares against it.

    The step count is padded up to a whole number of segments and the surplus iterations are made
    into no-ops by a **static** mask on the index, not by a data-dependent branch: shapes stay
    static and nothing in the differentiated path depends on a comparison against the data.
    """
    if levels not in (0, 2, 3):
        raise ValueError(f"levels must be 0 (none), 2 or 3, got {levels}")
    if levels == 0:
        out, _ = jax.lax.scan(lambda c, i: (step(c, i), None), carry, jnp.arange(n_steps))
        return out

    if segment is None:
        if residual_factor is None:
            raise ValueError("pass `segment`, or `residual_factor` so it can be derived (§7.4). "
                             "Neither may be guessed: the segment length sets the recompute cost.")
        segment = optimal_segment(n_steps, residual_factor)
    if not 1 <= segment <= n_steps:
        raise ValueError(f"segment {segment} outside [1, {n_steps}]")

    outer = math.ceil(n_steps / segment)
    inner_step = jax.checkpoint(step) if levels == 3 else step

    def run_segment(inner_carry: Carry, outer_index: jax.Array) -> tuple[Carry, None]:
        def body(c: Carry, offset: jax.Array) -> tuple[Carry, None]:
            index = outer_index * segment + offset
            # Padding only: with outer*segment == n_steps this is always True and folds away.
            return jax.lax.cond(index < n_steps, lambda: inner_step(c, index), lambda: c), None

        out, _ = jax.lax.scan(body, inner_carry, jnp.arange(segment))
        return out, None

    # `jax.checkpoint` on the segment is what makes this two-level: the outer scan keeps only the
    # carry at each segment boundary, and the segment's interior is replayed during the backward
    # pass. Applying it to the step function alone gives L = 1, which is a member of this same
    # family and, at the measured k, its optimum.
    final, _ = jax.lax.scan(jax.checkpoint(run_segment), carry, jnp.arange(outer))
    return final


def persistent_residuals(fn: Callable[[Any], Any], argument: Any,
                         field_bytes: int) -> tuple[float, int]:
    """Field-equivalents JAX keeps for the WHOLE backward pass, and the byte count.

    Measured, not modelled: `jax.linearize` partially evaluates the function and the residuals are
    exactly the constants the linearised part closes over. Same method that produced k at M2.3.

    **This is the persistent term only.** Anything a checkpointed segment recomputes and frees is
    not a jaxpr constant and is invisible here — see `peak_field_equivalents`. Reading this number
    as peak memory would understate a checkpointed run badly: at N = 50, L = 5 it reports 12 field
    equivalents where the peak is about 1756. What it *is* good for is verifying that the schedule
    is structurally doing what it claims, because it should equal ceil(N/L) and nothing else.
    """
    import numpy as np

    _, linear = jax.linearize(fn, argument)
    consts = jax.make_jaxpr(linear)(jax.tree.map(jnp.zeros_like, argument)).consts
    total = sum(int(np.prod(np.shape(c))) * np.asarray(c).dtype.itemsize for c in consts)
    return total / field_bytes, total
