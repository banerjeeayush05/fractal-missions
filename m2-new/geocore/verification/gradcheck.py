"""The gradient harness. PRD §7.1, §8.5 — V14, V15 and V16.

Built against functions whose gradients are known in closed form, BEFORE a solver exists. A
harness first built against the solver gives an ambiguous pass: it could mean the gradient is
right, or it could mean the harness is blind.

**V14, the Taylor remainder, is the primary gate.** For an objective J and a claimed gradient g,
perturb along a direction v and measure

    R(h) = | J(theta + h*v) - J(theta) - h*<g, v> |

If g is right, the linear term cancels exactly and the leading survivor is the quadratic one:
R ~ (h^2/2) v^T H v, a log-log slope of **2**. If g is wrong by eps, the linear term does NOT
cancel and R ~ h*|<eps, v>|, a slope of **1**. The gate is the slope.

Three things make that harder than it sounds, and each is a decision with a failure behind it.

**1. Parameter scales are required, not optional.** The perturbation is
`v_i = delta_i * max(|theta_i|, scale_i)`. Without it, a parameter sitting at 1e-12 receives a
1e-12 perturbation and reports a pass having probed nothing at all.

**2. The window is anchored at the MEASURED floor** (decision I7). Sweep eight decades of h and
record the whole curve, then fit the two decades immediately above the noise floor. The top of
the range is not quadratic — the solver's map is only piecewise smooth, and a slope of 1.29 up
there says nothing about the gradient. The bottom is where a correct `(h^2/2) v^T H v` and a
wrong `|eps.v| h` separate most. The floor is MEASURED, not modelled: B19's `C*sqrt(N)*eps*|J|`
estimate reads 51-65x high, and every factor of ten in a discard threshold costs a decade of
usable window.

**3. The remainder can notch.** |R| must grow with h; where it does not, two terms are cancelling
and the log-log slope is meaningless — it reads about 1.6 on a gradient that is exactly right.
`_cancellation_ceiling` caps the window below the first notch. That is safe ONLY because the
window is anchored at the floor: the cut removes points ABOVE the notch, never below, and below
is where a first-order error dominates. Do not "fix" a failing V14 by widening the band.

**Scoring is two-zone** (decision B15/B21), per direction, each passing on its own. There is no
upper bound to enforce: a wrong gradient leaves the first-order term uncancelled and tends toward
slope 1, so a slope above 2.2 is a direction of near-zero curvature, not a failure. It is
recorded as `degenerate_direction` and the fraction is reported, because a JUMP in that fraction
between runs is itself a finding.

**V15 and V16 are transpose checks, not derivative checks** (decision §5, finding A9). JAX builds
reverse mode by linearising with the forward-mode JVP rules and transposing them, so `jvp` and
`vjp` share those rules for all primitives. Corrupt a `custom_jvp` rule and both agree on the
wrong derivative. Only V14 compares a derivative against the function itself.

Numbers here are authoritative; `registry.py` records them descriptively for the ledger.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Final

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

# ------------------------------------------------------------------- scoring (decision B15/B21)
CLEAN: Final[str] = "clean_quadratic"
DEGENERATE: Final[str] = "degenerate_direction"
INSUFFICIENT: Final[str] = "insufficient_signal"
SHALLOW: Final[str] = "slope_too_low"

SLOPE_FAIL_BELOW: Final[float] = 1.8
CLEAN_UPPER: Final[float] = 2.2
MIN_POINTS_IN_WINDOW: Final[int] = 3

# ------------------------------------------------------------------- sweep (decision I7)
N_DIRECTIONS: Final[int] = 20
H_MAX: Final[float] = 1e-2          # a 1% perturbation, in scaled units
H_MIN: Final[float] = 1e-10         # where the floor is measured
POINTS_PER_DECADE: Final[int] = 5
WINDOW_DECADES: Final[float] = 2.0
FLOOR_MARGIN: Final[float] = 10.0   # signal must clear the floor by this factor to be scored
FLOOR_DECADES: Final[float] = 1.0   # lowest decade(s) of h used to MEASURE the floor

# The floor is measured as the MAXIMUM of |R| over the lowest decade, not the median or the mean.
# A noise floor is an AMPLITUDE: what the window needs is a level the signal reliably clears, and
# a median sits in the middle of the noise rather than above it.
#
# Found by a check failing. With a median of three samples the floor read 4.3e-16 while |R| in
# that same region ranged to 4.8e-15 -- 10x higher. The window then opened inside the noise, the
# curve there is not monotone, `_cancellation_ceiling` fired on ordinary jitter, and roughly one
# direction in twenty scored `insufficient_signal` with a perfectly correct gradient.
#
# The lowest decade is safe to use for this: at h = 1e-9 the true remainder (h^2/2) v^T H v is
# about 2e-18 against a roundoff level near 1e-15, so that region is pure noise by three orders
# of magnitude.

# ------------------------------------------------------------------- V15 / V16
JVP_VJP_RTOL: Final[float] = 1e-10


@dataclasses.dataclass(frozen=True)
class CheckResult:
    passed: bool
    message: str
    measured: dict[str, Any]

    def __bool__(self) -> bool:
        return self.passed


# --------------------------------------------------------------------------------- helpers


def perturbation_scale(theta: Any, scales: Any) -> jnp.ndarray:
    """`s_i = max(|theta_i|, scale_i)`. PRD §7.1 — required, not optional.

    A parameter at 1e-12 perturbed by 1e-12 moves the objective by nothing measurable, and the
    remainder sits on the noise floor at every h. That reports a pass having probed nothing.
    """
    flat, _ = ravel_pytree(theta)
    scale_flat, _ = ravel_pytree(scales)
    if scale_flat.shape != flat.shape:
        raise ValueError(f"scales have shape {scale_flat.shape}, parameters {flat.shape}")
    if not bool(jnp.all(scale_flat > 0)):
        raise ValueError("every parameter scale must be strictly positive")
    return jnp.maximum(jnp.abs(flat), scale_flat)


def _directions(key: jax.Array, n: int, dim: int) -> jnp.ndarray:
    """Unit directions. Random rather than coordinate axes: an error confined to one parameter
    is visible along an axis, but an error in a ROTATION of the parameter space is not."""
    raw = jax.random.normal(key, (n, dim), dtype=jnp.float64)
    return raw / jnp.linalg.norm(raw, axis=1, keepdims=True)


def reverse_gradient(f: Callable[[Any], Any], theta: Any) -> Any:
    return jax.grad(f)(theta)


def _flatten_fn(f: Callable[[Any], Any], theta: Any):
    flat, unravel = ravel_pytree(theta)
    return (lambda x: f(unravel(x))), flat, unravel


def _h_sweep() -> np.ndarray:
    decades = np.log10(H_MAX) - np.log10(H_MIN)
    n = int(round(decades * POINTS_PER_DECADE)) + 1
    return np.logspace(np.log10(H_MIN), np.log10(H_MAX), n)


def _cancellation_ceiling(hs: np.ndarray, rs: np.ndarray, start: int) -> int:
    """Index of the first NOTCH at or above `start` — the first place |R| stops growing.

    A notch means two terms are cancelling; the local slope there is meaningless and reads about
    1.6 on a gradient that is exactly right. Everything above the notch is discarded. Nothing
    below it is, which is what keeps this safe: below is where a first-order error dominates.
    """
    for i in range(start, len(rs) - 1):
        if rs[i + 1] <= rs[i]:
            return i
    return len(rs) - 1


def _safe(reduce_fn, values: list[float]) -> float | None:
    """Reduce over the finite entries, or None if there are none. A direction with no scorable
    window has no h range to report, and reporting 0.0 would read as a measurement."""
    finite = [v for v in values if np.isfinite(v)]
    return float(reduce_fn(finite)) if finite else None


def _fit_slope(hs: np.ndarray, rs: np.ndarray) -> float:
    logs = np.log10(np.maximum(rs, np.finfo(float).tiny))
    return float(np.polyfit(np.log10(hs), logs, 1)[0])


# ----------------------------------------------------------------------------- V14: Taylor


def taylor_test(f: Callable[[Any], Any], theta: Any, gradient: Any, scales: Any,
                key: jax.Array, n_directions: int = N_DIRECTIONS) -> CheckResult:
    """V14. Scored per direction; every direction must pass on its own."""
    f_flat, flat, _ = _flatten_fn(f, theta)
    g_flat, _ = ravel_pytree(gradient)
    if g_flat.shape != flat.shape:
        raise ValueError(f"gradient has shape {g_flat.shape}, parameters {flat.shape}")

    s = perturbation_scale(theta, scales)
    hs = _h_sweep()
    hs_j = jnp.asarray(hs)
    j0 = f_flat(flat)

    dirs = _directions(key, n_directions, flat.shape[0])
    slopes: list[float] = []
    labels: list[str] = []
    windows: list[tuple[float, float]] = []
    floors: list[float] = []
    curves: list[list[float]] = []

    for d in dirs:
        v = d * s                                   # scaled perturbation
        directional = jnp.dot(g_flat, v)
        # Evaluate the whole sweep at once; h is the only thing that varies.
        js = jax.vmap(lambda h: f_flat(flat + h * v))(hs_j)
        rs = np.asarray(jnp.abs(js - j0 - hs_j * directional))
        curves.append([float(r) for r in rs])

        # 1. MEASURE the floor over the lowest decade, where the true remainder is negligible.
        n_floor = max(2, int(round(FLOOR_DECADES * POINTS_PER_DECADE)) + 1)
        floor = float(np.max(rs[:n_floor]))
        floors.append(floor)

        # 2. Anchor the window at the first h whose signal clears the floor.
        above = np.flatnonzero(rs > FLOOR_MARGIN * max(floor, np.finfo(float).tiny))
        if above.size == 0:
            slopes.append(float("nan"))
            labels.append(INSUFFICIENT)
            windows.append((float("nan"), float("nan")))
            continue
        start = int(above[0])

        # 3. Cap below the first notch, then take two decades from the anchor.
        ceiling = _cancellation_ceiling(hs, rs, start)
        h_lo = hs[start]
        h_hi = min(h_lo * 10.0**WINDOW_DECADES, hs[ceiling])
        sel = (hs >= h_lo) & (hs <= h_hi) & (rs > FLOOR_MARGIN * max(floor, np.finfo(float).tiny))
        windows.append((float(h_lo), float(h_hi)))

        if int(sel.sum()) < MIN_POINTS_IN_WINDOW:
            slopes.append(float("nan"))
            labels.append(INSUFFICIENT)
            continue

        slope = _fit_slope(hs[sel], rs[sel])
        slopes.append(slope)
        labels.append(CLEAN if slope <= CLEAN_UPPER else DEGENERATE)
        if slope < SLOPE_FAIL_BELOW:
            labels[-1] = SHALLOW

    failures = [i for i, lab in enumerate(labels) if lab in (INSUFFICIENT, SHALLOW)]
    finite = [s for s in slopes if np.isfinite(s)]
    degenerate_fraction = labels.count(DEGENERATE) / len(labels)

    measured = {
        "n_directions": len(labels),
        "slope_min": min(finite) if finite else None,
        "slope_max": max(finite) if finite else None,
        "classifications": labels,
        "degenerate_fraction": degenerate_fraction,
        "h_range_swept": [float(hs[0]), float(hs[-1])],
        # Aggregated across directions, so this pair is the UNION of the scored windows and is
        # informational. The invariant that matters is per-direction, below.
        "h_window_min": _safe(min, [w[0] for w in windows]),
        "h_window_max": _safe(max, [w[1] for w in windows]),
        "window_decades_max": _safe(
            max, [np.log10(w[1] / w[0]) for w in windows
                  if np.isfinite(w[0]) and np.isfinite(w[1]) and w[0] > 0]
        ),
        "noise_floor_median": float(np.median(floors)),
    }

    if failures:
        worst = ", ".join(
            f"dir {i}: {labels[i]}"
            + (f" (slope {slopes[i]:.3f})" if np.isfinite(slopes[i]) else "")
            for i in failures[:5]
        )
        return CheckResult(
            False,
            f"V14 FAILED on {len(failures)}/{len(labels)} direction(s): {worst}. "
            f"A wrong gradient leaves the first-order term uncancelled and tends to slope 1. "
            f"Do NOT widen the window to fix this.",
            measured,
        )
    return CheckResult(
        True,
        f"V14 passed on {len(labels)} directions; slopes "
        f"[{measured['slope_min']:.3f}, {measured['slope_max']:.3f}], "
        f"degenerate fraction {degenerate_fraction:.2f}",
        measured,
    )


# ------------------------------------------------------------- V15: forward versus reverse


def forward_reverse_test(f: Callable[[Any], Any], theta: Any, key: jax.Array,
                         n_directions: int = N_DIRECTIONS,
                         rtol: float = JVP_VJP_RTOL,
                         gradient_fn: Callable[[Callable, Any], Any] | None = None
                         ) -> CheckResult:
    """V15. `jax.jvp` against `jax.vjp` on the same directional derivative.

    A TRANSPOSE check (finding A9). It cannot catch a wrong JVP rule, because reverse mode is
    built by transposing that same rule — both would agree on the same wrong answer.

    `gradient_fn` supplies the REVERSE side. It exists so V19 can inject a corrupted reverse-mode
    gradient: without it the canary could only reach V14, since V15 would quietly recompute a
    correct gradient of its own and compare that.
    """
    f_flat, flat, _ = _flatten_fn(f, theta)
    grad_of = gradient_fn if gradient_fn is not None else (lambda fn, th: jax.grad(fn)(th))
    g_flat, _ = ravel_pytree(grad_of(f, theta))
    dirs = _directions(key, n_directions, flat.shape[0])

    worst, worst_pair = 0.0, (0.0, 0.0)
    for d in dirs:
        try:
            _, fwd = jax.jvp(f_flat, (flat,), (d,))
        except Exception as exc:
            # Finding A9: a function carrying a `custom_vjp` rule cannot be differentiated in
            # forward mode at all, so jax.jvp raises. That is a FAILURE of V15, not a skip --
            # the check could not be performed, and a check that could not run must never be
            # recorded as a check that passed.
            return CheckResult(
                False,
                f"V15 FAILED: forward mode is unavailable on this function "
                f"({type(exc).__name__}: {exc}). A custom_vjp rule with no matching JVP makes "
                f"V15 and V16 inapplicable -- they cannot vouch for the reverse-mode rule.",
                {"forward_mode_available": False, "error": f"{type(exc).__name__}: {exc}"},
            )
        rev = jnp.dot(g_flat, d)
        denom = max(abs(float(fwd)), abs(float(rev)), 1.0)
        rel = abs(float(fwd) - float(rev)) / denom
        if rel > worst:
            worst, worst_pair = rel, (float(fwd), float(rev))

    measured = {"worst_relative_difference": worst, "rtol": rtol,
                "worst_pair": list(worst_pair), "n_directions": n_directions}
    if worst > rtol:
        return CheckResult(False, f"V15 FAILED: jvp/vjp differ by {worst:.3e} > {rtol:.1e} "
                                  f"(forward {worst_pair[0]:.12g}, reverse {worst_pair[1]:.12g})",
                           measured)
    return CheckResult(True, f"V15 passed: worst jvp/vjp difference {worst:.3e}", measured)


# -------------------------------------------------------------------- V16: dot-product test


def dot_product_test(vector_map: Callable[[Any], Any], theta: Any, key: jax.Array,
                     n_trials: int = N_DIRECTIONS,
                     rtol: float = JVP_VJP_RTOL,
                     cotangent_transform: Callable[[jnp.ndarray], jnp.ndarray] | None = None
                     ) -> CheckResult:
    """V16. For random u and w: `<w, Ju> == <J^T w, u>`, one side by `jvp`, the other by `vjp`.

    Needs a VECTOR-valued map: with a scalar objective w is a scalar and the identity collapses
    into V15. Nearly free while no custom VJP rule exists, and essential the moment one does.

    `cotangent_transform` corrupts the `J^T w` side, for the same reason `gradient_fn` exists on
    V15: a canary that cannot reach a check cannot show that check works.
    """
    _, flat, unravel = _flatten_fn(vector_map, theta)
    f_flat = lambda x: vector_map(unravel(x))
    y0, vjp_fn = jax.vjp(f_flat, flat)
    out_flat, _ = ravel_pytree(y0)

    ku, kw = jax.random.split(key)
    us = _directions(ku, n_trials, flat.shape[0])
    ws = _directions(kw, n_trials, out_flat.shape[0])

    worst, worst_pair = 0.0, (0.0, 0.0)
    for u, w in zip(us, ws):
        _, ju = jax.jvp(f_flat, (flat,), (u,))
        ju_flat, _ = ravel_pytree(ju)
        (jtw,) = vjp_fn(unravel_like(w, y0))
        if cotangent_transform is not None:
            jtw = cotangent_transform(jtw)
        lhs = float(jnp.dot(w, ju_flat))
        rhs = float(jnp.dot(jtw, u))
        denom = max(abs(lhs), abs(rhs), 1.0)
        rel = abs(lhs - rhs) / denom
        if rel > worst:
            worst, worst_pair = rel, (lhs, rhs)

    measured = {"worst_relative_difference": worst, "rtol": rtol,
                "worst_pair": list(worst_pair), "n_trials": n_trials}
    if worst > rtol:
        return CheckResult(False, f"V16 FAILED: <w,Ju> and <J^T w,u> differ by {worst:.3e} "
                                  f"> {rtol:.1e}", measured)
    return CheckResult(True, f"V16 passed: worst transpose difference {worst:.3e}", measured)


def unravel_like(flat_vector: jnp.ndarray, template: Any) -> Any:
    """Reshape a flat cotangent to the structure of `template`."""
    _, unravel = ravel_pytree(template)
    return unravel(flat_vector)
