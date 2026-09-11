"""Gradient verification harness: V14 (Taylor remainder), V15 and V16 (transpose consistency).

Each check is handed the function plus a *claimed* reverse-mode derivative (a gradient, or a
vector–Jacobian product function) and decides whether the claim is right. Primal evaluations and
forward-mode ``jax.jvp`` are computed here. The claim is injected by the caller, so V19 (§7.2) can
corrupt exactly what production code would hand in.

**What each check can actually see** (decision §5, from OQ A9). JAX builds every reverse-mode
derivative by linearising with the forward-mode JVP rules and transposing them, so ``jvp`` and
``vjp`` share those rules for *all* primitives, not only custom ones. V15 and V16 therefore verify
**transpose consistency**, not derivative correctness. V14 is the only check here that compares a
derivative against the function itself; V14a/V14b/V14c (analytic sensitivities) carry the rest.

**V14 scoring** (decision B15/B21, replacing the earlier three-zone rule). A wrong gradient leaves
the first-order term uncancelled, so its remainder goes as h and the slope tends to 1. No incorrect
gradient produces a slope above 2, so there is no upper bound to enforce:

    fewer than 3 points above the noise floor -> FAIL, "insufficient signal"
    slope < 1.8                               -> FAIL
    slope in [1.8, 2.2]                       -> PASS, "clean_quadratic"
    slope > 2.2                               -> PASS, "degenerate_direction"

The five-decade span is a condition on the *clean_quadratic* classification, not on pass or fail:
an O(h³) remainder reaches the floor sooner and can never span five decades, so scoring it
span-first would fail a correct gradient. Every direction's classification, and the degenerate
fraction for the run, go into the ledger: if that fraction jumps between runs, something changed
even though everything is green.

**Directions** are drawn in relative units using declared parameter scales (decision B20):
δ_i ∝ max(|θ_i|, scale_i). Scales are **required**, because an optional field is omitted exactly
where it matters — a parameter sitting at 1e-12 would otherwise get a 1e-12 perturbation, probing
nothing while reporting a pass.

Protocol constants live in ``m2.constants``. Callers may make a check stricter but never looser
(§11); a looser argument raises.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

import m2  # noqa: F401  (enables fp64)
from m2.constants import (
    DTYPE,
    TAYLOR_H_MAX,
    TAYLOR_MIN_DECADES,
    TAYLOR_MIN_DIRECTIONS,
    TAYLOR_MIN_POINTS_ABOVE_FLOOR,
    TAYLOR_N_H,
    TAYLOR_NOISE_FLOOR_C,
    TAYLOR_NOISE_FLOOR_REPEATS,
    TAYLOR_SLOPE_BAND,
    V15_MIN_DIRECTIONS,
    V15_RTOL,
    V16_MIN_PAIRS,
    V16_RTOL,
)

PyTree = Any
EPS = float(np.finfo(np.float64).eps)

CLEAN = "clean_quadratic"
DEGENERATE = "degenerate_direction"
NARROW = "narrow_span"  # slope in band but fewer than five decades survive (OPEN_QUESTIONS B24)


@dataclass(frozen=True)
class CheckResult:
    check: str
    passed: bool
    message: str
    measured: dict[str, Any] = field(default_factory=dict)


# --- reverse-mode providers (what production code hands to the checks) --------------------


def reverse_gradient(J: Callable[[PyTree], Any], theta: PyTree) -> PyTree:
    return jax.grad(J)(theta)


def reverse_vjp(F: Callable[[PyTree], Any], theta: PyTree) -> Callable[[PyTree], PyTree]:
    _, pullback = jax.vjp(F, theta)
    return lambda w: pullback(w)[0]


# --- noise floor ------------------------------------------------------------------------


def estimate_noise_floor(J: Callable[[jax.Array], Any], x0: jax.Array, *, n_steps: int = 1,
                         repeats: int = TAYLOR_NOISE_FLOOR_REPEATS) -> tuple[float, float]:
    """Spread of J over repeated evaluations at fixed θ, and the floor used for exclusion.

    A deterministic J gives a spread of exactly zero, so the floor falls back to accumulated
    roundoff: under a random-walk model over ``n_steps`` solver steps that is ~√N·eps·|J|, taken
    with a safety factor C (decision B19). Keeping the √N scaling matters because N changes by a
    factor of a few between grid refinements. A Monte Carlo J (M3) gives a real spread instead.
    """
    values = np.array([float(J(x0)) for _ in range(repeats)])
    spread = float(values.max() - values.min())
    roundoff = TAYLOR_NOISE_FLOOR_C * np.sqrt(max(n_steps, 1)) * EPS * max(abs(float(values[0])), 1.0)
    return spread, max(spread, float(roundoff))


# --- V14 ------------------------------------------------------------------------------------


def taylor_test(
    J: Callable[[PyTree], Any],
    theta: PyTree,
    grad: PyTree,
    *,
    scales: PyTree,
    key: jax.Array,
    n_directions: int = TAYLOR_MIN_DIRECTIONS,
    decades: float = TAYLOR_MIN_DECADES,
    n_h: int = TAYLOR_N_H,
    h_max: float = TAYLOR_H_MAX,
    slope_band: tuple[float, float] = TAYLOR_SLOPE_BAND,
    noise_floor: float | None = None,
    n_steps: int = 1,
) -> CheckResult:
    """V14 (§7.1, decisions §4 and B15/B21): R(h) = |J(θ+hδ) − J(θ) − h⟨g, δ⟩| must fall as O(h²)."""
    if n_directions < TAYLOR_MIN_DIRECTIONS:
        raise ValueError(f"n_directions={n_directions} < {TAYLOR_MIN_DIRECTIONS} (PRD §7.1)")
    if decades < TAYLOR_MIN_DECADES:
        raise ValueError(f"decades={decades} < {TAYLOR_MIN_DECADES} (PRD §7.1)")
    if n_h < TAYLOR_N_H:
        raise ValueError(f"n_h={n_h} < {TAYLOR_N_H} (OPEN_QUESTIONS B3)")
    lo, hi = slope_band
    if lo < TAYLOR_SLOPE_BAND[0] or hi > TAYLOR_SLOPE_BAND[1] or lo >= hi:
        raise ValueError(f"slope_band={slope_band} is looser than {TAYLOR_SLOPE_BAND} (PRD §7.1, §11)")

    x0, unravel = _ravel(theta, "theta")
    g, _ = _ravel(grad, "grad")
    if g.shape != x0.shape:
        raise ValueError(f"gradient has {g.size} components, parameters have {x0.size}")
    Jx = jax.jit(lambda x: J(unravel(x)))
    J0 = float(Jx(x0))
    spread, floor = ((float("nan"), noise_floor) if noise_floor is not None
                     else estimate_noise_floor(Jx, x0, n_steps=n_steps))
    hs = h_max * np.logspace(0.0, -float(decades), n_h)
    deltas = _directions(key, n_directions, x0, scales)

    slopes: list[float] = []
    classes: list[str] = []
    failures: list[str] = []
    kept_counts: list[int] = []
    for i, d in enumerate(deltas):
        dd = jnp.asarray(d)
        lin = float(jnp.dot(g, dd))
        Jh = np.array([float(Jx(x0 + h * dd)) for h in hs])
        R = np.abs(Jh - J0 - hs * lin)
        keep = np.isfinite(R) & (R > floor)
        kept_counts.append(int(keep.sum()))
        if keep.sum() < TAYLOR_MIN_POINTS_ABOVE_FLOOR:
            slopes.append(float("nan"))
            classes.append("insufficient_signal")
            failures.append(f"δ{i}: insufficient signal — {int(keep.sum())} points above the noise "
                            f"floor {floor:.3e}; raise h_max, lower the noise, or use an objective "
                            f"with curvature (linear functionals belong in V14a–V14c)")
            continue
        slope = float(np.polyfit(np.log10(hs[keep]), np.log10(R[keep]), 1)[0])
        span = float(np.log10(hs[keep].max()) - np.log10(hs[keep].min()))
        slopes.append(slope)
        if slope < lo:
            classes.append("first_order_error")
            failures.append(f"δ{i}: slope {slope:.3f} < {lo} — first-order term present: gradient error")
        elif slope <= hi:
            classes.append(CLEAN if span >= decades else NARROW)
        else:
            classes.append(DEGENERATE)

    passed = not failures
    finite = np.array([s for s in slopes if np.isfinite(s)])
    degenerate = [i for i, c in enumerate(classes) if c == DEGENERATE]
    measured = {
        "slopes": [round(v, 4) if np.isfinite(v) else "nan" for v in slopes],
        "classifications": classes,
        "degenerate_fraction": round(len(degenerate) / max(len(classes), 1), 4),
        "slope_min": float(finite.min()) if finite.size else "nan",
        "slope_max": float(finite.max()) if finite.size else "nan",
        "n_directions": n_directions,
        "h_range": [float(hs[-1]), float(hs[0])],
        "n_h": n_h,
        "noise_floor": floor,
        "noise_spread": spread,
        "n_steps": n_steps,
        "points_kept_min": int(min(kept_counts)),
        "J0": J0,
    }
    msg = "; ".join(failures[:5]) + (" …" if len(failures) > 5 else "")
    if passed:
        counts = {c: classes.count(c) for c in sorted(set(classes))}
        msg = f"all {n_directions} directions pass: {counts}"
    return CheckResult("V14", passed, msg, measured)


# --- V15 ------------------------------------------------------------------------------------


def forward_reverse_test(
    J: Callable[[PyTree], Any],
    theta: PyTree,
    grad: PyTree,
    *,
    scales: PyTree,
    key: jax.Array,
    n_directions: int = V15_MIN_DIRECTIONS,
    rtol: float = V15_RTOL,
) -> CheckResult:
    """V15 (§8.5): forward-mode ⟨∇J, δ⟩ must equal ⟨g, δ⟩ from the claimed gradient.

    A transpose-consistency check, not a derivative check: see the module docstring.
    """
    if n_directions < V15_MIN_DIRECTIONS:
        raise ValueError(f"n_directions={n_directions} < {V15_MIN_DIRECTIONS} (OPEN_QUESTIONS B4)")
    if rtol > V15_RTOL:
        raise ValueError(f"rtol={rtol} is looser than {V15_RTOL} (PRD §8.5, §11)")
    x0, unravel = _ravel(theta, "theta")
    g, _ = _ravel(grad, "grad")
    scale = float(jnp.linalg.norm(g))  # ‖∇J‖·‖δ‖ with ‖δ‖ = 1 (decision B4)
    fwd = jax.jit(lambda x, v: jax.jvp(lambda y: J(unravel(y)), (x,), (v,))[1])
    errs = []
    for d in _directions(key, n_directions, x0, scales):
        dd = jnp.asarray(d)
        try:
            a = float(fwd(x0, dd))
        except Exception as e:  # noqa: BLE001  forward mode unavailable is a FAIL, not a skip
            return CheckResult("V15", False, f"forward mode unavailable: {type(e).__name__}: {e}"[:400],
                               {"error": type(e).__name__})
        errs.append(_rel(a, float(jnp.dot(g, dd)), scale))
    return _tolerance_result("V15", errs, rtol, "jvp vs claimed gradient")


# --- V16 ------------------------------------------------------------------------------------


def dot_product_test(
    F: Callable[[PyTree], Any],
    theta: PyTree,
    vjp_fn: Callable[[PyTree], PyTree],
    *,
    scales: PyTree,
    key: jax.Array,
    n_pairs: int = V16_MIN_PAIRS,
    rtol: float = V16_RTOL,
) -> CheckResult:
    """V16 (§8.5): ⟨w, J u⟩ via jax.jvp must equal ⟨Jᵀw, u⟩ via the claimed VJP."""
    if n_pairs < V16_MIN_PAIRS:
        raise ValueError(f"n_pairs={n_pairs} < {V16_MIN_PAIRS} (OPEN_QUESTIONS B5)")
    if rtol > V16_RTOL:
        raise ValueError(f"rtol={rtol} is looser than {V16_RTOL} (OPEN_QUESTIONS B5, §11)")
    x0, unravel_in = _ravel(theta, "theta")
    y0, unravel_out = _ravel(F(theta), "F(theta)")
    fwd = jax.jit(lambda x, v: ravel_pytree(jax.jvp(lambda y: F(unravel_in(y)), (x,), (v,))[1])[0])
    ku, kw = jax.random.split(key)
    us = _directions(ku, n_pairs, x0, scales)
    ws = _directions(kw, n_pairs, jnp.ones_like(y0), jnp.ones_like(y0))  # output space has no θ scale
    errs = []
    for u, w in zip(us, ws):
        u, w = jnp.asarray(u), jnp.asarray(w)
        try:
            lhs = float(jnp.dot(w, fwd(x0, u)))
        except Exception as e:  # noqa: BLE001
            return CheckResult("V16", False, f"forward mode unavailable: {type(e).__name__}: {e}"[:400],
                               {"error": type(e).__name__})
        jtw, _ = _ravel(vjp_fn(unravel_out(w)), "vjp output")
        errs.append(_rel(lhs, float(jnp.dot(jtw, u)), float(jnp.linalg.norm(jtw))))
    return _tolerance_result("V16", errs, rtol, "<w, Ju> vs <Jᵀw, u>")


# --- helpers ----------------------------------------------------------------------------------


def _ravel(tree: PyTree, name: str) -> tuple[jax.Array, Callable]:
    flat, unravel = ravel_pytree(tree)
    if flat.dtype != jnp.dtype(DTYPE):
        raise TypeError(f"{name} must be {DTYPE} throughout (PRD §7.5), got {flat.dtype}")
    return flat, unravel


def perturbation_scale(x0: jax.Array, scales: PyTree) -> np.ndarray:
    """max(|θ_i|, scale_i) per component (decision B20). Scales are required, never inferred."""
    if scales is None:
        raise ValueError("parameter scales are required (decision B20): declare a typical magnitude "
                         "for every parameter, or a parameter near zero is never probed")
    s = np.abs(np.asarray(ravel_pytree(scales)[0], dtype=float))
    if s.shape != tuple(np.shape(x0)):
        raise ValueError(f"scales have {s.size} components, parameters have {np.size(x0)}")
    if not np.all(s > 0):
        raise ValueError(f"every declared scale must be positive, got {s}")
    return np.maximum(np.abs(np.asarray(x0)), s)


def _directions(key: jax.Array, n: int, x0: jax.Array, scales: PyTree) -> np.ndarray:
    """Unit-norm directions with δ_i ∝ max(|θ_i|, scale_i) (decisions §4 and B20)."""
    scale = perturbation_scale(x0, scales)
    d = np.asarray(jax.random.normal(key, (n, np.size(x0)), dtype=jnp.float64)) * scale
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("a random direction collapsed to zero under the declared scales")
    return d / norms


def _rel(a: float, b: float, scale: float) -> float:
    """Difference normalised by ‖∇J‖·‖δ‖ (‖δ‖ = 1), not by the directional derivative (B4)."""
    if scale == 0.0:
        return 0.0 if a == b else float("inf")
    return abs(a - b) / scale


def _tolerance_result(check: str, errs: list[float], rtol: float, what: str) -> CheckResult:
    e = np.array(errs)
    bad = np.flatnonzero(~(e <= rtol))
    passed = bad.size == 0
    msg = (f"{what}: max rel. error {e.max():.2e} <= {rtol:g}" if passed
           else f"{what}: {bad.size}/{e.size} directions exceed {rtol:g} (max {e.max():.2e})")
    return CheckResult(check, passed, msg, {"max_rel_error": float(e.max()), "rtol": rtol, "n": int(e.size),
                                            "n_failed": int(bad.size)})
