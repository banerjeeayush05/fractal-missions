"""V19 — corrupted-gradient canary (PRD §7.2, §8.5).

Procedure (OPEN_QUESTIONS B13):
1. Control: the true reverse-mode gradient/VJP must PASS V14, V15 and V16. Without this, a
   harness that always fails would "pass" the canary.
2. For every component k in turn, scale the claimed gradient (and the claimed VJP output)
   component k by 1.05. V14, V15 and V16 must each FAIL.
3. A degenerate direction must not become a hiding place: under the decision B15/B21 scoring a
   slope above the band passes, so `tests/test_v19_canary.py` also runs a genuinely degenerate
   objective with a corrupted gradient and requires it to fail.
Precondition: the SCALED sensitivity max(|θ_k|, scale_k)·|∂J/∂θ_k| ≥ V19_MIN_GRAD_COMPONENT at θ₀.
A 5 % error on a component the harness barely perturbs is invisible to every check, so that case is
refused rather than reported as caught.

The same procedure runs against the real solver's objectives from M2.3 on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
from jax.flatten_util import ravel_pytree

from m2.constants import V19_CORRUPTION_FACTOR, V19_MIN_GRAD_COMPONENT
from m2.verification.gradcheck import (
    CheckResult,
    perturbation_scale,
    dot_product_test,
    forward_reverse_test,
    reverse_gradient,
    reverse_vjp,
    taylor_test,
)

PyTree = Any


def corrupt_component(tree: PyTree, index: int, factor: float = V19_CORRUPTION_FACTOR) -> PyTree:
    flat, unravel = ravel_pytree(tree)
    return unravel(flat.at[index].multiply(factor))


def corrupt_vjp(vjp_fn: Callable[[PyTree], PyTree], index: int,
                factor: float = V19_CORRUPTION_FACTOR) -> Callable[[PyTree], PyTree]:
    return lambda w: corrupt_component(vjp_fn(w), index, factor)


@dataclass
class CanaryReport:
    control: dict[str, CheckResult]
    corrupted: dict[int, dict[str, CheckResult]]
    taylor_detection: dict[str, Any] = field(default_factory=dict)

    @property
    def control_passed(self) -> bool:
        return all(r.passed for r in self.control.values())

    @property
    def all_caught(self) -> bool:
        return all(not r.passed for per_k in self.corrupted.values() for r in per_k.values())

    @property
    def caught(self) -> bool:
        return self.control_passed and self.all_caught

    def measured(self) -> dict[str, Any]:
        return {
            "corruption_factor": V19_CORRUPTION_FACTOR,
            "n_components": len(self.corrupted),
            "control": {c: {"passed": r.passed, **r.measured} for c, r in self.control.items()},
            "corrupted_detected": {f"component_{k}": {c: (not r.passed) for c, r in per_k.items()}
                                   for k, per_k in self.corrupted.items()},
            "corrupted_V14_slope_min": {f"component_{k}": per_k["V14"].measured.get("slope_min")
                                        for k, per_k in self.corrupted.items()},
            "corrupted_V15_max_rel_error": {f"component_{k}": per_k["V15"].measured.get("max_rel_error")
                                            for k, per_k in self.corrupted.items()},
            "corrupted_V16_max_rel_error": {f"component_{k}": per_k["V16"].measured.get("max_rel_error")
                                            for k, per_k in self.corrupted.items()},
            "taylor_detection": self.taylor_detection,
        }


def run_canary(J: Callable[[PyTree], Any], F: Callable[[PyTree], Any], theta: PyTree, *,
               scales: PyTree, key: jax.Array) -> CanaryReport:
    g = reverse_gradient(J, theta)
    vjp = reverse_vjp(F, theta)
    gflat = np.asarray(ravel_pytree(g)[0])
    # Scaled sensitivity max(|θ_k|, scale_k)·|∂J/∂θ_k| (decisions B13, B20): a 5 % error on a
    # component the harness barely perturbs is undetectable by any check.
    sensitivity = perturbation_scale(np.asarray(ravel_pytree(theta)[0]), scales) * np.abs(gflat)
    if np.any(sensitivity < V19_MIN_GRAD_COMPONENT):
        raise ValueError(f"V19 precondition: scaled sensitivity {sensitivity} has components below "
                         f"{V19_MIN_GRAD_COMPONENT}; a 5 % error there is undetectable (B13, B20)")
    k14, k15, k16 = jax.random.split(key, 3)

    def three(grad, vjp_fn):
        return {
            "V14": taylor_test(J, theta, grad, scales=scales, key=k14),
            "V15": forward_reverse_test(J, theta, grad, scales=scales, key=k15),
            "V16": dot_product_test(F, theta, vjp_fn, scales=scales, key=k16),
        }

    control = three(g, vjp)
    corrupted = {k: three(corrupt_component(g, k), corrupt_vjp(vjp, k)) for k in range(gflat.size)}
    return CanaryReport(control, corrupted)


def taylor_detection_threshold(J: Callable[[PyTree], Any], theta: PyTree, *, scales: PyTree,
                               key: jax.Array,
                               factors_minus_one=(1e-1, 5e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
                               ) -> dict[str, Any]:
    """Smallest relative corruption ε of each gradient component that V14 still catches.

    Reported, not asserted: it measures how fine the Taylor gate is at this h range. For V15 and
    V16 the threshold is their 1e-10 tolerance by construction.
    """
    g = reverse_gradient(J, theta)
    n = ravel_pytree(g)[0].size
    out = {}
    for k in range(n):
        caught = [eps for eps in factors_minus_one
                  if not taylor_test(J, theta, corrupt_component(g, k, 1.0 + eps), scales=scales,
                                     key=key).passed]
        out[f"component_{k}"] = min(caught) if caught else None
    return {"smallest_relative_error_caught": out, "eps_grid": list(factors_minus_one)}
