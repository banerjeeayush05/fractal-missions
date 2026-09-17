"""V19 — the corrupted-gradient canary. PRD §7.2, §8.5. The M2.0 gate.

Built against a trivial analytic function; no solver exists yet.

The argument, in one line: **a verification suite that has never been shown to fail has not been
verified.** V14, V15 and V16 all pass on this codebase. So would a harness that returned `True`
unconditionally. V19 is what distinguishes the two — inject a 5% error into one gradient
component and require all three checks to go red, with an unmutated control passing.

Both halves are load-bearing. Without the control, a harness that always fails would "pass" V19.

Two preconditions the canary ASSERTS rather than assumes:

* **No gradient component may be near zero** (decision B13). A multiplicative corruption of a
  zero component is not a corruption at all — `1.05 * 0 == 0` — so such a case would report an
  undetectable error as a harness failure.
* **The corruption must actually change the gradient.** A no-op corruption makes the mutant
  identical to the control, and every check passes on both. `tests/test_v19_mutants.py` sabotages
  the canary five ways and requires each sabotage to turn V19 red.

`run_canary` takes its three checks as ARGUMENTS. That is what makes the mutation guard possible:
the mutant test passes in deliberately broken versions and requires the verdict to notice.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from geocore.constants import V19_CORRUPTION_FACTOR
from geocore.verification.gradcheck import (
    CheckResult,
    dot_product_test,
    forward_reverse_test,
    taylor_test,
)

NEAR_ZERO: float = 1e-6


def corrupt_component(gradient: Any, index: int, factor: float = V19_CORRUPTION_FACTOR) -> Any:
    """Multiply one flattened component of a gradient PyTree by `factor`."""
    flat, unravel = ravel_pytree(gradient)
    return unravel(flat.at[index].multiply(factor))


def _assert_detectable(gradient: Any) -> list[float]:
    flat, _ = ravel_pytree(gradient)
    values = [float(v) for v in flat]
    tiny = [i for i, v in enumerate(values) if abs(v) < NEAR_ZERO]
    if tiny:
        raise ValueError(
            f"gradient components {tiny} are within {NEAR_ZERO} of zero; a multiplicative "
            f"corruption of a zero component is undetectable by ANY check, so the canary would "
            f"report an impossible task as a harness failure (decision B13)"
        )
    return values


@dataclasses.dataclass(frozen=True)
class CanaryReport:
    control: dict[str, CheckResult]
    mutants: list[dict[str, CheckResult]]
    gradient_values: list[float]
    factor: float

    @property
    def control_passes(self) -> bool:
        return all(r.passed for r in self.control.values())

    def mutant_detected_by(self, index: int) -> dict[str, bool]:
        """Per check: did it FAIL on this mutant? Failing is the desired outcome."""
        return {name: (not r.passed) for name, r in self.mutants[index].items()}

    @property
    def undetected(self) -> list[tuple[int, str]]:
        """(component, check) pairs where a corrupted gradient still passed."""
        return [(i, name) for i, m in enumerate(self.mutants)
                for name, r in m.items() if r.passed]

    def verdict(self) -> CheckResult:
        measured = {
            "factor": self.factor,
            "n_components": len(self.mutants),
            "control": {k: v.passed for k, v in self.control.items()},
            "control_measured": {k: v.measured for k, v in self.control.items()},
            "detection": [self.mutant_detected_by(i) for i in range(len(self.mutants))],
            "undetected": [[i, n] for i, n in self.undetected],
        }
        if not self.control_passes:
            failed = [k for k, v in self.control.items() if not v.passed]
            return CheckResult(
                False,
                f"V19 FAILED: the UNMUTATED control did not pass {failed}. Without a passing "
                f"control, a harness that always fails would 'pass' this check.",
                measured,
            )
        if self.undetected:
            missed = ", ".join(f"component {i} slipped past {n}" for i, n in self.undetected)
            return CheckResult(
                False,
                f"V19 FAILED: a {self.factor}x corruption was NOT caught -- {missed}. The "
                f"check(s) named cannot be relied on to detect a wrong gradient.",
                measured,
            )
        return CheckResult(
            True,
            f"V19 passed: control clean; every component's {self.factor}x corruption caught by "
            f"V14, V15 and V16",
            measured,
        )


def run_canary(objective: Callable[[Any], Any], vector_map: Callable[[Any], Any], theta: Any,
               scales: Any, key: jax.Array, factor: float = V19_CORRUPTION_FACTOR,
               *,
               taylor: Callable[..., CheckResult] = taylor_test,
               forward_reverse: Callable[..., CheckResult] = forward_reverse_test,
               dot_product: Callable[..., CheckResult] = dot_product_test,
               corrupt: Callable[[Any, int, float], Any] = corrupt_component,
               components: list[int] | None = None) -> CanaryReport:
    """Run the control, then one mutant per gradient component.

    The three checks are injectable so `tests/test_v19_mutants.py` can sabotage them.
    """
    true_gradient = jax.grad(objective)(theta)
    values = _assert_detectable(true_gradient)
    n = len(values)
    indices = list(range(n)) if components is None else list(components)

    control = {
        "V14": taylor(objective, theta, true_gradient, scales, key),
        "V15": forward_reverse(objective, theta, key),
        "V16": dot_product(vector_map, theta, key),
    }

    mutants: list[dict[str, CheckResult]] = []
    for i in indices:
        bad = corrupt(true_gradient, i, factor)

        # The SAME corruption has to reach all three, or V19 would only ever exercise V14:
        # V15 and V16 build their own reverse mode and would silently recompute a correct one.
        def bad_gradient(_fn: Callable, _th: Any, _bad: Any = bad) -> Any:
            return _bad

        def bad_cotangent(jtw: jnp.ndarray, _i: int = i, _f: float = factor) -> jnp.ndarray:
            return jtw.at[_i].multiply(_f)

        mutants.append({
            "V14": taylor(objective, theta, bad, scales, key),
            "V15": forward_reverse(objective, theta, key, gradient_fn=bad_gradient),
            "V16": dot_product(vector_map, theta, key, cotangent_transform=bad_cotangent),
        })

    return CanaryReport(control=control, mutants=mutants, gradient_values=values, factor=factor)


def taylor_detection_threshold(objective: Callable[[Any], Any], theta: Any, scales: Any,
                               key: jax.Array, index: int = 0,
                               factors: tuple[float, ...] = (1.05, 1.02, 1.01, 1.005, 1.002,
                                                             1.001)) -> float | None:
    """The smallest corruption V14 still catches, recorded in the ledger.

    Not a pass/fail criterion — a SENSITIVITY figure. It is the honest answer to "how wrong would
    the gradient have to be before we noticed", and a change in it between runs is a finding.
    """
    true_gradient = jax.grad(objective)(theta)
    smallest: float | None = None
    for factor in factors:
        bad = corrupt_component(true_gradient, index, factor)
        if not taylor_test(objective, theta, bad, scales, key).passed:
            smallest = factor
        else:
            break
    return smallest
