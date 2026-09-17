"""Objectives whose gradients are known in CLOSED FORM.

These exist so the harness can be validated against truth rather than against `jax.grad`.
Comparing `jax.grad` to `jax.grad` is a tautology; comparing it to a derivative worked out by
hand is evidence. The M2.0 gate runs the whole gradient harness against these, before a solver
exists — see `canary.py`.

`objective` is deliberately CUBIC in `a`. A quadratic would have a constant Hessian and an exactly
zero third-order term, which would make the Taylor remainder unnaturally clean and let a harness
bug hide. `vector_map` is vector-valued because V16's identity collapses into V15 for a scalar.

`linear_objective` carries finding **A10**: its Hessian is exactly zero, so the remainder of a
CORRECT gradient sits on the roundoff floor at every h. The [1.8, 2.2] band cannot pass it. This
is not a harness bug — it is a documented limit, and `tests/test_gradcheck.py` pins the behaviour
so it cannot be "fixed" into a silent pass.
"""

from __future__ import annotations

import jax.numpy as jnp

# Two properties are required of this point, and both were found by a check failing, not by
# foresight.
#
# 1. NO GRADIENT COMPONENT NEAR ZERO. A multiplicative corruption of a zero component is not a
#    corruption (1.05 * 0 == 0), so V19 would report an impossible task as a harness failure.
#    `canary.py` asserts this rather than assuming it (decision B13).
#
# 2. THE HESSIAN MUST BE DEFINITE HERE. For this J, d2J/da_i^2 = 6*a_i, so a single negative a_i
#    makes the Hessian indefinite -- and an indefinite Hessian guarantees directions with
#    v^T H v = 0, where a CORRECT gradient has no quadratic term for the Taylor remainder to
#    find. With a = (0.7, -1.3, 2.0) the eigenvalues were [-7.88, 3.10, 3.88, 4.08, 5.11, 12.12]
#    and roughly 1 direction in 20 scored `insufficient_signal`, failing the V19 control on 2 of
#    3 seeds. All a_i positive makes every 2x2 block [[6a_i, 1], [1, 4]] positive definite.
#
#    This is finding A10, and the fixture is what changed -- NOT the scoring rule. The limitation
#    itself is real, still present, and pinned by `linear_objective` below so it cannot quietly
#    become a passing case.
_A0 = (0.7, 1.3, 2.0)
_B0 = (1.1, 0.4, -0.9)


def theta0() -> dict[str, jnp.ndarray]:
    return {"a": jnp.array(_A0, dtype=jnp.float64), "b": jnp.array(_B0, dtype=jnp.float64)}


def scales() -> dict[str, jnp.ndarray]:
    """Per-parameter perturbation scales. PRD §7.1 — required, not optional."""
    return {"a": jnp.ones(3, dtype=jnp.float64), "b": jnp.ones(3, dtype=jnp.float64)}


def objective(theta: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """J = sum(a^3) + 2*sum(b^2) + sum(a*b)."""
    a, b = theta["a"], theta["b"]
    return jnp.sum(a**3) + 2.0 * jnp.sum(b**2) + jnp.sum(a * b)


def objective_gradient(theta: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    """The gradient of `objective`, BY HAND. dJ/da = 3a^2 + b; dJ/db = 4b + a."""
    a, b = theta["a"], theta["b"]
    return {"a": 3.0 * a**2 + b, "b": 4.0 * b + a}


def vector_map(theta: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """A vector-valued map for V16: F(a, b) = [a^2 ; b^3 ; a*b]."""
    a, b = theta["a"], theta["b"]
    return jnp.concatenate([a**2, b**3, a * b])


def linear_objective(theta: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Zero curvature in every direction. Finding A10's case, made executable."""
    return jnp.sum(2.0 * theta["a"]) + jnp.sum(3.0 * theta["b"])


def linear_gradient(theta: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    return {"a": jnp.full_like(theta["a"], 2.0), "b": jnp.full_like(theta["b"], 3.0)}
