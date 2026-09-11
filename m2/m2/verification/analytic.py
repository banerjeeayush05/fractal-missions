"""Trivial analytic functions for verifying the verification harness itself. No physics.

``objective`` is strictly convex (positive-definite Hessian everywhere), so every direction
has curvature and a correct gradient gives a clean O(h²) Taylor remainder (compare
OPEN_QUESTIONS A10). ``vector_map`` is a nonlinear map R⁵ → R⁶ for the V16 dot-product test.
Parameters are a dict pytree with mixed leaf shapes, as M2's parameter PyTrees will be.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

import m2  # noqa: F401

_rng = np.random.default_rng(20260911)
_A = jnp.asarray(_rng.normal(size=(4, 5)))
_M = _rng.normal(size=(5, 5))
_Q = jnp.asarray(_M @ _M.T / 5 + 0.5 * np.eye(5))  # SPD, λ_min > 0.5
_B = jnp.asarray(_rng.normal(size=(6, 5)) / 2)
_C = jnp.asarray(_rng.normal(size=(6, 5)) / 2)


def theta0() -> dict[str, jax.Array]:
    """Base point, chosen so every gradient component is well away from zero (B13)."""
    return {
        "a": jnp.asarray(0.7, dtype=jnp.float64),
        "b": jnp.asarray([-0.4, 1.1], dtype=jnp.float64),
        "c": jnp.asarray([0.9, -1.3], dtype=jnp.float64),
    }


def scales() -> dict[str, jax.Array]:
    """Declared typical magnitudes, required by the harness (decision B20). Same tree as theta0."""
    return {
        "a": jnp.asarray(1.0, dtype=jnp.float64),
        "b": jnp.ones(2, dtype=jnp.float64),
        "c": jnp.ones(2, dtype=jnp.float64),
    }


def _x(theta) -> jax.Array:
    return ravel_pytree(theta)[0]


def objective(theta) -> jax.Array:
    """J(θ) = logsumexp(Aθ) + ½θᵀQθ + Σ exp(0.3 θᵢ). Strictly convex."""
    x = _x(theta)
    return jax.nn.logsumexp(_A @ x) + 0.5 * x @ _Q @ x + jnp.sum(jnp.exp(0.3 * x))


def vector_map(theta) -> jax.Array:
    """F(θ) = tanh(Bθ) + 0.1 (Cθ)², a nonlinear map R⁵ → R⁶."""
    x = _x(theta)
    return jnp.tanh(_B @ x) + 0.1 * (_C @ x) ** 2
