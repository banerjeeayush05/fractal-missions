"""The two analytic velocity models M2 ships for its own testing (PRD §5.5).

These return an **etch rate R in nm/s: positive removes material** (decision §1). M2 applies the
sign once, in the advection equation φ_t − R|∇φ| = 0, so no velocity model — here, in M3 or in M5 —
ever carries it.

M2 ships analytic models only. Flux, visibility and shadowing are M3 (§3), and a placeholder here
would create a contract M3 then has to fight.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import m2  # noqa: F401  (enables fp64)
from m2.constants import DIRECTIONAL_MIN_P
from m2.schema import PyTree, VelocityRequest


def isotropic(request: VelocityRequest, params: PyTree) -> jax.Array:
    """R = v_iso everywhere: the neutral-etch stand-in, independent of orientation."""
    return params["v_iso"] * jnp.ones(request.weights.shape, dtype=request.weights.dtype)


def directional(request: VelocityRequest, params: PyTree) -> jax.Array:
    """R = v_iso + v_dir·max(0, n·ẑ)^p — a cosine-law ion flux with no shadowing.

    ẑ points toward the plasma, so up-facing surfaces etch and a downward-facing overhang does not
    (checked by V1a). p must exceed 1: at p = 1 the law has a kink exactly at n·ẑ = 0, which is the
    vertical sidewall — the most common surface in a trench.

    `0^p` is the trap here. Its derivative with respect to p is −∞·0 in the naive form, and a NaN in
    an untaken branch still poisons the gradient, so the base is made benign *before* the power.
    """
    from m2.constants import Z_AXIS, Z_HAT_SIGN

    mu = request.normals[..., Z_AXIS] * Z_HAT_SIGN
    positive = mu > 0.0
    safe_base = jnp.where(positive, mu, 1.0)  # double-where: no 0^p reaches the derivative
    directional_term = jnp.where(positive, safe_base ** params["p"], 0.0)
    return params["v_iso"] + params["v_dir"] * directional_term


MODELS = {"isotropic": isotropic, "directional": directional}


def model_for(name: str):
    if name not in MODELS:
        raise KeyError(f"unknown velocity model {name!r}; M2 ships {sorted(MODELS)} (§5.5)")
    return MODELS[name]


def check_parameters(name: str, params: PyTree) -> None:
    """Structural checks the config layer also performs, for parameters built in code."""
    expected = {"isotropic": {"v_iso"}, "directional": {"v_iso", "v_dir", "p"}}[name]
    if set(params) != expected:
        raise KeyError(f"model {name!r} needs exactly {sorted(expected)}, got {sorted(params)}")
    if "p" in params and float(params["p"]) <= DIRECTIONAL_MIN_P:
        raise ValueError(f"p must be > {DIRECTIONAL_MIN_P}: at p = 1 the law has a kink on vertical "
                         f"sidewalls, where the derivative dies")
