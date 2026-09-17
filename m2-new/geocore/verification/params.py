"""Perturbation scales for the solver's differentiable parameters. PRD §7.1.

Required, not optional: V14 perturbs parameter i by `delta_i * max(|theta_i|, scale_i)`. A parameter
sitting near zero would otherwise receive a perturbation near zero and report a pass having probed
nothing.

A scale is the size of change that is PHYSICALLY meaningful for that parameter, not a number chosen
to make a check pass. Changing one is a Class A change and needs its reason written here.
"""

from __future__ import annotations

from typing import Final

import jax.numpy as jnp

# nm/s. Etch rates in the reference cases are a few nm/s; 1 nm/s is a meaningful change.
V0_SCALE_NM_PER_S: Final[float] = 1.0

# Dimensionless directional exponent. p runs from about 2 to 64 (V3); 1 is a meaningful change.
P_SCALE: Final[float] = 1.0


def scales_for(params: dict) -> dict:
    """A scales PyTree with the same structure as `params`. Unknown keys raise, so a new
    parameter cannot enter V14 without someone deciding its scale."""
    known = {"v0_nm_per_s": V0_SCALE_NM_PER_S, "p": P_SCALE}
    missing = sorted(set(params) - set(known))
    if missing:
        raise KeyError(f"no perturbation scale declared for {missing}; add one in params.py "
                       f"with the reason it is physically meaningful")
    return {k: jnp.asarray(known[k], dtype=jnp.float64) for k in params}
