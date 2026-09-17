"""The two analytic velocity models M2 ships for its own testing. PRD §5.5.

Both satisfy the `VelocityModel` contract: given a `VelocityRequest`, return an etch rate in nm/s
per entry, shape `(K,)`, positive removing material.

These are NOT physics. M2's velocity is prescribed and analytic by design (PRD §3): flux,
shadowing, re-emission and redeposition are M3, and building a placeholder version here would
create a contract M3 then has to fight.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from geocore.constants import VERTICAL_AXIS, Z_HAT_SIGN
from geocore.schema import VelocityRequest


def isotropic(request: VelocityRequest, params: Any) -> Array:
    """`R = v0`, the same everywhere. V1's law."""
    return jnp.full((request.capacity,), params["v0_nm_per_s"])


def directional(request: VelocityRequest, params: Any) -> Array:
    """`R = v0 * max(0, n . z_hat)^p`. A cosine-law ion flux with no shadowing.

    Up-facing surfaces etch at the full rate; a downward-facing overhang underside does not move
    at all. With axis 0 increasing toward the plasma, `n . z_hat` is just the vertical component
    of the normal.

    **Two traps, both in one expression.**

    The law has a KINK at `n.z = 0`, which is exactly a vertical sidewall -- the most common
    surface in a trench. `p > 1` moves that kink out of the differentiated path, and `config.py`
    refuses to load a case with `p <= 1`.

    `0**p` is finite but its DERIVATIVE IN p is `0**p * log(0)`, which is NaN. A plain
    `jnp.where(c > 0, c**p, 0.0)` does not help: the untaken branch is still evaluated and its
    NaN still propagates through the gradient. Hence the double-where -- the exponent only ever
    sees a strictly positive base.
    """
    p = params["p"]
    cosine = request.normals[:, VERTICAL_AXIS] * Z_HAT_SIGN
    facing = jnp.maximum(cosine, 0.0)
    safe = jnp.where(facing > 0.0, facing, 1.0)          # never 0 inside the power
    flux = jnp.where(facing > 0.0, safe**p, 0.0)
    return params["v0_nm_per_s"] * flux


MODELS = {"isotropic": isotropic, "directional": directional}


def model_by_name(name: str):
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f"unknown velocity model {name!r}; M2 ships {sorted(MODELS)}") from None
