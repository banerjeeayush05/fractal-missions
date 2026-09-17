"""The frozen velocity contract for M3. PRD §5.5, deliverable 6.

M3 imports from HERE and from nowhere else in `geocore`. Everything M3 needs is re-exported, so
the surface M2 must keep stable is one module rather than the whole package.

Status: **v0.3, provisional.** PRD §5.5 (decision §3) says the contract may not freeze until the
M3 owner signs off, and finding S3.1 records that §5.0 and §5.5 disagree about whether it is
v0.2 or v0.3. Until that is answered, `assert_frozen` refuses.

What a velocity model may rely on:

* `positions` are closest-point projections onto the zero level set, in nm.
* `normals` are unit and point from solid into open volume.
* `weights` are in [0, 1] and are exactly 0 for padding.
* `cell_id` is stable across steps and OPAQUE -- seed an RNG with it, never index geometry.
* Entries beyond `n_active` carry benign values, not NaN.

What it may NOT rely on:

* That it is called once per step. TVD-RK calls it once per STAGE.
* That two calls with the same inputs return the same value (M3's will not).
* Any relationship between row order and geometry.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from geocore.rng import request_keys
from geocore.schema import CONTRACT_PROVISIONAL, CONTRACT_VERSION, VelocityModel, VelocityRequest
from geocore.velocity import MODELS, directional, isotropic, model_by_name

__all__ = [
    "CONTRACT_VERSION", "CONTRACT_PROVISIONAL", "VelocityRequest", "VelocityModel",
    "request_keys", "isotropic", "directional", "MODELS", "model_by_name",
    "validate_response", "assert_frozen",
]


class ContractViolation(ValueError):
    pass


def validate_response(request: VelocityRequest, rate: Array) -> Array:
    """Check a velocity model's return value at trace time. Shapes only -- values are tracers.

    Cheap, and it turns the most common integration mistake (returning a grid-shaped field
    instead of a per-request vector) into an error naming both shapes.
    """
    rate = jnp.asarray(rate)
    if rate.shape != (request.capacity,):
        raise ContractViolation(
            f"a velocity model must return one rate per request entry, shape "
            f"({request.capacity},); got {rate.shape}"
        )
    return rate


def assert_frozen() -> None:
    """Raise while the contract is still provisional. Called by anything that would depend on it
    being stable -- so the dependency cannot be created by accident."""
    if CONTRACT_PROVISIONAL:
        raise ContractViolation(
            f"the velocity contract is at v{CONTRACT_VERSION} and is PROVISIONAL (PRD §5.5, "
            f"decision §3): it may not freeze before the M3 owner signs off, and finding S3.1 "
            f"records that §5.0 and §5.5 disagree on the version number"
        )
