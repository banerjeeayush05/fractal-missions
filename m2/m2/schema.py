"""Data contracts (PRD §5.0). Built before any physics.

Field names and types follow §5.0 verbatim. Validation is structural only (shapes, dtypes,
§5.1 boundary rules), so it also works on JAX tracers. Value checks on concrete arrays live in
``check_geometry_values``.

Velocity contract v0.3 (decisions §3 and A16/B14): `run_seed`, `stage_index` and a stable, opaque
`cell_id` are carried explicitly; `positions` are closest-point projections onto the zero level set
in a fixed-capacity padded set of size K; `weights` carry smooth band membership and are exactly
zero for padding, so a cell entering or leaving the band contributes nothing at the moment it does;
`n_active` reports how many entries are real. Still provisional until the M3 owner signs off
(OQ B9; CROSS_MISSION X1, and the six-item M3 list there).

Padding is a NaN hazard: `0 * NaN` is NaN and poisons the whole gradient, so padded entries carry a
benign position and normals use the double-where guard regardless of weight (decision B14).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import m2  # noqa: F401  (enables fp64)
from m2.constants import DTYPE, MATERIAL_SUM_ATOL

Array = jax.Array
PyTree = Any
"""The differentiable parameter set. Every parameter M2 differentiates with respect to must
live here and never be captured in a closure (§5.0, §7.3)."""


class SchemaError(ValueError):
    """A data contract was violated."""


@dataclass(frozen=True)
class Grid:
    shape: tuple[int, ...]  # (nz, ny, nx) in 3D; (nz, nx) in 2D
    spacing_nm: float  # isotropic
    periodic: tuple[bool, ...]  # lateral periodic, vertical not

    def __post_init__(self) -> None:
        shape, periodic = tuple(self.shape), tuple(self.periodic)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "periodic", periodic)
        if len(shape) not in (2, 3):
            raise SchemaError(f"Grid.shape must be 2D or 3D, got {shape}")
        if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in shape):
            raise SchemaError(f"Grid.shape entries must be positive ints, got {shape}")
        s = self.spacing_nm
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0:
            raise SchemaError(f"Grid.spacing_nm must be a finite positive number, got {s!r}")
        if len(periodic) != len(shape) or not all(isinstance(p, bool) for p in periodic):
            raise SchemaError(f"Grid.periodic must be one bool per axis, got {periodic}")
        # §5.1: axis 0 is vertical (non-periodic, Neumann); all lateral axes periodic.
        if periodic[0]:
            raise SchemaError("Grid.periodic[0] (vertical axis) must be False (PRD §5.1)")
        if not all(periodic[1:]):
            raise SchemaError("lateral axes must be periodic (PRD §5.1; OPEN_QUESTIONS Q2)")

    @property
    def ndim(self) -> int:
        return len(self.shape)


@dataclass(frozen=True)
class Material:
    name: str
    index: int
    is_mask: bool
    is_void: bool = False  # decision §10: the open volume is an explicit material (OQ B17)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise SchemaError(f"Material.name must be a non-empty str, got {self.name!r}")
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise SchemaError(f"Material.index must be an int >= 0, got {self.index!r}")
        for flag in ("is_mask", "is_void"):
            if not isinstance(getattr(self, flag), bool):
                raise SchemaError(f"Material.{flag} must be a bool, got {getattr(self, flag)!r}")
        if self.is_void and self.is_mask:
            raise SchemaError("a material cannot be both void and mask")


def validate_materials(materials: Sequence[Material]) -> tuple[Material, ...]:
    """Indices must be exactly 0..n-1 (they index the fraction axis, B8); names unique.

    Decision §10: exactly one material is the void, so fractions sum to 1 everywhere including the
    open volume, and a single-solid-material stack is an instance of the general code rather than
    a special case.
    """
    mats = tuple(materials)
    if not mats:
        raise SchemaError("at least one Material is required")
    if sorted(m.index for m in mats) != list(range(len(mats))):
        raise SchemaError(f"Material indices must be 0..{len(mats) - 1}, got {[m.index for m in mats]}")
    if len({m.name for m in mats}) != len(mats):
        raise SchemaError(f"Material names must be unique, got {[m.name for m in mats]}")
    n_void = sum(m.is_void for m in mats)
    if n_void != 1:
        raise SchemaError(f"exactly one material must be the void (decision §10), got {n_void}")
    return tuple(sorted(mats, key=lambda m: m.index))


@dataclass
class Geometry:
    phi: Array  # signed distance, negative inside solid
    material: Array  # mollified material fractions, see §5.6. Shape (n_materials, *grid.shape)
    grid: Grid

    def __post_init__(self) -> None:
        _require_fp64("Geometry.phi", self.phi)
        _require_fp64("Geometry.material", self.material)
        if tuple(self.phi.shape) != self.grid.shape:
            raise SchemaError(f"Geometry.phi shape {self.phi.shape} != grid.shape {self.grid.shape}")
        if self.material.ndim != self.grid.ndim + 1 or tuple(self.material.shape[1:]) != self.grid.shape:
            raise SchemaError(
                f"Geometry.material shape {self.material.shape} != (n_materials, *{self.grid.shape})"
            )


@dataclass(frozen=True)
class VelocityRequest:  # what M2 hands to a velocity model. Contract v0.3, decisions §3 and A16.
    positions: Array  # closest-point projections onto the zero level set, (K, d), padded
    normals: Array  # (K, d)
    material_fractions: Array  # (K, n_materials)
    weights: Array  # (K,) smooth band membership; exactly 0 for padding and beyond the band
    cell_id: Array  # (K,) int: FLATTENED GRID INDEX the point came from. Seeds the RNG; see below.
    n_active: Array  # scalar int: how many entries are real, so padding is visible and K is sizeable
    time: float
    step_index: int  # REQUIRED — see §7.6 on stochastic velocity
    stage_index: int  # RK stage within the step; draws are independent per stage by default
    run_seed: int  # run-level seed. key = f(run_seed, step_index, stage_index, cell_id)

    # The stable id and the smooth weight are TWO HALVES OF ONE MECHANISM, not two safeguards
    # (decision A16). `cell_id` stops a cell's arrival in the band from shifting everyone else's
    # random draws; the weight reaching zero at the band edge stops the arriving cell's own draw
    # from entering discontinuously. Remove either and J is discontinuous in θ, and the Taylor
    # test cannot be run at all. Do not "simplify" one of them away.
    #
    # `cell_id` is OPAQUE to the velocity model: it seeds an RNG and is never used to index
    # geometry. Otherwise M3 acquires a dependency on M2's grid layout and the two missions stop
    # being independently replaceable.
    #
    # Common random numbers DO NOT SURVIVE A CHANGE OF GRID SPACING: ids mean different things at
    # different dx. Harmless in M2, where velocity is deterministic; a real constraint on M3's
    # convergence and cross-resolution studies.

    def __post_init__(self) -> None:
        for name in ("positions", "normals", "material_fractions", "weights"):
            _require_fp64(f"VelocityRequest.{name}", getattr(self, name))
        if not jnp.issubdtype(jnp.asarray(self.cell_id).dtype, jnp.integer):
            raise SchemaError(f"VelocityRequest.cell_id must be an integer array, got {self.cell_id}")
        p, n, m, w = self.positions, self.normals, self.material_fractions, self.weights
        if tuple(jnp.shape(self.cell_id)) != tuple(p.shape[:-1]):
            raise SchemaError(f"VelocityRequest.cell_id shape {jnp.shape(self.cell_id)} != batch "
                              f"shape {p.shape[:-1]}")
        if jnp.ndim(self.n_active) != 0:
            raise SchemaError(f"VelocityRequest.n_active must be a scalar, got {jnp.shape(self.n_active)}")
        if p.ndim < 1 or p.shape[-1] not in (2, 3):
            raise SchemaError(f"VelocityRequest.positions must be (..., d) with d in (2, 3), got {p.shape}")
        if n.shape != p.shape:
            raise SchemaError(f"VelocityRequest.normals shape {n.shape} != positions shape {p.shape}")
        if m.ndim != p.ndim or m.shape[:-1] != p.shape[:-1]:
            raise SchemaError(
                f"VelocityRequest.material_fractions batch shape {m.shape[:-1]} != positions {p.shape[:-1]}"
            )
        if tuple(w.shape) != tuple(p.shape[:-1]):
            raise SchemaError(f"VelocityRequest.weights shape {w.shape} != batch shape {p.shape[:-1]}")
        for name in ("step_index", "stage_index", "run_seed"):
            if getattr(self, name) is None:
                raise SchemaError(f"VelocityRequest.{name} is required (contract v0.2, §7.6)")

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.positions.shape[:-1])


# Velocity model signature. M2 ships analytic implementations only.
VelocityModel = Callable[[VelocityRequest, PyTree], Array]  # -> speed in nm/s


def check_velocity_output(request: VelocityRequest, speed: Array) -> None:
    """A velocity model must return one fp64 speed per requested point."""
    _require_fp64("velocity model output", speed)
    if tuple(speed.shape) != request.batch_shape:
        raise SchemaError(f"velocity model returned shape {speed.shape}, expected {request.batch_shape}")


def check_geometry_values(geom: Geometry) -> None:
    """Concrete-value checks (not traceable): finite φ; fractions in [0, 1] summing to 1."""
    phi, frac = np.asarray(geom.phi), np.asarray(geom.material)
    if not np.all(np.isfinite(phi)):
        raise SchemaError("Geometry.phi contains non-finite values")
    if np.any(frac < 0) or np.any(frac > 1):
        raise SchemaError("Geometry.material fractions must lie in [0, 1]")
    err = float(np.max(np.abs(frac.sum(axis=0) - 1.0)))
    if err > MATERIAL_SUM_ATOL:
        raise SchemaError(f"Geometry.material fractions must sum to 1 (max error {err:.3e})")


def _require_fp64(name: str, x: Any) -> None:
    dtype = getattr(x, "dtype", None)
    if dtype is None or jnp.dtype(dtype) != jnp.dtype(DTYPE):
        raise SchemaError(f"{name} must be {DTYPE} (PRD §7.5), got {dtype}")


# --- pytree registration -----------------------------------------------------------------
# Unflattening bypasses __init__ so JAX can rebuild these with placeholder leaves (tracers,
# None, sentinels) during transformations without tripping validation.


def _unflatten(cls, names, aux, children):
    obj = object.__new__(cls)
    for name, value in zip(names, children):
        object.__setattr__(obj, name, value)
    for name, value in aux:
        object.__setattr__(obj, name, value)
    return obj


def _register(cls, data_fields: tuple[str, ...]) -> None:
    meta_fields = tuple(f.name for f in fields(cls) if f.name not in data_fields)

    def flatten(obj):
        return tuple(getattr(obj, n) for n in data_fields), tuple((n, getattr(obj, n)) for n in meta_fields)

    jax.tree_util.register_pytree_node(
        cls, flatten, lambda aux, ch: _unflatten(cls, data_fields, aux, ch)
    )


_register(Geometry, ("phi", "material"))
_register(VelocityRequest, ("positions", "normals", "material_fractions", "weights", "cell_id",
                           "n_active", "time", "step_index", "stage_index", "run_seed"))
