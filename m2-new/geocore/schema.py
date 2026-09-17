"""Data contracts. PRD §5.0 — "build these first, before any physics".

Nothing here computes anything. These types are the shapes that the solver, the velocity models
and M3 all agree on, and the reason they come before the physics is that `VelocityRequest` is a
contract with another mission: M3 writes velocity models against it, so it has to be right before
there is code depending on it being wrong.

Two JAX-specific concerns shape the design:

**Static versus traced.** `Grid` holds no arrays — a shape, a spacing, a boundary flag. It is a
frozen dataclass of hashable fields, so it can be a `static_argnums` argument and its contents can
drive Python-level control flow (loop bounds, stencil offsets). `Geometry` and `VelocityRequest`
hold arrays and must survive `jit` and `lax.scan`, so they are registered as PyTrees: the arrays
are children that JAX traces, and the `Grid` rides along as metadata that it does not.

**Validation runs at trace time, not run time.** Shapes are static in JAX, so the checks in
`__post_init__` execute once while the graph is built and cost nothing per step. They are written
defensively because PyTree unflattening reconstructs these objects, sometimes with placeholders
rather than arrays, and a validator that insists on `.shape` would break `jax.eval_shape`.

Deviations from the literal §5.0 text are recorded at the bottom of this docstring.

---

**S3.1 — the contract version is stated twice, differently.** Class A (PRD error).
§5.0's code comment says "Contract v0.3". §5.5's amendment says "The contract is at **v0.2** and
stays `provisional`". The fields §5.5 attributes to v0.2 (`run_seed`, `stage_index`, padded
`positions`, `weights`) are all present in §5.0, which additionally carries `cell_id` and
`n_active` from decisions A16 and B14 — so the likely reading is that v0.3 = v0.2 + those two and
§5.5 was not updated. *Not resolved by guessing:* this module declares `CONTRACT_VERSION = "0.3"`
because §5.0 is the block that actually lists the fields being implemented, and carries
`CONTRACT_PROVISIONAL = True` because §5.5's *substantive* claim — not frozen until the M3 owner
signs off — is unaffected either way. Needs an owner ruling before M2.3 freezes anything.

**S3.2 — the material-fraction axis order is unspecified.** Class D.
§5.0 types `VelocityRequest.material_fractions` as `(K, n_materials)` — materials trailing — but
gives `Geometry.material` no shape at all. *Mechanical rule applied:* match the one that is
specified, so `Geometry.material` is `(*grid.shape, n_materials)`. Trailing also keeps the
spatial axes contiguous, which is what every stencil indexes.

**S3.3 — `Geometry` is frozen here; §5.0 leaves it mutable.** Class A.
§5.0 marks `Grid`, `Material` and `VelocityRequest` `frozen=True` and `Geometry` plain. A mutable
PyTree node is a hazard under `jit`: assigning to a field of a traced object silently captures a
tracer. The solver threads geometry through `lax.scan` as a carry, which is a functional update
anyway, so nothing needs mutation. If a case for mutability appears, this reverts.

**S3.4 — `time` and `step_index` cannot be Python scalars inside the solve.** Class A.
§5.0 types them `float` and `int`. Inside `lax.scan` they arrive as traced scalar arrays, so the
annotations here are `Array`. This is deliberate and load-bearing: a Python `int` step index would
have to come from an unrolled loop, and PRD §5.2 requires a static graph. It also means a velocity
model may not branch on them in Python — which is the intended constraint, not a side effect.
"""

from __future__ import annotations

import dataclasses
import numbers
from typing import Any, Callable, Final

import jax
from jax import Array

from geocore.constants import VERTICAL_AXIS

# PRD §5.5, decision §3. Provisional until the M3 owner signs off; see S3.1 above.
CONTRACT_VERSION: Final[str] = "0.3"
CONTRACT_PROVISIONAL: Final[bool] = True


def _is_shaped(value: Any) -> bool:
    """True for anything array-like enough to validate. PyTree unflattening can pass sentinels
    that are not arrays, and a validator that assumed otherwise would break `jax.eval_shape`."""
    return hasattr(value, "shape")


# --------------------------------------------------------------------------------- Grid


@dataclasses.dataclass(frozen=True)
class Grid:
    """Static grid metadata. Hashable, so it can be a `static_argnums` argument.

    `shape` is `(nz, nx)` in 2D and `(nz, ny, nx)` in 3D — the vertical axis is 0 in both, which
    is what lets one code path serve both dimensions (PRD §5.1).
    """

    shape: tuple[int, ...]
    spacing_nm: float
    periodic: tuple[bool, ...]

    def __post_init__(self) -> None:
        # Rejected, not coerced. `frozen=True` means normalising a list to a tuple would need
        # object.__setattr__, and handing back an object that is not the one you passed in is
        # its own small surprise. Caught here because a list passes every check below -- it has
        # a length and it iterates -- and then fails far away and confusingly, when something
        # tries to use the Grid as a hashable static argument to a jitted function.
        for name, value in (("shape", self.shape), ("periodic", self.periodic)):
            if not isinstance(value, tuple):
                raise TypeError(
                    f"{name} must be a tuple, got {type(value).__name__}: Grid is used as a "
                    f"static (hashable) argument and a list is not hashable"
                )
        # A float shape is hashable, so the tuple check above would pass it. It would then
        # survive `n < 2` as well, and only fail once an array constructor saw 16.0.
        if not all(
            isinstance(n, numbers.Integral) and not isinstance(n, bool) for n in self.shape
        ):
            raise TypeError(f"shape must contain integers, got {self.shape}")
        if not all(isinstance(flag, bool) for flag in self.periodic):
            raise TypeError(f"periodic must contain bools, got {self.periodic}")
        if len(self.shape) not in (2, 3):
            raise ValueError(f"grid must be 2D or 3D, got shape {self.shape}")
        if any(n < 2 for n in self.shape):
            raise ValueError(f"every axis needs at least 2 cells, got {self.shape}")
        if len(self.periodic) != len(self.shape):
            raise ValueError(
                f"periodic has {len(self.periodic)} entries for a {len(self.shape)}D grid"
            )
        if self.spacing_nm <= 0.0:
            raise ValueError(f"spacing_nm must be positive, got {self.spacing_nm}")
        # PRD §5.1: lateral boundaries periodic, top and bottom Neumann. A periodic vertical axis
        # would wrap the plasma side onto the substrate side — the etch front would re-enter from
        # below, which produces a plausible-looking field and a meaningless result.
        if self.periodic[VERTICAL_AXIS]:
            raise ValueError("the vertical axis must be Neumann, not periodic (PRD §5.1)")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def n_cells(self) -> int:
        n = 1
        for axis in self.shape:
            n *= axis
        return n


# ----------------------------------------------------------------------------- Material


@dataclasses.dataclass(frozen=True)
class Material:
    """One layer in the stack. `index` is its column in the fraction field.

    `is_mask` is carried separately from the name because the mask is not just another material:
    from M2.6 it is a solid body in phi with an etch-rate multiplier of exactly zero (PRD §5.6).
    """

    name: str
    index: int
    is_mask: bool = False

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"material index must be non-negative, got {self.index}")
        if not self.name:
            raise ValueError("material name must be non-empty")


# ----------------------------------------------------------------------------- Geometry


@dataclasses.dataclass(frozen=True)
class Geometry:
    """The evolving state: the level set plus what each cell is made of.

    `phi` has the grid's shape. `material` is `(*grid.shape, n_materials)` — mollified FRACTIONS
    summing to 1 along the last axis, never integer IDs. PRD §5.6: an integer ID makes the etch
    rate jump as a step function when the interface crosses a layer boundary, and the derivative
    of a step is zero everywhere and undefined at one point, which destroys the gradient with
    respect to anything controlling *when* the interface arrives there.
    """

    phi: Array
    material: Array
    grid: Grid

    def __post_init__(self) -> None:
        if not (_is_shaped(self.phi) and _is_shaped(self.material)):
            return
        if tuple(self.phi.shape) != tuple(self.grid.shape):
            raise ValueError(f"phi has shape {self.phi.shape}, grid says {self.grid.shape}")
        if tuple(self.material.shape[:-1]) != tuple(self.grid.shape):
            raise ValueError(
                f"material has spatial shape {self.material.shape[:-1]}, "
                f"grid says {self.grid.shape} (expected (*shape, n_materials))"
            )

    @property
    def n_materials(self) -> int:
        return int(self.material.shape[-1])


jax.tree_util.register_dataclass(Geometry, data_fields=["phi", "material"], meta_fields=["grid"])


# ---------------------------------------------------------------------- VelocityRequest


@dataclasses.dataclass(frozen=True)
class VelocityRequest:
    """What M2 hands a velocity model. The contract with M3.

    A FIXED-CAPACITY PADDED SET of K entries, not a list of the cells that happen to be near the
    interface. K is static because JAX needs static shapes; the interface lengthens as a trench
    deepens, so K is sized from the WORST step, not the first, and overflow aborts like the CFL
    assertion rather than silently truncating (PRD §5.0).

    `weights` and `cell_id` are two halves of ONE mechanism, not two safeguards (decision A16):

    * `cell_id` is a stable flattened grid index. It seeds the RNG, so a cell entering the band
      does not shift every other cell's random draws. It is OPAQUE to the velocity model — seed
      with it, never index geometry with it, or M3 acquires a dependency on M2's grid layout.
    * `weights` is smooth band membership reaching exactly zero BEFORE the band edge, so the
      arriving cell's own draw does not enter discontinuously.

    Remove either and the objective is discontinuous in the parameters.

    A zero weight does NOT protect against NaN: `0 * NaN` is NaN and poisons the whole gradient.
    Padded entries therefore carry a BENIGN POSITION, not merely a zero weight (decision B14).
    """

    positions: Array            # (K, d)  closest-point projections onto the zero level set
    normals: Array              # (K, d)  unit, pointing from solid into open volume
    material_fractions: Array   # (K, n_materials)
    weights: Array              # (K,)    smooth band membership; exactly 0 for padding
    cell_id: Array              # (K,)    int, stable flattened grid index; seeds the RNG
    n_active: Array             # scalar int: how many entries are real
    time: Array                 # scalar, nm/s-consistent time in seconds
    step_index: Array           # scalar int — REQUIRED, see PRD §7.6 on stochastic velocity
    stage_index: Array          # scalar int: RK stage; draws are independent per stage
    run_seed: Array             # scalar int: run-level RNG seed

    def __post_init__(self) -> None:
        fields = (self.positions, self.normals, self.material_fractions,
                  self.weights, self.cell_id)
        if not all(_is_shaped(f) for f in fields):
            return
        capacity = self.positions.shape[0]
        for name, value, rank in (
            ("positions", self.positions, 2),
            ("normals", self.normals, 2),
            ("material_fractions", self.material_fractions, 2),
            ("weights", self.weights, 1),
            ("cell_id", self.cell_id, 1),
        ):
            if len(value.shape) != rank:
                raise ValueError(f"{name} must be rank {rank}, got shape {value.shape}")
            if value.shape[0] != capacity:
                raise ValueError(
                    f"{name} has leading dimension {value.shape[0]}, expected K={capacity}"
                )
        if self.positions.shape[1] != self.normals.shape[1]:
            raise ValueError(
                f"positions are {self.positions.shape[1]}D but normals are "
                f"{self.normals.shape[1]}D"
            )

    @property
    def capacity(self) -> int:
        """K — the padded size. Static; `n_active` is how many entries are real."""
        return int(self.positions.shape[0])

    @property
    def ndim(self) -> int:
        return int(self.positions.shape[1])

    @property
    def n_materials(self) -> int:
        return int(self.material_fractions.shape[1])


jax.tree_util.register_dataclass(
    VelocityRequest,
    data_fields=["positions", "normals", "material_fractions", "weights", "cell_id",
                 "n_active", "time", "step_index", "stage_index", "run_seed"],
    meta_fields=[],
)


# -------------------------------------------------------------------------- the callable

# PRD §5.0. Returns an ETCH RATE in nm/s per request entry, shape (K,): positive removes
# material. The second argument is the differentiable parameter PyTree — every parameter M2
# differentiates with respect to must live in it, never captured in a closure, or `jax.grad`
# cannot see it and silently returns a zero column (PRD §7.3).
VelocityModel = Callable[["VelocityRequest", Any], Array]
