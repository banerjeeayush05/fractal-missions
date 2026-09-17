"""The velocity band: which cells the model is called on, and how its answer reaches the stencils.

PRD §5.4, decision C10. This is the bridge between M2's grid and M3's contract, and it is the
single largest step away from the capstone -- which evaluated velocity on all 65,536 cells.

**Two bands, with different jobs.**

* The EVALUATION band (default 1.5 cells) is where the velocity model is actually CALLED. It stays
  thin because cells deeper in the band project to nearly the same surface point, so calling M3
  for them buys the same expensive Monte Carlo estimate several times over.
* The EXTENSION band (default 8 cells, tapering over the outer 2) is where a valid rate must
  EXIST, because the stencils reach there: the interface moves up to `cfl * reinit_every` cells
  between reinitialisations, reinitialisation propagates about `n_reinit` cells, and the upwind
  stencil reaches one more.

Between them sits the closest-point gather: M3 evaluates on the thin set, M2 spreads the answer
outward across the wide one at negligible cost.

**The padded set.** JAX needs static shapes, but the number of cells near the interface changes
every step -- a trench lengthens its own perimeter as it deepens. So the request is always exactly
K entries: the K cells with the smallest |phi|, of which `n_active` are genuinely in the band and
the rest carry weight zero. K is sized from the WORST step, not the first, and overflow aborts
like the CFL assertion rather than silently truncating the interface.

**`cell_id` and `weights` are two halves of one mechanism, not two safeguards** (decision A16).
The id stops a cell's arrival from shifting everyone else's random draws; the weight reaching
exactly zero before the band edge stops the arriving cell's own contribution from entering
discontinuously. Remove either and the objective is discontinuous in the parameters.

**The weights are part of the forward map.** PRD §11 forbids `stop_gradient` on them. They look
like a mask, which is exactly what a future reader will wrap to make a test pass.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.ndimage import map_coordinates

from geocore.config import BandConfig
from geocore.constants import NORMAL_EPS
from geocore.interface import validate_response
from geocore.schema import Grid, VelocityModel, VelocityRequest
from geocore.stencils import central_gradient


class BandOverflow(RuntimeError):
    """More cells were in the evaluation band than the request had room for."""


# --------------------------------------------------------------------------------- weights


def smoothstep(t: Array) -> Array:
    """Quintic smoothstep: 0 at t<=0, 1 at t>=1, with ZERO first AND second derivative at both
    ends.

    The cubic `3t^2 - 2t^3` is the usual choice and is only C1. V14 measures a Taylor remainder
    and needs the second derivative to exist, so a C1 taper would put a discontinuity in exactly
    the quantity the primary gate is fitting. The quintic costs two extra multiplies.
    """
    s = jnp.clip(t, 0.0, 1.0)
    return s * s * s * (s * (6.0 * s - 15.0) + 10.0)


def _membership(phi: Array, grid: Grid, radius_cells: float, taper_cells: float) -> Array:
    """1 in the core, tapering smoothly to EXACTLY 0 at `radius_cells` from the interface."""
    distance_cells = jnp.abs(phi) / grid.spacing_nm
    inner = radius_cells - taper_cells
    taper = jnp.maximum(taper_cells, 1e-12)
    return jnp.where(
        distance_cells <= inner,
        1.0,
        smoothstep((radius_cells - distance_cells) / taper),
    )


def evaluation_weight(phi: Array, grid: Grid, bands: BandConfig) -> Array:
    """Smooth membership of the thin band where the model is called.

    The taper is the whole band: at 1.5 cells wide there is no room for a flat core, and the point
    is that a cell entering contributes nothing at the moment it enters.
    """
    return _membership(phi, grid, bands.evaluation_cells, bands.evaluation_cells)


def extension_weight(phi: Array, grid: Grid, bands: BandConfig) -> Array:
    """Smooth membership of the wide band where a rate must exist for the stencils."""
    return _membership(phi, grid, bands.extension_cells, bands.taper_cells)


# ------------------------------------------------------------------- normals and projection


def unit_normal(phi: Array, grid: Grid) -> Array:
    """`n = grad(phi)/|grad(phi)|`, pointing from solid into open volume. Shape `(ndim, *shape)`.

    `|grad phi|` vanishes on the medial axis -- a trench centreline, a circle's centre -- and the
    quotient is then 0/0. Two guards, and BOTH are needed:

    * `NORMAL_EPS` inside the sqrt keeps the magnitude off zero, because `d/dx sqrt(x)` is
      infinite at x = 0 and one such cell is enough to NaN every parameter.
    * the double-where keeps the DIVISION away from a zero denominator. `jnp.where` alone does
      not save you: both branches are evaluated, and a NaN in the branch not taken propagates
      through the gradient exactly as if it had been.
    """
    gradient = central_gradient(phi, grid)
    magnitude = jnp.sqrt(jnp.sum(gradient**2, axis=0) + NORMAL_EPS)
    safe = jnp.where(magnitude > jnp.sqrt(NORMAL_EPS), magnitude, 1.0)
    normal = gradient / safe
    return jnp.where(magnitude > jnp.sqrt(NORMAL_EPS), normal, 0.0)


def closest_points(phi: Array, grid: Grid) -> Array:
    """Project every cell onto the zero level set: `p = x - phi * n`. Shape `(ndim, *shape)`, nm.

    Exact when phi is a true signed distance function, which is why `initial.py` goes to the
    trouble of producing one and why reinitialisation exists to keep it that way.
    """
    from geocore.initial import coordinate_fields

    coords = jnp.stack(coordinate_fields(grid), axis=0)
    return coords - phi[None, ...] * unit_normal(phi, grid)


# ------------------------------------------------------------------------- request assembly


def occupancy(phi: Array, grid: Grid, bands: BandConfig) -> Array:
    """How many cells are genuinely in the evaluation band. A traced scalar; check it on the host."""
    return jnp.sum(evaluation_weight(phi, grid, bands) > 0.0)


def _fractions_at(material: Array, positions_nm: Array, grid: Grid) -> Array:
    """Material fractions at the request POSITIONS -- the closest surface points -- by multilinear
    interpolation. Shape `(K, n_materials)`.

    Not the fractions at the cell. A band cell sits up to a band-width from the surface, and near a
    layer boundary the material there is not the material the surface is actually in. Evaluating at
    the cell would make the rate change when the BAND reaches a layer, a band-width too early.
    """
    coords = [positions_nm[a] / grid.spacing_nm - 0.5 for a in range(grid.ndim)]
    return jnp.stack([map_coordinates(material[..., m], coords, order=1, mode="nearest")
                      for m in range(material.shape[-1])], axis=-1)


def build_request(phi: Array, material: Array, grid: Grid, bands: BandConfig, capacity: int,
                  time: Array, step_index: Array, stage_index: int,
                  run_seed: int) -> tuple[VelocityRequest, Array]:
    """Assemble the fixed-capacity padded request. Returns it with the live band occupancy.

    Selection is `lax.top_k` on `-|phi|`, which is a static-shape way of saying "the K cells
    closest to the interface". Sorting the whole grid would also work and costs O(n log n) on
    2.7 million cells every stage.

    Padding entries are REAL cells that happen to be far from the interface, so their positions
    are ordinary finite coordinates. That matters: a zero weight does not protect against NaN,
    because `0 * NaN` is NaN (decision B14).
    """
    if capacity > grid.n_cells:
        raise ValueError(
            f"request capacity K={capacity} exceeds the {grid.n_cells} cells in the grid. K is "
            f"the number of cells the velocity model is called on, so it can never usefully "
            f"exceed the grid; size it from the peak band occupancy of a trial run instead."
        )
    if capacity < 1:
        raise ValueError(f"request capacity must be at least 1, got {capacity}")

    weight_field = evaluation_weight(phi, grid, bands)
    flat_weight = weight_field.reshape(-1)
    flat_phi = phi.reshape(-1)

    # Rank by |phi|: the K nearest the interface, in a static shape.
    _, cell_id = jax.lax.top_k(-jnp.abs(flat_phi), capacity)

    points = closest_points(phi, grid).reshape(grid.ndim, -1)
    normals = unit_normal(phi, grid).reshape(grid.ndim, -1)

    return VelocityRequest(
        positions=points[:, cell_id].T,
        normals=normals[:, cell_id].T,
        material_fractions=_fractions_at(material, points[:, cell_id], grid),
        weights=flat_weight[cell_id],
        cell_id=cell_id,
        n_active=jnp.sum(flat_weight[cell_id] > 0.0),
        time=jnp.asarray(time, dtype=jnp.float64),
        step_index=jnp.asarray(step_index, dtype=jnp.int32),
        stage_index=jnp.asarray(stage_index, dtype=jnp.int32),
        run_seed=jnp.asarray(run_seed, dtype=jnp.int32),
    ), jnp.sum(flat_weight > 0.0)


# ---------------------------------------------------------------------------------- gather


def gather_to_band(rate: Array, request: VelocityRequest, phi: Array, grid: Grid,
                   bands: BandConfig) -> Array:
    """Spread the evaluated rates from the thin set across the extension band.

    Each request entry's rate is scattered back to its own cell, weighted by its band membership.
    Every extension-band cell then reads the rate at ITS closest point on the interface by
    multilinear interpolation, normalised by the interpolated weight -- so the result is a
    weighted average of nearby evaluated rates rather than a sum diluted by empty cells.

    Second order, which caps the whole scheme's order at two even under WENO5 (finding J2 in the
    existing tree). Recorded rather than hidden: a fifth-order Hamiltonian fed a second-order
    rate field cannot beat second order.
    """
    weighted = jnp.zeros(grid.shape, dtype=jnp.float64).reshape(-1)
    weighted = weighted.at[request.cell_id].add(rate * request.weights)
    mass = jnp.zeros(grid.shape, dtype=jnp.float64).reshape(-1)
    mass = mass.at[request.cell_id].add(request.weights)

    # Index-space coordinates of every cell's closest point: coord = (i + 0.5) * dx.
    index_coords = closest_points(phi, grid) / grid.spacing_nm - 0.5
    sample = [index_coords[a] for a in range(grid.ndim)]

    numerator = map_coordinates(weighted.reshape(grid.shape), sample, order=1, mode="nearest")
    denominator = map_coordinates(mass.reshape(grid.shape), sample, order=1, mode="nearest")

    safe = jnp.where(denominator > 1e-12, denominator, 1.0)
    gathered = jnp.where(denominator > 1e-12, numerator / safe, 0.0)
    return gathered * extension_weight(phi, grid, bands)


# ------------------------------------------------------------------------- the rate field


@dataclasses.dataclass(frozen=True)
class BandedRateField:
    """Adapts a `VelocityModel` to the solver's `RateField`: request, evaluate, gather.

    Returns `(rate_on_grid, occupancy)`. The occupancy rides out of the scan as a per-step
    auxiliary and is checked on the host by `assert_capacity` -- the same pattern as the CFL
    assertion, and for the same reason.
    """

    model: VelocityModel
    grid: Grid
    bands: BandConfig
    capacity: int
    material: Array
    run_seed: int = 0

    def __call__(self, phi: Array, params: Any, ctx) -> tuple[Array, Array]:
        request, live = build_request(
            phi, self.material, self.grid, self.bands, self.capacity,
            ctx.time, ctx.step_index, ctx.stage_index, self.run_seed,
        )
        rate = validate_response(request, self.model(request, params))
        return gather_to_band(rate, request, phi, self.grid, self.bands), live


def assert_capacity(occupancy_per_step: Array, capacity: int) -> int:
    """Host-side overflow check. PRD §5.0: overflow aborts like the CFL assertion."""
    peak = int(jnp.max(occupancy_per_step))
    if peak > capacity:
        step = int(jnp.argmax(occupancy_per_step))
        raise BandOverflow(
            f"the evaluation band held {peak} cells at step {step}, over a capacity of "
            f"{capacity}. K must be sized from the WORST step, not the first: the interface "
            f"lengthens as the trench deepens, so an overflow that does not fire at step 0 can "
            f"still fire at step 900 of 1000."
        )
    return peak


def single_material(grid: Grid) -> Array:
    """A one-material fraction field. Multi-material arrives at stage 17."""
    return jnp.ones((*grid.shape, 1), dtype=jnp.float64)
