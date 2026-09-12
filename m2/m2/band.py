"""Which points the velocity model is asked about, and how its answer reaches the whole band.

Two bands (decision C10):
- the **evaluation** band (default 1.5 cells) is the only set the velocity model is called on. It
  is thin on purpose: cells deeper in the band project to nearly the same surface point, so calling
  M3 for them buys the same expensive Monte Carlo estimate several times.
- the **extension** band (default 8 cells, tapering to zero over the outer 2) is where a valid
  velocity must exist, because that is where the advection and reinitialisation stencils reach.

§5.4's closest-point gather bridges them, and by the 2026-09-11 scope decision it is built here at
M2.1 rather than M2.2, so V1 and V2 — which have exact answers — are what first exercise it.

Two halves of one mechanism (decision A16), and neither may be removed:
- `cell_id`, the flattened grid index, keeps a point's RNG draw stable when other cells join or
  leave the band;
- the smooth weight, which reaches exactly zero before the band edge, keeps an arriving cell's own
  contribution from entering discontinuously.

Padding: `jnp.nonzero(..., size=K)` fills unused slots with index 0 — a real cell — so padded
entries are given weight exactly zero and a finite position. `0 * NaN` is NaN, so "finite" is not
optional (decision B14).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

import m2  # noqa: F401  (enables fp64)
from m2.config import BandConfig
from m2.schema import Grid, VelocityRequest

WEIGHT_FLOOR = 1e-12  # below this a gathered weight is treated as absent


def _smoothstep(t: jax.Array) -> jax.Array:
    """C¹ ramp on [0, 1], flat at both ends."""
    t = jnp.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def evaluation_weight(phi: jax.Array, grid: Grid, bands: BandConfig) -> jax.Array:
    """Smooth membership of the evaluation band: 1 at the interface, exactly 0 at the band edge.

    Written in φ² rather than |φ| so the weight is smooth *through* the interface; |φ| has a kink
    at φ = 0, which would put a kink in the differentiated path (§11).
    """
    b = bands.evaluation_cells * grid.spacing_nm
    return _smoothstep(1.0 - (phi / b) ** 2)


def extension_weight(phi: jax.Array, grid: Grid, bands: BandConfig) -> jax.Array:
    """1 in the interior of the extension band, tapering to exactly 0 over its outer cells."""
    dx = grid.spacing_nm
    outer = bands.extension_cells * dx
    inner = max(outer - bands.extension_taper_cells * dx, 0.0)
    return _smoothstep((outer - jnp.abs(phi)) / max(outer - inner, dx * 1e-6))


def closest_point(phi: jax.Array, grid: Grid, unit_normal: jax.Array) -> jax.Array:
    """x − φ·n: the projection onto the zero level set, exact where φ is a signed distance.

    These are the `positions` the velocity model receives (decision §3): M3 needs flux at the
    surface, and a cell centre is up to a cell away and, on the solid side, inside the material.
    """
    from m2.stencils import cell_centres

    return cell_centres(grid) - phi[None, ...] * unit_normal


def estimate_capacity(grid: Grid, bands: BandConfig, *, margin: float = 1.5) -> int:
    """K for the padded set, sized for the WORST step rather than the first (decision C10).

    The interface lengthens as the trench deepens, so a K sized from the initial geometry fires its
    overflow assertion late in a gate run. The bound here is the whole vertical extent of the domain
    interpreted as interface length — far above any single feature — times the band thickness, times
    a margin. Generous on purpose: K costs memory, an overflow costs a run. OPEN_QUESTIONS B26.
    """
    shell_cells = 2.0 * bands.evaluation_cells + 1.0  # thickness of the |φ| < b shell, in cells
    lateral = math.prod(grid.shape[1:])
    # Upper bound on interface length for one feature: the domain perimeter in cells. A trench is
    # floor + two sidewalls ≈ CD/dx + 2·depth/dx, which this comfortably exceeds.
    perimeter_cells = 2.0 * (grid.shape[0] + lateral)
    return int(math.ceil(margin * shell_cells * perimeter_cells))


@dataclass(frozen=True)
class BandSelection:
    request: VelocityRequest
    n_active: jax.Array  # scalar: how many entries are real
    capacity: int


def assemble_request(
    phi: jax.Array,
    material: jax.Array,
    grid: Grid,
    bands: BandConfig,
    *,
    capacity: int,
    time: float,
    step_index,
    stage_index,
    run_seed,
) -> BandSelection:
    """Build the padded `VelocityRequest` for one RK stage (contract v0.3)."""
    from m2.stencils import normals

    unit_normal, _ = normals(phi, grid)
    weight = evaluation_weight(phi, grid, bands)
    points = closest_point(phi, grid, unit_normal)

    flat_weight = weight.reshape(-1)
    (cell_id,) = jnp.nonzero(flat_weight > 0.0, size=capacity, fill_value=0)
    n_active = jnp.sum(flat_weight > 0.0)

    # Padded slots land on cell 0, a real cell, so its weight would otherwise be counted twice.
    live = jnp.arange(capacity) < n_active
    gathered_weight = jnp.where(live, flat_weight[cell_id], 0.0)

    d = grid.ndim
    request = VelocityRequest(
        positions=points.reshape(d, -1).T[cell_id],
        normals=unit_normal.reshape(d, -1).T[cell_id],
        material_fractions=material.reshape(material.shape[0], -1).T[cell_id],
        weights=gathered_weight,
        cell_id=cell_id,
        n_active=n_active,
        time=time,
        step_index=step_index,
        stage_index=stage_index,
        run_seed=run_seed,
    )
    return BandSelection(request, n_active, capacity)


def scatter_to_grid(values: jax.Array, cell_id: jax.Array, grid: Grid) -> jax.Array:
    """Sum padded per-point values back onto the grid. Padding carries zero, so it adds nothing."""
    flat = jnp.zeros(math.prod(grid.shape), dtype=values.dtype).at[cell_id].add(values)
    return flat.reshape(grid.shape)


def _sample_linear(field: jax.Array, index_coords: jax.Array, grid: Grid) -> jax.Array:
    """Bilinear sample with per-axis boundary handling: wrap laterally, clamp vertically."""
    coords = [
        jnp.mod(index_coords[a], n) if periodic else jnp.clip(index_coords[a], 0.0, n - 1.0)
        for a, (n, periodic) in enumerate(zip(grid.shape, grid.periodic))
    ]
    return map_coordinates(field, coords, order=1, mode="nearest")


def gather_to_band(
    speed: jax.Array,
    request: VelocityRequest,
    phi: jax.Array,
    grid: Grid,
    bands: BandConfig,
) -> jax.Array:
    """Closest-point gather: spread the speeds evaluated on the thin set across the extension band.

    Each cell samples the weighted speed field at its own projection onto the interface, and
    normalises by the sampled weight — so a cell whose stencil straddles the edge of the evaluation
    band is not diluted by the zeros outside it. The division is double-where guarded.
    """
    from m2.stencils import normals

    weighted = scatter_to_grid(request.weights * speed, request.cell_id, grid)
    weights = scatter_to_grid(request.weights, request.cell_id, grid)

    unit_normal, _ = normals(phi, grid)
    projection = closest_point(phi, grid, unit_normal) / grid.spacing_nm
    flat_proj = projection.reshape(grid.ndim, -1)

    num = _sample_linear(weighted, flat_proj, grid).reshape(grid.shape)
    den = _sample_linear(weights, flat_proj, grid).reshape(grid.shape)
    safe = jnp.where(den > WEIGHT_FLOOR, den, 1.0)
    extended = jnp.where(den > WEIGHT_FLOOR, num / safe, 0.0)
    return extended * extension_weight(phi, grid, bands)
