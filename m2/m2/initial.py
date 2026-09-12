"""Initial signed-distance fields (PRD §5.1).

φ < 0 inside solid, φ > 0 in the open volume, and every constructor here returns an exact signed
distance so that |∇φ| = 1 holds at t = 0 without reinitialisation (which is M2.2).

Axis 0 is vertical and increases toward the plasma, so "deeper into the wafer" is a smaller index
and `surface_nm` is measured in the same coordinates as `m2.stencils.cell_centres`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import m2  # noqa: F401  (enables fp64)
from m2.schema import Geometry, Grid, Material
from m2.stencils import cell_centres


def half_space(grid: Grid, surface_nm: float, *, solid_below: bool = True) -> jax.Array:
    """A flat interface at `surface_nm`. Solid below it (a wafer) or above it (a ceiling).

    `solid_below=False` is what V1a uses for a downward-facing surface: under the directional law
    it must not move at all.
    """
    z = cell_centres(grid)[0]
    phi = z - surface_nm
    return phi if solid_below else -phi


def disk(grid: Grid, centre_nm: tuple[float, ...], radius_nm: float) -> jax.Array:
    """A solid disk (2D) or sphere (3D): φ = |x − c| − r, negative inside. V1's geometry."""
    coords = cell_centres(grid)
    centre = jnp.asarray(centre_nm, dtype=jnp.float64).reshape((-1,) + (1,) * grid.ndim)
    return jnp.sqrt(jnp.sum((coords - centre) ** 2, axis=0)) - radius_nm


def trench(grid: Grid, surface_nm: float, cd_nm: float, depth_nm: float,
           centre_nm: float | None = None) -> jax.Array:
    """A rectangular trench already cut into the surface, as an exact signed distance.

    The mask is not modelled (decision §2), so the opening is pre-cut into φ: solid below
    `surface_nm`, minus a slot of width `cd_nm` and depth `depth_nm`. Lateral axis 1 only, which is
    all the coupon's line trenches need.
    """
    coords = cell_centres(grid)
    z, x = coords[0], coords[1]
    lateral_extent = grid.shape[1] * grid.spacing_nm
    centre = 0.5 * lateral_extent if centre_nm is None else centre_nm

    wafer = z - surface_nm  # negative inside the solid
    # Signed distance to the open slot: negative inside the slot, measured as a rectangle.
    dx_out = jnp.abs(x - centre) - 0.5 * cd_nm
    dz_out = (surface_nm - depth_nm) - z
    outside = jnp.sqrt(jnp.maximum(dx_out, 0.0) ** 2 + jnp.maximum(dz_out, 0.0) ** 2)
    inside = jnp.minimum(jnp.maximum(dx_out, dz_out), 0.0)
    slot = outside + inside  # < 0 inside the slot
    return jnp.maximum(wafer, -slot)  # solid minus slot, in signed-distance form


def uniform_material(grid: Grid, materials: tuple[Material, ...]) -> jax.Array:
    """Fraction field for a single-solid-material stack: void 0, the solid 1 (decision §10).

    Phase 1 is a one-solid-material instance of the general code, not a special case: the array has
    a row per material and sums to 1 everywhere. Layered stacks arrive at M2.6.
    """
    solids = [m for m in materials if not m.is_void]
    if len(solids) != 1:
        raise ValueError(f"uniform_material expects exactly one solid material, got {len(solids)}")
    fractions = jnp.zeros((len(materials), *grid.shape), dtype=jnp.float64)
    return fractions.at[solids[0].index].set(1.0)


def geometry(phi: jax.Array, grid: Grid, materials: tuple[Material, ...]) -> Geometry:
    return Geometry(phi=phi, material=uniform_material(grid, materials), grid=grid)
