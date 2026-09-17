"""Initial level sets. PRD §5.1.

Every shape here is an EXACT signed distance function: `|grad phi| = 1` everywhere except on the
medial axis, where no distance function is differentiable. That is a stronger requirement than
"phi is zero on the right surface", and it is the whole reason this module is more than four
lines.

**Why exactness matters, concretely.** The obvious way to build a trench is boolean algebra on
half-spaces -- `max(film, -void)` -- which is what the capstone does. The zero level set is
correct, so a picture of it looks right. But away from the surface the value is wrong near every
corner: `max` of two distance functions overestimates distance outside a concave corner and
underestimates it outside a convex one. Three consequences:

* V11 ("reinitialisation restores distance") cannot distinguish a scheme that failed from an
  initial condition that was never a distance function.
* V13 initialises an EXACT trapezoid of known CD, depth and sidewall angle and requires the
  extracted values back to 0.1 nm. A composed approximation has no exact CD to compare against.
* The velocity band is measured in cells of distance. Where phi is not a distance, the band is
  not the width it says it is.

So the composed shapes are built from an exact polygon distance instead: distance to the nearest
point of the boundary, signed by inside/outside. That is exact by construction, corners included.

**Coordinates.** Cell `i` along an axis sits at `(i + 0.5) * dx`, so a grid of `n` cells spans
`[0, n*dx]`. Positions are given in nm, never in cells.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from geocore.constants import VERTICAL_AXIS, VERTICAL_BUFFER_CELLS
from geocore.schema import Grid


class DomainTooSmall(ValueError):
    """The feature plus its etch depth plus the buffer does not fit in the grid."""


# --------------------------------------------------------------------------- coordinates


def axis_coordinates(grid: Grid, axis: int) -> Array:
    """Cell-centre positions in nm along one axis."""
    return (jnp.arange(grid.shape[axis], dtype=jnp.float64) + 0.5) * grid.spacing_nm


def coordinate_fields(grid: Grid) -> tuple[Array, ...]:
    """One array per axis, each of the grid's shape, holding that axis's coordinate."""
    return tuple(jnp.meshgrid(*[axis_coordinates(grid, a) for a in range(grid.ndim)],
                              indexing="ij"))


def lateral_centre(grid: Grid) -> tuple[float, ...]:
    return tuple(grid.shape[a] * grid.spacing_nm / 2.0 for a in range(1, grid.ndim))


def require_headroom(grid: Grid, surface_z_nm: float, etch_depth_nm: float) -> None:
    """PRD §5.1: stack + maximum etch depth + a 10-cell buffer must fit.

    Checked when the initial condition is built, not only at config load, because a test that
    constructs a Grid directly never passes through the config loader -- and an interface that
    reaches the Neumann boundary stalls, which looks like a physics result.
    """
    buffer_nm = VERTICAL_BUFFER_CELLS * grid.spacing_nm
    lowest = surface_z_nm - etch_depth_nm
    if lowest - buffer_nm < 0.0:
        raise DomainTooSmall(
            f"a {etch_depth_nm:g} nm etch from z = {surface_z_nm:g} nm reaches "
            f"z = {lowest:g} nm, inside the {VERTICAL_BUFFER_CELLS}-cell "
            f"({buffer_nm:g} nm) bottom buffer"
        )
    top_nm = grid.shape[VERTICAL_AXIS] * grid.spacing_nm
    if surface_z_nm + buffer_nm > top_nm:
        raise DomainTooSmall(
            f"the surface at z = {surface_z_nm:g} nm is within the "
            f"{VERTICAL_BUFFER_CELLS}-cell buffer of the top boundary at z = {top_nm:g} nm"
        )


# ------------------------------------------------------------------------- exact shapes


def plane(grid: Grid, surface_z_nm: float) -> Array:
    """Solid below, open above. `phi = z - surface_z`, exact by inspection."""
    return coordinate_fields(grid)[VERTICAL_AXIS] - surface_z_nm


def sphere(grid: Grid, centre_nm: tuple[float, ...], radius_nm: float) -> Array:
    """A solid ball. `phi = |x - c| - r`, exact everywhere but the centre.

    Circle in 2D, sphere in 3D, same expression. This is V1's geometry: under a constant rate
    the radius must satisfy `r(t) = r0 - R*t`.
    """
    if len(centre_nm) != grid.ndim:
        raise ValueError(f"centre has {len(centre_nm)} components for a {grid.ndim}D grid")
    coords = coordinate_fields(grid)
    squared = sum((c - centre_nm[a]) ** 2 for a, c in enumerate(coords))
    return jnp.sqrt(squared) - radius_nm


def polygon_2d(grid: Grid, vertices_nm: Array) -> Array:
    """Exact signed distance to a closed 2D polygon. Negative INSIDE.

    `vertices_nm` is `(M, 2)` in `(z, x)` order, listed in sequence; the polygon closes
    automatically. Concave corners are fine.

    Two independent computations, which is what makes it exact rather than approximate:

    * MAGNITUDE — the minimum distance from the point to any edge SEGMENT. Clamping the
      projection parameter to [0, 1] is what makes a vertex the nearest point when it is;
      projecting onto the infinite edge LINE instead is the usual bug, and it under-reports
      distance in the wedge outside every convex corner.
    * SIGN — a crossing-number test, computed separately and multiplied in. Distance never has
      to know about inside or outside, and the sign never has to know about distance.
    """
    if grid.ndim != 2:
        raise ValueError("polygon_2d is 2D only; use extrude() for a 3D trench")
    verts = jnp.asarray(vertices_nm, dtype=jnp.float64)
    if verts.ndim != 2 or verts.shape[1] != 2:
        raise ValueError(f"vertices must be (M, 2) in (z, x) order, got {verts.shape}")

    z, x = coordinate_fields(grid)
    point = jnp.stack([z, x], axis=-1)                      # (nz, nx, 2)

    start = verts                                           # (M, 2)
    end = jnp.roll(verts, -1, axis=0)
    edge = end - start                                      # (M, 2)

    w = point[..., None, :] - start                         # (nz, nx, M, 2)
    t = jnp.clip(
        jnp.sum(w * edge, axis=-1) / jnp.sum(edge * edge, axis=-1), 0.0, 1.0
    )                                                       # (nz, nx, M)
    offset = w - t[..., None] * edge
    distance = jnp.sqrt(jnp.min(jnp.sum(offset * offset, axis=-1), axis=-1))

    # Crossing number: count edges crossing the horizontal ray from the point.
    pz, px = point[..., 0:1], point[..., 1:2]
    sz, sx = start[:, 0], start[:, 1]
    ez, ex = end[:, 0], end[:, 1]
    straddles = (sx > px) != (ex > px)
    z_at_crossing = sz + (px - sx) * (ez - sz) / jnp.where(ex - sx == 0.0, 1.0, ex - sx)
    crossings = jnp.sum(jnp.where(straddles & (pz < z_at_crossing), 1, 0), axis=-1)
    inside = (crossings % 2) == 1

    return jnp.where(inside, -distance, distance)


def extrude(profile: Array, grid: Grid) -> Array:
    """Extend a 2D `(nz, nx)` profile along the remaining lateral axis of a 3D grid.

    Exact for a long trench: the nearest boundary point of an infinite extrusion has the same
    in-plane distance at every position along the extruded axis.
    """
    if grid.ndim != 3:
        raise ValueError("extrude targets a 3D grid")
    nz, nx = profile.shape
    if (nz, nx) != (grid.shape[0], grid.shape[2]):
        raise ValueError(f"profile {profile.shape} does not match grid (nz, nx) "
                         f"({grid.shape[0]}, {grid.shape[2]})")
    return jnp.broadcast_to(profile[:, None, :], grid.shape)


# ------------------------------------------------------------------- composed features


def trapezoid(grid: Grid, surface_z_nm: float, depth_nm: float, top_cd_nm: float,
              bottom_cd_nm: float, centre_x_nm: float | None = None) -> Array:
    """A trapezoidal trench cut into a film. V13's fixture, exact everywhere.

    The SOLID is described directly, as one closed polygon, rather than composed from a film and
    a void. Composition is the obvious approach and it is wrong -- see below -- so this is the
    one shape in the module that is worth reading carefully.

    **Why not `max(film, -void)`.** That was the first implementation and a brute-force check
    against the true boundary caught it. Just above the opening, `film = z - surface_z` reports
    the distance to the plane `z = surface_z`, as though the film's top surface continued across
    the opening. It does not: across the opening there is no film, and the nearest solid is the
    mask corner or the sidewall. At one cell 0.5 nm above the surface the composed field said
    0.50 nm where the true distance is 3.53 nm -- a 7x error, 0.5 nm from the interface, in
    exactly the region the velocity band lives in.

    **The skirts.** The polygon is extended far past the grid on the left, right and bottom, so
    those artificial edges are never the nearest boundary to any cell inside the domain. Without
    them, cells near the domain edge would measure their distance to the edge of the array, which
    is not a material interface at all -- the solid continues beyond it.

    Sidewall angle follows from the geometry -- `atan(2*depth / (top_cd - bottom_cd))` -- so a
    test can state the angle it expects instead of measuring the thing it is checking.
    """
    if grid.ndim != 2:
        raise ValueError("trapezoid is 2D; wrap it in extrude() for 3D")
    centre = lateral_centre(grid)[0] if centre_x_nm is None else centre_x_nm
    require_headroom(grid, surface_z_nm, depth_nm)
    if bottom_cd_nm <= 0.0 or top_cd_nm <= 0.0:
        raise ValueError("both CDs must be positive")

    width_nm = grid.shape[1] * grid.spacing_nm
    height_nm = grid.shape[0] * grid.spacing_nm
    skirt = 10.0 * max(width_nm, height_nm)

    top, bottom = top_cd_nm / 2.0, bottom_cd_nm / 2.0
    floor_z = surface_z_nm - depth_nm

    solid = jnp.array([
        [-skirt,        -skirt],              # skirt: far below and left
        [surface_z_nm,  -skirt],
        [surface_z_nm,  centre - top],        # mask corner, left
        [floor_z,       centre - bottom],     # left sidewall down to the floor
        [floor_z,       centre + bottom],     # across the floor
        [surface_z_nm,  centre + top],        # right sidewall up to the mask corner
        [surface_z_nm,  width_nm + skirt],
        [-skirt,        width_nm + skirt],    # skirt: far below and right
    ], dtype=jnp.float64)

    return polygon_2d(grid, solid)


def trench(grid: Grid, surface_z_nm: float, depth_nm: float, opening_nm: float,
           centre_x_nm: float | None = None) -> Array:
    """A vertical-sidewall trench: the trapezoid with equal top and bottom CD."""
    return trapezoid(grid, surface_z_nm, depth_nm, opening_nm, opening_nm, centre_x_nm)


def sidewall_angle_deg(depth_nm: float, top_cd_nm: float, bottom_cd_nm: float) -> float:
    """The angle a `trapezoid` was built with, from its dimensions. 90 deg is vertical."""
    half_run = (top_cd_nm - bottom_cd_nm) / 2.0
    return float(jnp.degrees(jnp.arctan2(depth_nm, half_run))) if half_run != 0.0 else 90.0
