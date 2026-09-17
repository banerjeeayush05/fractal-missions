"""Differentiable geometry extraction. PRD §5.7, milestone M2.5.

Turns a level set into the numbers an etch engineer measures -- depth, CD at fixed heights, sidewall
angle, bow, mask remaining -- in a form JAX can differentiate. The product sells d(CD)/d(recipe), not
d(volume)/d(recipe); this module is where the level set meets metrology.

**Rule 1: sub-cell only.** Every position is a linear interpolation of the zero crossing between two
cells. Counting cells quantises the output to dx and makes its derivative zero almost everywhere.

**Rule 2: fixed ABSOLUTE heights and windows** (decision C5). CD is read at z = 1200 nm, not at "half
the current depth". With an absolute height the row-interpolation weights are constants, so nothing in
the extraction switches discretely as the parameters move. A depth-relative height walks across grid
rows as the depth changes and puts a jump in the gradient. It is also what metrology measures.

**Rule 3: sidewall angle by least-squares fit** over a fixed window with a fixed number of samples, not
from two CDs. The sample heights are constants, so the fitted slope is a LINEAR function of the
crossing positions -- smooth wherever they are.

**Known, unavoidable roughness.** A crossing is linear interpolation between the two cells that
straddle it. When the surface moves past a grid node, the pair of cells switches, so the extracted
position is continuous but its derivative jumps there. V14 on CD is therefore touchier than on
`solid_volume`; check this before hunting a bug (m2/CLAUDE.md).

**Invalid extraction returns NaN, deliberately.** If a trench has no wall at the requested height --
the height is above the surface, or the trench has closed -- the result is NaN rather than a plausible
number built from index 0. A NaN objective is loud; a wrong CD is not.

Scope: 2D profiles `(nz, nx)` with the vertical axis first. A 3D LINE feature (an extruded trench)
reduces to a profile with `line_profile`. Holes and other genuinely 3D features are not handled here.

scikit-image is the independent reference for V13 ONLY and is never imported by this module
(decision A11).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array

from geocore.constants import VERTICAL_AXIS
from geocore.schema import Grid


def _require_profile(phi: Array, grid: Grid) -> None:
    if phi.ndim != 2:
        raise ValueError(f"extraction works on a 2D profile (nz, nx); got shape {phi.shape}. "
                         f"Reduce a 3D line feature with line_profile().")
    if VERTICAL_AXIS != 0:
        raise AssertionError("extraction assumes the vertical axis is axis 0")


def line_profile(phi: Array, lateral_axis: int = 1) -> Array:
    """Average a 3D line feature along its length. Exact for an extruded profile, and linear, so it
    passes gradients straight through."""
    return jnp.mean(phi, axis=lateral_axis)


def _interpolation(position_nm: float, spacing_nm: float, n: int) -> tuple[int, float]:
    """Lower cell index and weight for a FIXED position. Python numbers: resolved at trace time."""
    continuous = position_nm / spacing_nm - 0.5
    lower = min(max(int(math.floor(continuous)), 0), n - 2)
    return lower, continuous - lower


def row_at_height(phi: Array, grid: Grid, z_nm: float) -> Array:
    """phi along x at an absolute height, interpolated between the two rows that bracket it."""
    j, t = _interpolation(z_nm, grid.spacing_nm, phi.shape[0])
    return (1.0 - t) * phi[j] + t * phi[j + 1]


def column_at(phi: Array, grid: Grid, x_nm: float) -> Array:
    """phi along z at an absolute lateral position."""
    i, t = _interpolation(x_nm, grid.spacing_nm, phi.shape[1])
    return (1.0 - t) * phi[:, i] + t * phi[:, i + 1]


def _first_crossing(values: Array, open_to_solid: bool) -> tuple[Array, Array]:
    """Fractional index of the first sign change along `values`, and whether one exists.

    Looks for the first index k with a value on the far side of zero; the crossing lies between k-1
    and k. `argmax` picks k and carries no gradient; the interpolation fraction does.
    """
    beyond = values < 0.0 if open_to_solid else values >= 0.0
    k = jnp.argmax(beyond)
    found = jnp.any(beyond) & (k > 0)
    k_safe = jnp.maximum(k, 1)
    before, after = values[k_safe - 1], values[k_safe]
    denominator = before - after
    safe = jnp.where(found & (denominator != 0.0), denominator, 1.0)
    return (k_safe - 1) + before / safe, found


def wall_positions(phi: Array, grid: Grid, centre_x_nm: float, z_nm: float) -> tuple[Array, Array]:
    """(left wall x, right wall x) in nm at an absolute height, searching outward from the trench
    centre. NaN where no wall exists at that height."""
    _require_profile(phi, grid)
    row = row_at_height(phi, grid, z_nm)
    dx = grid.spacing_nm
    centre = min(max(int(round(centre_x_nm / dx - 0.5)), 1), row.shape[0] - 2)

    right_frac, right_found = _first_crossing(row[centre:], open_to_solid=True)
    left_frac, left_found = _first_crossing(row[centre::-1], open_to_solid=True)

    right_x = (centre + right_frac + 0.5) * dx
    left_x = (centre - left_frac + 0.5) * dx
    ok = right_found & left_found & (row[centre] >= 0.0)
    return jnp.where(ok, left_x, jnp.nan), jnp.where(ok, right_x, jnp.nan)


def cd_at(phi: Array, grid: Grid, centre_x_nm: float, z_nm: float) -> Array:
    """Critical dimension -- trench width -- at an absolute height, nm."""
    left, right = wall_positions(phi, grid, centre_x_nm, z_nm)
    return right - left


def floor_height(phi: Array, grid: Grid, centre_x_nm: float) -> Array:
    """Absolute height of the trench floor at the trench centre, nm. Scans upward from the bottom
    of the domain: solid below, open above."""
    _require_profile(phi, grid)
    frac, found = _first_crossing(column_at(phi, grid, centre_x_nm), open_to_solid=False)
    return jnp.where(found, (frac + 0.5) * grid.spacing_nm, jnp.nan)


def depth(phi: Array, grid: Grid, centre_x_nm: float, reference_z_nm: float) -> Array:
    """Trench depth below a FIXED absolute reference height (normally the initial top surface).

    Measured from a fixed reference rather than from the current top surface, which also moves:
    until the mask is modelled (stage 17), the field etches along with the floor (finding L2).
    """
    return reference_z_nm - floor_height(phi, grid, centre_x_nm)


def sidewall_angle(phi: Array, grid: Grid, centre_x_nm: float, z_low_nm: float,
                   z_high_nm: float, n_samples: int = 9) -> Array:
    """Mean sidewall angle of both walls in degrees, 90 = vertical, < 90 = wider at the top.

    Least-squares slope dx/dz of each wall over `n_samples` fixed heights in [z_low, z_high]. With the
    heights fixed, the slope is sum(w_i x_i) with constant weights -- linear in the crossings. The
    angle from vertical-slope s is atan2(1, s); matches `initial.sidewall_angle_deg`.
    """
    if not (z_high_nm > z_low_nm and n_samples >= 3):
        raise ValueError("need z_high > z_low and at least 3 samples")
    heights = [z_low_nm + (z_high_nm - z_low_nm) * i / (n_samples - 1) for i in range(n_samples)]
    mean_z = sum(heights) / n_samples
    spread = sum((z - mean_z) ** 2 for z in heights)
    weights = jnp.array([(z - mean_z) / spread for z in heights])

    walls = [wall_positions(phi, grid, centre_x_nm, z) for z in heights]
    left = jnp.stack([w[0] for w in walls])
    right = jnp.stack([w[1] for w in walls])
    slope_right = jnp.sum(weights * right)          # widens upward: positive
    slope_left = -jnp.sum(weights * left)           # mirror, so the same convention applies
    angle = lambda s: jnp.degrees(jnp.arctan2(1.0, s))
    return 0.5 * (angle(slope_right) + angle(slope_left))


def bow(phi: Array, grid: Grid, centre_x_nm: float, z_low_nm: float, z_high_nm: float) -> Array:
    """Bulge of the trench at mid-window relative to its ends, nm. Positive = wider in the middle.

    `CD(z_mid) - (CD(z_low) + CD(z_high)) / 2`. The PRD names bow without defining it (finding
    S16.1, provisional). The common "max CD in the window" definition uses a max, which is a kink in
    the differentiated path (§11); this definition is linear in three CDs and is zero for any straight
    taper, which is what bow should measure.
    """
    mid = 0.5 * (z_low_nm + z_high_nm)
    return (cd_at(phi, grid, centre_x_nm, mid)
            - 0.5 * (cd_at(phi, grid, centre_x_nm, z_low_nm)
                     + cd_at(phi, grid, centre_x_nm, z_high_nm)))


def mask_remaining(phi: Array, grid: Grid, column_x_nm: float, mask_bottom_z_nm: float) -> Array:
    """Thickness of material above `mask_bottom_z` at a column outside the trench, nm.

    Scans DOWN from the top of the domain to the first solid. Meaningful when the geometry carries a
    mask (`materials.masked_trench`) and `mask_bottom_z` is the film top. With the mask at infinite
    selectivity the model predicts zero loss, so a measured loss on a real wafer is direct evidence that
    the zero multiplier is wrong (PRD §5.7).
    """
    _require_profile(phi, grid)
    column = column_at(phi, grid, column_x_nm)[::-1]
    frac, found = _first_crossing(column, open_to_solid=True)
    top = (phi.shape[0] - 1 - frac + 0.5) * grid.spacing_nm
    return jnp.where(found, top - mask_bottom_z_nm, jnp.nan)
