"""Materials: what each point in space is made of, and how that changes the etch rate. PRD §5.6.

**Fractions, never integer IDs.** Every point carries a fraction per material, summing to 1. An integer
ID makes the etch rate a step function of position: the moment the surface reaches a new layer the
rate jumps, and the derivative of a step is zero everywhere and undefined at one point. The gradient
with respect to anything that controls WHEN the surface arrives at the layer -- which, for an etch
stop, is the thing worth predicting -- is destroyed.

**Mollified boundaries.** A layer's fraction ramps from 0 to 1 over `w_mat` (config, default 2 cells)
with the same C2 quintic smoothstep the band weights use. The ramp is NUMERICAL, not physical; a real
SiGe layer has a sharp edge. That is why the `w_mat` sensitivity study exists and is reported whatever
it shows: a gradient that depends strongly on a smoothing width is a finding, not something to tune.

**Materials do not move.** Fractions are a fixed field in space. The surface moves THROUGH them.

**Where the mask material is.** Fractions say what a solid WOULD be made of at a point, so they extend
into open space. The mask material is the layer ABOVE the film top, across the whole width -- the
opening included. The lateral pattern of the mask lives in phi, not in the fractions. This matters:
defining the mask fraction from the mask BODY puts the mask's own surface on the edge of its fraction
ramp, where the fraction is 0.5, and a "zero-rate" mask would etch at half rate. As a layer, every mask
surface sits deep inside mask material, and the only material boundary is the horizontal contact
between mask and film.

**The mask is geometry with a zero rate** (owner amendment 2026-09-13). The mask is a solid body in phi,
marked as mask material, whose etch-rate multiplier is exactly 0 -- not a parameter, not calibrated.
That restores pattern transfer (the field under the mask stays put, so a trench deepens rather than
translates, finding H2), allows undercut beneath the mask edge, and makes the mask corner a real
corner. The front CONTACTS the mask but never CROSSES into it, so the mask needs no `w_mat` study; if
its multiplier is ever made non-zero, it becomes a second crossing and does.

Rate blend: `R = R_model * sum_m(fraction_m * multiplier_m)`, with the multipliers in the params PyTree
(so a selectivity can later be fitted) and the mask's forced to 0.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import jax.numpy as jnp
from jax import Array

from geocore.band import smoothstep
from geocore.constants import VERTICAL_AXIS
from geocore.initial import coordinate_fields, polygon_2d
from geocore.schema import Grid, Material, VelocityModel, VelocityRequest

MULTIPLIER_KEY = "material_rate"


@dataclasses.dataclass(frozen=True)
class Layer:
    """A horizontal layer occupying [bottom_z, top_z] in absolute nm. `None` extends it without limit."""

    material: Material
    bottom_z_nm: float | None
    top_z_nm: float | None


def _step_up(z: Array, at_nm: float, width_nm: float) -> Array:
    """0 below `at`, 1 above, C2 ramp of total width `width` centred on `at`."""
    return smoothstep((z - at_nm) / width_nm + 0.5)


def layer_fractions(grid: Grid, layers: Sequence[Layer], n_materials: int,
                    w_mat_cells: float) -> Array:
    """Fraction field `(*grid.shape, n_materials)` for a stack of horizontal layers.

    Each layer is `step_up(bottom) - step_up(top)`. Adjacent layers share a boundary, so the sum
    telescopes: with the lowest layer unbounded below and the highest unbounded above, the fractions sum
    to exactly 1 everywhere, ramps included.
    """
    if w_mat_cells <= 0:
        raise ValueError("w_mat_cells must be positive")
    ordered = sorted(layers, key=lambda l: -1e300 if l.bottom_z_nm is None else l.bottom_z_nm)
    if ordered[0].bottom_z_nm is not None or ordered[-1].top_z_nm is not None:
        raise ValueError("the lowest layer must be unbounded below and the highest unbounded above, "
                         "or the fractions do not sum to 1 at the ends of the domain")
    for below, above in zip(ordered[:-1], ordered[1:]):
        if below.top_z_nm != above.bottom_z_nm:
            raise ValueError(f"layers must be contiguous: {below.material.name} ends at "
                             f"{below.top_z_nm} nm, {above.material.name} starts at "
                             f"{above.bottom_z_nm} nm")

    z = coordinate_fields(grid)[VERTICAL_AXIS]
    width = w_mat_cells * grid.spacing_nm
    fractions = jnp.zeros((*grid.shape, n_materials), dtype=jnp.float64)
    for layer in ordered:
        low = 1.0 if layer.bottom_z_nm is None else _step_up(z, layer.bottom_z_nm, width)
        high = 0.0 if layer.top_z_nm is None else _step_up(z, layer.top_z_nm, width)
        fractions = fractions.at[..., layer.material.index].add(low - high)
    return fractions


def masked_trench(grid: Grid, film_top_nm: float, mask_thickness_nm: float, depth_nm: float,
                  top_cd_nm: float, bottom_cd_nm: float, centre_x_nm: float | None = None) -> Array:
    """phi for a film with a trench, under a mask whose opening equals the trench's top CD.

    One closed polygon for the whole solid -- mask and film together -- so it is an exact distance
    function, including at the mask corner and where the mask sidewall meets the trench wall
    (the stage-10 lesson: composing solids with min/max is wrong near openings).
    """
    from geocore.initial import lateral_centre, require_headroom

    if grid.ndim != 2:
        raise ValueError("masked_trench is 2D; extrude for 3D")
    centre = lateral_centre(grid)[0] if centre_x_nm is None else centre_x_nm
    mask_top = film_top_nm + mask_thickness_nm
    require_headroom(grid, mask_top, mask_thickness_nm + depth_nm)
    width = grid.shape[1] * grid.spacing_nm
    skirt = 10.0 * max(width, grid.shape[0] * grid.spacing_nm)
    top, bottom = top_cd_nm / 2.0, bottom_cd_nm / 2.0
    floor = film_top_nm - depth_nm
    solid = jnp.array([
        [-skirt, -skirt],
        [mask_top, -skirt],
        [mask_top, centre - top],          # mask top, left corner
        [film_top_nm, centre - top],       # down the mask sidewall
        [floor, centre - bottom],          # down the trench wall
        [floor, centre + bottom],
        [film_top_nm, centre + top],
        [mask_top, centre + top],          # up the mask sidewall, right corner
        [mask_top, width + skirt],
        [-skirt, width + skirt],
    ], dtype=jnp.float64)
    return polygon_2d(grid, solid)


def with_selectivity(model: VelocityModel, materials: Sequence[Material]) -> VelocityModel:
    """Blend a velocity model's rate by material: `R * sum(fraction * multiplier)`.

    Multipliers come from `params[MULTIPLIER_KEY]`, shape `(n_materials,)`. The mask's is overwritten
    with 0 inside the model, so infinite selectivity cannot be undone by a parameter and receives no
    gradient -- it is a modelling decision, not a fitted quantity.
    """
    masks = [m.index for m in materials if m.is_mask]
    if len(masks) > 1:
        raise ValueError("at most one material may be the mask")

    def blended(request: VelocityRequest, params: Any) -> Array:
        multipliers = params[MULTIPLIER_KEY]
        if masks:
            multipliers = multipliers.at[masks[0]].set(0.0)
        return model(request, params) * (request.material_fractions @ multipliers)

    return blended
