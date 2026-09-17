"""Measurement helpers shared by the invariant checks. Host-side numpy, never differentiated.

**Area is the sub-cell zero-contour area** (finding G2, carried over from the existing tree). Three
candidate measures differ by an order of magnitude on the same run -- a cell count is quantised to
whole cells, and a mollified-Heaviside integral also responds to how phi is reshaped NEAR the
interface, not only to where the interface is. V8, V9 and V12 are statements about the REGION, so
they use the measure of the region.

The region is computed exactly for the piecewise-linear interpolant of phi: the dual grid between
cell centres is split into triangles, on each of which phi is linear and the negative part has a
closed-form area.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def _negative_area_of_triangles(f1, f2, f3, tri_area):
    f = np.sort(np.stack([f1, f2, f3], axis=-1), axis=-1)
    a, b, c = f[..., 0], f[..., 1], f[..., 2]
    area = np.zeros_like(a)
    all_neg = c < 0.0
    one_neg = (a < 0.0) & (b >= 0.0)
    two_neg = (b < 0.0) & (c >= 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        area = np.where(all_neg, tri_area, area)
        area = np.where(one_neg, tri_area * a * a / ((a - b) * (a - c)), area)
        area = np.where(two_neg, tri_area - tri_area * c * c / ((c - a) * (c - b)), area)
    return np.nan_to_num(area)


def solid_area(phi, dx: float) -> float:
    """Area of {phi < 0} for the piecewise-linear interpolant, in nm^2. 2D only."""
    p = np.asarray(phi, dtype=float)
    tri = 0.5 * dx * dx
    p00, p01, p10, p11 = p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]
    return float(np.sum(_negative_area_of_triangles(p00, p01, p11, tri)
                        + _negative_area_of_triangles(p00, p10, p11, tri)))


def symmetric_difference_area(phi_a, phi_b, dx: float) -> float:
    """Area of the region in exactly one of the two solids: |A u B| - |A n B|."""
    a, b = np.asarray(phi_a, dtype=float), np.asarray(phi_b, dtype=float)
    return solid_area(np.minimum(a, b), dx) - solid_area(np.maximum(a, b), dx)


def radii_along_rays(phi, dx: float, centre_nm, n_angles: int = 72, r_max_nm: float | None = None):
    """Sub-cell radius of the first - to + crossing along each ray from `centre_nm` (z, x)."""
    p = np.asarray(phi, dtype=float)
    r_max = r_max_nm if r_max_nm is not None else 0.45 * min(p.shape) * dx
    s = np.linspace(0.0, r_max, 4000)
    radii = []
    for ang in np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False):
        z = (centre_nm[0] + s * np.cos(ang)) / dx - 0.5
        x = (centre_nm[1] + s * np.sin(ang)) / dx - 0.5
        vals = map_coordinates(p, [z, x], order=1, mode="nearest")
        k = int(np.argmax(vals >= 0.0))
        radii.append(s[k - 1] + (s[k] - s[k - 1]) * (-vals[k - 1]) / (vals[k] - vals[k - 1]))
    return np.array(radii)


def grad_mag_error_near_interface(phi, grid, band_cells: float = 3.0) -> float:
    """max ||grad phi| - 1| over cells within `band_cells` of the zero level set (V11's measure)."""
    import jax.numpy as jnp
    from geocore.stencils import central_gradient

    mag = jnp.sqrt(jnp.sum(central_gradient(phi, grid) ** 2, axis=0))
    near = (jnp.abs(phi) <= band_cells * grid.spacing_nm)
    near = near.at[:2].set(False).at[-2:].set(False)
    return float(jnp.max(jnp.where(near, jnp.abs(mag - 1.0), 0.0)))
