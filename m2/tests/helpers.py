"""Test-only helpers: synthetic configs and 1D sub-cell crossings.

These are not part of the package. The crossing helper is a 1D linear zero crossing along a grid
line, which is all V1/V2 need to locate an interface; the differentiable extraction proper (CD at
fixed heights, sidewall angle) is M2.5.
"""

from __future__ import annotations

import numpy as np

from m2.config import BandConfig, CaseSpec, ExtensionConfig, M2Config, Parameter, VelocityConfig
from m2.schema import Grid, Material

MATERIALS = (Material("void", 0, False, is_void=True), Material("silicon", 1, False))


def synthetic_config(
    shape=(64, 64),
    spacing_nm: float = 10.0,
    *,
    n_steps: int = 200,
    depth_nm: float = 1000.0,
    model: str = "isotropic",
    v_iso: float = 5.83,
    v_dir: float = 5.25,
    p: float = 2.0,
    bands: BandConfig | None = None,
    seed: int = 0,
) -> M2Config:
    """A config for a synthetic check (V1, V2, V1a), not a coupon case.

    Built directly rather than loaded from YAML: these geometries are disks and planes, not the
    §2 trench ladder, and inventing case files for them would imply a coupon that does not exist.
    """
    params = {"v_iso": Parameter(v_iso, max(v_iso, 1.0))}
    if model == "directional":
        params |= {"v_dir": Parameter(v_dir, max(v_dir, 1.0)), "p": Parameter(p, 2.0)}
    grid = Grid(shape, spacing_nm, tuple([False] + [True] * (len(shape) - 1)))
    return M2Config(
        case=CaseSpec("synthetic", 1, "calibrate", depth_nm, None, None, (1.0,)),
        dimension=len(shape),
        grid=grid,
        n_steps=n_steps,
        cfl_target=0.4,
        seed=seed,
        materials=MATERIALS,
        extension=ExtensionConfig("closest_point", None),
        velocity=VelocityConfig(model, params),
        bands=bands or BandConfig(8.0, 2.0, 1.5, None),
    )


def crossing_along(values: np.ndarray, spacing_nm: float, *, start: float = 0.0) -> float:
    """First φ = 0 crossing along a 1D array, by linear interpolation between the bracketing cells.

    Returns the coordinate in nm. Raises if the line never crosses, so a silent 'no interface' can
    never be read as a position.
    """
    values = np.asarray(values)
    above = np.flatnonzero(values > 0)
    if above.size == 0 or above[0] == 0:
        raise ValueError("no sign change along this line: the interface is not where it was expected")
    i = int(above[0])
    frac = -values[i - 1] / (values[i] - values[i - 1])
    return start + spacing_nm * (i - 1 + frac)


def disk_radius(phi: np.ndarray, spacing_nm: float, centre_index: int) -> float:
    """Mean radius of a disk, from the four axis-aligned crossings through its centre."""
    phi = np.asarray(phi)
    rays = [
        phi[centre_index, centre_index:],
        phi[centre_index, : centre_index + 1][::-1],
        phi[centre_index:, centre_index],
        phi[: centre_index + 1, centre_index][::-1],
    ]
    return float(np.mean([crossing_along(r, spacing_nm) for r in rays]))


# --- test-only velocity model and geometry (decision C2 approved the rotation model) -----------
# A rigid-rotation field is a test harness, not a flux model, so PRD §3's ban on transport does not
# bite. It exists because V8 and V9 need a position-dependent velocity, which neither of §5.5's two
# analytic models provides. It must never leave tests/.

def rigid_rotation_model(centre_nm, omega_rad_per_s):
    """R = −(u·n) for u = ω × (x − c), the normal speed of a rigid rotation (axes: 0 = z, 1 = x).

    The sign follows from M2's convention: advection is φ_t = +R|∇φ|, while pure transport by u is
    φ_t = −(u·n)|∇φ|.
    """
    import jax.numpy as jnp

    cz, cx = centre_nm

    def model(request, params):
        del params
        z, x = request.positions[..., 0], request.positions[..., 1]
        u_z = -omega_rad_per_s * (x - cx)
        u_x = omega_rad_per_s * (z - cz)
        u_dot_n = u_z * request.normals[..., 0] + u_x * request.normals[..., 1]
        return -u_dot_n

    return model


def notched_disk(grid, centre_nm, radius_nm, slot_width_nm, slot_top_nm):
    """Zalesak's disk: a disk with a rectangular slot cut out of it (V8's geometry)."""
    import jax.numpy as jnp

    from m2.stencils import cell_centres

    coords = cell_centres(grid)
    z, x = coords[0], coords[1]
    cz, cx = centre_nm
    disk = jnp.sqrt((z - cz) ** 2 + (x - cx) ** 2) - radius_nm

    dx_out = jnp.abs(x - cx) - 0.5 * slot_width_nm
    dz_out = z - slot_top_nm  # slot open downward from its top
    outside = jnp.sqrt(jnp.maximum(dx_out, 0.0) ** 2 + jnp.maximum(dz_out, 0.0) ** 2)
    inside = jnp.minimum(jnp.maximum(dx_out, dz_out), 0.0)
    slot = outside + inside
    return jnp.maximum(disk, -slot)


def mollified_area(phi, spacing_nm, width_cells=1.5):
    """Enclosed area via a mollified Heaviside — the measure V8/V9 use for area and overlap."""
    import numpy as _np

    h = 0.5 * (1.0 + _np.tanh(_np.asarray(phi) / (width_cells * spacing_nm)))
    return float(_np.sum(1.0 - h) * spacing_nm ** 2)


def radius_at_angle(phi, spacing_nm, centre_index, angle_rad, max_cells=None):
    """Sub-cell radius along an arbitrary ray, by bilinear sampling (for V10's anisotropy)."""
    import numpy as _np

    arr = _np.asarray(phi)
    n = arr.shape[0] if max_cells is None else max_cells
    ts = _np.arange(0, n) * 0.25
    zz = centre_index + ts * _np.cos(angle_rad)
    xx = centre_index + ts * _np.sin(angle_rad)
    z0, x0 = _np.floor(zz).astype(int), _np.floor(xx).astype(int)
    inside = (z0 >= 0) & (x0 >= 0) & (z0 < arr.shape[0] - 1) & (x0 < arr.shape[1] - 1)
    zz, xx, z0, x0, ts = zz[inside], xx[inside], z0[inside], x0[inside], ts[inside]
    fz, fx = zz - z0, xx - x0
    vals = ((1 - fz) * (1 - fx) * arr[z0, x0] + (1 - fz) * fx * arr[z0, x0 + 1]
            + fz * (1 - fx) * arr[z0 + 1, x0] + fz * fx * arr[z0 + 1, x0 + 1])
    above = _np.flatnonzero(vals > 0)
    if above.size == 0 or above[0] == 0:
        raise ValueError("no crossing along this ray")
    i = int(above[0])
    frac = -vals[i - 1] / (vals[i] - vals[i - 1])
    return spacing_nm * (ts[i - 1] + frac * (ts[i] - ts[i - 1]))


# --- test-only profile extraction (M2.5 owns the differentiable version) -----------------------
# 1D sub-cell zero crossings along grid rows: enough to measure CD, depth and sidewall angle for a
# diagnostic. Never a substitute for M2.5's extraction.

def half_width_profile(phi, grid, centre_index=None):
    """Half-width of the open slot at each row, in nm, by sub-cell crossing from the centre out."""
    import numpy as _np

    arr = _np.asarray(phi)
    dx = grid.spacing_nm
    c = arr.shape[1] // 2 if centre_index is None else centre_index
    widths = _np.full(arr.shape[0], _np.nan)
    for i in range(arr.shape[0]):
        row = arr[i, c:]
        if row[0] <= 0:
            continue
        below = _np.flatnonzero(row < 0)
        if below.size == 0:
            continue
        j = int(below[0])
        frac = row[j - 1] / (row[j - 1] - row[j])
        widths[i] = dx * (j - 1 + frac)
    return widths


def cd_at_depth(phi, grid, surface_nm, depth_nm):
    """CD (full width) at a fixed absolute depth below the original surface, in nm."""
    import numpy as _np

    widths = half_width_profile(phi, grid)
    z = _np.arange(len(widths)) * grid.spacing_nm
    i = int(_np.argmin(_np.abs(z - (surface_nm - depth_nm))))
    return 2.0 * widths[i]


def floor_depth(phi, grid, surface_nm):
    """Depth of the trench floor below the original surface, in nm (centre column)."""
    import numpy as _np

    col = _np.asarray(phi)[:, _np.asarray(phi).shape[1] // 2]
    above = _np.flatnonzero(col > 0)
    if above.size == 0:
        return _np.nan
    i = int(above[0])
    frac = -col[i - 1] / (col[i] - col[i - 1])
    return surface_nm - grid.spacing_nm * (i - 1 + frac)


def sidewall_angle_deg(phi, grid, surface_nm, window_nm):
    """Least-squares sidewall angle from vertical, over a fixed absolute depth window (P2)."""
    import numpy as _np

    widths = half_width_profile(phi, grid)
    z = _np.arange(len(widths)) * grid.spacing_nm
    depth = surface_nm - z
    sel = (depth >= window_nm[0]) & (depth <= window_nm[1]) & _np.isfinite(widths)
    if sel.sum() < 3:
        return _np.nan
    slope = _np.polyfit(z[sel], widths[sel], 1)[0]
    return float(_np.degrees(_np.arctan(slope)))


def consistent_config(travel_nm, spacing_nm=10.0, shape=(64, 64), **kw):
    """A synthetic config whose derived timing is self-consistent.

    `synthetic_config` takes an explicit step count, so pairing it with an unrelated depth silently
    produces a huge dt — CFL 5 rather than 0.4. This derives both together: depth = travel, and
    N = ceil(travel / (cfl_target * dx)), exactly as the config loader does for a coupon case.
    """
    import math

    n_steps = int(math.ceil(travel_nm / (0.4 * spacing_nm)))
    return synthetic_config(shape, spacing_nm, n_steps=n_steps, depth_nm=travel_nm, **kw)
