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
