"""VTK input and output, for the V22 cross-code comparison. PRD §8.8.

VTK files are the ONLY thing that crosses between this code and ViennaPS. ViennaPS is GPL-3.0: never
link, import, vendor or copy it. It runs in its own process, from its own repository, under its own
licence, and writes files; we read files. Nothing here imports it, and nothing here is derived from it.

Level sets are written as a VTK image (the phi field on its grid) so the other side can contour it with
its own tools, and contours are written as polylines so a comparison needs no shared code at all.
"""

from __future__ import annotations

import pathlib

import numpy as np

from geocore.schema import Grid


def write_level_set(path: str | pathlib.Path, phi, grid: Grid) -> pathlib.Path:
    """Write phi as a VTK image. Spacing is in nm and the origin is the first cell CENTRE, matching
    `initial.coordinate_fields`; a reader that assumes cell corners is off by half a cell."""
    import pyvista as pv

    array = np.asarray(phi, dtype=np.float64)
    if array.shape != tuple(grid.shape):
        raise ValueError(f"phi has shape {array.shape}, grid says {grid.shape}")
    dims = array.shape if array.ndim == 3 else (array.shape[0], array.shape[1], 1)
    image = pv.ImageData(dimensions=dims,
                         spacing=(grid.spacing_nm,) * 3,
                         origin=(0.5 * grid.spacing_nm,) * 3)
    image.point_data["phi"] = array.reshape(-1, order="F")
    path = pathlib.Path(path)
    image.save(path)
    return path


def read_level_set(path: str | pathlib.Path):
    """Read a VTK image written by either side. Returns `(phi, spacing_nm, origin_nm)`."""
    import pyvista as pv

    image = pv.read(str(path))
    dims = tuple(int(d) for d in image.dimensions)
    phi = np.asarray(image.point_data["phi"]).reshape(dims, order="F")
    if dims[2] == 1:
        phi = phi[:, :, 0]
    return phi, float(image.spacing[0]), tuple(float(o) for o in image.origin)


def contour_points(phi, grid: Grid) -> np.ndarray:
    """Points on the zero level set, in nm. `(M, 2)` in 2D, `(M, 3)` in 3D.

    Sub-cell, by linear interpolation along grid edges: for every pair of neighbouring cells whose phi
    straddles zero, the crossing point. Deliberately NOT marching squares -- the comparison needs a
    point set, and the simplest construction is the one least likely to disagree for a reason that is
    about the extraction rather than about the two solvers.
    """
    array = np.asarray(phi, dtype=np.float64)
    dx = grid.spacing_nm
    points = []

    # Nodes sitting exactly on the surface. A strict sign change misses them, and a plane placed on a
    # row of cell centres is entirely made of them -- which is how this was found.
    exact = np.argwhere(array == 0.0)
    if exact.size:
        points.append((exact + 0.5).astype(np.float64) * dx)

    for axis in range(array.ndim):
        lower = np.take(array, np.arange(array.shape[axis] - 1), axis=axis)
        upper = np.take(array, np.arange(1, array.shape[axis]), axis=axis)
        crossing = np.argwhere((lower * upper) < 0.0)
        if crossing.size == 0:
            continue
        index = tuple(crossing[:, a] for a in range(array.ndim))
        a_values, b_values = lower[index], upper[index]
        fraction = a_values / (a_values - b_values)
        coordinates = (crossing + 0.5).astype(np.float64)
        coordinates[:, axis] += fraction
        points.append(coordinates * dx)
    return np.vstack(points) if points else np.empty((0, array.ndim))


def write_contour(path: str | pathlib.Path, phi, grid: Grid) -> pathlib.Path:
    """Write the zero level set as a VTK point set."""
    import pyvista as pv

    points = contour_points(phi, grid)
    padded = np.zeros((points.shape[0], 3))
    padded[:, :points.shape[1]] = points
    path = pathlib.Path(path)
    pv.PolyData(padded).save(path)
    return path
