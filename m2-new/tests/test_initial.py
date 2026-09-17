"""Initial level sets are EXACT signed distance functions, not fields that merely vanish on the
right surface.

The reference here is brute force: sample the true boundary densely, take the minimum distance to
it, sign it by an independent inside test. Slow, obviously correct, and completely unlike the
implementation -- which is what makes it evidence rather than a restatement.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from geocore.initial import (
    DomainTooSmall, coordinate_fields, extrude, plane, polygon_2d, require_headroom,
    sidewall_angle_deg, sphere, trapezoid, trench,
)
from geocore.schema import Grid
from geocore.stencils import central_gradient

G2 = Grid(shape=(48, 48), spacing_nm=1.0, periodic=(False, True))
G3 = Grid(shape=(48, 48, 48), spacing_nm=1.0, periodic=(False, True, True))
SURFACE, DEPTH, TOP_CD, BOTTOM_CD, CENTRE = 36.0, 18.0, 20.0, 12.0, 24.0


def _grad_mag(phi, grid):
    return jnp.sqrt(jnp.sum(central_gradient(phi, grid) ** 2, axis=0))


def _brute_force_trapezoid(grid: Grid) -> np.ndarray:
    """The same shape, computed a completely different way."""
    floor = SURFACE - DEPTH
    corners = [(SURFACE, 0.0), (SURFACE, CENTRE - TOP_CD / 2), (floor, CENTRE - BOTTOM_CD / 2),
               (floor, CENTRE + BOTTOM_CD / 2), (SURFACE, CENTRE + TOP_CD / 2),
               (SURFACE, grid.shape[1] * grid.spacing_nm)]
    samples = []
    for a, b in zip(corners[:-1], corners[1:]):
        a, b = np.array(a), np.array(b)
        n = int(np.linalg.norm(b - a) * 200) + 2
        samples.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
    boundary = np.vstack(samples)

    z, x = (np.asarray(c) for c in coordinate_fields(grid))
    points = np.stack([z.ravel(), x.ravel()], axis=1)
    distance = np.min(np.linalg.norm(points[:, None, :] - boundary[None, :, :], axis=2),
                      axis=1).reshape(z.shape)
    half = BOTTOM_CD / 2 + (TOP_CD - BOTTOM_CD) / 2 * (z - floor) / DEPTH
    in_void = (z >= floor) & (z <= SURFACE) & (np.abs(x - CENTRE) <= half)
    return np.where((z < SURFACE) & ~in_void, -distance, distance)


# ------------------------------------------------------------------- shapes with no corners


def test_a_plane_is_exactly_a_distance_function():
    phi = plane(G2, SURFACE)
    interior = _grad_mag(phi, G2)[2:-2]
    assert float(jnp.max(jnp.abs(interior - 1.0))) == 0.0


@pytest.mark.parametrize("ndim", [2, 3], ids=["2D", "3D"])
def test_the_sphere_residual_is_second_order_truncation_not_a_construction_error(ndim):
    """`|grad phi| = 1` holds for a sphere only up to the stencil's truncation error, so an
    absolute threshold would be a number invented to fit whatever was measured.

    The falsifiable claim is the ORDER: central differences are second order, so halving dx must
    quarter the residual. A construction error would not obey that -- it would stay put, or
    scale with the first power. This needs no magic constant and is a stronger statement.
    """
    domain_nm, radius_nm, centre_nm = 48.0, 14.0, 24.0
    errors = []
    for cells in (48, 96):
        dx = domain_nm / cells
        grid = Grid(shape=(cells,) * ndim, spacing_nm=dx,
                    periodic=(False,) + (True,) * (ndim - 1))
        phi = sphere(grid, (centre_nm,) * ndim, radius_nm)
        near = jnp.abs(phi) < 4.0
        err = jnp.where(near, jnp.abs(_grad_mag(phi, grid) - 1.0), 0.0)
        margin = int(round(4.0 / dx))
        errors.append(float(jnp.max(err[margin:-margin])))

    ratio = errors[0] / errors[1]
    assert ratio == pytest.approx(4.0, rel=0.25), (
        f"residual {errors[0]:.3e} -> {errors[1]:.3e} is a ratio of {ratio:.2f}; second-order "
        f"truncation requires ~4"
    )


def test_a_sphere_wider_than_its_periodic_axis_is_corrupted_by_the_wrap():
    """A CONSTRAINT ON V1's GEOMETRY, pinned before V1 exists to use it.

    V1 runs a circle or sphere under periodic lateral boundaries. Periodicity is correct for a
    repeating trench and wrong for an isolated ball: where the sphere reaches the lateral edge,
    the neighbour lookup wraps to the far side and the field there is not a distance function.

    So V1's configuration must keep the sphere clear of the lateral boundaries. This test states
    the failure explicitly rather than leaving it to be rediscovered as a mysterious 5% error in
    a convergence study.
    """
    cramped = Grid(shape=(48, 8, 48), spacing_nm=1.0, periodic=(False, True, True))
    phi = sphere(cramped, (24.0, 4.0, 24.0), 14.0)       # 28 nm across an 8 nm periodic axis
    near = jnp.abs(phi) < 4.0
    err = jnp.where(near, jnp.abs(_grad_mag(phi, cramped) - 1.0), 0.0)[3:-3]
    assert float(jnp.max(err)) > 1e-2
    interior_y = err[:, 2:-2, :]
    assert float(jnp.max(interior_y)) < float(jnp.max(err)), \
        "the corruption must be at the periodic edge, not throughout"


def test_sphere_rejects_a_centre_of_the_wrong_dimension():
    with pytest.raises(ValueError, match="components"):
        sphere(G2, (1.0, 2.0, 3.0), 5.0)


# ---------------------------------------------------------------------------- the trapezoid


def test_the_trapezoid_matches_a_brute_force_distance_field():
    """The headline. Two unrelated computations of the same quantity must agree."""
    mine = np.asarray(trapezoid(G2, SURFACE, DEPTH, TOP_CD, BOTTOM_CD, CENTRE))
    truth = _brute_force_trapezoid(G2)
    near = np.abs(truth) < 4.0
    assert np.max(np.abs(mine - truth)[near]) < 1e-3


def test_the_composed_min_max_construction_is_wrong_near_the_opening():
    """FINDING, pinned. `max(film, -void)` -- the obvious construction, and the capstone's -- is
    not a distance function above the opening: it reports the distance to the plane `z = surface`
    as though the film continued across the opening, where in truth the nearest solid is the mask
    corner. The error is largest exactly where the velocity band lives.

    This test exists so the shortcut is not reintroduced as a simplification.
    """
    z, x = coordinate_fields(G2)
    floor = SURFACE - DEPTH
    composed = jnp.maximum(
        z - SURFACE,
        -jnp.maximum(jnp.maximum(floor - z, z - SURFACE),
                     jnp.abs(x - CENTRE) - TOP_CD / 2),
    )
    truth = _brute_force_trapezoid(G2)
    just_above_opening = (np.asarray(z) > SURFACE) & (np.asarray(z) < SURFACE + 1.5) & \
                         (np.abs(np.asarray(x) - CENTRE) < TOP_CD / 2 - 1.0)
    composed_err = np.max(np.abs(np.asarray(composed) - truth)[just_above_opening])
    exact_err = np.max(np.abs(np.asarray(trapezoid(G2, SURFACE, DEPTH, TOP_CD, BOTTOM_CD,
                                                   CENTRE)) - truth)[just_above_opening])
    assert composed_err > 1.0, "the composed field should be badly wrong here"
    assert exact_err < 1e-3
    assert composed_err > 100 * max(exact_err, 1e-9)


def test_the_gradient_residual_at_corners_is_intrinsic_not_a_construction_error():
    """A cell straddling a sharp corner cannot show |grad phi| = 1 under central differences, for
    ANY distance function. The claim is not that the residual is absent -- it is that it is the
    same residual the exact field has."""
    grid = G2
    mine = trapezoid(grid, SURFACE, DEPTH, TOP_CD, BOTTOM_CD, CENTRE)
    truth = jnp.asarray(_brute_force_trapezoid(grid))
    near = (jnp.abs(truth) < 4.0).at[:3].set(False).at[-3:].set(False)
    err_mine = float(jnp.max(jnp.where(near, jnp.abs(_grad_mag(mine, grid) - 1.0), 0.0)))
    err_true = float(jnp.max(jnp.where(near, jnp.abs(_grad_mag(truth, grid) - 1.0), 0.0)))
    assert err_mine == pytest.approx(err_true, rel=0.05)


def test_the_zero_level_set_has_the_dimensions_it_was_asked_for():
    """Sub-cell crossings along the surface row and the floor row must return the CDs given."""
    phi = np.asarray(trapezoid(G2, SURFACE, DEPTH, TOP_CD, BOTTOM_CD, CENTRE))
    x = np.asarray(coordinate_fields(G2)[1])[0]

    def width_at(z_nm: float) -> float:
        row = phi[int(z_nm - 0.5)]
        crossings = [x[i] + (x[i + 1] - x[i]) * (-row[i]) / (row[i + 1] - row[i])
                     for i in range(len(row) - 1) if row[i] * row[i + 1] < 0]
        return crossings[-1] - crossings[0]

    assert width_at(SURFACE - 0.5) == pytest.approx(TOP_CD, abs=0.6)
    assert width_at(SURFACE - DEPTH + 0.5) == pytest.approx(BOTTOM_CD, abs=0.6)


def test_a_trench_has_vertical_sidewalls():
    assert sidewall_angle_deg(DEPTH, TOP_CD, TOP_CD) == 90.0
    assert sidewall_angle_deg(18.0, 20.0, 12.0) == pytest.approx(77.47, abs=0.01)
    vertical = trench(G2, SURFACE, DEPTH, TOP_CD, CENTRE)
    tapered = trapezoid(G2, SURFACE, DEPTH, TOP_CD, BOTTOM_CD, CENTRE)
    assert not bool(jnp.allclose(vertical, tapered))


# -------------------------------------------------------------------------------- 3D, sizing


def test_extrude_repeats_the_profile_along_the_second_lateral_axis():
    profile = trapezoid(Grid((G3.shape[0], G3.shape[2]), 1.0, (False, True)),
                        SURFACE, DEPTH, TOP_CD, BOTTOM_CD, CENTRE)
    volume = extrude(profile, G3)
    assert volume.shape == G3.shape
    for y in range(G3.shape[1]):
        assert bool(jnp.array_equal(volume[:, y, :], profile))


def test_extrude_rejects_a_mismatched_profile():
    with pytest.raises(ValueError, match="does not match"):
        extrude(jnp.zeros((8, 8)), G3)


def test_headroom_is_required_below_the_etch_and_above_the_surface():
    """PRD §5.1. An interface that reaches the Neumann boundary stalls, which looks like physics."""
    require_headroom(G2, surface_z_nm=36.0, etch_depth_nm=18.0)
    with pytest.raises(DomainTooSmall, match="bottom buffer"):
        require_headroom(G2, surface_z_nm=36.0, etch_depth_nm=30.0)
    with pytest.raises(DomainTooSmall, match="top boundary"):
        require_headroom(G2, surface_z_nm=44.0, etch_depth_nm=4.0)


def test_polygon_is_2d_only():
    with pytest.raises(ValueError, match="2D only"):
        polygon_2d(G3, jnp.zeros((4, 2)))
