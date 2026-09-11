"""PRD §5.0 data contracts and the §5.5 velocity contract (v0.2, decision §3)."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from m2 import interface
from m2.schema import (
    Geometry,
    Grid,
    Material,
    SchemaError,
    VelocityRequest,
    check_geometry_values,
    check_velocity_output,
    validate_materials,
)


# --- Grid ---------------------------------------------------------------------------------


def test_grid_2d_and_3d():
    assert Grid((64, 32), 2.0, (False, True)).ndim == 2
    assert Grid([64, 32, 32], 2, [False, True, True]).shape == (64, 32, 32)


@pytest.mark.parametrize(
    "shape,spacing,periodic,match",
    [
        ((8,), 1.0, (False,), "2D or 3D"),
        ((8, 0), 1.0, (False, True), "positive ints"),
        ((8, 4.0), 1.0, (False, True), "positive ints"),
        ((8, 4), 0.0, (False, True), "spacing_nm"),
        ((8, 4), float("nan"), (False, True), "spacing_nm"),
        ((8, 4), 1.0, (False,), "one bool per axis"),
        ((8, 4), 1.0, (True, True), "vertical axis"),
        ((8, 4, 4), 1.0, (False, True, False), "lateral axes must be periodic"),
    ],
)
def test_grid_rejects(shape, spacing, periodic, match):
    with pytest.raises(SchemaError, match=match):
        Grid(shape, spacing, periodic)


# --- Material -----------------------------------------------------------------------------


def _mats():
    return [Material("silicon", 1, False), Material("void", 0, False, is_void=True)]


def test_materials_validate_and_sort():
    mats = validate_materials(_mats())
    assert [m.name for m in mats] == ["void", "silicon"]
    assert mats[0].is_void and not mats[1].is_void


@pytest.mark.parametrize(
    "mats,match",
    [
        ([], "at least one"),
        ([Material("a", 0, False, is_void=True), Material("b", 2, False)], "indices"),
        ([Material("a", 0, False, is_void=True), Material("a", 1, False)], "unique"),
        ([Material("a", 0, False), Material("b", 1, False)], "exactly one material must be the void"),
        ([Material("a", 0, False, is_void=True), Material("b", 1, False, is_void=True)],
         "exactly one material must be the void"),
    ],
)
def test_materials_reject(mats, match):
    with pytest.raises(SchemaError, match=match):
        validate_materials(mats)


def test_material_field_types():
    with pytest.raises(SchemaError):
        Material("", 0, False)
    with pytest.raises(SchemaError):
        Material("a", True, False)
    with pytest.raises(SchemaError):
        Material("a", 0, 1)
    with pytest.raises(SchemaError, match="both void and mask"):
        Material("a", 0, True, is_void=True)


# --- Geometry -----------------------------------------------------------------------------


def _geom(n_mat=2):
    grid = Grid((6, 4), 2.0, (False, True))
    frac = jnp.zeros((n_mat, 6, 4)).at[0].set(1.0)
    return Geometry(jnp.linspace(-1, 1, 24).reshape(6, 4), frac, grid)


def test_geometry_shapes():
    g = _geom()
    check_geometry_values(g)
    with pytest.raises(SchemaError, match="grid.shape"):
        Geometry(jnp.zeros((5, 4)), jnp.ones((1, 6, 4)), g.grid)
    with pytest.raises(SchemaError, match="n_materials"):
        Geometry(jnp.zeros((6, 4)), jnp.ones((6, 4)), g.grid)


def test_geometry_value_checks():
    g = _geom()
    bad = Geometry(g.phi, g.material.at[1, 0, 0].set(0.5), g.grid)
    with pytest.raises(SchemaError, match="sum to 1"):
        check_geometry_values(bad)
    with pytest.raises(SchemaError, match="non-finite"):
        check_geometry_values(Geometry(g.phi.at[0, 0].set(jnp.nan), g.material, g.grid))


def test_geometry_is_a_pytree_through_jit():
    g = _geom()
    out = jax.jit(lambda geo: Geometry(geo.phi * 2.0, geo.material, geo.grid))(g)
    assert isinstance(out, Geometry) and out.grid == g.grid
    assert len(jax.tree_util.tree_leaves(g)) == 2  # grid is static metadata, not a leaf


def test_geometry_is_generic_in_material_count():
    """Decision §10: phase 1 (void + silicon) and phase 2 (+ SiGe) use the same interface."""
    for n_mat in (2, 3):
        check_geometry_values(_geom(n_mat))


# --- VelocityRequest / VelocityModel (contract v0.2) ---------------------------------------


def _request(k=5, d=2, n_mat=2, step=0, stage=0, seed=7):
    return VelocityRequest(
        positions=jnp.zeros((k, d)),
        normals=jnp.ones((k, d)) / jnp.sqrt(d),
        material_fractions=jnp.ones((k, n_mat)) / n_mat,
        weights=jnp.concatenate([jnp.ones(k - 1), jnp.zeros(1)]),  # last entry is padding
        cell_id=jnp.arange(k),
        n_active=jnp.asarray(k - 1),
        time=0.0,
        step_index=step,
        stage_index=stage,
        run_seed=seed,
    )


def _bare(**kw):
    """Positional-ish builder for the rejection tests."""
    base = dict(positions=jnp.zeros((5, 2)), normals=jnp.zeros((5, 2)),
                material_fractions=jnp.ones((5, 1)), weights=jnp.ones(5), cell_id=jnp.arange(5),
                n_active=jnp.asarray(5), time=0.0, step_index=0, stage_index=0, run_seed=1)
    base.update(kw)
    return VelocityRequest(**base)


def test_velocity_request_fields_are_contract_v0_3():
    names = [f.name for f in dataclasses.fields(VelocityRequest)]
    assert names == ["positions", "normals", "material_fractions", "weights", "cell_id",
                     "n_active", "time", "step_index", "stage_index", "run_seed"]
    assert VelocityRequest.__dataclass_params__.frozen


def test_velocity_request_validation():
    _request(k=3, d=3)
    with pytest.raises(SchemaError, match="normals"):
        _bare(normals=jnp.zeros((5, 3)))
    with pytest.raises(SchemaError, match="batch shape"):
        _bare(material_fractions=jnp.ones((4, 1)))
    with pytest.raises(SchemaError, match="weights"):
        _bare(weights=jnp.ones(4))
    with pytest.raises(SchemaError, match="cell_id"):
        _bare(cell_id=jnp.arange(4))
    with pytest.raises(SchemaError, match="cell_id must be an integer"):
        _bare(cell_id=jnp.zeros(5))
    with pytest.raises(SchemaError, match="n_active"):
        _bare(n_active=jnp.arange(5))
    for missing in ("step_index", "stage_index", "run_seed"):
        with pytest.raises(SchemaError, match=missing):
            _bare(**{missing: None})


def test_cell_id_is_a_stable_identifier_not_a_row_index():
    """Decision A16: the RNG key uses the flattened grid index, so a cell joining the band does not
    shift anyone else's draws. Here cell 41 keeps its id when cell 17 appears ahead of it."""
    before = _bare(cell_id=jnp.asarray([12, 41, 63, 77, 0]), weights=jnp.asarray([1.0, 1, 1, 1, 0]))
    after = _bare(cell_id=jnp.asarray([12, 17, 41, 63, 77]), weights=jnp.ones(5))
    assert int(before.cell_id[1]) == 41 and int(after.cell_id[2]) == 41
    key_of = lambda req, cid: (int(req.run_seed), int(req.step_index), int(req.stage_index), cid)  # noqa: E731
    assert key_of(before, 41) == key_of(after, 41)


def test_zero_weight_does_not_protect_against_nan():
    """Decision B14: 0 * NaN is NaN and poisons the whole gradient, so padded entries need a benign
    position and normals need the double-where guard regardless of weight."""
    weights = jnp.asarray([1.0, 0.0])
    poisoned = jnp.asarray([1.0, jnp.nan])
    assert jnp.isnan(jnp.sum(weights * poisoned))  # masking after the fact does not help
    # The double-where pattern: make the untaken branch harmless before differentiating.
    def guarded(x):
        safe = jnp.where(weights > 0, x, 1.0)
        return jnp.sum(jnp.where(weights > 0, jnp.sqrt(safe), 0.0))
    g = jax.grad(guarded)(jnp.asarray([4.0, 0.0]))
    assert not jnp.any(jnp.isnan(g)), g


def test_padding_is_identifiable_by_zero_weight_and_counted_by_n_active():
    """Capacity K is fixed, so the tail is padding; weight 0 is how a model can tell, and n_active
    is the separate diagnostic needed to size K and see overflow coming (decision B14)."""
    req = _request(k=5)
    assert float(req.weights[-1]) == 0.0
    assert int(req.n_active) == 4
    assert req.batch_shape == (5,)


def test_velocity_request_crosses_scan_with_traced_indices():
    """step_index, stage_index and run_seed are traced inside lax.scan."""

    def model(req: VelocityRequest, params) -> jax.Array:  # analytic stand-in
        del req
        return params["v0"] * jnp.ones(5)

    def body(carry, i):
        v = None
        for stage in range(2):  # one call per RK stage (§5.5)
            req = _request(step=i, stage=stage)
            v = model(req, {"v0": carry}) * req.weights  # padding contributes nothing
            check_velocity_output(req, v)
        return carry, v.sum()

    _, out = jax.lax.scan(body, jnp.asarray(2.0), jnp.arange(3))
    assert out.shape == (3,)


def test_velocity_output_contract():
    req = _request()
    check_velocity_output(req, jnp.zeros(5))
    with pytest.raises(SchemaError, match="shape"):
        check_velocity_output(req, jnp.zeros(4))
    with pytest.raises(SchemaError, match="float64"):
        check_velocity_output(req, jnp.zeros(5, jnp.float32))


def test_contract_fingerprint_matches_version():
    """Changing the velocity contract requires bumping CONTRACT_VERSION and the fingerprint in the
    same commit (OPEN_QUESTIONS B9)."""
    assert interface.contract_fingerprint() == interface.CONTRACT_FINGERPRINT, (
        "velocity contract changed: bump CONTRACT_VERSION and CONTRACT_FINGERPRINT together"
    )
    assert interface.CONTRACT_VERSION == "0.3.0"
    # Provisional until the M3 owner signs off; do not freeze at M2.3 without it (decision §3).
    assert interface.CONTRACT_STATUS == "provisional"
