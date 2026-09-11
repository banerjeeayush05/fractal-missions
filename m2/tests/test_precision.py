"""PRD §7.5: fp64 for all level-set fields. fp32 must never be the silent default."""

import jax
import jax.numpy as jnp
import pytest

import m2  # noqa: F401
from m2.schema import Geometry, Grid, SchemaError
from m2.verification.gradcheck import taylor_test


def test_x64_enabled_on_import():
    assert jax.config.jax_enable_x64
    assert jnp.zeros(3).dtype == jnp.float64
    assert jnp.asarray(1.0).dtype == jnp.float64


def test_geometry_rejects_fp32():
    grid = Grid((4, 3), 2.0, (False, True))
    with pytest.raises(SchemaError, match="float64"):
        Geometry(jnp.zeros((4, 3), jnp.float32), jnp.ones((1, 4, 3)), grid)
    with pytest.raises(SchemaError, match="float64"):
        Geometry(jnp.zeros((4, 3)), jnp.ones((1, 4, 3), jnp.float32), grid)


def test_gradient_harness_rejects_fp32_parameters():
    theta = {"a": jnp.asarray(1.0, jnp.float32)}
    with pytest.raises(TypeError, match="float64"):
        taylor_test(lambda t: t["a"] ** 2, theta, theta, scales=theta, key=jax.random.PRNGKey(0))
