"""Sign and orientation conventions (PRD §5.1; decision §1, resolving OQ A1).

φ < 0 inside solid. Etch rate R > 0 removes material, with dφ/dt = +R|∇φ|. Axis 0 increases
toward the plasma, so ẑ = +e₀. There is no solver yet, so these tests assert the conventions and
the consequences downstream code relies on, using geometry alone. The forward check that a trench
floor actually recedes is V1a at M2.1.
"""

import jax
import jax.numpy as jnp
import numpy as np

import m2.constants as C


def disk_sdf(x, r=1.0):
    """Signed distance to a SOLID disk of radius r, under the §5.1 convention."""
    return jnp.linalg.norm(x) - r


def test_constants_encode_negative_inside_solid():
    assert C.PHI_SOLID_SIGN == -1
    assert C.PHI_OPEN_SIGN == +1
    assert (C.NORMAL_POINTS_FROM, C.NORMAL_POINTS_TO) == ("solid", "open")


def test_constants_encode_etch_sign_and_orientation():
    assert C.DPHI_DT_RATE_SIGN == +1  # φ_t − R|∇φ| = 0
    assert (C.Z_AXIS, C.Z_HAT_SIGN) == (0, +1)
    assert C.AXIS0_INCREASES_TOWARD == "plasma"


def test_disk_sdf_matches_convention():
    inside, outside = jnp.array([0.2, -0.1]), jnp.array([1.5, 0.3])
    assert np.sign(disk_sdf(inside)) == C.PHI_SOLID_SIGN
    assert np.sign(disk_sdf(outside)) == C.PHI_OPEN_SIGN


def test_normal_points_from_solid_to_open():
    # At the interface point (1, 0) of a solid unit disk, open space is at +x.
    x = jnp.array([1.0, 0.0])
    g = jax.grad(disk_sdf)(x)
    n = g / jnp.linalg.norm(g)
    np.testing.assert_allclose(n, [1.0, 0.0], atol=1e-15)
    assert np.sign(disk_sdf(x + 1e-3 * n)) == C.PHI_OPEN_SIGN
    assert np.sign(disk_sdf(x - 1e-3 * n)) == C.PHI_SOLID_SIGN


def test_positive_rate_removes_solid():
    """dφ/dt = +R|∇φ| with R > 0 must turn solid (φ<0) into open volume (φ>0)."""
    phi = jnp.asarray(-0.05)  # just inside the solid
    grad_norm, rate, dt = 1.0, 1.0, 0.1
    phi_next = phi + C.DPHI_DT_RATE_SIGN * rate * grad_norm * dt
    assert np.sign(phi_next) == C.PHI_OPEN_SIGN, "positive R must etch, not deposit"


def test_directional_law_selects_surfaces_facing_the_plasma():
    """R = v₀·max(0, n·ẑ)^p with ẑ = +e₀: a trench floor etches, an overhang underside does not."""
    z_hat = jnp.zeros(2).at[C.Z_AXIS].set(C.Z_HAT_SIGN)
    v0, p = 1.0, 2.0

    def rate(n):
        return v0 * jnp.maximum(0.0, jnp.dot(n, z_hat)) ** p

    # Trench floor: solid below, open volume above (toward the plasma) → n = +ẑ.
    assert float(rate(z_hat)) > 0.0
    # Downward-facing overhang: n = −ẑ → no etch.
    assert float(rate(-z_hat)) == 0.0
    # Vertical sidewall: n ⟂ ẑ → no etch under this law.
    assert float(rate(jnp.array([0.0, 1.0]))) == 0.0


def test_trench_floor_recedes_away_from_the_plasma():
    """The floor's interface moves toward decreasing axis 0, i.e. deeper into the wafer."""
    # φ for a floor at height z0 with solid below: φ(z) = z − z0 (negative below).
    z0, rate, dt = 0.0, 1.0, 0.1
    phi_at = lambda z: z - z0  # noqa: E731
    # dφ/dt = +R|∇φ| = +R shifts φ up, so the zero crossing moves to z0 − R·dt.
    new_zero = z0 - C.DPHI_DT_RATE_SIGN * rate * dt
    assert new_zero < z0
    assert np.sign(phi_at(z0 - 0.5 * rate * dt)) == C.PHI_SOLID_SIGN  # was solid, now etched


def test_p_must_exceed_one_to_avoid_a_sidewall_kink():
    """max(0, n·ẑ)^p is non-differentiable at n·ẑ = 0 — on vertical walls — unless p > 1."""
    assert C.DIRECTIONAL_MIN_P == 1.0
    assert C.DIRECTIONAL_MIN_P_FOR_GRADIENT_TESTS >= 2.0
    d_rate = jax.grad(lambda s, p: jnp.maximum(0.0, s) ** p)
    assert np.isfinite(float(d_rate(0.0, 2.0)))  # p = 2: derivative exists and is 0
    assert float(d_rate(0.0, 2.0)) == 0.0


def test_derivative_in_p_is_finite_at_zero_argument():
    """0^p must not produce NaN in the p-derivative (decision, 'four things the file missed')."""
    safe_pow = lambda s, p: jnp.where(s > 0, jnp.where(s > 0, s, 1.0) ** p, 0.0)  # double-where
    d_dp = jax.grad(safe_pow, argnums=1)
    assert float(d_dp(0.0, 2.0)) == 0.0
    assert np.isfinite(float(d_dp(0.0, 2.0)))
    assert np.isfinite(float(d_dp(0.5, 2.0)))
