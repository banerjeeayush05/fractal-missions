"""Stage 2 gate — the conventions in geocore/constants.py hold together.

Most of what you can assert about a constants file is tautology: `assert PHI_SOLID_SIGN == -1.0`
restates the definition in a second place and then breaks whenever someone edits the value,
which catches a typo rather than a mistake. So almost nothing here asserts a literal.

What is asserted instead is (a) RELATIONSHIPS between constants, which are derivations and can
genuinely be got wrong, and (b) CONSEQUENCES — small worked examples where the convention has to
produce the physically right answer as a system rather than as four independent numbers.

These tests need no stencil, no solver and no velocity model. They use the fact that an exact
signed distance function has |grad(phi)| = 1, which makes one advection step exact arithmetic.
"""

import ast
import pathlib

import jax.numpy as jnp
import pytest

from geocore import constants as c

# --------------------------------------------------------------------------- relationships


def test_phi_signs_are_opposite_and_nonzero():
    """Solid and open must be on opposite sides of zero. A convention where both were positive,
    or where one was 0.0, would still import and would still let every test below run."""
    assert c.PHI_SOLID_SIGN * c.PHI_OPEN_SIGN < 0.0
    assert c.PHI_SOLID_SIGN < 0.0 < c.PHI_OPEN_SIGN


def test_upwind_selector_is_the_negative_of_the_advection_sign():
    """A derivation, not a value. We advect phi_t - R|grad phi| = 0; the Godunov branch is
    selected by the sign of c in the standard form phi_t + c|grad phi| = 0, so c = -R.

    If someone changes the etch-sign convention and updates ADVECTION_RATE_SIGN without also
    updating this, the solver upwinds from the downstream side: stable-looking, plausible
    output, wrong answer. That is the failure this file exists to make loud.
    """
    assert c.UPWIND_SELECTOR_SIGN == -c.ADVECTION_RATE_SIGN


def test_cfl_ceiling_leaves_room_for_the_case_target():
    """PRD §5.2: configs carry a target of 0.4 and this is the hard ceiling asserted per step.
    A ceiling at or below the target would make every conforming case abort."""
    assert c.CFL_MAX == 0.5
    assert c.CFL_MAX > 0.4


def test_guard_epsilons_are_positive_and_negligible():
    """Both sit inside a sqrt to keep its argument off zero. They must be strictly positive to
    do that job, and far below any length scale in nm so they never perturb a result."""
    assert c.GRAD_MAG_EPS > 0.0
    assert c.NORMAL_EPS > 0.0
    assert c.GRAD_MAG_EPS < 1e-20
    assert c.NORMAL_EPS < 1e-9


def test_v19_corruption_is_a_five_percent_error():
    """PRD §7.2. Small enough to resemble a real bug, large enough that a working harness must
    see it. A factor of exactly 1.0 would make the canary vacuous."""
    assert c.V19_CORRUPTION_FACTOR != 1.0
    assert abs(c.V19_CORRUPTION_FACTOR - 1.0) == pytest.approx(0.05)


# --------------------------------------------------------------------------- consequences


def test_a_positive_rate_shrinks_a_solid_disk():
    """The whole sign convention, exercised end to end with no solver.

    phi = r - r0 is the exact signed distance to a solid disk: negative inside, and |grad| = 1
    everywhere except the centre. That makes one advection step exact arithmetic --
    phi <- phi + dt*R*1 -- so this tests the CONVENTION and nothing else.

    A positive etch rate must make the disk smaller. Flip any one of PHI_SOLID_SIGN or
    ADVECTION_RATE_SIGN and the disk grows instead, which is deposition.
    """
    r0, dt, rate = 10.0, 0.1, 2.0
    x = jnp.linspace(-20.0, 20.0, 81)
    X, Y = jnp.meshgrid(x, x, indexing="ij")
    r = jnp.sqrt(X**2 + Y**2)
    phi = r - r0

    assert phi[40, 40] < 0.0, "the centre of a solid disk must be inside solid"

    phi_next = phi + c.ADVECTION_RATE_SIGN * dt * rate  # |grad phi| = 1 for an exact SDF

    solid_before = int(jnp.sum(phi < 0.0))
    solid_after = int(jnp.sum(phi_next < 0.0))
    assert solid_after < solid_before, "a positive etch rate must REMOVE material"

    # And by exactly the analytic amount: the zero level set moves to r = r0 - dt*rate.
    r_expected = r0 - dt * rate
    on_new_surface = jnp.abs(r - r_expected) < 1e-12
    if bool(jnp.any(on_new_surface)):
        assert float(jnp.max(jnp.abs(phi_next[on_new_surface]))) < 1e-12
    assert float(phi_next[40, 40]) == -(r0 - dt * rate)


def test_an_upward_face_etches_and_an_overhang_underside_does_not():
    """PRD §5.1, decision §1, and the only convention V14-V16 cannot check.

    Axis 0 increases toward the plasma, so z_hat = +e_0. Under the directional law
    R = v0 * max(0, n.z_hat)^p, a surface whose normal points at the plasma etches at full
    rate, and a downward-facing surface -- the underside of an overhang -- does not move.

    An inverted axis convention swaps the two, which looks entirely plausible in a plot: the
    trench still deepens, it just deepens the wrong feature.
    """
    z = jnp.linspace(-5.0, 5.0, 11)

    # Solid below, plasma above: phi = z - surface. Its gradient along axis 0 is +1.
    facing_plasma = z - 0.0
    n_dot_z = jnp.gradient(facing_plasma)[5] * c.Z_HAT_SIGN
    assert float(n_dot_z) > 0.0
    assert float(jnp.maximum(0.0, n_dot_z) ** 2) > 0.0, "an up-facing surface must etch"

    # Solid above, open below -- an overhang underside. Gradient along axis 0 is -1.
    overhang_underside = -(z - 0.0)
    n_dot_z = jnp.gradient(overhang_underside)[5] * c.Z_HAT_SIGN
    assert float(n_dot_z) < 0.0
    assert float(jnp.maximum(0.0, n_dot_z) ** 2) == 0.0, "an overhang underside must not move"


def test_the_vertical_axis_is_the_first_axis():
    """2D is (nz, nx) and 3D is (nz, ny, nx), so the vertical axis is 0 in both. A convention
    that put it last would make every 2D/3D shared code path index differently by dimension."""
    assert c.VERTICAL_AXIS == 0
    for ndim in (2, 3):
        assert 0 <= c.VERTICAL_AXIS < ndim


# --------------------------------------------------------------------------- structure


def test_constants_module_imports_nothing():
    """Its docstring claims it is the bottom of the dependency graph. This checks that claim,
    because a constants module that imports a sibling is how an import cycle starts."""
    source = pathlib.Path(c.__file__).read_text()
    imported = {
        node.module.split(".")[0] if isinstance(node, ast.ImportFrom) else
        node.names[0].name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    assert imported <= {"typing"}, f"constants.py should import nothing but typing: {imported}"
