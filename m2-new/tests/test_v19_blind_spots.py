"""Finding A9, made executable: V19's "all three fail" holds for ONE of three injection points.

PRD §8.5 claimed `jax.jvp` and `jax.vjp` are "separate code paths", so agreement between them is
real evidence. That is not true for custom rules. JAX derives reverse mode by linearising with
the forward-mode JVP rule and TRANSPOSING it, so both share that rule.

The consequence is not academic: it decides what V15 and V16 are worth. These tests run against
the installed JAX rather than asserting the claim, so the day the behaviour changes, they say so.

M2 has no custom rules today. The moment one is added, this file is the reason someone knows
V15 and V16 stopped covering it.
"""

import jax
import jax.numpy as jnp
import pytest

from geocore.verification.gradcheck import forward_reverse_test, taylor_test

KEY = jax.random.PRNGKey(19)
THETA = jnp.array([0.7, 1.3, 2.0], dtype=jnp.float64)
SCALES = jnp.ones(3, dtype=jnp.float64)
BAD = 1.05


# ---------------------------------------------------------- injection point 1: custom_jvp
@jax.custom_jvp
def cubed_sum(x):
    return jnp.sum(x**3)


@cubed_sum.defjvp
def _corrupted_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    return cubed_sum(x), BAD * jnp.sum(3.0 * x**2 * t)   # the 1.05 is the injected error


# ---------------------------------------------------------- injection point 2: custom_vjp
@jax.custom_vjp
def cubed_sum_vjp(x):
    return jnp.sum(x**3)


def _fwd(x):
    return cubed_sum_vjp(x), x


def _bwd(res, g):
    return (BAD * 3.0 * res**2 * g,)


cubed_sum_vjp.defvjp(_fwd, _bwd)


def test_corrupt_jvp_rule_is_caught_by_v14_only():
    """The headline of A9. Reverse mode is the transpose of this same rule, so jvp and vjp agree
    -- on the wrong derivative. V15 passes. Only V14, which compares against the FUNCTION, fails.
    """
    assert not taylor_test(cubed_sum, THETA, jax.grad(cubed_sum)(THETA), SCALES, KEY).passed, \
        "V14 must catch a corrupted JVP rule"
    assert forward_reverse_test(cubed_sum, THETA, KEY).passed, \
        "V15 is expected to PASS here -- that is precisely finding A9"


def test_corrupt_vjp_rule_makes_forward_mode_unavailable():
    """A `custom_vjp` function cannot be differentiated in forward mode at all, so jax.jvp raises.

    The harness must report that as a FAILURE, not a skip. A check that could not run has not
    passed, and recording it as a pass would be the most expensive kind of wrong.
    """
    with pytest.raises(Exception):
        jax.jvp(cubed_sum_vjp, (THETA,), (THETA,))

    result = forward_reverse_test(cubed_sum_vjp, THETA, KEY)
    assert not result.passed
    assert result.measured["forward_mode_available"] is False
    assert not taylor_test(cubed_sum_vjp, THETA, jax.grad(cubed_sum_vjp)(THETA),
                           SCALES, KEY).passed


def test_uncorrupted_custom_rules_pass_both():
    """The control. Without it, the two tests above would be satisfied by a harness that failed
    on any function carrying a custom rule at all."""
    @jax.custom_jvp
    def honest(x):
        return jnp.sum(x**3)

    @honest.defjvp
    def _honest_jvp(primals, tangents):
        (x,), (t,) = primals, tangents
        return honest(x), jnp.sum(3.0 * x**2 * t)

    assert taylor_test(honest, THETA, jax.grad(honest)(THETA), SCALES, KEY).passed
    assert forward_reverse_test(honest, THETA, KEY).passed
