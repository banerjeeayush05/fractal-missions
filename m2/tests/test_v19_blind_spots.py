"""Where the corruption sits decides which checks can see it (OPEN_QUESTIONS A9).

§8.5 says jvp and vjp are "separate code paths", so V15/V16 agreement is independent evidence.
That holds for the literal V19 corruption (of the returned gradient, see test_v19_canary.py).
It does not hold inside custom differentiation rules:
- custom_jvp: JAX derives the VJP by transposing the JVP. Both agree on the wrong derivative;
  V15/V16 pass and only V14 catches it.
- custom_vjp: forward mode is unavailable, so V15/V16 cannot run and must report FAIL.
These tests pin down the installed JAX's behaviour. If one starts failing, JAX changed and A9
needs revisiting.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from m2.verification import analytic as A
from m2.verification.gradcheck import (
    dot_product_test,
    forward_reverse_test,
    reverse_gradient,
    reverse_vjp,
    taylor_test,
)

KEY = jax.random.PRNGKey(3)
_SCALE = jnp.ones(5).at[2].set(1.05)  # 5 % error on component 2 only


@jax.custom_jvp
def sin_bad_jvp(x):
    return jnp.sin(x)


@sin_bad_jvp.defjvp
def _sin_bad_jvp_rule(primals, tangents):
    (x,), (t,) = primals, tangents
    return jnp.sin(x), _SCALE * jnp.cos(x) * t


@jax.custom_vjp
def sin_bad_vjp(x):
    return jnp.sin(x)


sin_bad_vjp.defvjp(lambda x: (jnp.sin(x), x), lambda x, ct: (_SCALE * jnp.cos(x) * ct,))


def _pair(elementwise):
    def J(th):
        x = ravel_pytree(th)[0]
        return A.objective(th) + jnp.sum(elementwise(x))

    def F(th):
        x = ravel_pytree(th)[0]
        return jnp.concatenate([A.vector_map(th), elementwise(x)])

    return J, F


def test_custom_jvp_corruption_only_v14_sees_it():
    J, F = _pair(sin_bad_jvp)
    th = A.theta0()
    x2 = float(ravel_pytree(th)[0][2])
    assert abs(np.cos(x2)) > 0.1  # the corrupted term is not ~0
    g = reverse_gradient(J, th)
    assert not taylor_test(J, th, g, scales=A.scales(), key=KEY).passed
    assert forward_reverse_test(J, th, g, scales=A.scales(), key=KEY).passed  # blind
    assert dot_product_test(F, th, reverse_vjp(F, th), scales=A.scales(), key=KEY).passed  # blind


def test_custom_vjp_corruption_v15_v16_cannot_run_and_fail():
    J, F = _pair(sin_bad_vjp)
    th = A.theta0()
    g = reverse_gradient(J, th)
    assert not taylor_test(J, th, g, scales=A.scales(), key=KEY).passed
    r15 = forward_reverse_test(J, th, g, scales=A.scales(), key=KEY)
    r16 = dot_product_test(F, th, reverse_vjp(F, th), scales=A.scales(), key=KEY)
    assert not r15.passed and "forward mode unavailable" in r15.message
    assert not r16.passed and "forward mode unavailable" in r16.message
