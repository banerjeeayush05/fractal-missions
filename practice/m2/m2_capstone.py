import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)     # fp64. V15 needs 1e-10; fp32 can't.

# ---------------- config: chosen by a human, never computed ----------------
n, dx = 256, 1.0
T, N  = 50.0, 300
REINIT_EVERY, REINIT_ITERS = 5, 5
film_top, mask_top, opening = 0.0, 40.0, 40.0
VMAX = 1.9          # largest v_iso + v_dir you will ever pass in. Used by the
                    # CFL assert: for this model max|V| = v_iso + v_dir exactly.

# for quick iteration while developing, uncomment:
# n, T, N = 128, 24.0, 120

dt = T / N
x = (jnp.arange(n) - n/2 + 0.5) * dx
X, Y = jnp.meshgrid(x, x, indexing="ij")


# ------------------------------------------------------------------ stencil
def grad_mag(phi, w, eps=1e-30):
    """|grad phi| by Godunov upwinding. Only sign(w) is used:
    V for advection, S for reinit.

    eps sits INSIDE the sqrt. At a Godunov 'valley' cell both branches are
    exactly 0 and d/dx sqrt(x) is infinite there -- one such cell makes
    jax.grad return NaN for every parameter.
    """
    a = (phi - jnp.roll(phi,  1, 0)) / dx
    b = (jnp.roll(phi, -1, 0) - phi) / dx
    g = jnp.concatenate([phi[:, :1], phi, phi[:, -1:]], axis=1)   # Neumann in y
    c = (phi - g[:, :-2]) / dx
    d = (g[:, 2:] - phi) / dx

    pos = (jnp.maximum(a,0)**2 + jnp.minimum(b,0)**2 +
           jnp.maximum(c,0)**2 + jnp.minimum(d,0)**2)
    neg = (jnp.minimum(a,0)**2 + jnp.maximum(b,0)**2 +
           jnp.minimum(c,0)**2 + jnp.maximum(d,0)**2)

    return jnp.sqrt(jnp.where(w > 0, pos, neg) + eps)


def reinit(phi, iters=REINIT_ITERS):
    phi0 = phi                                   # no .copy() -- JAX is immutable
    S    = phi0 / jnp.sqrt(phi0**2 + dx**2)
    dtau = 0.5 * dx
    for _ in range(iters):                       # fixed count, from config
        phi = phi - dtau * S * (grad_mag(phi, S) - 1.0)
    return phi


# ----------------------------------------------------------------- velocity
def velocity(phi, v_iso, v_dir, p, eps=1e-12):
    gx = (jnp.roll(phi, -1, 0) - jnp.roll(phi, 1, 0)) / (2*dx)
    g  = jnp.concatenate([phi[:, :1], phi, phi[:, -1:]], axis=1)
    gy = (g[:, 2:] - g[:, :-2]) / (2*dx)
    nz = gy / jnp.sqrt(gx**2 + gy**2 + eps)      # eps INSIDE the sqrt
    ion = jnp.maximum(nz, 0.0) ** p              # ions can't hit downward faces
    return -(v_iso + v_dir * ion)                # NEGATIVE: etching shrinks solid


# --------------------------------------------------------------- extraction
def cross_up(v, coord):
    """First  - -> +  crossing, sub-cell. argmax = first True, FIXED shape."""
    k = jnp.argmax((v[:-1] < 0) & (v[1:] >= 0))
    v0, v1 = v[k], v[k+1]
    return coord[k] + dx * (-v0) / (v1 - v0)


def cross_dn(v, coord):
    """Last  + -> -  crossing, sub-cell."""
    m = (v[:-1] >= 0) & (v[1:] < 0)
    k = m.shape[0] - 1 - jnp.argmax(m[::-1])
    v0, v1 = v[k], v[k+1]
    return coord[k] + dx * (v0) / (v0 - v1)


def measure(phi):
    y_bot = cross_up(phi[n//2, :], x)            # trench bottom
    depth = film_top - y_bot

    y_mid = (film_top + y_bot) / 2               # CD at mid-depth
    j0 = jnp.clip(jnp.searchsorted(x, y_mid) - 1, 0, n - 2)
    t  = (y_mid - x[j0]) / dx
    row = (1 - t) * phi[:, j0] + t * phi[:, j0+1]   # interpolate the ROW too

    return depth, cross_dn(row, x) - cross_up(row, x)


# ------------------------------------------------------------- forward map
def initial_phi():
    film = Y - film_top
    mask = jnp.maximum(jnp.maximum(film_top - Y,        # above film_top
                                   Y - mask_top),      # below mask_top
                                   opening - jnp.abs(X))  # outside the opening
    return reinit(jnp.minimum(film, mask), 5)


@jax.checkpoint                                  # recompute internals on backward pass
def step(phi, params):
    v_iso, v_dir, p = params[0], params[1], params[2]
    V    = velocity(phi,  v_iso, v_dir, p)
    phi1 = phi + dt * (-V * grad_mag(phi, V))
    V1   = velocity(phi1, v_iso, v_dir, p)       # recompute V at stage 2
    return 0.5*phi + 0.5*(phi1 + dt * (-V1 * grad_mag(phi1, V1)))


def simulate(params):
    phi = initial_phi()
    for k in range(N):                           # exactly N, always
        phi = step(phi, params)
        if (k + 1) % REINIT_EVERY == 0:          # condition on the INDEX, safe
            phi = reinit(phi)
    return phi


# ------------------------------------------------------------- objective
theta_true = jnp.array([0.2, 1.0, 2.0])          # the recipe to recover

DEPTH_TARGET, CD_TARGET = measure(simulate(theta_true))   # targets FROM the sim


def objective(params):
    depth, cd = measure(simulate(params))
    return (cd - CD_TARGET)**2 + (depth - DEPTH_TARGET)**2


# ================================================================= run it
if __name__ == "__main__":
    # CFL, asserted once: max|V| = v_iso + v_dir analytically for this model
    assert VMAX * dt / dx < 0.5, f"CFL {VMAX*dt/dx:.3f} -- raise N or lower T"

    print(f"targets:  depth {float(DEPTH_TARGET):.4f}   CD {float(CD_TARGET):.4f}")

    theta = jnp.array([0.35, 0.70, 2.0])         # start somewhere wrong
    d, c_ = measure(simulate(theta))
    print(f"at start: depth {float(d):.4f}   CD {float(c_):.4f}   J {float(objective(theta)):.4f}")

    g = jax.grad(objective)(theta)
    print(f"grad J  = {np.asarray(g)}")
    print(f"finite? {bool(jnp.all(jnp.isfinite(g)))}")