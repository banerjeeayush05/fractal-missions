"""Conventions and fixed numerical constants.

Everything here is a value that is **the same for the whole product**, fixed by a decision or a
finding. Per-case values — grid spacing, target depth, band widths, `n_reinit`, `w_mat`, the
CFL target — live in `config.py` instead, because they legitimately differ between cases.

The distinction matters because of one anti-requirement (PRD §11): *do not tune `w_mat`, CFL,
band width or `n_reinit` to make a gradient test pass.* A value that sits in a config file next
to values people are supposed to change is a value someone will eventually change. Putting the
fixed ones here, with the decision that fixed them named in the comment, makes the difference
visible at the point of use.

Nothing in this module imports anything. It is the bottom of the dependency graph.
"""

from typing import Final

# --------------------------------------------------------------------------- sign of phi
# PRD §5.1. phi is a signed distance function, NEGATIVE inside solid material and positive in
# the open volume. The outward normal n = grad(phi)/|grad(phi)| therefore points FROM the solid
# INTO the open volume.
#
# This is asserted rather than assumed because a sign error here inverts every gradient in the
# product, and inverts it silently: the solver still runs, the fields still look like a trench,
# and the optimiser still converges — to the wrong recipe.
PHI_SOLID_SIGN: Final[float] = -1.0
PHI_OPEN_SIGN: Final[float] = +1.0

# --------------------------------------------------------------------------- orientation
# PRD §5.1, decision §1. Axis 0 is vertical and INCREASES TOWARD THE PLASMA, so the unit vector
# pointing at the plasma is z_hat = +e_0.
#
# Consequence: a trench floor recedes AWAY from the plasma as it etches (its phi rises), and the
# underside of an overhang, whose normal points away from the plasma, does not move under a
# directional law at all. Check V1a is the only thing in the suite that can catch an inversion
# of this — V14, V15 and V16 all pass happily on a sign-flipped world.
VERTICAL_AXIS: Final[int] = 0
Z_HAT_SIGN: Final[float] = +1.0

# --------------------------------------------------------------------------- etch sign
# Decision §1. Velocity models return an ETCH RATE R in nm/s: R > 0 removes material, R < 0
# deposits (legal, unused in M2). M2 advects
#
#     d(phi)/dt - R |grad(phi)| = 0        equivalently     phi <- phi + dt * R * |grad(phi)|
#
# so with R > 0 and phi negative inside solid, phi rises and the solid shrinks. The sign flip
# lives HERE, in M2, exactly once — not in every M3 and M5 velocity model, where it would have
# to be got right independently each time.
ADVECTION_RATE_SIGN: Final[float] = +1.0

# The Godunov upwind branch is selected by the sign of the coefficient in the standard form
# phi_t + c|grad(phi)| = 0, which for the equation above is c = -R. Recorded so that stage 8
# does not have to re-derive it, and so a test can assert it.
UPWIND_SELECTOR_SIGN: Final[float] = -1.0

# --------------------------------------------------------------------------- fixed numerics
# PRD §5.2, decision D8. Hard ceiling, asserted at EVERY step, not a target. The per-case target
# (0.4) is a config field; this is the line that must never be crossed.
CFL_MAX: Final[float] = 0.5

# PRD §5.3, decision C2. Reinitialisation pseudo-timestep, d(tau) = REINIT_DTAU_OVER_DX * dx.
# Fixed rather than configurable: it is a property of the reinitialisation scheme's stability,
# not of the case being run.
REINIT_DTAU_OVER_DX: Final[float] = 0.5

# PRD §5.7, decision A2. Width of the mollified Heaviside in the smooth volumetric functional
# solid_volume = integral of (1 - H(phi)), in units of dx. Fixed so that the §8.7 diagnostic
# means the same thing in every run that cites it.
HEAVISIDE_WIDTH_CELLS: Final[float] = 1.5

# PRD §5.1. The domain covers the initial stack plus the maximum expected etch depth plus this
# buffer, so the interface never reaches the Neumann boundary.
VERTICAL_BUFFER_CELLS: Final[int] = 10

# --------------------------------------------------------------------------- NaN guards
# The capstone learned both of these the hard way and they are product-wide, so they are fixed
# here rather than rediscovered per module.
#
# GRAD_MAG_EPS sits INSIDE the sqrt in the Godunov Hamiltonian. At a Godunov "valley" cell both
# one-sided branches are exactly zero, and d/dx sqrt(x) is infinite at x = 0: one such cell
# makes jax.grad return NaN for every parameter. Added inside, it never reaches zero.
GRAD_MAG_EPS: Final[float] = 1e-30

# NORMAL_EPS likewise sits inside the sqrt when normalising grad(phi). |grad(phi)| -> 0 on the
# medial axis — the trench centreline, a circle's centre — and jnp.where does NOT save you: a
# NaN in the untaken branch still poisons the gradient. Use the double-where pattern as well.
NORMAL_EPS: Final[float] = 1e-12

# --------------------------------------------------------------------------- WENO5
# PRD §5.2 (built post-M2.3), finding J0. The smoothness-indicator regulariser is FIXED. Jiang & Peng
# scale it by max(v1^2 ... v5^2) over the stencil; that max is a kink in the differentiated path,
# which §11 forbids. Fixing it is safe because the indicators are built from v = delta(phi)/dx, which
# is about |grad phi| ~ 1 on a distance function, so the published scaling would be ~1 anyway. A test
# moves this by 100x each way and requires the observed order not to move.
WENO_EPS: Final[float] = 1e-6

# Finding J3. Within this many cells of a NON-periodic boundary WENO5 falls back to the two-point
# one-sided difference. The Neumann clamp replicates the edge value; WENO reads that flat run as
# smooth data and extrapolates from it, which in the original tree manufactured a phantom solid blob
# at a domain corner. The mask depends on the grid index alone, so it adds no data-dependent branch.
WENO_BOUNDARY_FALLBACK_CELLS: Final[int] = 3

# --------------------------------------------------------------------------- verification
# PRD §7.2, §8.5. V19 injects this multiplicative error into one gradient component and requires
# V14, V15 and V16 to fail. 1.05 is a 5% corruption: large enough that a working harness must
# see it, small enough that it is the size of error a real bug produces.
V19_CORRUPTION_FACTOR: Final[float] = 1.05
