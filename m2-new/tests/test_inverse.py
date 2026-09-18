"""M2.8 — inverse sanity: recover a known parameter set from a noisy synthetic profile, cold start.

The last milestone, and the first time the whole chain runs the way the product will use it: forward
solve, extraction, gradient, fit. `scipy.optimize` is permitted here and only here (PRD §10, decision
§12); `tests/test_environment.py` asserts the package never imports it.

**Noise model (proposal P8, `provisional: true`).** Independent Gaussian noise per observable, 1 sigma
from the coupon's metrology repeatability once measured. Until then CD 1.0 nm, depth 5 nm, sidewall
angle 0.2 deg -- the right order for CD-SEM and cross-section metrology at this feature size, not
measurements. P8 also requires that **V14 passes at the recovered parameters**, because recovery is
never itself evidence of gradient correctness.

**"Within the noise floor" is computed, not guessed.** Propagating the observable noise through the
Jacobian gives the parameter covariance `(J^T W J)^-1` with `W = diag(1/sigma^2)`, so the tolerance is
the interval the metrology itself implies: 0.25 nm/s in v0 and 0.21 in p for this case.

**Two caveats that belong on any green M2.8.**

1. This is an INVERSE CRIME: the synthetic measurement comes from the same solver being fitted. That is
   deliberate -- it isolates "can the fitter find the answer" from "is the physics right" -- but it
   means systematic errors shared by both sides cancel. In particular S12.1's reinitialisation drift
   subtracts out exactly, so a green M2.8 is NOT evidence that the drift is harmless. Against a real
   wafer it would not cancel; that is the mechanism S12.1 warns about.
2. The noise is metrology REPEATABILITY, not accuracy. A fit can sit inside the interval and still be
   biased.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.optimize import minimize

from geocore import extraction as X
from geocore.band import BandedRateField
from geocore.config import BandConfig
from geocore.materials import Layer, layer_fractions, masked_trench, with_selectivity
from geocore.schema import Grid, Material
from geocore.solver import SolvePlan, evolve
from geocore.velocity import directional
from geocore.verification.gradcheck import reverse_gradient, taylor_test
from geocore.verification.params import scales_for

pytestmark = pytest.mark.nightly

FILM, MASK = Material("film", 0), Material("mask", 1, is_mask=True)
GRID = Grid((80, 32), 10.0, (False, True))
FILM_TOP, CENTRE = 600.0, 160.0
# Heights that hold a wall across the WHOLE search domain. At v0 = 1, p = 4 the floor sits at 430.0, so
# a height of 430 returns NaN, the objective becomes NaN, and L-BFGS-B stops at its starting point
# reporting zero iterations -- a "fit" that is really the cold start echoed back (S19.2).
HEIGHTS = (460.0, 500.0, 540.0)
SIGMA = np.array([1.0, 1.0, 1.0, 5.0, 0.2])       # CD x3 (nm), depth (nm), sidewall angle (deg)
TRUE = np.array([2.0, 2.0])                        # v0 nm/s, p
BOUNDS = [(0.5, 8.0), (1.2, 6.0)]


@pytest.fixture(scope="module")
def problem():
    fractions = layer_fractions(GRID, [Layer(FILM, None, FILM_TOP), Layer(MASK, FILM_TOP, None)],
                                2, 2.0)
    phi0 = masked_trench(GRID, FILM_TOP, 100.0, 150.0, 140.0, 100.0, CENTRE)
    field = BandedRateField(with_selectivity(directional, [FILM, MASK]), GRID, BandConfig(), 600,
                            fractions)
    plan = SolvePlan(GRID, 1.0, 20, reinit_every=5)

    def observables(theta):
        params = {"v0_nm_per_s": theta[0], "p": theta[1], "material_rate": jnp.array([1.0, 1.0])}
        phi = evolve(phi0, params, field, plan)[0]
        return jnp.array([X.cd_at(phi, GRID, CENTRE, z) for z in HEIGHTS]
                         + [X.depth(phi, GRID, CENTRE, FILM_TOP),
                            X.sidewall_angle(phi, GRID, CENTRE, HEIGHTS[0], HEIGHTS[-1])])

    observe = jax.jit(observables)
    truth = np.asarray(observe(jnp.asarray(TRUE)))
    jacobian = np.asarray(jax.jacfwd(observe)(jnp.asarray(TRUE)))
    covariance = np.linalg.inv(jacobian.T @ np.diag(1.0 / SIGMA**2) @ jacobian)
    return observe, truth, np.sqrt(np.diag(covariance))


def _fit(observe, measured, start, gradient_scale=None):
    """L-BFGS-B on the weighted least-squares cost, with our own gradient."""
    def cost(theta):
        residual = (observe(jnp.asarray(theta)) - measured) / SIGMA
        return jnp.sum(residual**2)

    value_and_grad = jax.jit(jax.value_and_grad(cost))

    def objective(theta):
        value, gradient = value_and_grad(jnp.asarray(theta))
        gradient = np.asarray(gradient, dtype=float)
        if gradient_scale is not None:
            gradient = gradient * gradient_scale
        return float(value), gradient

    result = minimize(objective, np.asarray(start, dtype=float), jac=True, method="L-BFGS-B",
                      bounds=BOUNDS)
    return result.x, result


def test_recovery_is_within_the_noise_floor(problem):
    """Eight independent noise draws. Each fit must land inside the interval the metrology implies.

    Asserted on the DISTRIBUTION, not one draw: with a single draw a 1-sigma test fails about a third
    of the time by construction, and a seed chosen to pass would be choosing the result.
    """
    observe, truth, sigma_parameter = problem
    errors = []
    for seed in range(8):
        measured = jnp.asarray(truth + np.random.default_rng(seed).normal(0.0, SIGMA))
        theta, result = _fit(observe, measured, start=[1.0, 4.0])
        assert result.nit > 0, "a fit that never iterated is the cold start echoed back"
        errors.append(theta - TRUE)
    errors = np.array(errors)
    scaled = np.abs(errors) / sigma_parameter

    assert np.median(scaled, axis=0).max() < 1.0, f"median error {np.median(scaled, axis=0)} sigma"
    assert scaled.max() < 3.0, f"worst error {scaled.max():.2f} sigma"


def test_the_answer_does_not_depend_on_the_cold_start(problem):
    """Three starting points across the search domain, including the far corner. A fit that depends on
    where it started is reporting the optimiser, not the data."""
    observe, truth, _ = problem
    measured = jnp.asarray(truth + np.random.default_rng(0).normal(0.0, SIGMA))
    answers = [_fit(observe, measured, start)[0] for start in ([1.0, 4.0], [6.0, 1.5], [3.0, 3.0])]
    for other in answers[1:]:
        assert np.allclose(answers[0], other, atol=1e-4)


def test_v14_passes_at_the_recovered_parameters(problem):
    """P8's second condition. The Taylor check is re-run where the fit landed, because a fit lands
    somewhere whether or not the gradient that guided it was right."""
    observe, truth, _ = problem
    measured = jnp.asarray(truth + np.random.default_rng(0).normal(0.0, SIGMA))
    theta, _ = _fit(observe, measured, start=[1.0, 4.0])

    recovered = {"v0_nm_per_s": jnp.float64(theta[0]), "p": jnp.float64(theta[1])}
    objective = jax.jit(lambda t: jnp.sum(observe(jnp.array([t["v0_nm_per_s"], t["p"]]))))
    result = taylor_test(objective, recovered, reverse_gradient(objective, recovered),
                         scales_for(recovered), jax.random.PRNGKey(28))
    assert result.passed, result.message


def test_convergence_is_not_evidence_that_the_gradient_is_right(problem):
    """PRD §11's first anti-requirement, made executable.

    Corrupt one gradient component by 5 % -- the same corruption V19 injects -- and the fit still lands
    on the SAME answer to four decimals. The optimum is a property of the objective; the gradient only
    chooses the path taken to it. So "the optimiser converged" says nothing about gradient correctness,
    which is why M2.8 asserts V14 at the recovered point instead.
    """
    observe, truth, _ = problem
    measured = jnp.asarray(truth + np.random.default_rng(0).normal(0.0, SIGMA))
    honest, _ = _fit(observe, measured, start=[1.0, 4.0])
    corrupted, result = _fit(observe, measured, start=[1.0, 4.0],
                             gradient_scale=np.array([1.05, 1.0]))
    assert result.nit > 0
    assert np.allclose(honest, corrupted, atol=1e-3), (
        "if a corrupted gradient DID move the answer, this test's claim would be wrong and the "
        "anti-requirement would need rewording")


def test_the_objective_stays_finite_across_the_search_domain(problem):
    """Finding S19.2, pinned. Extraction returns NaN where no wall exists at the requested height, by
    design. A NaN objective makes L-BFGS-B stop at its starting point and report success, so the
    observation heights must hold a wall everywhere the optimiser may look."""
    observe, truth, _ = problem
    for v0, p in ((0.5, 1.2), (0.5, 6.0), (8.0, 1.2), (8.0, 6.0), (2.0, 2.0)):
        values = np.asarray(observe(jnp.array([v0, p])))
        assert np.all(np.isfinite(values)), f"observables are not finite at v0={v0}, p={p}: {values}"
