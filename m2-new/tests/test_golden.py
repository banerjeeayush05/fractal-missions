"""V21 — golden profile regression. PRD §8.6, fast tier.

A frozen 2D profile, reproduced to 1e-10. Its job is not to prove the profile is right; it is to make
any numerical change VISIBLE IN REVIEW. Change a stencil, a weight, a default, and this test fails
until someone regenerates the golden file in the same commit and says why.

Regenerate deliberately, never casually:

    GEOCORE_REGENERATE_GOLDEN=1 uv run pytest tests/test_golden.py

and put the reason in the commit message. The file is small (a 64x32 fp64 field, 16 KB) and tracked.
"""

import os
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

from geocore.band import BandedRateField
from geocore.config import BandConfig
from geocore.materials import Layer, layer_fractions, masked_trench, with_selectivity
from geocore.schema import Grid, Material
from geocore.solver import SolvePlan, solve
from geocore.velocity import directional

GOLDEN = pathlib.Path(__file__).parent / "golden" / "profile_2d.npz"
FILM, MASK = Material("film", 0), Material("mask", 1, is_mask=True)
TOLERANCE = 1e-10


def _reference_run():
    """Frozen configuration. Every number here is part of the golden file's meaning: changing one is
    changing what the check compares, so it belongs in the same commit as a regenerated file."""
    grid = Grid((64, 32), 10.0, (False, True))
    fractions = layer_fractions(grid, [Layer(FILM, None, 400.0), Layer(MASK, 400.0, None)], 2, 2.0)
    phi0 = masked_trench(grid, 400.0, 100.0, 60.0, 140.0, 100.0, 160.0)
    field = BandedRateField(with_selectivity(directional, [FILM, MASK]), grid, BandConfig(), 1200,
                            fractions)
    params = {"v0_nm_per_s": jnp.float64(2.0), "p": jnp.float64(2.0),
              "material_rate": jnp.array([1.0, 1.0])}
    plan = SolvePlan(grid, 1.0, 40, reinit_every=5, n_reinit=5, spatial_scheme="godunov",
                     temporal_scheme="rk2")
    return grid, solve(phi0, params, field, plan)


@pytest.mark.check("V21")
def test_v21_golden_profile(ledger_measure):
    grid, result = _reference_run()
    phi = np.asarray(result.phi)

    if os.environ.get("GEOCORE_REGENERATE_GOLDEN"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GOLDEN, phi=phi, shape=np.array(grid.shape),
                            spacing_nm=grid.spacing_nm)
        pytest.skip(f"regenerated {GOLDEN.name}; commit it with the reason for the change")

    assert GOLDEN.exists(), (
        f"{GOLDEN} is missing. Generate it with GEOCORE_REGENERATE_GOLDEN=1 and commit it.")
    stored = np.load(GOLDEN)
    assert tuple(stored["shape"]) == grid.shape
    assert float(stored["spacing_nm"]) == grid.spacing_nm

    scale = float(np.max(np.abs(stored["phi"])))
    worst = float(np.max(np.abs(phi - stored["phi"]))) / scale
    ledger_measure(relative_difference=worst, tolerance=TOLERANCE, grid=list(grid.shape),
                   n_steps=40, scheme="godunov")
    assert worst < TOLERANCE, (
        f"the profile moved by {worst:.2e} relative. If the change is intended, regenerate the "
        f"golden file in the SAME commit and say why.")


def test_the_golden_file_is_not_trivially_reproducible():
    """A golden check that compares a field of zeros, or one the solver never touched, would pass
    forever. This pins that the reference run actually etched something."""
    stored = np.load(GOLDEN)["phi"]
    assert float(np.min(stored)) < 0.0 < float(np.max(stored)), "the profile has an interface"
    assert float(np.std(stored)) > 1.0
