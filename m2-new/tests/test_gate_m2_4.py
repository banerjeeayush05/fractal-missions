"""M2.4 gate, gate tier. What can run anywhere runs here; the rest needs an H100.

Validates the plan before GPU time is paid for: geometry builds, CFL holds, the request capacity covers
the WORST step (not the first), and the checkpoint schedule comes from measured k. The original tree
found two defects this way that would each have wasted a paid session (findings L1, L2).
"""

import pytest

from geocore.verification.gate import (
    MEMORY_BUDGET_GB, build_s03, forward_plan, memory_model,
)

pytestmark = pytest.mark.gate


@pytest.fixture(scope="module")
def run():
    return build_s03()


def test_the_s03_geometry_builds_and_the_first_steps_are_stable(run):
    """Ten steps is enough to catch a bad capacity, a CFL violation or a broken geometry."""
    result = forward_plan(run, n_steps=10)
    assert result["max_cfl"] <= 0.5
    assert result["occupancy_worst"] <= result["capacity"]
    assert result["occupancy_worst"] > result["occupancy_first"], \
        "the interface lengthens as the trench deepens; K must be sized from the worst step"
    print(f"\nS03 forward, 10 steps: max CFL {result['max_cfl']:.3f}, occupancy "
          f"{result['occupancy_first']} -> {result['occupancy_worst']} of capacity "
          f"{result['capacity']}, {result['ms_per_step']:.0f} ms/step")


def test_the_checkpoint_schedule_fits_the_budget(run):
    model = memory_model(run.grid, run.case.n_steps)
    assert model["segment"] == 1
    assert model["peak_gb"] < MEMORY_BUDGET_GB
    assert model["peak_gb"] < model["sqrt_n_gb"] / 5, "L from measured k must beat L = sqrt(N)"
    print(f"\nS03 memory model: L = {model['segment']}, {model['persistent_fields']} persistent "
          f"(exact) + {model['transient_fields']:.0f} transient (MODELLED) fields x "
          f"{model['field_gb']*1000:.1f} MB = {model['peak_gb']:.1f} GB of {MEMORY_BUDGET_GB:g} GB; "
          f"unchecked {model['unchecked_gb']:,.0f} GB, L=sqrt(N) {model['sqrt_n_gb']:.0f} GB")
