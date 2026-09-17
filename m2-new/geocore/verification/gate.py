"""The M2.4 gate run on case S03 in 3D, as library code. PRD §6, decision I5(a), findings I4/K1/K2.

No script, no generated report file: `build_s03`, `forward_plan` and `memory_model` return values, the
gate-tier tests in `tests/test_gate_m2_4.py` assert on them, and whatever needs a human decision goes to
OPEN_QUESTIONS.md.

**What is measured and what is modelled.** Persistent checkpoint storage -- one phi per segment -- is
exact by construction. Transient storage, the residuals alive while one segment is replayed, is `L * k`
from a k measured on smaller 3D grids (341-344 from 2,700 to 172,800 cells). The transient term
dominates, so on a CPU the peak is a MODEL. Only a GPU run turns it into a measurement, and that number
never enters the ledger of record.

On an H100:

    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_deterministic_ops=true \\
    uv run pytest -m gate -s

sampling `nvidia-smi --query-gpu=memory.used` alongside it until the peak is instrumented here.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import time
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from geocore.band import BandedRateField, occupancy
from geocore.checkpoint import peak_fields, segment_length_for
from geocore.config import CaseConfig, load_case
from geocore.initial import extrude
from geocore.materials import Layer, layer_fractions, masked_trench, with_selectivity
from geocore.schema import Grid
from geocore.solver import SolvePlan, assert_cfl, evolve
from geocore.velocity import directional

__all__ = ["build_s03", "forward_plan", "gradient_run", "memory_model", "device_peak_gb",
           "K_STEP_3D_GODUNOV", "K_REINIT_3D_GODUNOV", "MEMORY_BUDGET_GB"]

# Measured in 3D, Godunov (tests/test_3d.py pins the grid-size independence).
K_STEP_3D_GODUNOV = 344.0
K_REINIT_3D_GODUNOV = 141.0
MEMORY_BUDGET_GB = 40.0
CONFIG = pathlib.Path(__file__).resolve().parents[2] / "configs/dev/S03.yaml"


@dataclasses.dataclass(frozen=True)
class GateRun:
    case: CaseConfig
    grid: Grid
    phi0: Array
    fractions: Array
    params: dict[str, Any]


def build_s03() -> GateRun:
    """S03 with the mask modelled. The layout is derived, not chosen: the film top sits one buffer plus
    the full etch depth above the bottom, so the mask top lands one buffer below the top (PRD §5.1)."""
    case = load_case(CONFIG)
    grid = case.grid
    nz, _, nx = grid.shape
    dx = grid.spacing_nm
    film_top = 10 * dx + case.target_depth_nm
    profile_grid = Grid((nz, nx), dx, (False, True))
    profile = masked_trench(profile_grid, film_top, case.mask_thickness_nm, 100.0, 500.0, 500.0,
                            nx * dx / 2.0)
    film = next(m for m in case.materials if not m.is_mask)
    stack = layer_fractions(profile_grid, [Layer(film, None, film_top),
                                           Layer(case.mask, film_top, None)],
                            len(case.materials), case.w_mat_cells)
    fractions = jnp.stack([extrude(stack[..., m], grid) for m in range(stack.shape[-1])], axis=-1)
    params = {"v0_nm_per_s": jnp.float64(case.velocity.params["v0_nm_per_s"]),
              "p": jnp.float64(case.velocity.params["p"]),
              "material_rate": jnp.ones(len(case.materials))}
    return GateRun(case, grid, extrude(profile, grid), fractions, params)


def forward_plan(run: GateRun, n_steps: int | None = None, capacity: int | None = None) -> dict:
    """Run the forward solve and report what a GPU run needs to know: CFL, the WORST-step band
    occupancy that sizes K, and warm wall-clock."""
    case, grid = run.case, run.grid
    steps = n_steps or case.n_steps
    k = capacity or min(grid.n_cells, 3 * int(occupancy(run.phi0, grid, case.bands)))
    base = SolvePlan.from_case(case)
    plan = SolvePlan(grid, base.dt_s, steps, reinit_every=base.reinit_every,
                     n_reinit=base.n_reinit, spatial_scheme=base.spatial_scheme)
    field = BandedRateField(with_selectivity(directional, case.materials), grid, case.bands, k,
                            run.fractions)
    solve = jax.jit(lambda p: evolve(run.phi0, p, field, plan))

    start = time.perf_counter()
    _, diagnostics = solve(run.params)
    float(diagnostics.cfl[-1])
    first = time.perf_counter() - start
    start = time.perf_counter()
    _, diagnostics = solve(run.params)
    float(diagnostics.cfl[-1])
    warm = time.perf_counter() - start

    return {"steps": steps, "capacity": k, "max_cfl": assert_cfl(diagnostics.cfl, plan),
            "occupancy_first": int(diagnostics.occupancy[0]),
            "occupancy_worst": int(jnp.max(diagnostics.occupancy)),
            "first_call_s": first, "warm_s": warm, "ms_per_step": warm / steps * 1000.0}


def device_peak_gb() -> float | None:
    """Peak device memory actually used, in GB, or None on a backend that does not report it.

    `memory_stats()` is the number the 40 GB gate is about. On CPU it returns nothing, which is why the
    peak is MODELLED here and MEASURED only on the GPU. Requires
    `XLA_PYTHON_CLIENT_PREALLOCATE=false`, or the allocator grabs most of the card up front and the
    figure means nothing.
    """
    device = jax.local_devices()[0]
    stats = getattr(device, "memory_stats", lambda: None)()
    if not stats:
        return None
    peak = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use")
    return None if peak is None else peak / 1e9


def gradient_run(run: GateRun, n_steps: int | None = None, capacity: int | None = None,
                 segment: int | None = None) -> dict:
    """The checkpointed gradient on S03: warm timings, adjoint ratio and measured peak memory.

    This is the run the M2.4 gate is about, and it does not fit on a laptop: the model says about
    26 GB. On a CPU it will either swap or be killed.
    """
    case, grid = run.case, run.grid
    steps = n_steps or case.n_steps
    k = capacity or min(grid.n_cells, 3 * int(occupancy(run.phi0, grid, case.bands)))
    length = segment or segment_length_for(steps, K_STEP_3D_GODUNOV)
    base = SolvePlan.from_case(case)
    plan = SolvePlan(grid, base.dt_s, steps, reinit_every=base.reinit_every, n_reinit=base.n_reinit,
                     spatial_scheme=base.spatial_scheme, checkpoint_segment=length)
    field = BandedRateField(with_selectivity(directional, case.materials), grid, case.bands, k,
                            run.fractions)

    from geocore.functionals import solid_volume

    objective = jax.jit(lambda p: solid_volume(evolve(run.phi0, p, field, plan)[0], grid))
    gradient = jax.jit(jax.grad(objective))

    float(objective(run.params))                      # compile
    start = time.perf_counter(); float(objective(run.params)); forward = time.perf_counter() - start
    float(gradient(run.params)["p"])                  # compile
    start = time.perf_counter(); float(gradient(run.params)["p"]); adjoint = time.perf_counter() - start

    return {"steps": steps, "segment": length, "capacity": k, "forward_s": forward,
            "adjoint_s": adjoint, "adjoint_ratio": adjoint / forward,
            "peak_gb_measured": device_peak_gb()}


def memory_model(grid: Grid, n_steps: int, k_step: float = K_STEP_3D_GODUNOV,
                 k_reinit: float = K_REINIT_3D_GODUNOV) -> dict:
    """Peak memory for the checkpoint schedule derived from measured k (finding I4)."""
    field_gb = math.prod(grid.shape) * 8 / 1e9
    segment = segment_length_for(n_steps, k_step)
    persistent = math.ceil(n_steps / segment)
    transient = segment * k_step + k_reinit
    return {"field_gb": field_gb, "segment": segment, "persistent_fields": persistent,
            "transient_fields": transient, "peak_gb": (persistent + transient) * field_gb,
            "unchecked_gb": n_steps * k_step * field_gb,
            "sqrt_n_gb": peak_fields(n_steps, round(math.sqrt(n_steps)), k_step) * field_gb,
            "budget_gb": MEMORY_BUDGET_GB}


def main(argv: list[str] | None = None) -> int:
    """The M2.4 gate run. On an H100:

        XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_deterministic_ops=true \
        uv run python -m geocore.verification.gate --full

    Add `--v14` for the full-resolution 3D Taylor check (much longer). Copy the whole printed block
    back; it is the gate's evidence, and it does NOT go into the ledger of record (hardware-gated
    numbers never do).
    """
    import argparse

    parser = argparse.ArgumentParser(description="M2.4 gate run on case S03 in 3D")
    parser.add_argument("--full", action="store_true", help="also run the checkpointed gradient")
    parser.add_argument("--v14", action="store_true", help="also run full-resolution 3D V14")
    parser.add_argument("--steps", type=int, default=None, help="truncate the run (smoke test)")
    args = parser.parse_args(argv)

    run = build_s03()
    model = memory_model(run.grid, run.case.n_steps)
    devices = [str(d) for d in jax.devices()]
    print("=" * 78)
    print("M2.4 GATE — case S03, 3D")
    print("=" * 78)
    print(f"devices              {devices}")
    print(f"grid                 {run.grid.shape} at {run.grid.spacing_nm:g} nm, "
          f"{model['field_gb'] * 1000:.1f} MB per fp64 field")
    print(f"N (derived)          {run.case.n_steps}   mask {run.case.mask_thickness_nm:g} nm "
          f"(provisional)   scheme {SolvePlan.from_case(run.case).spatial_scheme}")

    forward = forward_plan(run, n_steps=args.steps)
    print()
    print("FORWARD")
    print(f"  max CFL            {forward['max_cfl']:.3f}   (ceiling 0.5)")
    print(f"  band occupancy     {forward['occupancy_first']} first step, "
          f"{forward['occupancy_worst']} worst of capacity {forward['capacity']}")
    print(f"  warm wall-clock    {forward['warm_s']:.1f} s for {forward['steps']} steps "
          f"({forward['ms_per_step']:.0f} ms/step)          GATE: < 60 s at N = {run.case.n_steps}")
    print(f"  first call         {forward['first_call_s']:.1f} s (compile + run, reported separately)")

    print()
    print("MEMORY MODEL (from k measured in 3D on small grids)")
    print(f"  L from measured k  {model['segment']}   "
          f"(sqrt(N) would be {round(model['budget_gb'] and (run.case.n_steps ** 0.5)):.0f})")
    print(f"  persistent         {model['persistent_fields']} fields (exact by construction)")
    print(f"  transient          {model['transient_fields']:.0f} fields (MODELLED)")
    print(f"  modelled peak      {model['peak_gb']:.1f} GB of {model['budget_gb']:g} GB budget")
    print(f"  unchecked would be {model['unchecked_gb']:,.0f} GB")

    if args.full:
        gradient = gradient_run(run, n_steps=args.steps)
        print()
        print("GRADIENT (checkpointed)")
        print(f"  segment L          {gradient['segment']}")
        print(f"  warm forward       {gradient['forward_s']:.2f} s")
        print(f"  warm gradient      {gradient['adjoint_s']:.2f} s")
        print(f"  adjoint ratio      {gradient['adjoint_ratio']:.2f}x"
              f"                          GATE: <= 4x")
        measured = gradient["peak_gb_measured"]
        print(f"  peak memory        "
              + (f"{measured:.1f} GB                        GATE: < 40 GB"
                 if measured is not None else
                 "not reported by this backend (CPU); run on the GPU"))

    if args.v14:
        from geocore.functionals import solid_volume
        from geocore.verification.gradcheck import reverse_gradient, taylor_test
        from geocore.verification.params import scales_for

        base = SolvePlan.from_case(run.case)
        plan = SolvePlan(run.grid, base.dt_s, args.steps or run.case.n_steps,
                         reinit_every=base.reinit_every, n_reinit=base.n_reinit,
                         checkpoint_segment=segment_length_for(run.case.n_steps, K_STEP_3D_GODUNOV))
        field = BandedRateField(with_selectivity(directional, run.case.materials), run.grid,
                                run.case.bands, 3 * int(occupancy(run.phi0, run.grid,
                                                                  run.case.bands)), run.fractions)
        objective = jax.jit(lambda p: solid_volume(evolve(run.phi0, p, field, plan)[0], run.grid))
        result = taylor_test(objective, run.params, reverse_gradient(objective, run.params),
                             scales_for(run.params), jax.random.PRNGKey(14))
        print()
        print("FULL-RESOLUTION 3D V14")
        print(f"  {result.message}")

    print()
    print("Paste this whole block back. It is gate evidence and does not enter the ledger of record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
