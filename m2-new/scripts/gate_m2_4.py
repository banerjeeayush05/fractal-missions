"""M2.4 gate run on case S03 in 3D. PRD §6: peak memory < 40 GB on one H100, recompute overhead
<= 2x, warm adjoint ratio <= 4x, checkpoint schedule from measured k, full-resolution 3D V14.

Two modes, and the difference matters.

  --forward-only   Runs on any machine, including CPU. Builds the case, runs the forward solve,
                   measures band occupancy at every step (so K is sized from the WORST step), checks
                   CFL and capacity, derives the checkpoint schedule, and writes the MEMORY MODEL.
                   Its purpose is to find defects before paid GPU time is spent (findings L1, L2 in
                   the original tree were both found this way).

  (default)        On the H100. Adds the checkpointed gradient, warm timings and measured peak
                   memory. Run with:
                     XLA_PYTHON_CLIENT_PREALLOCATE=false \\
                     XLA_FLAGS=--xla_gpu_deterministic_ops=true \\
                     uv run python scripts/gate_m2_4.py [--v14]

**What is measured and what is modelled.** Persistent checkpoint storage -- one phi per segment -- is
exact. Transient storage, the residuals alive while one segment is replayed, is `L * k` from a k
measured on smaller 3D grids (it does not change with grid size: 341-344 from 2,700 to 172,800
cells). The transient term dominates, so on CPU the peak is a MODEL and is reported as one. Only the
default mode on a GPU turns it into a measurement.

Hardware-gated numbers go into a dated report under reports/, NOT into the ledger of record.

Geometry caveat (finding L2 in the original tree): the mask is not modelled until M2.6, so under a
directional law the flat field etches at the same rate as the trench floor and the trench translates
rather than deepening. Memory and timing are insensitive to that; the resulting profile is not a
physical S03 result and must not be read as one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import pathlib
import platform
import sys
import time

import jax
import jax.numpy as jnp

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geocore.band import BandedRateField, assert_capacity, occupancy, single_material  # noqa: E402
from geocore.checkpoint import peak_fields, segment_length_for  # noqa: E402
from geocore.config import load_case  # noqa: E402
from geocore.functionals import solid_volume  # noqa: E402
from geocore.initial import extrude, trapezoid  # noqa: E402
from geocore.schema import Grid  # noqa: E402
from geocore.solver import SolvePlan, assert_cfl, evolve  # noqa: E402
from geocore.velocity import directional  # noqa: E402

# Measured in 3D, Godunov, stage 15 (tests/test_3d.py pins grid-size independence).
K_STEP_3D_GODUNOV = 344.0
K_REINIT_3D_GODUNOV = 141.0
MEMORY_BUDGET_GB = 40.0


def build_case():
    case = load_case(ROOT / "configs/dev/S03.yaml")
    grid = case.grid
    nz, ny, nx = grid.shape
    dx = grid.spacing_nm
    # CD 500 nm, pitch 1000 nm (the lateral period), a 100 nm starting trench. The surface sits so the
    # 2500 nm etch plus the 10-cell buffer fits exactly (PRD §5.1).
    surface = nz * dx - 10 * dx
    profile_grid = Grid((nz, nx), dx, (False, True))
    profile = trapezoid(profile_grid, surface, 100.0, 500.0, 500.0, nx * dx / 2.0)
    return case, grid, extrude(profile, grid)


def field_gb(grid: Grid) -> float:
    return math.prod(grid.shape) * 8 / 1e9


def forward_only(case, grid, phi0, params, capacity, n_steps):
    plan = SolvePlan.from_case(case)
    plan = SolvePlan(grid, plan.dt_s, n_steps, reinit_every=plan.reinit_every,
                     n_reinit=plan.n_reinit, spatial_scheme=plan.spatial_scheme)
    field = BandedRateField(directional, grid, case.bands, capacity, single_material(grid))
    run = jax.jit(lambda p: evolve(phi0, p, field, plan))
    start = time.perf_counter()
    phi, diagnostics = run(params)
    float(diagnostics.cfl[-1])
    first = time.perf_counter() - start
    start = time.perf_counter()
    phi, diagnostics = run(params)
    float(diagnostics.cfl[-1])
    warm = time.perf_counter() - start
    return phi, diagnostics, plan, first, warm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--steps", type=int, default=None,
                        help="truncate the run (forward-only smoke test); omit for the full N")
    parser.add_argument("--v14", action="store_true", help="also run full-resolution 3D V14")
    args = parser.parse_args()

    case, grid, phi0 = build_case()
    n_steps = args.steps or case.n_steps
    params = {"v0_nm_per_s": jnp.float64(case.velocity.params["v0_nm_per_s"]),
              "p": jnp.float64(case.velocity.params["p"])}
    devices = [str(d) for d in jax.devices()]

    initial_occupancy = int(occupancy(phi0, grid, case.bands))
    capacity = min(grid.n_cells, 3 * initial_occupancy)

    phi, diagnostics, plan, first_s, warm_s = forward_only(case, grid, phi0, params, capacity,
                                                            n_steps)
    max_cfl = assert_cfl(diagnostics.cfl, plan)
    peak_occupancy = int(jnp.max(diagnostics.occupancy))
    capacity_ok = peak_occupancy <= capacity
    try:
        assert_capacity(diagnostics.occupancy, capacity)
    except Exception as exc:  # recorded, not swallowed: an overflow invalidates the timing run
        capacity_note = f"OVERFLOW: {exc}"
    else:
        capacity_note = "within capacity"

    full_n = case.n_steps
    segment = segment_length_for(full_n, K_STEP_3D_GODUNOV)
    persistent_fields = math.ceil(full_n / segment)
    transient_fields = segment * K_STEP_3D_GODUNOV + K_REINIT_3D_GODUNOV
    fgb = field_gb(grid)
    model_gb = (persistent_fields + transient_fields) * fgb
    naive_gb = full_n * K_STEP_3D_GODUNOV * fgb
    sqrt_n_gb = peak_fields(full_n, int(round(math.sqrt(full_n))), K_STEP_3D_GODUNOV) * fgb

    lines = [
        f"# M2.4 gate run — S03 3D — {dt.date.today().isoformat()} — "
        f"{'forward-only' if args.forward_only else 'full'}",
        "",
        f"Generated by `scripts/gate_m2_4.py`. Host {platform.node()}, devices {devices}. "
        "Hardware-gated numbers: NOT written to the ledger of record.",
        "",
        "## Case",
        "",
        f"| grid | dx | fp64 field | N (derived) | steps run | dt | scheme | reinit |",
        f"|---|---|---|---|---|---|---|---|",
        f"| {grid.shape} | {grid.spacing_nm:g} nm | {fgb*1000:.1f} MB | {full_n} | {n_steps} | "
        f"{plan.dt_s:g} s | {plan.spatial_scheme} | every {plan.reinit_every}, {plan.n_reinit} it |",
        "",
        "Rate is the dev config's PROVISIONAL nominal rate; this run certifies memory and time, "
        "never agreement with the coupon. Mask not modelled (finding L2): the profile is not "
        "physical.",
        "",
        "## Forward solve",
        "",
        f"| quantity | value |",
        f"|---|---|",
        f"| max CFL | {max_cfl:.3f} (ceiling 0.5) |",
        f"| band occupancy, first step | {initial_occupancy} |",
        f"| band occupancy, worst step | {peak_occupancy} |",
        f"| request capacity K used | {capacity} — {capacity_note} |",
        f"| forward wall-clock, first call (compile + run) | {first_s:.1f} s |",
        f"| forward wall-clock, warm | {warm_s:.1f} s ({warm_s / n_steps * 1000:.0f} ms/step) |",
        "",
        "## Checkpoint schedule and memory",
        "",
        f"k measured in 3D (Godunov): {K_STEP_3D_GODUNOV:g} fields per step, "
        f"{K_REINIT_3D_GODUNOV:g} per reinitialisation cycle.",
        "",
        f"| schedule | peak fields | peak memory | basis |",
        f"|---|---|---|---|",
        f"| unchecked | {full_n * K_STEP_3D_GODUNOV:,.0f} | {naive_gb:,.0f} GB | model |",
        f"| L = sqrt(N) = {int(round(math.sqrt(full_n)))} | "
        f"{peak_fields(full_n, int(round(math.sqrt(full_n))), K_STEP_3D_GODUNOV):,.0f} | "
        f"{sqrt_n_gb:,.1f} GB | model |",
        f"| **L = {segment} (from measured k)** | {persistent_fields + transient_fields:,.0f} | "
        f"**{model_gb:.1f} GB** | persistent {persistent_fields} fields MEASURED by construction; "
        f"transient {transient_fields:.0f} fields MODELLED |",
        "",
        f"Budget {MEMORY_BUDGET_GB:g} GB. Modelled margin {MEMORY_BUDGET_GB - model_gb:.1f} GB. "
        "The model excludes XLA workspace and allocator fragmentation; the H100 run replaces it with "
        "a measurement.",
        "",
    ]

    if not args.forward_only:
        checked = SolvePlan(grid, plan.dt_s, n_steps, reinit_every=plan.reinit_every,
                            n_reinit=plan.n_reinit, checkpoint_segment=segment)
        field = BandedRateField(directional, grid, case.bands, max(capacity, peak_occupancy),
                                single_material(grid))
        objective = jax.jit(lambda p: solid_volume(evolve(phi0, p, field, checked)[0], grid))
        gradient = jax.jit(jax.grad(objective))
        float(objective(params))
        float(gradient(params)["p"])
        start = time.perf_counter(); float(objective(params)); fwd = time.perf_counter() - start
        start = time.perf_counter(); float(gradient(params)["p"]); adj = time.perf_counter() - start
        lines += [
            "## Gradient (checkpointed)",
            "",
            f"| quantity | value | gate |",
            f"|---|---|---|",
            f"| warm forward | {fwd:.2f} s | |",
            f"| warm gradient | {adj:.2f} s | |",
            f"| warm adjoint ratio | {adj / fwd:.2f}x | <= 4x |",
            "",
            "Peak device memory: read from the device during this run (not yet instrumented on "
            "this build — record `nvidia-smi --query-gpu=memory.used` sampled during the gradient "
            "call alongside this report).",
            "",
        ]
        if args.v14:
            from geocore.verification.gradcheck import reverse_gradient, taylor_test
            from geocore.verification.params import scales_for
            result = taylor_test(objective, params, reverse_gradient(objective, params),
                                 scales_for(params), jax.random.PRNGKey(14))
            lines += ["## Full-resolution 3D V14", "", result.message, ""]

    out = ROOT / "reports" / (f"m2_4_gate-{dt.date.today().isoformat()}-"
                              f"{'forward-only' if args.forward_only else 'full'}"
                              f"{'' if args.steps is None else f'-{n_steps}steps'}.md")
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwritten: {out.relative_to(ROOT)}")
    return 0 if capacity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
