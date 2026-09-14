"""M2.4 gate measurements on the real case, and the two that need an H100 (PRD §6, decision X13).

Run modes:

    uv run python scripts/gate_m2_4.py --forward-only    # CPU-feasible; validates the run plan
    uv run python scripts/gate_m2_4.py                   # full gate; needs ~40 GB of device memory

`--forward-only` exists so that every part of this that *can* be checked without paying for an
instance is checked before one is rented. It runs the real gate geometry (S03 in 3D, 270×100×100,
N = 625) through a full forward solve, which fits in 8 GB, and reports peak band occupancy and CFL.
The gradient does not fit, and neither does the memory profile — those are what the instance is for.

**Peak memory is read from the device allocator**, not modelled. That is the whole point of the
instance: `checkpoint.peak_field_equivalents` can only model the transient replay term on CPU (see
its docstring), and the 40 GB gate is mostly that term. `memory_stats()["peak_bytes_in_use"]` is a
measurement of the thing the gate names.

Output is a dated markdown report under `reports/`. It is **not** written into the verification
ledger of record: per the repo rule, CI on a clean checkout is the only writer of that, and this is
a hand-driven run on rented hardware. The report is cited by `verification.md` as an
externally-produced measurement instead, with the hardware and driver recorded, so the provenance of
these numbers stays explicit rather than blurring into the normal ledger (owner decision,
2026-09-14).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import m2  # noqa: E402,F401  (enables fp64)
from m2 import functionals, initial, solver  # noqa: E402
from m2.band import estimate_capacity  # noqa: E402
from m2.checkpoint import optimal_segment, peak_field_equivalents, persistent_residuals  # noqa: E402
from m2.config import (  # noqa: E402
    BandConfig,
    CaseSpec,
    ExtensionConfig,
    M2Config,
    Parameter,
    VelocityConfig,
    parameter_tree,
)
from m2.schema import Grid, Material  # noqa: E402
from m2.verification.gradcheck import reverse_gradient, taylor_test  # noqa: E402

# Coupon case S03 (decision §2): CD 500 nm, pitch 1000 nm, depth 2500 nm, dx 10 nm.
SHAPE_3D = (270, 100, 100)
DX = 10.0
SURFACE_NM = 2600.0
CD_NM = 500.0
INITIAL_DEPTH_NM = 100.0
TARGET_DEPTH_NM = 2500.0
RATE_NM_S = 5.83  # configs/dev nominal, provisional: may not certify coupon agreement
MATERIALS = (Material("film", 0, False),)
BUDGET_GB = 40.0
RATIO_TARGET = 4.0


def masked_directional(request, params):
    """Directional etch through a **test-only** lateral mask window.

    Without this the gate case is not a trench. Decision §2 leaves the mask unmodelled, so a
    directional etch removes the flat field at exactly the rate it removes the trench floor: the
    whole surface translates downward and the feature does not deepen. Measured on this case over
    625 steps, the surface fell 2480 nm while the trench went from 100 nm deep to **60 nm** — it
    eroded (finding L2, and OPEN_QUESTIONS H2, which anticipated this).

    A memory and cost gate measured on that geometry would be measured on a translating plane, and
    its peak band occupancy would sit at the initial 36,000 rather than the 180,000 a 2500 nm trench
    reaches. The capacity and K figures the gate is meant to establish would be optimistic.

    So the mask is emulated here as a static lateral window on the rate — the same test-only device
    the half-day diagnostic used. It is **not** production: H2 is resolved as "the mask becomes real
    geometry at M2.6", and this function should be deleted when that lands. It is static in x, so it
    carries no parameter dependence and cannot move under differentiation.
    """
    from m2.velocity import directional

    centre = SHAPE_3D[1] * DX / 2.0
    open_here = jnp.abs(request.positions[..., 1] - centre) < CD_NM / 2.0
    return directional(request, params) * open_here.astype(jnp.float64)


def build_config(scheme: str, shape=SHAPE_3D) -> M2Config:
    grid = Grid(shape, DX, tuple([False] + [True] * (len(shape) - 1)))
    travel = TARGET_DEPTH_NM - INITIAL_DEPTH_NM
    n_steps = int(np.ceil(travel / (0.4 * DX)))
    return M2Config(
        case=CaseSpec("S03", 3, "calibrate", TARGET_DEPTH_NM, CD_NM, 1000.0, (1.0,)),
        dimension=len(shape), grid=grid, n_steps=n_steps, cfl_target=0.4, seed=0,
        materials=MATERIALS, extension=ExtensionConfig("closest_point", None),
        velocity=VelocityConfig("directional", {
            "v_iso": Parameter(0.58, 1.0), "v_dir": Parameter(5.25, 5.25),
            "p": Parameter(2.0, 2.0)}),
        bands=BandConfig(8.0, 2.0, 1.5, None), spatial_scheme=scheme)


def device_peak_bytes() -> int | None:
    """Peak device allocation, or None on a backend that does not report it (CPU)."""
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:  # noqa: BLE001  a backend without stats is not an error here
        return None
    return None if stats is None else stats.get("peak_bytes_in_use")


def fastest(fn, argument, reps=3) -> float:
    fn(argument)  # warm: compile time is excluded and reported separately
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(fn(argument))
        best = min(best, time.perf_counter() - start)
    return best


def _depths(phi, grid) -> tuple[float, float]:
    """Surface height under the mask and at the trench centre, in nm, by first φ > 0 crossing."""
    array = np.asarray(phi)
    field = array[:, 5, 50] if grid.ndim == 3 else array[:, 5]
    centre = array[:, 50, 50] if grid.ndim == 3 else array[:, 50]
    top = lambda column: grid.spacing_nm * float(np.flatnonzero(column > 0)[0])  # noqa: E731
    return top(field), top(centre)


def run_forward(scheme: str, masked: bool = True) -> dict:
    """Full forward solve on the gate case. Fits in 8 GB; validates capacity, CFL and geometry."""
    cfg = build_config(scheme)
    capacity = estimate_capacity(cfg.grid, cfg.bands)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi0 = initial.trench(cfg.grid, SURFACE_NM, CD_NM, INITIAL_DEPTH_NM)
    params = parameter_tree(cfg)[0]
    model = masked_directional if masked else None

    start = time.perf_counter()
    result = solver.solve(cfg, phi0, material, params, capacity=capacity, model=model)
    elapsed = time.perf_counter() - start

    phi = np.asarray(result.phi)
    field0, centre0 = _depths(phi0, cfg.grid)
    field1, centre1 = _depths(result.phi, cfg.grid)
    return {
        "masked": masked,
        "initial_depth_nm": field0 - centre0, "final_depth_nm": field1 - centre1,
        "surface_fell_nm": field0 - field1,
        "scheme": scheme, "grid": list(cfg.grid.shape), "n_steps": cfg.n_steps,
        "dt_s": cfg.dt_s, "max_cfl": result.max_cfl, "cfl_bound": 0.5,
        "capacity": capacity, "peak_occupancy": result.peak_occupancy,
        "capacity_headroom": 1.0 - result.peak_occupancy / capacity,
        "wall_clock_s": elapsed, "field_mb": phi.nbytes / 1e6,
        "phi_finite": bool(np.all(np.isfinite(phi))),
        "phi_min": float(phi.min()), "phi_max": float(phi.max()),
    }


def run_gradient(scheme: str, levels: int, segment: int | None) -> dict:
    """The gate's gradient measurements. Needs device memory this laptop does not have."""
    cfg = build_config(scheme)
    capacity = estimate_capacity(cfg.grid, cfg.bands)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi0 = initial.trench(cfg.grid, SURFACE_NM, CD_NM, INITIAL_DEPTH_NM)
    params, scales = parameter_tree(cfg)

    def J(p):
        return functionals.solid_volume(
            solver.final_phi(cfg, phi0, material, p, capacity=capacity,
                             levels=levels, segment=segment), cfg.grid)

    field_bytes = int(np.prod(cfg.grid.shape)) * 8
    persistent, _ = persistent_residuals(J, params, field_bytes)

    forward = fastest(jax.jit(J), params)
    before = device_peak_bytes()
    adjoint = fastest(jax.jit(jax.value_and_grad(J)), params)
    peak = device_peak_bytes()

    grad = reverse_gradient(J, params)
    taylor = taylor_test(J, params, grad, scales=scales, key=jax.random.PRNGKey(2026),
                         n_steps=cfg.n_steps)

    return {
        "scheme": scheme, "levels": levels, "segment": segment,
        "forward_s": forward, "adjoint_s": adjoint, "adjoint_ratio": adjoint / forward,
        "ratio_target": RATIO_TARGET, "ratio_met": bool(adjoint / forward <= RATIO_TARGET),
        "peak_device_bytes": peak, "peak_device_gb": None if peak is None else peak / 1e9,
        "peak_before_adjoint_bytes": before,
        "budget_gb": BUDGET_GB,
        "budget_met": None if peak is None else bool(peak / 1e9 <= BUDGET_GB),
        "persistent_field_equivalents": persistent,
        "gradient": {k: float(v) for k, v in grad.items()},
        "v14_passed": taylor.passed, "v14_message": taylor.message,
        "v14_slope_min": taylor.measured["slope_min"],
        "v14_slope_max": taylor.measured["slope_max"],
    }


def environment() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=False).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=False).stdout.strip())
    except Exception:  # noqa: BLE001
        sha, dirty = "unknown", True
    device = jax.local_devices()[0]
    return {
        "git_sha": sha, "working_tree_dirty": dirty,
        "jax_version": jax.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "backend": jax.default_backend(),
        "device_kind": device.device_kind, "device_count": jax.device_count(),
        "x64_enabled": jax.config.jax_enable_x64,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def render(env: dict, forwards: list[dict], gradients: list[dict], forward_only: bool) -> str:
    lines = [
        "# M2.4 gate measurements — case S03 in 3D",
        "",
        f"Generated {env['timestamp_utc']} by `scripts/gate_m2_4.py`"
        f"{' (--forward-only)' if forward_only else ''}.",
        "",
        "**Provenance.** This is a hand-driven run, not CI. It is therefore *not* written into the",
        "verification ledger of record — CI on a clean checkout is the only writer of that. This",
        "report is the externally-produced measurement `verification.md` cites instead, so the",
        "hardware-specific numbers keep their provenance (owner decision, 2026-09-14).",
        "",
        "## Environment",
        "",
        "| key | value |",
        "|---|---|",
    ]
    lines += [f"| {k} | `{v}` |" for k, v in env.items()]

    lines += ["", "## Forward solve (the full gate geometry)", "",
              "| scheme | mask | N | max CFL | capacity | peak occ. | headroom | depth nm | wall |",
              "|---|:---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in forwards:
        lines.append(
            f"| {row['scheme']} | {'emulated' if row['masked'] else 'none'} | {row['n_steps']} | "
            f"{row['max_cfl']:.3f} | {row['capacity']:,} | {row['peak_occupancy']:,} | "
            f"{row['capacity_headroom']:.1%} | {row['initial_depth_nm']:.0f}→"
            f"{row['final_depth_nm']:.0f} | {row['wall_clock_s']:.0f} s |")
    lines += ["",
              "The mask is a **test-only** static lateral window on the rate (finding L2). Decision",
              "§2 leaves the mask unmodelled, so without it a directional etch removes the field at",
              "the same rate as the trench floor and the feature erodes instead of deepening. H2 is",
              "resolved as 'the mask becomes real geometry at M2.6'; this emulation goes away then."]

    if forward_only:
        lines += ["", "## Gradient measurements", "",
                  "**Not run.** `--forward-only`. The gate's two remaining items — peak device",
                  "memory under 40 GB and warm adjoint ratio at or under 4x — require an H100;",
                  "see decision X13. Projections from the measured residual factor are in",
                  "`OPEN_QUESTIONS.md` finding K3 and are explicitly *not* a substitute."]
    else:
        lines += ["", "## Gradient, memory and cost", "",
                  "| scheme | levels | L | adjoint ratio | <=4x | peak device | <40 GB | V14 |",
                  "|---|---:|---:|---:|:---:|---:|:---:|:---:|"]
        for row in gradients:
            peak = "n/a" if row["peak_device_gb"] is None else f"{row['peak_device_gb']:.1f} GB"
            lines.append(
                f"| {row['scheme']} | {row['levels']} | {row['segment']} | "
                f"{row['adjoint_ratio']:.2f}x | {'yes' if row['ratio_met'] else '**no**'} | {peak} | "
                f"{'yes' if row['budget_met'] else '**no**'} | "
                f"{'pass' if row['v14_passed'] else '**FAIL**'} |")

    lines += ["", "## Raw", "", "```json",
              json.dumps({"environment": env, "forward": forwards, "gradient": gradients},
                         indent=2, default=str), "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-only", action="store_true",
                        help="skip everything needing more memory than a laptop has")
    parser.add_argument("--schemes", default="godunov,weno5")
    parser.add_argument("--unmasked", action="store_true",
                        help="no mask emulation: the surface translates and the trench erodes (L2)")
    parser.add_argument("--both-geometries", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    env = environment()
    print(f"backend={env['backend']} device={env['device_kind']} x64={env['x64_enabled']}")

    forwards, gradients = [], []
    for scheme in schemes:
        for masked in (True, False) if args.both_geometries else (not args.unmasked,):
            print(f"[forward] {scheme} masked={masked} ...", flush=True)
            row = run_forward(scheme, masked=masked)
            print(f"          CFL {row['max_cfl']:.3f}, occupancy {row['peak_occupancy']:,}"
                  f"/{row['capacity']:,}, depth {row['initial_depth_nm']:.0f} -> "
                  f"{row['final_depth_nm']:.0f} nm, {row['wall_clock_s']:.1f} s", flush=True)
            forwards.append(row)

    if not args.forward_only:
        for scheme in schemes:
            k = 349.2 if scheme == "godunov" else 1049.5
            cfg = build_config(scheme)
            for levels, segment in ((2, optimal_segment(cfg.n_steps, k)), (3, 25)):
                print(f"[gradient] {scheme} levels={levels} L={segment} ...", flush=True)
                row = run_gradient(scheme, levels, segment)
                row["projected_peak_gb"] = peak_field_equivalents(
                    cfg.n_steps, k, levels=levels, segment=segment) * 21.6 / 1000.0
                print(f"           ratio {row['adjoint_ratio']:.2f}x, peak "
                      f"{row['peak_device_gb']}, V14 {row['v14_passed']}", flush=True)
                gradients.append(row)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    suffix = "-forward-only" if args.forward_only else ""
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "reports" / f"m2_4_gate-{stamp}{suffix}.md")
    out.write_text(render(env, forwards, gradients, args.forward_only))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
