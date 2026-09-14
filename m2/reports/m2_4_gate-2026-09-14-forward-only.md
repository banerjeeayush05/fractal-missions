# M2.4 gate measurements — case S03 in 3D

Generated 2026-09-14T17:34:15.283577+00:00 by `scripts/gate_m2_4.py` (--forward-only).

**Provenance.** This is a hand-driven run, not CI. It is therefore *not* written into the
verification ledger of record — CI on a clean checkout is the only writer of that. This
report is the externally-produced measurement `verification.md` cites instead, so the
hardware-specific numbers keep their provenance (owner decision, 2026-09-14).

## Environment

| key | value |
|---|---|
| git_sha | `cfc1cd348a8a78cda14472ddff1f099ca8cbfd1d` |
| working_tree_dirty | `True` |
| jax_version | `0.11.1` |
| python | `3.13.5` |
| platform | `macOS-26.3.1-arm64-arm-64bit-Mach-O` |
| backend | `cpu` |
| device_kind | `cpu` |
| device_count | `1` |
| x64_enabled | `True` |
| timestamp_utc | `2026-09-14T17:34:15.283577+00:00` |

## Forward solve (the full gate geometry)

| scheme | mask | N | max CFL | capacity | peak occ. | headroom | depth nm | wall |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| godunov | emulated | 600 | 0.417 | 444,000 | 155,400 | 65.0% | 100→2290 | 192 s |
| godunov | none | 600 | 0.417 | 444,000 | 36,000 | 91.9% | 100→60 | 192 s |

The mask is a **test-only** static lateral window on the rate (finding L2). Decision
§2 leaves the mask unmodelled, so without it a directional etch removes the field at
the same rate as the trench floor and the feature erodes instead of deepening. H2 is
resolved as 'the mask becomes real geometry at M2.6'; this emulation goes away then.

## Gradient measurements

**Not run.** `--forward-only`. The gate's two remaining items — peak device
memory under 40 GB and warm adjoint ratio at or under 4x — require an H100;
see decision X13. Projections from the measured residual factor are in
`OPEN_QUESTIONS.md` finding K3 and are explicitly *not* a substitute.

## Raw

```json
{
  "environment": {
    "git_sha": "cfc1cd348a8a78cda14472ddff1f099ca8cbfd1d",
    "working_tree_dirty": true,
    "jax_version": "0.11.1",
    "python": "3.13.5",
    "platform": "macOS-26.3.1-arm64-arm-64bit-Mach-O",
    "backend": "cpu",
    "device_kind": "cpu",
    "device_count": 1,
    "x64_enabled": true,
    "timestamp_utc": "2026-09-14T17:34:15.283577+00:00"
  },
  "forward": [
    {
      "masked": true,
      "initial_depth_nm": 100.0,
      "final_depth_nm": 2290.0,
      "surface_fell_nm": 0.0,
      "scheme": "godunov",
      "grid": [
        270,
        100,
        100
      ],
      "n_steps": 600,
      "dt_s": 0.714694110920526,
      "max_cfl": 0.4166666666666669,
      "cfl_bound": 0.5,
      "capacity": 444000,
      "peak_occupancy": 155400,
      "capacity_headroom": 0.65,
      "wall_clock_s": 191.73898629203904,
      "field_mb": 21.6,
      "phi_finite": true,
      "phi_min": -1142.2455543569506,
      "phi_max": 272.01966053574364
    },
    {
      "masked": false,
      "initial_depth_nm": 100.0,
      "final_depth_nm": 60.0,
      "surface_fell_nm": 2480.0,
      "scheme": "godunov",
      "grid": [
        270,
        100,
        100
      ],
      "n_steps": 600,
      "dt_s": 0.714694110920526,
      "max_cfl": 0.4166666666666669,
      "cfl_bound": 0.5,
      "capacity": 444000,
      "peak_occupancy": 36000,
      "capacity_headroom": 0.9189189189189189,
      "wall_clock_s": 192.05733354197582,
      "field_mb": 21.6,
      "phi_finite": true,
      "phi_min": -548.6279909568041,
      "phi_max": 1461.166397390938
    }
  ],
  "gradient": []
}
```
