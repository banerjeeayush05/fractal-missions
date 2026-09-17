# OPEN_QUESTIONS — M2 (m2-new build)

Ambiguities, deviations and findings, with the evidence behind each. Never resolve an ambiguity by
guessing; write it here instead, classified:

- **A** — PRD error (arithmetic, cross-reference, or a statement shown to be false)
- **B** — placeholder, carrying `provisional: true` and the mechanical rule that produced it
- **C** — gap blocking a later milestone
- **D** — minor

Separately, per `WORKING_AGREEMENT.md` §2, a change is **Class A** (adjusts a requirement) or
**Class B** (changes or compromises the project; stop and ask). Findings are numbered by build stage.

Findings from stages 1–11 are recorded in the docstrings of the modules they concern
(`pyproject.toml` S1.1, `geocore/schema.py` S3.1–S3.4, `geocore/config.py` S4.1–S4.3).

---

## S12.1 — Reinitialisation drift accumulates, and V12 cannot see it

**Status:** KNOWN DISCREPANCY — owner accepted 2026-09-16, option c. Re-measure V1, V9 and V14a
after WENO5 (stage 14) and revisit with those numbers. Decision record:
`decisions/2026-09-16-s12-1-and-s13-1.md`.
**Additional question: decided 2026-09-16** — check V12a added (see S12.3).
**Classification:** C (M2.2 gate accepted with this exception). **Working-agreement class: B** — it changes what a
gate proves.
**Gate affected:** M2.2. V1 (full travel) and V9 fail; both are recorded `xfail(strict=True)`.

### The issue

Reinitialisation restores φ to a signed distance function every 5 steps. Its one obligation is to
do that **without moving the zero level set** (PRD §5.3, §8.3).

Check V12 tests that obligation for **one** reinitialisation cycle, on an exact distance function,
with no advection. It passes. A real run applies a cycle every `reinit_every` steps — 40 cycles in
V1's 200 steps — and the small motion of each cycle **accumulates**. Nothing in the check registry
bounds the accumulated motion directly. V1 at full travel is the first check that sees it, and it
fails.

In plain terms: each reinitialisation cycle nudges the surface inward by an amount too small for
V12 to fail. Forty of them shrink an etched disk by about 3 %, and they shrink it more along the grid
diagonals than along the axes, so a round feature also turns slightly square.

### Why it matters

PRD §8.3, on V12: reinitialisation that shifts the interface "produces a systematic etch-rate bias
that looks like physics, calibrates away into the closure parameters, and then fails to transfer."
That is the failure mode here. Specifically:

- **The bias is far above metrology.** 3 % of a 150 nm radius is about 5 nm. PRD H1 puts the
  metrology floor at 0.3 nm for CD and 0.1 nm for depth.
- **Gradient checks cannot catch it.** V14, V15 and V16 verify the derivative of the model against
  the model. A model with drift is still smooth, so its gradient is correct *for the wrong model*.
  Only forward checks with exact answers (V1, V9) see it.
- **Calibration will absorb it.** A fit will lower an etch rate or yield by about 3 % to compensate,
  and that parameter will then fail on any case with a different depth or number of cycles.
- **It depends on grid spacing**, so parameters calibrated at one resolution do not transfer to
  another.
- **It is anisotropic.** Error is concentrated on diagonal surfaces, so sidewall angle and corners —
  direct product metrics — are affected most, and a grid artefact can be mistaken for physical
  anisotropy.
- **It grows with run length.** The S03 coupon case runs 625 steps (125 cycles), three times V1.

### What was tested, and what each run established

All runs: isotropic etch of a solid disk, `R = 5.83 nm/s`, `n_reinit = 5`, `reinit_every = 5`,
first-order Godunov, TVD-RK2. Area is the sub-cell zero-contour area (finding G2). Radius is read
along 72 rays from the disk centre.

**Run 1 — first full-travel etch.** P5 geometry: r₀ = 300 nm, dx = 10 nm, T = 25.7 s, 200 steps,
exact final radius 150.169 nm.

| configuration | radius | error |
|---|---|---|
| no reinitialisation | 223.9 nm | stalls (the known band limit, finding G0) |
| reinitialisation every 5 steps | 149.08 nm | −0.72 % |

Measured along a **single grid row**. Looked like a pass.

**Run 2 — the same field, radius measured in 72 directions.**

| statistic | radius | error |
|---|---|---|
| mean | 145.41 nm | **−3.17 %** |
| along a grid axis | 148.92 nm | −0.84 % |
| along a diagonal | 143.23 nm | −4.62 % |

First evidence of the problem. Run 1's single row happened to lie along the grid axis, which is the
direction with the least error.

**Run 3 — isolating the cause.** Three variants of the same etch at three grid spacings. Exactly one
thing changes between adjacent columns. Error is in the mean radius.

| dx | rate everywhere, no reinit | rate everywhere, + reinit | band, + reinit |
|---|---|---|---|
| 10 nm | −1.02 % | −3.17 % | −3.17 % |
| 5 nm | −0.50 % | −1.56 % | −1.56 % |
| 2.5 nm | −0.25 % | −0.77 % | −0.77 % |

- **Columns 2 and 3 agree to the third decimal.** The velocity band (stage 11) contributes no error.
- **Column 1 to column 2:** adding reinitialisation roughly triples the error. Reinitialisation is
  the dominant cause, contributing about −2.15 % at dx = 10 nm against −1.02 % from advection.
- **Down each column:** halving dx halves the error. Both contributions are first-order
  discretisation error, not a defect that grows without bound.

**Run 4 — one cycle in isolation (this is V12).** Exact disk, r = 200 nm, 64×64 at dx = 10 nm, one
cycle, no advection.

| measure | value | tolerance |
|---|---|---|
| area change | 0.066 % | 0.1 % |

A single cycle passes. The accumulated effect of 40 cycles, from Run 3, is about 2 % in radius.

**Run 5 — V9 (reversibility).** Disk r = 200 nm, 64×64, etch 100 nm then deposit 100 nm, 100 steps
each. Symmetric-difference area relative to the starting area:

| dx | no reinit | + reinit | tolerance |
|---|---|---|---|
| 10 nm | 3.11 % | **7.76 %** | 3 % |
| 5 nm | 1.51 % | 3.89 % | 3 % |
| 2.5 nm | 0.74 % | 1.95 % | 3 % |

Same pattern: reinitialisation multiplies the error by about 2.5, and both columns converge at first
order. V9 passes only at dx ≤ 2.5 nm. V9's geometry is not specified by the PRD; this one was chosen
before it was measured, using V1's grid vocabulary with the disk clear of every boundary. This is
recorded as S12.2 in `tests/test_invariants.py` and shares S12.1's cause.

### Conclusions

1. The failures of V1 (full travel) and V9 are real, reproducible, and share one cause: cumulative
   interface motion from reinitialisation.
2. The velocity band is not involved.
3. The error is first-order discretisation error, converging with dx. It is too large at the
   resolutions the checks run at (10 nm, and 5 nm per finding G3).
4. **V12 as specified cannot detect this.** It bounds one cycle; the product runs tens to hundreds.
   That is a gap in the check design, independent of how the drift itself is resolved.
5. **Whether V1 passes depends on an unspecified definition.** PRD §8.1 does not say which radius.
   Along a grid axis V1 passes (−0.84 %); averaged over directions it fails (−3.17 %). The mean is
   asserted, because the axis is the single direction the grid favours and selecting it would
   select the flattering answer.
6. **Unexplained discrepancy with the original tree.** `m2/` recorded Godunov V1 at 0.38 % (finding
   G0) and 1.08 % (finding J2), both better than 3.17 % here. Either this reinitialisation drifts more
   than the original, or the original measured radius differently (for example along an axis). This
   has not been investigated.

### Options

| option | cost | what it buys | risk |
|---|---|---|---|
| **a.** Compare against `m2/`: its radius measure and its reinitialisation | about an hour | Determines whether this is a defect in this build or a measurement difference | None |
| **b.** Subcell-fix reinitialisation (Russo–Smereka): hold interface-adjacent cells to φ₀ | a day or more | Removes the cause | Selects between stencils near the interface; must be shown clean under V14 at stage 13 |
| **c.** Record as a known discrepancy; re-measure after WENO5 (stage 14) | none now | `m2/` recorded WENO5 cutting V1 radius error 120× | Reinitialisation drift may not respond to a better advection scheme |
| **d.** Run V1 and V9 at dx = 2.5 nm | 4–8× runtime | Both pass | Chooses resolution to make checks pass; finding G3 set 5 nm, where both still fail |
| **e.** Reinitialise less often or with fewer iterations | none | Less accumulated drift | Tuning `reinit_every` or `n_reinit` to pass a check — forbidden by PRD §11; also reintroduces the band limit |
| **f.** Measure V1's radius along a grid axis | none | V1 passes | Hides exactly the bias PRD §8.3 warns about |

**Recommendation: a first, then b or c.** Option a is cheap and decides whether the fix belongs in
this code or in the check definition. If this reinitialisation is worse than the original, fix it
(b). If the difference is measurement, the owner must choose V1's radius definition, then choose
between b and c.

Options d, e and f are rejected: each turns a failing check green without removing the cause.

### Additional question for the owner — decided 2026-09-16

Should the registry gain a check that bounds **accumulated** reinitialisation drift over a full run,
rather than per cycle? **Yes.** Added as V12a; see S12.3.

### Reproduce

```bash
cd m2-new
uv run pytest tests/test_reinit.py tests/test_invariants.py -rx
```

| test | check | expected result |
|---|---|---|
| `tests/test_reinit.py::test_v1_full_travel_with_reinitialisation` | V1 | xfail — mean radius −3.17 % |
| `tests/test_reinit.py::test_v12_reinitialisation_does_not_move_the_interface` | V12 | pass — 0.066 % |
| `tests/test_reinit.py::test_the_band_adds_no_error_over_evaluating_everywhere` | — | pass — band cleared |
| `tests/test_reinit.py::test_cumulative_reinit_drift_is_first_order_in_dx` | — | pass — error halves with dx |
| `tests/test_reinit.py::test_reinitialisation_is_what_removes_the_band_limit` | — | pass — finding G0 |
| `tests/test_invariants.py::test_v9_reversibility` | V9 | xfail — 7.76 % |

The `xfail` markers are `strict=True`: if either check starts passing, the suite fails, so a fix
cannot land unnoticed. Tolerances are unchanged.

---

## S12.3 — V12a's tolerance

**Status:** provisional placeholder, `provisional: true`. **Classification:** B.
**Working-agreement class: A**, announced when made. Owner set the check, not its tolerance.

V12a compares two runs of V1's geometry that differ only in whether they reinitialise, with the rate
applied everywhere so the run without reinitialisation does not stall at the band edge. The
difference in final radius is attributable to reinitialisation.

**Mechanical rule applied:** mean shift over 72 directions < **1 %** of the final radius — V1's own
forward tolerance. If reinitialisation alone may move the interface further than the check it runs
inside allows, that check's verdict is decided by reinitialisation rather than by the physics.

| dx | cycles | mean shift | max shift (diagonal) | signed mean |
|---|---|---|---|---|
| 10 nm | 40 | **2.15 %** | 3.06 % | −3.23 nm |
| 5 nm | 80 | 1.05 % | 1.55 % | −1.58 nm |

V12a fails at dx = 10 nm and is recorded `xfail(strict=True)` under the S12.1 known discrepancy.
Note that the rule was chosen before V12a's own measurement but after S12.1's Run 3, which already
implied about 2.15 %; it was not chosen to pass.

**Owner decision needed:** confirm or replace the 1 % rule.

---

## PRD location

`M2_PRD.md` now lives in this folder and is amended in place by `m2-new` decisions (header, §6, §8,
§8.0, §8.1, §8.3, §8.5; each passage marked "Amended 2026-09-16"). The original tree's copy is no
longer in the working directory. Decision record: `decisions/2026-09-16-drift-check-and-prd.md`.

---

## Gate note — M2.3 built with the M2.2 gate open

On 2026-09-16 the owner directed work to continue to stage 13 (M2.3) while S12.1 was unresolved, and
the same day accepted S12.1 as a known discrepancy (option c). The M2.2 gate is therefore accepted
with that exception: V1 (full travel) and V9 remain `xfail(strict=True)`. Nothing
in M2.3 depends on how S12.1 is resolved, except V14a, which inherits it (below).

---

## S13.1 — The PRD gives the analytic sensitivities V14a–V14c no tolerance

**Status:** DECIDED — owner accepted the 1 % rule 2026-09-16. Decision record:
`decisions/2026-09-16-s12-1-and-s13-1.md`. **Classification:** B (was).
**Working-agreement class: A** (adjusts a requirement), announced when made.

Decision §5 added V14a, V14b and V14c "with closed-form answers" and gives no tolerance. The check
registry previously said "relative error < 1e-6". That figure was written at stage 5 with no basis
in the PRD and is **withdrawn**.

**Mechanical rule applied: 1 % relative error, for all three.** It follows from three constraints:

1. The checks exist to catch what V14–V16 cannot: a wrong sign, a wrong normal convention, or a
   derivative wrong by the size V19 injects (5 %). The tolerance must sit clearly below 5 %.
2. A discrete derivative carries the same discretisation error as the discrete forward map, so it
   cannot be held tighter than the forward tolerance of its own configuration. V1's is 1 %.
3. One number for all three, so no check has a tolerance fitted to its own result.

**This rule was written after the results were measured.** It is recorded as such. Under it V14b and
V14c pass and V14a fails, so it was not chosen to make every check pass.

| check | derivative | measured | expected | relative error | result |
|---|---|---|---|---|---|
| V14b | d(depth)/dR, plane | 19.9993 | +20 | 3.5e-5 | pass |
| V14c | dz/dv0, 30° facet | −17.348 | −17.321 | 1.6e-3 | pass |
| V14c | dz/dp, 30° facet | 4.972 | 4.983 | 2.1e-3 | pass |
| V14a | dr/dR, disk, full travel | −26.19 | −25.7 | 1.9e-2 | **xfail (S12.1)** |

V14a's failure is attributed to S12.1 by experiment, not assumption:
`test_v14a_derivative_is_correct_without_cumulative_reinit_drift` runs the same derivative with the
travel inside the band and reinitialisation off, and it passes within 1 %. The gradient machinery is
sound; the error belongs to the forward map.

**Owner decision:** accepted, 2026-09-16.

---

## S13.2 — Which refinement pairs define "observed order"

**Status:** open, minor. **Classification:** D.

PRD §8.2 requires an observed order but does not say how it is computed from several refinement
levels. The tests take the **finest pair** (the asymptotic regime) and record every pair.

This decides V7:

| refinement pair | observed order |
|---|---|
| 1st (coarsest) | 1.80 |
| 2nd | 1.91 |
| 3rd (finest) | **1.97** |
| least-squares fit, all four levels | 1.895 |

The requirement is ≥ 1.9. The finest pair passes; a least-squares fit over all levels misses by 0.005.
The coarsest pair is pre-asymptotic, and the finest-pair convention is standard practice, but because
it is the difference between pass and fail it is stated here rather than left implicit.

V5 and V6 pass under either convention:

| check | L1 orders | L∞ orders | requirement |
|---|---|---|---|
| V5 (manufactured solution) | 1.07, 1.00, 0.99 | 1.80, 1.25, 1.01 | scheme order, 1 |
| V6 (spatial, full product path) | 0.97, 0.99, 1.02 | 0.94, 0.95, 0.99 | ≥ 0.9 |

---

## S13.3 — Residual factor k and adjoint ratio (measured, not gated)

**Status:** recorded for M2.4. **Classification:** C (input to the M2.4 checkpoint schedule).

Decision I5(a): M2.3 measures and reports the adjoint cost; it does not gate on it. Measured on a
40 × 32 grid, directional law, first-order Godunov, CPU:

| quantity | value |
|---|---|
| k, residuals per solver step, in φ-sized fields | 206 |
| k, residuals per reinitialisation cycle | 101 |
| warm adjoint / forward wall-clock ratio, unchecked, 20 steps | 3.46 |

**Proposed M2.4 schedule** (PRD §7.4): two-level checkpointing peak is `N/L + L·k`, minimised at
`L* = sqrt(N/k)`. For the S03 case, N = 625 and k ≈ 206 per step give `L* ≈ 1.7`, so **L = 1 or 2**:
checkpoint every step or every other step. This agrees in direction with the original tree's finding
K2 (L = 1 at k = 349).

Caveats, stated so the numbers are not over-read:

- k includes arrays sized by the request capacity K as well as by the grid, so it is not exactly
  grid-independent. Re-measure on the S03 grid before fixing L.
- The original tree measured k = 349 for Godunov; this build measures 206. The definitions may
  differ (for example whether reinitialisation is folded into the per-step figure). Not investigated.
- The adjoint ratio is a CPU laptop measurement of an unchecked run. Decision I5(a) sets the
  product-relevant figure as a checkpointed H100 run at M2.4; this number is not that.

Reproduce: `tests/test_adjoint_cost.py` keeps the k measurement executable.

---

## S13.4 — The local ledger is replaced wholesale by every run

**Status:** open, minor. **Classification:** D.

`Ledger.write` replaces `reports/local/verification_ledger.json` with the rows from the current run.
Running only the fast tier therefore removes the nightly rows (V5–V10) written by an earlier full run.
The ledger of record is unaffected, because CI writes it from a complete run. Locally it means the
file reflects the last run, not the last result of every check. Options: merge rows by check ID with
the newer row winning, or accept the current behaviour and document it. No change made.

---

## BUILD_ORDER correction — the smooth functional moved to stage 13

The M2.3 gate requires V14, V15 and V16 **on the smooth volumetric functional** (PRD §5.7, §6).
`BUILD_ORDER.md` placed `functionals.py` at stage 16 with extraction. `solid_volume` and its
mollified Heaviside were therefore built at stage 13; extraction (CD, sidewall angle, depth, bow,
mask remaining) stays at stage 16.

---

## Related, already decided

**G1 — V8 (Zalesak's disk) fails under first-order Godunov.** Owner accepted as a known discrepancy
on 2026-09-11 (recorded in `m2/OPEN_QUESTIONS.md`). This build measures 42.95 % area loss and 75 %
notch fill at 100², against 43.0 % recorded. `tests/test_invariants.py::test_v8_zalesak_disk` is
`xfail(strict=True)`. It is numerical diffusion from advection, distinct from S12.1, though
reinitialisation contributes to it.
