# OPEN_QUESTIONS — M2 (m2-new build)

Everything still undecided, with the evidence behind it. Never resolve an ambiguity by guessing; write
it here instead, classified:

- **A** — PRD error (arithmetic, cross-reference, or a statement shown to be false)
- **B** — placeholder, carrying `provisional: true` and the mechanical rule that produced it
- **C** — gap blocking a later milestone
- **D** — minor

Separately, per `WORKING_AGREEMENT.md` §2, a change is **Class A** (adjusts a requirement) or
**Class B** (changes or compromises the project; stop and ask).

**The pipeline:** an item lives here until it is settled, then it moves to `DECISIONS.md` and is deleted
from this file. Findings that are simply *recorded* — a measured property of the code, needing no
decision — live in the docstring of the module they concern, not here.

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

---

## S14.2 — WENO5 re-measurement after the fix is incomplete

**Status:** open. **Classification:** C (S12.1's revisit depends on it).

The owner declared stage 14 finished. The full re-measurement under the fixed scheme was stopped to
free the CPU (XLA compiles of about 5 minutes each). Confirmed after the fix: V8 (area −1.49 %, notch
17.7 %, both within tolerance) and the dx = 1 disk (no holes). Measured **before** the fix, with
forward-Euler reinitialisation, and not re-confirmed: V1 0.034 %, V12a 0.002 %, V9 0.009 %, V10 0.009 %,
V14a 0.36 %, V6 order about 2 (finding J2's cap). Those runs showed no instability, so the numbers are
likely to hold, but that is not a measurement. WENO5 variants of the check tests are not yet written.

S12.1 (option c) is to be revisited with these numbers. Until they are re-confirmed, it is not.

---

## S14.3 — WENO5 cost

**Status:** recorded. **Classification:** C (owner decision J5: whether WENO5 becomes the default).

| quantity (2D, 40×32, CPU) | Godunov | WENO5 |
|---|---|---|
| k per step | 206 | 378 |
| k per reinitialisation cycle | 101 | 529 (before RK3; RK3 raises it further) |
| warm adjoint ratio, unchecked | 3.46× | 7.40× |
| V14 | 20 clean | 18 clean, 2 degenerate (slope 2.24) |
| V16 | 3.7e-15 | 8.6e-11, against 1e-10 |

V16 under WENO5 passes with little margin. Godunov remains the default.

---

---

## S15.4 — M2.4 gate status: measured on an H100, one item short

**Status:** four of five items pass; the adjoint ratio misses narrowly and needs an owner decision.
Measured 2026-09-17 on one H100 80 GB (Lambda, Utah), `cuda:0`, Godunov, S03 at 290x100x100, N = 625.
Hardware-gated numbers: recorded here, never in the ledger of record.

| gate item | target | measured | |
|---|---|---|---|
| forward wall-clock, warm | < 60 s | **4.2 s** (7 ms/step) | pass, 14x inside |
| peak device memory | < 40 GB | **20.7 GB** | pass |
| checkpoint schedule from measured k | L from k, not sqrt(N) | **L = 1**, and measured best | pass |
| request capacity from the worst step | no overflow | 131,400 worst of 164,250 | pass |
| warm adjoint ratio, checkpointed | <= 4x | **4.65x** | **miss** |
| full-resolution 3D V14 | pass | not run | open |

Compile time, reported separately per PRD §6: 9.8 s for the forward.

**The memory model is conservative in the safe direction**: 25.8 GB modelled against 20.7 GB measured,
so k = 344 measured on small grids over-predicts by about 25 % at full size. Persistent storage tracked
the design exactly -- 23.2 MB per step, one field each, 6.3 GB baseline plus 625 fields.

**L = 1 is optimal, not merely derived.** Measured ratios: L = 1 gives 4.65x at 20.7 GB, L = 2 gives
7.89x at 27.1 GB, L = 3 gives 21.19x at 33.0 GB. Larger segments cost more time AND more memory here.

**The ratio does not grow with run length.** Scanned N = 5 to 625: 4.28, 4.47, 4.50, 4.51, 4.54, 4.57,
4.59, 4.62, 4.63. Flat to within 8 % over a 125x range, which rules out memory traffic, checkpoint
granularity and any single expensive operation (the component profile measures 1.0-2.2x for every part
of a step).

### The open question: 4.65x against a 4x target

The target comes from decision I5(a), set at M2.4 on a different machine before this build existed, and
this is a 16 % miss. Options:

| option | cost | what it buys |
|---|---|---|
| a. Accept 4.65x, record it with the target's provenance | none | an honest number; nothing downstream is blocked |
| b. Chase the last 16 % | days | `top_k` is half the forward step (S18.2); cutting it lifts BOTH the forward and the ratio |
| c. Re-derive the 4x target from what M8's optimiser actually needs | hours of thinking | a target with a reason rather than an inherited number |

**Recommend a now, b later.** The forward has 14x of headroom, so nothing is waiting on this.

### Still needing hardware

Full-resolution 3D V14 (`--full --v14`) was not run. It is the one remaining gate item that needs a GPU.

---

## S17.1 — w_mat sensitivity: small for the marker's own rate, large for the film's

**Status:** reported, and the resolution question DECIDED 2026-09-17 (dx = 2 nm, w_mat 2 cells). **Classification:** C — owner should read before M2.8
calibration. Report: `reports/w_mat_sensitivity.md` (generated by `scripts/w_mat_study.py`).

Masked trench, directional law, dx = 2 nm, floor crossing into a 30 nm SiGe marker (provisional).
Change relative to the default w_mat = 2 cells:

| w_mat | share of marker | depth | d/dv0 | d/d(SiGe rate) | d/d(film rate) |
|---|---|---|---|---|---|
| 1 cell | 6.7 % | −0.33 % | +0.53 % | +1.51 % | −4.88 % |
| 4 cells | 26.7 % | +0.43 % | −0.16 % | −1.71 % | +8.41 % |
| 8 cells | 53.3 % | +1.37 % | +0.18 % | −5.44 % | **+31.27 %** |

V14 passes at every width. The depth and its sensitivity to the overall rate and to the marker's rate
barely depend on w_mat. The sensitivity to the **film** rate does: wider ramps blend film into the
marker's boundary zone, and at 8 cells (over half the marker) it moves by 31 %. At the default
(13 % of the marker) the film-rate sensitivity differs from 1 cell by about 5 %.

Plain reading: the gradient that says "how much does the film etch rate matter while crossing into the
marker" is partly a property of the smoothing, not of the physics, and more so the wider the smoothing
is relative to the layer. A fit that calibrates a film rate from marker-crossing data will inherit that.

**Owner decision, 2026-09-17:** keep dx = 2 nm and the default w_mat of 2 cells (13 % of the marker).
Decision record: `decisions/2026-09-17-w-mat-resolution.md`. `w_mat` is never fitted; the standing
obligation is that any physical parameter fitted from marker-crossing data records the w_mat it was
fitted under.

---

## S17.5 — The mask thickness is provisional

**Status:** placeholder, `provisional: true`. **Classification:** B.

200 nm has no coupon measurement behind it. *Mechanical rule:* a conventional hard-mask-to-etch ratio
for a 2.5 um silicon trench is order 10 %, and 200 nm is 20 cells at dx = 10 nm, so the mask is resolved
rather than a two-cell sliver. S00 carries 100 nm on the same rule. Both are dev configs and may never
certify a claim of agreement with the coupon (`require_measured_rate`). When metrology supplies the real
thickness, the grid height follows from it: `ceil((2 * buffer + depth + mask) / dx)` cells.

Effect of the change on the M2.4 projection: field 21.6 -> 23.2 MB, modelled peak 24.0 -> 25.8 GB
against the 40 GB budget. Band occupancy at the worst step rose from 36,200 to 41,600 cells.

---

---

## S18.2 — `top_k` is half the step time on CPU

**Status:** recorded, optimisation candidate. **Classification:** D. Profile on a 256x256 grid:

| part | time |
|---|---|
| full step | 14.54 ms |
| `top_k` inside `build_request` | 7.27 ms |
| everything else in `build_request` | ~0 |
| `closest_points` | 0.31 ms |
| `unit_normal` | 0.42 ms |
| reinitialisation cycle (5 iterations) | 0.67 ms |

`top_k` ranks every cell by `-|phi|`, so its cost follows the GRID, not K: capacity 3,056 and 16,384
both give 14.5 ms per step. Band occupancy is 2.3 % of cells, so 97 % of the work is spent ranking cells
that will not be used. Fixing it means a selection that is static-shaped without a full ranking; nothing
is proposed yet, and it should be re-profiled on the GPU first, where `top_k` behaves differently.

---

## S18.3 — A single timing call is not a measurement

**Status:** fixed; recorded because it produced a wrong result that was reported as a gate failure.
**Classification:** D.

The same N = 625, L = 1 gradient measured **78.25 s** in one run and **18.78 s** in another, on the same
device with the same capacity and the same code: a 4x difference in a computation that did not change.
`gradient_run` timed a single call after one warm-up, so it recorded whatever the device allocator was
doing at that instant -- the 78 s reading was the first large-stack gradient on a fresh device, the
18.78 s one came after smaller runs had grown its pools.

That figure was reported as an 18x adjoint ratio and a failed gate item. It was neither. The true ratio
is 4.65x.

Fixed: the gradient is timed as the **median of three warm calls**, and the spread is printed beside it
(measured 0.01 s at N = 625, so the number is now stable). Lesson worth keeping: a benchmark that
reports one number from one call cannot distinguish a result from an allocator state.
