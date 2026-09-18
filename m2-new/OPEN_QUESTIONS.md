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

## S20.2 — V6 requires order >= 4 of WENO5, and two separate things stop it

**Status:** open, raised 2026-09-17 when WENO5 became the default. **Classification:** A (the
requirement is one of the things that is wrong, and the measurement convention is the other).
**Class:** A under `WORKING_AGREEMENT.md` §2 — it adjusts a stated requirement, so it is announced,
not acted on.

The registry's V6 row reads `godunov >= 0.9; weno5 >= 4`. The Godunov half is measured and passes.
The WENO5 half fails, and the measurement shows two distinct reasons rather than one.

Measured 2026-09-18, full product path, four levels at constant CFL:

| dx (nm) | L¹ error (nm) | L¹ order | L∞ order |
|---|---|---|---|
| 8 | 0.06996 | — | — |
| 4 | 0.01546 | 2.178 | 2.017 |
| 2 | 0.003716 | 2.057 | 1.856 |
| 1 | 0.001321 | 1.492 | 0.448 |

**Reason one: the coupled path is second order, and that is by design.** The 8→4 and 4→2 pairs read
2.18 and 2.06. Finding J2 says why: the velocity extension gathers rates to the band by normalised
multilinear interpolation, which is second order. A fifth-order Hamiltonian downstream of a
second-order gather gives a second-order path. The order of a composition is the order of its
weakest link, and V6 as written measures the composition while its tolerance describes one link.
The Hamiltonian's own order is verified separately at 5.13.

**Reason two: the finest pair is measuring the ruler, not the scheme.** At dx = 1 nm the error is
0.0013 nm — about a picometre, far below anything the sub-cell contour used by `radii_along_rays`
can resolve. The orders collapse there (1.49 and 0.45) because the numerator is extraction noise.
This never showed up under Godunov, whose errors at the same levels are two orders of magnitude
larger, so the floor is never reached.

That interacts with **S13.2**, settled 2026-09-17: "observed order is read from the finest
refinement pair." The convention is right in the asymptotic regime and wrong once a scheme is
accurate enough to hit the measurement floor, which is exactly what a better scheme does. Whatever
is decided about the tolerance, the convention needs a floor guard — for example, ignore any pair
whose finer error is within 10x of the extraction's own resolution, and fail if no pair survives.

| option | what it says | consequence |
|---|---|---|
| **a.** Require `weno5 >= 1.9` on the product path, read from the finest pair ABOVE the extraction floor, and check fifth order separately on the bare Hamiltonian | The requirement was measuring the wrong thing, and so was the convention | V6 passes on 2.06; one new small check covers the reconstruction itself |
| **b.** Keep `>= 4` and raise the extension to fifth order | The gather is the defect | Real work in `band.py`, and the gather is in the differentiated path, so V14 must be re-run |
| **c.** Keep `>= 4` and record the failure permanently | The product does not meet its own stated order | A standing red check, and the number it reports is noise |

**Recommendation: a.** The second-order gather is a deliberate design choice — it is what makes the
closest-point projection differentiable and cheap — not an accident, and no M2 deliverable depends
on fifth-order convergence of the coupled path. Option b is the only one that would change the
product, and nothing has asked for it. The floor guard is needed under any of the three.

The tolerance is UNCHANGED until the owner decides. The WENO5 case is `xfail(strict=True)`, so the
ledger records it as not passing and a fix cannot land unnoticed.

---

## S20.3 — One V14 direction lost its signal when the solver got more accurate

**Status:** open, raised 2026-09-17. **Classification:** D (minor, but it makes a check red).
**Class:** A under `WORKING_AGREEMENT.md` §2.

`tests/test_extraction.py::test_diagnostic_decomposition_volume_and_cd_both_pass`, V14 on
`solid_volume` over the etched trench fixture:

| | Godunov | WENO5 |
|---|---|---|
| clean directions | 20 / 20 | 19 / 20 |
| slope range | 1.987 – 2.014 | 1.998 – 2.014 |
| direction 18 | clean | `insufficient_signal` |
| noise floor | 4.56e-11 | 3.55e-11 |

`insufficient_signal` means the Taylor remainder for that one random direction never rose out of the
noise floor anywhere in the swept window. The check could not measure; it did not measure something
wrong. The same gradient vector scores 2.00 in the other nineteen directions, and a wrong gradient
fails towards slope 1 rather than towards silence.

The cause is the improvement itself. The remainder is about `(h^2/2) v^T H v`; WENO5 makes the
forward map smoother and the fixture's curvature along that direction smaller, so the signal sank
below what fp64 resolves through a 20-step solve.

| option | what it does | risk |
|---|---|---|
| **a.** Give this fixture more steps or a coarser `dx`, so there is more curvature to see | Restores signal by making the case less trivial | Changes a fixture after seeing its result — must be justified by the physics, not the score |
| **b.** Treat `insufficient_signal` as "not evidence" rather than "failure" when no direction is SHALLOW | Says plainly what the classification means | Weakens V14 unless the count of measurable directions is itself asserted |
| **c.** Leave it red | Honest, costs a permanently failing check | — |

**Recommendation: b, with a floor on measurable directions** (for example, at least 18 of 20 must be
clean AND none may be shallow). That is the statement V14 is actually making. It is not a tolerance
change — no slope bound moves — but it does change a pass rule, so it is the owner's call.

Widening the h window is not on the list. `gradcheck.py` warns against it in as many words, because
it is the one change that would turn a genuinely wrong gradient green.

---

## S15.4 — M2.4 gate status: measured on an H100, one item short

**Status:** two items open, both needing the H100: full-resolution 3D V14, and a **re-measurement
under WENO5**, which became the default later the same day (DECISIONS.md). Every number in this
section is Godunov. WENO5's k is about 1.8x larger per step and about 5x larger per reinitialisation
cycle, so the peak memory, the optimal segment L and the adjoint ratio all move, and by how much is
not known. The adjoint ratio of 4.65x against a 4x target was accepted by the owner on 2026-09-17
(DECISIONS.md) on the Godunov figure.
Measured 2026-09-17 on one H100 80 GB (Lambda, Utah), `cuda:0`, Godunov, S03 at 290x100x100, N = 625.
Hardware-gated numbers: recorded here, never in the ledger of record.

| gate item | target | measured | |
|---|---|---|---|
| forward wall-clock, warm | < 60 s | **4.2 s** (7 ms/step) | pass, 14x inside |
| peak device memory | < 40 GB | **20.7 GB** | pass |
| checkpoint schedule from measured k | L from k, not sqrt(N) | **L = 1**, and measured best | pass |
| request capacity from the worst step | no overflow | 131,400 worst of 164,250 | pass |
| warm adjoint ratio, checkpointed | <= 4x | **4.65x** | accepted 2026-09-17 |
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

### The adjoint ratio: 4.65x, ACCEPTED

Owner accepted 2026-09-17: 4.65x stands as the measured figure. See DECISIONS.md.

### Still needing hardware

1. Full-resolution 3D V14 (`--full --v14`) was not run.
2. Re-measure k, peak memory, the optimal L and the adjoint ratio under the WENO5 default, and
   update `gate.py`'s `K_STEP_3D_GODUNOV` / `K_REINIT_3D_GODUNOV` with a WENO5 pair beside them.
   Not blocking: the gate is an M2.4 artefact, M2.4 was signed off on Godunov, and the constants are
   named for the scheme they were measured on so nothing silently reads them as the default.

---

---

---

---

## S17.1 — w_mat sensitivity: small for the marker's own rate, large for the film's

**Status:** reported, and the resolution question DECIDED 2026-09-17 (dx = 2 nm, w_mat 2 cells).
**Classification:** C — owner should read before M2.8 calibration. The study is
`tests/test_w_mat_study.py::test_w_mat_sensitivity_study`, which writes its table to the ledger.
(It was `reports/w_mat_sensitivity.md`, generated by `scripts/w_mat_study.py`, until those two
directories were removed on owner instruction — everything lives in the code now.)

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

---

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

---

---

---

## S19.1 — P8's noise magnitudes are placeholders

**Status:** placeholder, `provisional: true`. **Classification:** B.

M2.8 recovers parameters "to within the noise floor", and the floor comes from proposal P8: independent
Gaussian noise per observable at CD 1.0 nm, depth 5 nm, sidewall angle 0.2 deg. P8 marks these
provisional: they are the right order for CD-SEM and cross-section metrology at this feature size, not
measurements. The real values come from repeated measurements of the same coupon feature, which is also
what would make M2.8's tolerance defensible rather than chosen.

Everything downstream follows from them: propagated through the Jacobian they give the parameter
tolerance this build asserts against, 0.25 nm/s in v0 and 0.21 in p. Tighter metrology means a tighter
test, and the test is only as meaningful as the numbers behind it.

Note also that these are metrology REPEATABILITY figures, not accuracy: a fit can sit inside the
interval and still be biased.

---

## S19.2 — A NaN objective makes the optimiser report success at its starting point

**Status:** recorded; pinned by a test. **Classification:** D.

Extraction returns NaN where no wall exists at the requested height -- deliberate, so a missing feature
is loud rather than a plausible number (S16.3). Inside a fit that becomes quiet: the first cold start
tried here put the trench floor at 430.007 nm while CD was being read at 430 nm, so the objective was
NaN, and L-BFGS-B returned its starting point after **zero iterations** with a successful status. The
"recovered" parameters were the initial guess echoed back, and every draw agreed with every other draw
because none of them had fitted anything.

Two guards now: the observation heights are chosen to hold a wall across the whole search domain, and
the fit asserts `result.nit > 0`. A test walks the corners of the domain and requires every observable
to be finite.
