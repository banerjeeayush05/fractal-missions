# Decision record — responses to `OPEN_QUESTIONS.md` (M2.0 gate)

Verbatim copy of the response document supplied by the mission owner on 2026-09-11 (Revision 3).
**Authoritative: supersedes the M2 PRD where they conflict.** Implementation status for each
section is tracked in `../OPEN_QUESTIONS.md`; cross-mission consequences in `../../CROSS_MISSION.md`.

---

**Revision 3.** §2 (target feature and domain) and §10 (materials) are rewritten around a
two-phase validation coupon. Phase 1 is a deliberately minimal single-material structure chosen so
that no milestone depends on physics M2 does not model. Phase 2 adds a Si/SiGe material transition
so that the multi-material work in M2.6 has a physical reference case. Everything else is unchanged.

Good file. The stop-and-ask discipline was correct, the arithmetic checks were correct, and
backing A9 with executable tests rather than assertion was the right instinct. Every class-A item
is accepted as a real finding.

**These answers are authoritative and supersede the M2 PRD where they conflict.** For each one,
update the PRD text in the same commit that implements it, with a one-line note in the commit
message saying which decision it implements. Do not leave the PRD and the code disagreeing.

A14: numbering mistake, disregard.

---

## 1. Sign and orientation (A1)

Velocity models return an **etch rate** `R` in nm/s. Positive removes material; negative deposits,
which is legal but unused in M2. M2 advects

    phi_t - R * |grad phi| = 0

so the sign flip lives in one place in M2 rather than in every velocity model M3 and M5 will
write. Axis 0 increases **toward the plasma**, so `z_hat = +e_0` and the directional model
`R = v0 * max(0, n . z_hat)^p` etches up-facing surfaces. Record both in `constants.py` and assert
them.

Add a forward check at M2.1 that fails on the wrong sign. Under the directional model, a trench
floor must move **away** from the plasma, and a downward-facing overhang must not move at all.
V14–V16 cannot catch a sign inversion, so this check is the only guard.

---

## 2. Target feature and domain (Q1, Q2, A5, A6) — REWRITTEN

The reference geometry is a coupon that will be fabricated and measured at the Stanford
Nanofabrication Facility: patterned on the ASML PAS 5500/60 i-line stepper, etched on the Lam TCP
9400 ("lampoly").

It comes in two phases on the same reticle and the same tool. **Phase 1** is single-material and
deliberately the simplest structure that supports a real transport claim. **Phase 2** adds one
epitaxial SiGe marker layer, giving M2.6 a physical reference case. Build phase 1 configs first;
phase 2 configs are needed by M2.6.

### Phase 1 geometry — single material

    Material:  single-crystal n-type silicon. ONE material.
    Feature:   long periodic trench, 1:1 line:space
    Depth:     2500 nm target
    Mask:      500 nm SiO2, treated as infinitely selective and NOT MODELLED

Three widths, identical in every other respect:

| ID | CD (nm) | Pitch (nm) | Nominal AR | dx (nm) | Lateral cells | Role |
|---|---|---|---|---|---|---|
| **S00** | open field | — | — | 10 | — | **calibrate** (blanket rate) |
| **S01** | 2000 | 4000 | 1.25 | 10 | 400 | **calibrate** |
| **S02** | 1000 | 2000 | 2.5 | 10 | 200 | predict |
| **S03** | 500 | 1000 | 5.0 | 10 | 100 | predict |

Vertical extent: 10-cell headroom + 2500 nm etch + 10-cell buffer, rounded up — approximately
270 cells at dx = 10 nm. The buffer sits below the maximum etch depth; the headroom above the
stack. Refine to dx = 5 nm for the convergence studies (V6, V7).

Three etch times at 1/3, 2/3 and 3/3 of the endpoint, so each width is compared as a trajectory
rather than a single profile.

### Phase 2 geometry — one material transition

Identical trenches, on a wafer carrying an epitaxial marker layer:

    void  /  epi Si ~400 nm  /  Si(1-x)Ge(x) marker ~30 nm  /  Si substrate

Both materials are conductive, so charge drains and charging never enters — that is the whole
reason for choosing SiGe over an oxide or nitride marker. The etch front crosses into the marker at
roughly 400 nm depth, back out of it at roughly 430 nm, then continues through plain silicon to the
same 2500 nm endpoint.

| ID | CD (nm) | Pitch (nm) | dx (nm) | Lateral cells | Role |
|---|---|---|---|---|---|
| **M01** | 2000 | 4000 | 2 | 2000 | **calibrate** |
| **M02** | 1000 | 2000 | 2 | 1000 | predict |
| **M03** | 500 | 1000 | 2 | 500 | predict |

Marker thickness and Ge fraction are ASSUMED and need the epitaxy owner's input: a strained SiGe
layer has a critical thickness above which it relaxes and becomes defective, and that thickness
falls sharply as Ge fraction rises. Treat ~30 nm as a placeholder, mark it `provisional: true`, and
make the layer thickness a config field rather than a constant.

**dx = 2 nm in phase 2, not 10 nm, and this matters.** The default `w_mat` of 2 cells is 20 nm at
dx = 10 nm, which is comparable to the marker thickness — the mollification would smear away the
entire layer being measured. At dx = 2 nm the default width is 4 nm and the layer is resolved over
15 cells.

This gives the §5.6 `w_mat` sensitivity study a physical anchor it did not have before. Run the
sweep at 1, 2, 4 and 8 cells **and** report `w_mat` as a fraction of marker thickness. If the
gradient depends strongly on `w_mat` once the smoothing width approaches the layer thickness, that
is a real finding about the limits of the mollified representation, and the PRD is explicit that it
must be reported rather than tuned away.

### The mask is not modelled

Put the trench through the mask into the initial phi and treat the mask as infinitely selective, in
both phases. This is the largest simplification here and it is deliberate: it keeps phase 1
strictly single-material, and it keeps phase 2 to exactly **one** material transition — the one
being studied — rather than three. Record it in `reports/simplifications.md` with this
justification.

The cost is that `mask_remaining` has no physical reference case in v0. It still gets implemented
and tested against synthetic geometry at M2.5; it just is not validated against the coupon.

### Q2

**Periodic single-feature domain, accepted.** Open fraction is 0.5 in all three widths because
line:space is 1:1, so pattern density is constant across the ladder and aspect ratio is the only
variable. Microloading is out of scope and no case probes it.

### The gate configuration

**Case S03 is the M2.4 gate config.** At dx = 10 nm in 3D that is roughly 270 x 100 x 100 cells,
2.7 million cells, **21.6 MB per fp64 field**. This is well inside any reasonable memory gate, so
the §4 dense-storage decision holds with a wide margin and Q1 is settled: at AR 5 the question
does not arise.

Note that the line cases are translationally invariant, so 2D is the correct representation and
3D exists only to exercise the 3D code path. Do not treat a 3D run of these cases as physically
meaningful.

### Step count

**Stop writing bare step counts into gates.** Configs specify a target CFL of 0.4 and

    N = ceil(D / (CFL * dx))

For the reference cases at dx = 10 nm and D = 2500 nm, N >= 625. At dx = 5 nm, N >= 1250. Rerun
`reports/dense_cost_table.md` against S03 at both grid spacings, not against the old 512-cube at
N = 500.

### Calibration and prediction roles are load-bearing

The roles in the tables are not decoration. In phase 1 the model is calibrated on S00 and S01 only,
and predicts S02 and S03 plus every case at the two earlier etch times. In phase 2 the transport
parameters carry over unchanged from phase 1; only the SiGe rate multiplier is fitted, on M01, and
M02 and M03 are predicted. When M2.8 and later M8 work happens, the fit must not touch a case
marked `predict`.

Encode this: add a `role` field to each case config with values `calibrate` or `predict`, and add
a test that any fitting routine refuses to consume a case whose role is not `calibrate`. This is
pre-registration written into the code rather than into a promise.

### What the extraction must match

The metrology reports CD at **fixed absolute heights** and sidewall angle over a **fixed absolute
depth window**. The extraction in §5.7 must use the same definitions, or every model-to-experiment
comparison is contaminated by depth error. See C5 below.

---

## 3. Velocity contract v0.2 (A7, A8, A15)

Being renegotiated with the M3 owner. Implement v0.2 and keep `CONTRACT_STATUS = "provisional"`
until that sign-off lands; do not freeze at M2.3 without it.

- Add `run_seed` and `stage_index` to `VelocityRequest`. The RNG key is a deterministic function of
  `(run_seed, step_index, stage_index, point_index)`. Never global, never stateful, never hashed
  from `time`.
- Default to **independent** draws per Heun stage. Passing `stage_index` lets a model choose to
  share them; M2 does not make that choice.
- `positions` is a **fixed-capacity padded set**, capacity `K` from config, obtained with
  `jnp.nonzero(..., size=K)`. M2 derives `K` from band width and grid, because M2 owns phi; M3
  accepts whatever `K` arrives.
- Membership is a **smooth weight** `w(phi)` reaching exactly zero before the band edge, so a cell
  entering or leaving the set contributes nothing at the moment it does. Overflow of `K` aborts
  with a clear message, handled like the CFL assertion.
- `positions` are **closest-point projections onto the zero level set**,
  `x - phi * grad phi / |grad phi|`, not cell centres. M3 needs flux at the surface; a cell centre
  is up to a cell away and, on the solid side, inside the material. This is the closest-point
  extension §5.4 already permits, so evaluation and extension become one step.
- Band width is a config parameter with a reported sensitivity study, in the manner of `w_mat`.

Evaluating velocity on every cell is rejected: roughly 80x wasted M3 cost.

CLAUDE.md: split three ways. A thin root file with the rules every mission shares (one autodiff
system; the GPL clean-room rule; ledgers are generated, never hand-written; never loosen a
tolerance to pass; stop at gates and wait). Then `m2/CLAUDE.md` and `m3/CLAUDE.md` for each
mission's scope and non-goals, since M2's non-goals are M3's goals.

---

## 4. Taylor-remainder pass rules (A10, B3, C3)

B3's "never exclude data" rule is the right instinct and the wrong mechanism: five decades of h is
ten decades of remainder, and over a several-hundred-step fp64 solve the bottom of that range sits
in roundoff whenever curvature is small.

- Declare a noise floor once in `constants.py`, estimated from repeated evaluations of J at fixed
  theta. Exclude points below it.
- Fit the slope on the remaining points. **Pass** if the slope is in [1.8, 2.2]. **Fail** with
  "insufficient signal" if fewer than five decades survive the floor.
- Also **pass**, logged in the ledger as `degenerate_direction` and never silently, if the
  remainder sits at the floor throughout or the slope is >= 2.8. A slope above 2.2 is never
  evidence of a bug: a wrong gradient gives slope 1.
- Directions in **relative units**: scale `delta_i` by `|theta_i|`. This answers C3's mixed-units
  question and is what makes B13's `|dJ/dtheta_k| >= 0.1` precondition meaningful.
- Every delta must still pass individually.

Keep linear functionals out of Taylor tests entirely; they are covered by §5.

---

## 5. V15/V16 are transpose checks, not derivative checks (A9)

Your finding is right and broader than stated. JAX builds every reverse-mode derivative by
linearising with the forward-mode JVP rules and transposing, so `jvp` and `vjp` share those rules
for **all** primitives, not only custom ones. Amend §8.5: V15 and V16 verify transpose
consistency. The PRD's claim that they are "separate code paths" is wrong, and M3's V32 repeats
it — flag that to the M3 owner.

That leaves V14 as the only check comparing a derivative against the function itself. Add
analytic-sensitivity checks with closed-form answers, fast tier:

- **V14a** — V1 isotropic growth: `dr/dV = T`.
- **V14b** — V2 plane translation: `dz/dV = T`.
- **V14c** — a tilted plane under the directional law, which moves along its normal at
  `v0 * cos^p(theta)`: derivatives in `v0` and `p` are exact. This also guards the sign convention
  in §1.

Do not use IDs V23 and above; M3 owns V23–V40.

---

## 6. GPU (Q6)

Lambda, on an **H100 80GB SXM** instance. Avoid their RTX 6000 and A6000 options, where fp64 runs
at a small fraction of fp32, and avoid B300-class parts. A100 80GB and B200 are acceptable
alternates. Gate runs happen on H100; the M1 dev machine runs 2D and small 3D on CPU only.

With the reference geometry now this small, most M2 work will fit on the dev machine. Reserve GPU
time for the M2.4 and M2.7 gates rather than for routine development.

---

## 7. Cost gates (A3, A4, A13)

Priority: **peak memory > recompute overhead > adjoint ratio.** Memory is a hard limit; recompute
is money.

Provisional replacements for the M2.4 gates, finalised at M2.3 once `k` is measured, all against
case S03:

- Peak memory under **40 GB** on one H100. The old 8 GB figure was sized on fp32 arithmetic and is
  withdrawn. The gate case is now small enough that this should be met with room to spare — if it
  is not, that is itself a finding worth reporting.
- Recompute overhead <= 2x with two-level checkpointing.
- Adjoint ratio <= **4x**, defined as warm-wall-clock `value_and_grad` divided by warm-wall-clock
  forward at the same config, compile time excluded and reported separately. The 3x figure in M2.3
  and §13 is achievable in 2D unchecked and is retained there only.

Two-level checkpointing means a **nested scan** — outer sqrt(N) segments, each checkpointed — not
`jax.checkpoint` on the step function alone. Your A4(a) is correct and the lesson plan repeats the
error; note it.

Measure `k` at M2.3 and propose final numbers then. Do not tune anything to hit these.

---

## 8. V18 (C4)

Restate it. Bitwise equality of the whole trajectory is not something XLA guarantees, and it is not
what M3 needs.

- RNG keys and sampled values: **bitwise identical** on recompute.
- Trajectory: agrees to **1e-12 relative**.
- CI enables deterministic GPU ops.

---

## 9. 2D first, endpoint-only (Q5, Q7)

First demo is 2D. 3D stays in M2 and comes off the demo's critical path. This matches the coupon:
every case is a long trench and is exactly represented in 2D.

Endpoint-only output: the differentiated solve returns only the final phi. Intermediate frames for
debugging and demos are dumped outside the gradient path.

---

## 10. Materials (Q3, Q4, C6) — REVISED

Q3: **mollified transitions accepted** for when they are needed. The `w_mat` study is the evidence
that confirms or overturns this, so run it honestly and report the result whatever it is. A sharp
boundary would require differentiating the crossing time implicitly, which is not in M2's budget.

Q4: **two materials, in two phases.** Phase 1 is single-material silicon with an unmodelled mask.
Phase 2 adds a SiGe marker layer, so the reference configs span one and two solid materials plus
void.

Consequences:

- Build the fraction field **generic in the number of materials** from the start. Phase 1 uses one
  and phase 2 uses two, and the interface must not change between them.
- **M2.6 stays on the critical path.** Do not defer it. Its gate — "gradient survives an interface
  crossing a material boundary" — is a numerical property and is tested first on a synthetic
  two-layer config, which does not wait for any wafer. But M2.6 now also owns a physical reference
  case (M01–M03), and the `w_mat` sensitivity study has a real layer thickness to be measured
  against. Q4's note about deferring M2.6 to shorten the mission by three weeks is **declined**.
- The open volume is an explicit **void** material so fractions sum to 1 everywhere, including in
  the single-material phase-1 configs. Do not special-case the single-material path — phase 1 is a
  one-material instance of the general code, not a separate branch, and a test should assert that.
- Selectivity is a per-material rate multiplier in the parameter PyTree. In phase 2 the SiGe
  multiplier is the one fitted parameter that phase 1 does not have.
- Do **not** use an oxide or nitride marker layer instead. Both are insulators, and etching down
  onto an insulator produces notching from charge accumulation — physics that is out of scope in
  both PRDs, and which would contaminate exactly the measurement M2.6 needs. SiGe is chosen because
  it is conductive.

---

## 11. V8 and V4 (C2, C7)

V8: your warning is credible — first-order level-set schemes lose notch area badly. If 2% is
unattainable with the mandated Godunov scheme, record it as a **finding** with the WENO5 result
alongside. Do not change the tolerance.

V4: **on hold.** Under a monotone cos^p law the vertical etch rate peaks at normal incidence, and
the classical facet-angle results assume a rate peaking off-normal, which needs a yield curve —
M5 physics. Do not spend time on V4 until someone derives the steady angle under this law. V14c
covers what V4 was really guarding.

V3: pick a finite p, suggest 64, and state it in the config. The 60 s gate is **warm**, compile
time reported separately, and is now defined against case S03. The old "500 nm trench" wording is
superseded by §2. Keep the V22 target at 2 nm.

---

## 12. Test-only optimiser (A12)

`scipy.optimize` (L-BFGS-B) is permitted inside `tests/` only for M2.8. It must never be imported
by the `m2` package. Add a test that asserts this.

M2.8 must also assert that **V14 passes at the recovered parameters**. A successful recovery is not
evidence the gradient is correct — §11 is explicit about that — it is a check that nothing
downstream is grossly broken, and the V14 assertion keeps the two claims separate.

---

## 13. Ledger (B7)

CI, on a clean checkout, is the **only writer of the ledger of record**. `verification.md` is
generated from that published ledger. Local runs write an untracked ledger. Everything else in B7
stands — pre-registered rows, skip counts as fail, partial collection writes nothing.

---

## Smaller items, all accepted as stated

B1, B2, B4–B6, B8–B13 accepted, with these amendments:

- **B4/B5 normalisation.** Normalise by `||grad J|| * ||delta||`, not by the directional derivative.
  A random delta nearly orthogonal to the gradient makes that derivative tiny and its relative
  error meaningless.
- **B12.** Rerun against case S03 at dx = 10 nm and dx = 5 nm.
- **D8.** Instead of `checkify`, return max CFL over all steps as an auxiliary output of the scan
  and assert on the host after the call. Avoids any question about how `checkify` composes with
  `grad`.
- **A2.** Rename to `solid_volume`, or define etched volume as the integral of `H(phi) - H(phi_0)`.
  For its only job — the §8.7 diagnostic — the sign is irrelevant, since Taylor passes or fails
  identically on J and -J. Fix the mollification width of H at `1.5 * dx` in `constants.py`, not
  config.
- **A11.** Accepted. CD at fixed heights needs only 1D sub-cell zero crossings along grid rows and
  columns, which are static-shape and JAX-native; full marching squares is not required.
  scikit-image stays as the independent reference for V13.
- **C1.** V2's 0.1% is relative to distance travelled. Propose V1 and V21 configurations, but base
  them on the case grid in §2 so everything shares one geometry vocabulary.
- **C2.** `dtau = 0.5 * dx`. A **test-only rigid-rotation velocity model** for V8 and V9 is
  approved — it is a test harness, not a flux model, so §3 does not bite.
- **C3.** Measure V6 in a band around the interface to avoid the kink at the circle centre. Use a
  same-dx, fine-dt reference for V7.
- **C4.** V17 relative, normalised as in B4. Peak memory from device memory statistics after the
  call.
- **C5.** CD at **fixed absolute heights**; sidewall angle over a **fixed absolute depth window**.
  Not fractions of depth — a depth-relative window changes the number of fit points discretely and
  produces a gradient artifact, and it would also not match the metrology. Propose the specific
  heights based on the 2500 nm target depth. `bow` and `mask_remaining`: propose definitions;
  `mask_remaining` has no physical reference case in either phase, since the mask is unmodelled
  (see §2), so it is tested against synthetic geometry only.
- **Phase 2 extraction.** The observable at a material crossing is a local perturbation of the
  sidewall and a change in the instantaneous etch rate, not a change in final depth. Propose what
  to extract — a sidewall profile residual against the phase-1 prediction is the obvious candidate —
  and raise it before M2.6 rather than during.
- **C8.** M2.8's noise model should use the metrology repeatability floor measured on the coupon.
  Until it exists, mark it provisional.
- **C9.** Correct not to create a nightly job that runs zero tests.

---

## Four things the file missed

- **§11 vs §5.2.** §11 forbids `max` on phi near the interface; §5.2 mandates the Godunov
  Hamiltonian, which is built from it. Fine in substance — `max(x,0)^2` is C1, so there is no
  kink — but the wording invites either a mollified Godunov or a stop-and-ask. Reword §11 as "no
  kinks or jumps in the differentiated path" and state that squared one-sided differences are
  allowed.
- **The directional model's kink is on the sidewalls.** `max(0, n . z_hat)^p` has `n . z_hat = 0`
  exactly on vertical walls, the most common surface in a trench. Require `p > 1`, use `p >= 2` in
  gradient tests, and add a test that the derivative in p does not produce NaN at `0^p`.
- **NaN poisoning in masked compute.** `grad phi / |grad phi|` blows up where `|grad phi| -> 0`,
  which happens on the medial axis — the trench centreline, the centre of the V1 circle.
  `jnp.where` does not save you: a NaN in the untaken branch still poisons the gradient. Use the
  double-where pattern and put a line about it in `m2/CLAUDE.md`.
- **Extraction is continuous but not smooth.** Linearly interpolated crossings have a derivative
  that jumps when a crossing passes a grid node. Expect V14 on CD to be touchier than on volume.
  If it fails only at the large-h end, check this before hunting a bug.

---

## Proceed

Order: §1 (sign) first, since every forward check depends on it; then §4 and §5, which change how
every subsequent gate is scored; then §2's phase-1 case configs, which every later milestone
consumes; then §3, marked provisional. Phase-2 configs are not needed until M2.6, but write the
fraction field generic in material count from the beginning so nothing has to be reworked then.

Re-run M2.0 and report. Do not build ahead of M2.1.
