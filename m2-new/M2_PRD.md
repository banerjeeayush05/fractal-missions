# M2 — Differentiable Geometry Core
## Product Requirements Document (agent-facing)

**Repo:** `fractal-m2`
**Owner:** Fractal Semiconductor
**Window:** 22 weeks (Path A, weeks 0–22)
**Path:** A (recipe solve) — critical path. Also required on Path B.
**Status of this document:** authoritative. Where this PRD conflicts with a habit or a
default, follow this PRD.
**Amended 2026-09-11** by the two 2026-09-11 owner decision records (now in git history at
`b5b320c:m2/decisions/`), which are authoritative over this document where
they differ. Amended passages cite the decision section.
**This copy belongs to the `m2-new` rebuild.** Amended 2026-09-16 by
`DECISIONS.md`; those
passages are marked "Amended 2026-09-16" and apply to `m2-new`. Further amended 2026-09-17 by
`DECISIONS.md`.

**Where cited records live in `m2-new`** (paths are relative to this file):

| cited as | where it is |
|---|---|
| decisions taken in this build | `DECISIONS.md` |
| open items (S-numbered) | `OPEN_QUESTIONS.md` |
| cross-mission impacts (X1–X19) | `../CROSS_MISSION.md` |
| verification ledger | `verification_ledger.json` at this folder's root |
| 2026-09-11 owner decisions | `git show b5b320c:m2/decisions/2026-09-11-open-questions-response-r3.md` and `…second-round-response.md` |
| lettered findings (A1–L2: "finding H2", "decision A11", "OPEN_QUESTIONS A16") | `git show b5b320c:m2/OPEN_QUESTIONS.md` |
| proposals (P5 etc.) | `git show b5b320c:m2/reports/proposals.md` |
| dense cost table | `git show b5b320c:m2/reports/dense_cost_table.md` |

The original `m2/` tree was deleted from the working tree on 2026-09-17; the records it held are still
in git history at commit `b5b320c`, which is why those rows are `git show` commands rather than paths.

**Amended 2026-09-17 (DECISIONS.md).** This build keeps no `decisions/`, `reports/` or `scripts/`
directory. §9's report deliverables are produced differently: the ledger keeps the machine-readable
record, findings that need a human live in `OPEN_QUESTIONS.md` and `DECISIONS.md`, and diagnostics that
would have written a report file are gate-tier tests (`tests/test_gate_m2_4.py`,
`tests/test_w_mat_study.py`). `verification.md` — the generated, customer-facing record — is
still owed, at stage 18.

---

## 0. How to use this document

Work **milestone by milestone** in the order in §6. Each has acceptance tests. **Stop at
each gate, run the tests, report results, and wait for review.** Do not build ahead.

If a requirement is ambiguous or you believe it is wrong, say so and stop. Do not resolve
ambiguity by guessing. Write the open-questions file as M1 did — that pattern worked and
caught real errors.

Write `CLAUDE.md` at repo root as your first action, summarising §3 (non-goals), §7
(numerical requirements) and §11 (anti-requirements).

---

## 1. What this mission is for

Everything Fractal sells reduces to one capability: **the derivative of a wafer feature
with respect to a process parameter.** Recipe solve is that derivative fed to an
optimiser. Calibration is that derivative fed to a fitter. Drift inversion is the same
derivative pointed at chamber state instead of recipe.

M2 builds the piece that computes it: 3D interface evolution with a gradient that has
been **verified**, not assumed.

M2 does **not** compute the surface velocity. That is M3. M2 consumes a velocity field
through a fixed contract (§5.5) and must be correct and testable with a prescribed
analytic velocity, so that M2 and M3 can fail independently.

---

## 2. The single most important design constraint

**A gradient that is wrong but plausible is worse than no gradient.**

An optimiser fed a subtly wrong gradient still converges. It converges to the wrong
recipe, confidently, and nothing in the output announces it. Every downstream mission
inherits the error, and it will surface as "the model doesn't transfer" at week 60 rather
than as a test failure at week 6.

Therefore: **the Taylor-remainder test is the gate, not the optimiser.** "The optimiser
converges" is never evidence of gradient correctness. Gradient tests go in CI from
milestone M2.0 and run on every commit.

---

## 3. Scope and non-goals

### In scope
- Level-set interface evolution, 2D and 3D, on a single-feature domain.
- Reverse-mode gradients through the full time integration, verified by Taylor test.
- Differentiable extraction of CD, depth, sidewall angle, mask remaining.
- Multi-material stacks with differentiable material transitions.
- Checkpointed adjoint with bounded memory.

### Explicitly NOT in scope — do not build these
- **Flux, visibility, shadowing, re-emission, redeposition.** All M3. M2 uses prescribed
  analytic velocity only.
- Surface chemistry closures, yield curves, sticking coefficients. M5.
- Reactor-scale anything. M6.
- Calibration, optimisers, posteriors. M8.
- Sparse / narrow-band data structures. See §4 — deliberately deferred.
- Charging, pulsing, stress, wet etch, crystallographic anisotropy.
- Wafer-scale or multi-feature domains.

If you want to add any of the above, stop and ask.

---

## 4. The architectural decision, and a revision to the mission plan

The mission plan specified a **narrow-band sparse** level set, citing ~10⁷ active cells
for a 1 µm³ domain at 1 nm. **That figure is for the wrong domain and this PRD revises
it.**

A single feature with periodic lateral boundaries is small. A 1 µm deep trench on a
256 nm pitch at 2 nm resolution is 512 × 128 × 128 ≈ 8.4 × 10⁶ cells — **34 MB per field
in fp32**. Dense is entirely affordable.

**Amended (decision §2).** That figure is fp32, and §7.5 mandates fp64. The reference geometry is
also no longer a 1 µm trench on a 256 nm pitch: it is coupon case S03 (CD 500 nm, pitch 1000 nm,
depth 2500 nm at dx = 10 nm), which is 270 × 100 × 100 in 3D — **21.6 MB per fp64 field**. The
dense decision therefore holds with a wide margin, and at AR 5 the sparsity question does not
arise. See `git show b5b320c:m2/reports/dense_cost_table.md`.

**Decision: dense storage, masked compute, in JAX.** Narrow-band sparsity is a
performance optimisation for larger domains and is explicitly deferred out of M2.

This is not a minor convenience. It removes three of the four hardest gradient problems:

| Problem under sparse | Under dense |
|---|---|
| Sparse structure has data-dependent topology; differentiating through it is bespoke work | Static shapes. JAX reverse-mode applies directly. |
| Reinitialisation adjoint had to be hand-derived (adjoint of fast marching, an Eikonal solve) | A **fixed number** of Hamilton–Jacobi reinitialisation iterations is just more differentiable PDE. Autodiff handles it. No hand-derived adjoint. |
| Narrow-band rebuild changes the active set discontinuously with parameters | No active set. |
| Warp's autodiff is less mature; silent wrongness is a live risk | JAX reverse-mode is heavily exercised and trustworthy. |

**Cost of the choice:** memory scales with volume, so domains beyond roughly 512³ become
awkward, and compute is wasted on cells far from the interface. Both are acceptable at
M2's scope and both are addressable later without changing the mathematics.

**Revisit sparsity only when** a mission needs multi-feature or wafer-scale domains. At
that point port hot kernels, keeping the dense implementation as the reference the sparse
one is tested against.

---

## 5. Technical specification

### 5.0 Data contracts — build these first, before any physics

```python
# m2/schema.py
@dataclass(frozen=True)
class Grid:
    shape: tuple[int, ...]        # (nz, ny, nx) in 3D; (nz, nx) in 2D
    spacing_nm: float             # isotropic
    periodic: tuple[bool, ...]    # lateral periodic, vertical not

@dataclass(frozen=True)
class Material:
    name: str
    index: int
    is_mask: bool

@dataclass
class Geometry:
    phi: Array                    # signed distance, negative inside solid
    material: Array               # mollified material fractions, see §5.6
    grid: Grid

@dataclass(frozen=True)
class VelocityRequest:            # what M2 hands to a velocity model. Contract v0.3.
    positions: Array              # closest-point projections onto the zero level set, (K, d)
    normals: Array                # (K, d)
    material_fractions: Array     # (K, n_materials)
    weights: Array                # (K,) smooth band membership; exactly 0 for padding
    cell_id: Array                # (K,) int: stable flattened grid index; seeds the RNG (A16)
    n_active: Array               # scalar int: how many entries are real (B14)
    time: float
    step_index: int               # REQUIRED — see §7.6 on stochastic velocity
    stage_index: int              # RK stage; draws are independent per stage by default
    run_seed: int                 # run-level RNG seed

# Velocity model signature. M2 ships analytic implementations only.
VelocityModel = Callable[[VelocityRequest, PyTree], Array]   # -> speed in nm/s
```

**The stable id and the smooth weight are two halves of one mechanism, not two safeguards
(decision A16).** `cell_id` stops a cell's arrival in the band from shifting everyone else's random
draws; the weight reaching zero at the band edge stops the arriving cell's own draw from entering
discontinuously. Remove either and J is discontinuous in θ. `cell_id` is **opaque** to the velocity
model: it seeds an RNG and never indexes geometry, or M3 acquires a dependency on M2's grid layout.
Common random numbers **do not survive a change of grid spacing**, since ids mean different things
at different dx — harmless in M2, a real constraint on M3's convergence studies.

`weights` is zero for padding, but **zero weight does not protect against NaN**: `0 * NaN` is NaN
and poisons the whole gradient, so padded entries carry a benign position and normals use the
double-where guard regardless of weight (decision B14).

`positions` is a **fixed-capacity padded set** of size K, derived by M2 from the *evaluation* band
and the grid, sized from the **worst step, not the first** — the interface lengthens as the trench
deepens, and overflow must not fire at step 900 of 1000. Peak occupancy is reported every run;
overflow aborts like the CFL assertion. Membership is a smooth weight reaching exactly zero
before the band edge, so a cell entering or leaving the band contributes nothing at the moment it
does. Evaluating velocity on every cell was rejected: roughly 80× wasted M3 cost (decision §3).

`PyTree` is the differentiable parameter set. **Every parameter M2 differentiates with
respect to must live in that PyTree**, never captured in a closure — a captured value is
invisible to `jax.grad` and produces a silently zero gradient column.

### 5.1 Level-set representation

- φ signed distance, **negative inside solid**, positive in the open volume. Fix this
  convention in `constants.py` and assert it in tests; sign errors here are the single
  most common source of an inverted gradient.
- **Etch sign and orientation (decision §1).** Velocity models return an **etch rate R** in nm/s:
  positive removes material, negative deposits (legal, unused in M2). M2 advects φ_t − R|∇φ| = 0,
  so the sign flip lives in one place in M2 rather than in every M3 and M5 velocity model. Axis 0
  increases **toward the plasma**, so ẑ = +e₀ and R = v₀·max(0, n·ẑ)^p etches up-facing surfaces.
  Both are recorded in `constants.py` and asserted. V14–V16 cannot catch a sign inversion, so the
  forward check **V1a** — a trench floor recedes from the plasma, a downward-facing overhang does
  not move — is the only guard.
- Vertical extent covers initial stack plus the maximum expected etch depth plus a
  10-cell buffer.
- Lateral boundaries **periodic**. Top and bottom **Neumann** (zero normal derivative).
- 2D and 3D from the same code path, dimension as a config field. Build 2D first: it runs
  in seconds and every validation case in §8 is visualisable there.

### 5.2 Time integration

Advection of φ under etch rate R:  ∂φ/∂t − R |∇φ| = 0   (decision §1; R > 0 removes material)

- Spatial: ~~Godunov upwind Hamiltonian for |∇φ|. WENO5 is an option behind a flag but is
  **not** required for M2 — first-order upwind with adequate resolution is sufficient to
  verify gradients, and the extra stencil complexity is a place for bugs to hide.~~
- **Amended (owner, 2026-09-17): WENO5 is the DEFAULT, in advection and in reinitialisation.**
  Godunov stays selectable and stays tested. The premise struck out above — that first-order upwind
  is sufficient to verify gradients — was shown false by measurement: under Godunov, V1 missed its
  1 % tolerance at −3.169 %, V9 missed 3 % at 7.76 %, V12a missed 1 % at 2.149 %, and V14a inherited
  the same drift at 1.9 %. Under WENO5 those read −0.031 %, 0.002 %, 0.000 % and pass, with no
  tolerance changed. Findings S12.1, S12.2 and G1 close; the cost is in §5.3 and §6.
- **Amended (owner, 2026-09-13): WENO5 is scheduled immediately after M2.3.** Measurement on the
  coupon geometry (finding H1, `git show b5b320c:m2/OPEN_QUESTIONS.md`) shows first-order costs ~1.4° of sidewall angle at dx = 10 nm
  — inside any metrology floor for CD (0.3 nm) and depth (0.1 nm), but ~3× V3's 0.5° requirement.
  It lands after the gradient harness exists so V14/V15/V16 verify it immediately, and deferring is
  nearly free because the adjoint is automatic: changing the forward scheme later means re-running
  the gradient checks, not rewriting an adjoint. V8's discrepancy is revisited then.
- **Built (findings J0–J6).** `spatial_scheme="weno5"` selects Jiang & Peng's HJ-WENO; only D⁻ and
  D⁺ change, the Godunov upwind selection on top is untouched, and the flag is threaded into
  reinitialisation as well as advection. The reconstruction is verified at order **5.13**. On
  product-shaped cases: V10 grid anisotropy 1.625 % → **0.013 %**, V1 radius error 1.081 % →
  **0.009 %**, V6 L¹ error 155× lower, and V14's worst slope *improves* from 1.802 to 1.996.
  Three qualifications, each with a finding: the scheme-level order is **~2.08, not the ≥4 this
  document requires** (J2, capped by the second-order velocity-extension path — logged `xfail`,
  requirement unchanged); V8's area loss falls to 1.23 % but its notch criterion still fails (J4);
  and ~~**`godunov` remains the default pending an owner decision** (J5)~~ — **decided 2026-09-17,
  WENO5 is the default**; J5's open worry, WENO5's effect on the residual factor k, is measured in
  2D (≈1.8× per step, ≈5× per reinitialisation cycle) and still unmeasured in 3D, which is why
  §6's gate figures are labelled Godunov-only rather than restated.
- **The regulariser deviates from the published scheme, deliberately** (J0). Jiang & Peng use
  ε = 1e-6·max(v₁²…v₅²); that `max` over the stencil is a kink in the differentiated path, which
  §11 forbids, so ε is fixed. Safe because the smoothness indicators are built from v = Δφ/dx ≈
  |∇φ| ≈ 1 for a signed distance. A test moves ε by ±100× and requires the observed order not to
  move, so the deviation is shown inert rather than asserted to be.
- **Within 3 cells of a non-periodic boundary WENO5 falls back to the two-point stencil** (J3).
  The Neumann condition replicates the edge value; WENO reads that flat run as smooth data and
  extrapolates from it. Unfixed, this manufactured a phantom solid blob at a domain corner — φ
  drifting from +67 to −2.96 — which the CFL assertion caught. The fallback mask is a function of
  the grid index alone, so it adds no data-dependent branch.
- Temporal: TVD-RK2 (Heun). RK3 behind a flag.

**Fixed step count. This is a hard requirement.**

The natural scheme is `dt = CFL · dx / max|V|`, integrating to a fixed final time T. But
`max|V|` depends on the parameters, so `dt` does, so the number of steps `N = ceil(T/dt)`
changes **discontinuously** with parameters. The gradient through a discontinuous step
count is wrong, and it is wrong in a way that looks like noise rather than like a bug.

Instead: fix `N`, set `dt = T/N`, and **assert** that `CFL = max|R|·dt/dx ≤ 0.5` at every step.
Fixed `N` also gives JAX a static computation graph, which the whole approach depends on.

**Amended (decision §2).** Do not write bare step counts into configs or gates. Configs carry a
target CFL of 0.4 and N is derived: `N = ceil(D / (CFL_target · dx))`, where D is the target etch
depth. For the reference cases that is N ≥ 625 at dx = 10 nm and N ≥ 1250 at dx = 5 nm. An
explicit `n_steps` below the derived minimum is rejected at load time.

**Amended (decision D8).** The per-step CFL is returned as an auxiliary output of the scan and
asserted on the host after the call, rather than with `checkify`, which avoids any question about
how `checkify` composes with `grad`. The abort message names N.

### 5.3 Reinitialisation

φ drifts away from a signed distance function under advection. Re-impose it by iterating
the reinitialisation PDE  ∂φ/∂τ + sign(φ₀)(|∇φ| − 1) = 0  for a **fixed** number of
iterations `n_reinit` (config, default 5), every `reinit_every` steps (default 5), with
`dτ = 0.5·dx` fixed in `constants.py` (decision C2).

- `sign(φ₀)` must be the **smoothed** sign, `φ₀ / sqrt(φ₀² + dx²)`. The exact sign
  function is non-differentiable at the interface and will corrupt the adjoint.
- Fixed iteration count is what makes this ordinary differentiable code. Do not implement
  a convergence-based stopping criterion — a data-dependent loop count reintroduces the
  §5.2 problem.
- Do **not** apply a stop-gradient to reinitialisation. It is a real part of the forward
  map and its derivative is real.

### 5.4 Velocity extension

**Implemented at M2.1, not M2.2** (2026-09-11 scope decision): the closest-point gather and the
request assembly it feeds are built alongside the first forward solve, so V1 and V2 — which have
exact answers and run in seconds — are what first exercise the contract. A temporary
evaluate-everywhere path would have been a second code path that M2.2 then deleted, never checked
against an analytic result.

**Two bands, not one (decision C10).** The *extension* band (default 8 cells, weight tapering to
zero over the outer 2) is where a valid velocity must exist for the advection and reinitialisation
stencils: the interface moves up to CFL·reinit_every ≈ 2 cells between reinitialisations,
reinitialisation propagates about n_reinit = 5 cells, and the upwind stencil reaches 1 more. The
*evaluation* band (default 1.5 cells) is where the velocity model is actually **called**, and it
must be thin: cells deeper in the band project to nearly the same surface point, so calling M3 for
them buys the same expensive Monte Carlo estimate several times. §5.4's closest-point gather is the
bridge — M3 evaluates on the thin set, M2 gathers outward across the full band at negligible cost.
The sensitivity study sweeps the **evaluation** band at 1, 1.5, 2 and 3 cells, since that is the one
with a cost consequence; the extension band only needs a check that 8 cells is sufficient.

Velocity is defined on the interface; advection needs it throughout the band. Extend by
solving  ∂V/∂τ + sign(φ)(∇φ/|∇φ|)·∇V = 0  for a fixed iteration count, or by the simpler
`closest-point` gather. Either is fine; both must be differentiable and use fixed counts.

### 5.5 The M3 velocity contract — freeze this early

M2 must be usable by M3 without modification. Freeze the `VelocityModel` signature above
by milestone M2.3 and version it.

**Amended (decision §3).** The contract is at **v0.2** and stays `provisional`: it is being
renegotiated with the M3 owner, and M2.3 must not freeze it before that sign-off lands. v0.2 adds
`run_seed` and `stage_index`, makes `positions` closest-point projections in a fixed-capacity
padded set, and adds `weights`. The RNG key is a deterministic function of
`(run_seed, step_index, stage_index, point_index)` — never global, never stateful, never hashed
from `time` — and draws are independent per RK stage by default. Whether `point_index` may be a
row in the padded array, which shifts when band membership changes and would break common random
numbers, is open: see finding A16 (`git show b5b320c:m2/OPEN_QUESTIONS.md`) and `../CROSS_MISSION.md` X3. M3 will supply a **stochastic, expensive** velocity;
M2's design must not assume velocity is cheap or deterministic. Specifically:

- Never call the velocity model more than once per RK stage.
- Never assume two calls with the same inputs return the same value.
- Pass `step_index` so the velocity model can seed deterministically (§7.6).

M2 ships two analytic models for its own testing: **isotropic** (V = constant) and
**directional** (V = v₀·max(0, n·ẑ)^p, a cosine-law ion flux with no shadowing).

### 5.6 Multi-material — and the discontinuity it creates

A stack of mask / film / stop layer. Represent as a per-material **fraction field**
summing to 1, not an integer ID field.

The hazard: etch rate jumps when the interface crosses a material boundary. An integer ID
makes that jump a step function, and its derivative is zero everywhere and undefined at
one point — the gradient with respect to anything controlling *when* the interface arrives
at that boundary is then destroyed.

**Mollify.** Material fraction transitions over a width `w_mat` (config, default 2 cells)
using a smoothed Heaviside. Effective velocity is the fraction-weighted blend.

**Amended (owner, 2026-09-13): the mask is modelled, as geometry.** Decision §2 originally left the
mask out to keep phase 1 single-material. Finding H2 showed that taken literally this makes a
patterned etch impossible to simulate: with nothing covering the field, a directional etch strips
the flat surface at the rate it deepens the floor, so the trench translates instead of deepening.
From M2.6 the mask is a solid body in φ, marked as mask material, with an etch-rate multiplier of
**zero** — infinite selectivity preserved, pattern transfer restored, undercut beneath the mask edge
allowed, and the mask corner a real geometric corner. M3 can also trace rays against it.

This adds a material **contact**, not a **crossing**: the front slides along and under the mask
(rate zero) but never passes into it, so the `w_mat` study still concerns exactly one crossing, the
SiGe marker, which is what decision §2 was protecting. If the mask's multiplier is ever made
non-zero, that changes and the mask corner becomes a second crossing to study.

**Amended (decision §2, §10).** Mollified transitions are accepted; the `w_mat` study is the
evidence that confirms or overturns that, so it is run honestly and reported whatever it shows.
The physical reference case is the phase-2 SiGe marker layer (~30 nm, `provisional`, a config
field not a constant) at dx = 2 nm, chosen because SiGe is conductive and so cannot introduce
charging. M2.6 stays on the critical path and is not deferred.

**Required diagnostic:** report gradient sensitivity to `w_mat` at 1, 2, 4 and 8 cells, **and**
report `w_mat` as a fraction of the marker thickness.
If the gradient depends strongly on a purely numerical smoothing width, that is a finding
and must be reported, not tuned away. See open question **Q3**.

### 5.7 Differentiable geometry extraction

Outputs an engineer cares about: `depth`, `CD` at three heights, `sidewall_angle`,
`mask_remaining`, `bow`.

**Amended 2026-09-17 (finding S16.1, owner accepted).** `bow = CD(z_mid) − (CD(z_low) + CD(z_high)) / 2`
over a fixed absolute depth window: zero for a straight taper, positive when the trench is wider in the
middle. The "maximum CD in the window" definition is not used because its `max` is a kink in the
differentiated path (§11).

**`mask_remaining` gains a physical reference case (owner, 2026-09-13):** the coupon metrology can
report mask loss. With the mask modelled at infinite selectivity the model predicts **zero** loss,
so any measured loss is direct evidence that the zero multiplier is wrong — and the size of the
mismatch sets the selectivity to fit. This matters because mask erosion drifts the top CD, which a
fit would otherwise absorb into a closure parameter and then fail to transfer.

- **Sub-cell extraction only.** Marching cubes / squares with linear interpolation of the
  zero crossing. Cell counting quantises the output to the grid spacing and its derivative
  is zero almost everywhere. This was the same requirement in M1 and for the same reason.
- Sidewall angle by least-squares fit to contour points in a specified depth window,
  not by finite-differencing two CD values.
- **Also expose a smooth volumetric functional**, `solid_volume = ∫(1 − H(φ)) dV` with mollified
  H, width fixed at `1.5·dx` in `constants.py` (decision A2: renamed, since with φ < 0 in solid
  this integral is the remaining solid, not the volume etched; for its only job — the §8.7
  diagnostic — the sign is irrelevant, as Taylor passes or fails identically on J and −J).
  This is smooth by construction and is the diagnostic in **§8.7**.
- **Amended (decision §2, C5, A11).** `CD` is reported at **fixed absolute heights** and sidewall
  angle over a **fixed absolute depth window**, matching the metrology; a depth-relative window
  changes the number of fit points discretely and produces a gradient artifact. Fixed-height CD
  needs only 1D sub-cell zero crossings along grid rows and columns, which are static-shape and
  JAX-native; full marching squares is not required, and scikit-image is the independent reference
  for V13 only, never in the differentiated path.

---

## 6. Milestones and gates

| # | Milestone | Gate |
|---|---|---|
| **M2.0** | Repo scaffold, `CLAUDE.md`, schema, config, CI, **verification ledger** | `pytest` green; sign convention asserted; **V19** passes (the canary is caught); ledger writes rows |
| **M2.1** | 2D forward: advection, fixed-N stepping, CFL assertion. **Plus the velocity-request assembly** — evaluation band, padded set with `weights`/`cell_id`/`n_active`, K sizing, closest-point projection and gather — moved here from §5.4/M2.2 by the 2026-09-11 scope decision, so the contract is first exercised by checks with closed-form answers | **V1, V2 in reduced form** (travel held inside the extension band; full travel needs reinitialisation and runs at M2.2 — owner decision 2026-09-11, option A), **V1a, V20** |
| **M2.2** | Reinitialisation, and the PDE-based extension option (§5.4's closest-point gather landed at M2.1) | **V8, V9, V10, V11, V12, V12a** — V12 especially, see §8.3. **Amended 2026-09-16:** V12a added (bounds drift accumulated over a run). **Amended 2026-09-17:** the known discrepancy is CLOSED, not accepted — V1, V9 and V12a all pass under the WENO5 default (S12.1, S12.2) |
| **M2.3** | **Reverse-mode adjoint, 2D.** Contract v0.2 stays provisional until the M3 owner signs off (decision §3) | **V14, V15, V16** on the smooth functional; **V14a/V14b/V14c** analytic sensitivities; ~~adjoint ≤3× forward in 2D unchecked~~ → **adjoint ratio measured and reported, not gated** (decision I5(a), owner 2026-09-13); k measured and the M2.4 numbers proposed. V5–V7 convergence orders reported |
| **M2.4** | Checkpointing + 3D | **V17, V18**; 3D **V14** on case S03; peak memory **<40 GB** on one H100; recompute overhead ≤2× with two-level checkpointing; adjoint ratio **≤4×** warm wall-clock, **set here and measured on a checkpointed H100 run** (decision I5(a): the M2.3 2D-unchecked figure was a laptop measurement of a different thing and is not the number the product cares about); **the checkpoint count comes from measured k, not √N** (finding I4) (decision §7; the old 8 GB figure was fp32-sized and is withdrawn) |
| **M2.5** | Differentiable extraction | **V13** including the sub-cell smoothness sweep; **V14** on `CD_mid` and `sidewall_angle`, not only the volumetric functional |
| **M2.6** | **Multi-material, including the mask as geometry** (amendment to decision §2, owner 2026-09-13) | Gradient survives an interface crossing a material boundary; `w_mat` sensitivity study reported; the mask is a solid body in φ with a zero rate multiplier, so pattern transfer, undercut beneath the mask edge and a real mask corner all work |
| **M2.7** | Performance + external verification | **V3, V22** (V4 on hold, decision §11); full-resolution 3D **V14**; case S03 in 3D <60 s **warm** on one H100, compile time reported separately; `verification.md` generated from the ledger |
| **M2.8** | Inverse sanity | Recover a known synthetic parameter set from a synthetic profile with noise, to within the noise floor, from a cold start |

---

## 7. Numerical requirements

These are correctness requirements, not style.

### 7.1 The Taylor-remainder test is the gate

For random direction δ and step h:

```
R(h) = | J(θ + hδ) − J(θ) − h ⟨∇J, δ⟩ |     must decay as O(h²)
```

Sweep h over at least five decades, fit the slope, require 1.8 ≤ slope ≤ 2.2, over ≥20
random δ. A first-order finite-difference check is **not sufficient** — it passes on
gradients that are first-order right and second-order wrong.

**Amended (decisions §4, B15/B21, B19, B20). The band is no longer a hard requirement:** a wrong
gradient leaves the first-order term uncancelled, so its remainder goes as h and the slope tends to
1. No incorrect gradient produces a slope above 2, so there is no upper bound to enforce. Scoring,
in this order:

    fewer than 3 points in the scored window -> FAIL, "insufficient signal"
    slope < 1.8                              -> FAIL
    slope in [1.8, 2.2]                      -> PASS, logged "clean_quadratic"
    slope > 2.2                              -> PASS, logged "degenerate_direction"

**Amended again (decision I7, owner 2026-09-13, findings I1/I6/I7). The slope is fitted in an
anchored window, not over a fixed h-range.** The sweep still covers eight decades below h_max and
the whole remainder curve is recorded; the fit is taken over the **two decades immediately above the
measured noise floor**. On the real solver at 50 steps the curve has three regimes, and only the
middle one carries information about the gradient:

    h = 1e-1 … 1e-3    slope 1.29   kinks in a piecewise-smooth map, higher-order geometry
    h = 1e-3 … 3e-7    slope 2.00   the quadratic regime          <- the window sits here
    h < 1e-7           slope 0      the fp64 floor

The bottom of the usable range is also where the check has the most **power**: a correct gradient's
remainder is ½h²δᵀHδ and a wrong one's is |ε·g·δ|·h, so the two separate as h falls. Measured, a
correct gradient scores 1.99–2.29 in the anchored window and a 5 % corruption of any single
component scores exactly 1.000. Five decades is unattainable on this solver and always was; the old
rule was written at M2.0 against a trivial analytic function that has no kinks.

- The window is **found, never assumed**: the anchor is the smallest swept h whose remainder still
  clears 100× the floor. Narrowing a test window is also how a test is blinded, so the rule carries
  an obligation, discharged at the hardest case in `tests/test_gradients_2d.py`: **the V19 canary
  must still catch a 5 % corruption in whatever window this produces.**
- The 100× margin is derived, not tuned: the weakest scored point then carries at most 1 % noise,
  biasing the fitted slope by about log10(1.01)/span ≈ 0.003, three orders below the ±0.2 band.
  Calibrated on an exact quadratic, whose slope is 2 by construction: it reads 1.9996–2.0016 at
  100×, and 1.983–2.005 at 10×.
- The two-decade span is a condition on the `clean_quadratic` **classification**, not on pass or
  fail: an O(h³) remainder reaches the floor sooner and can never span the full window.
- **The noise floor is measured, not modelled** (finding I6). It is the remainder evaluated at a
  step far below any signal (h ≈ 1e-10), where the true remainder is ~1e-20·|H| and what survives is
  the rounding error of one evaluation of J. The earlier rule —
  `max(measured spread, C·√N·eps·max(|J|, 1))` with C = 10 — reads 51–65× high against the real
  solver, and a conservative threshold for *discarding* data costs roughly two decades of usable
  window. That model is still computed and recorded beside the measurement on every run, so the two
  can be compared; it is no longer what excludes points.
- The ledger records the classification **per direction** and the **degenerate fraction per run**.
  If that fraction jumps between runs, something changed even though everything is green: treat it
  as a finding.
- Directions are drawn using **declared parameter scales**: δ_i ∝ max(|θ_i|, scale_i). Every
  declared parameter carries a **required** `scale`, its typical magnitude, because a parameter
  sitting at 1e-12 would otherwise get a 1e-12 perturbation, probing nothing while reporting a pass.
  Scales are a property of the parameter, not a test-harness detail: M8 needs the same numbers.
- Every δ must still pass individually. Keep linear functionals out of Taylor tests; V14a–V14c
  cover them.

### 7.2 Test the test

At M2.0, before any physics, add a test that injects a deliberately corrupted gradient
(scale one component by 1.05) and asserts the Taylor harness **fails**. A green test suite
that would stay green under a real bug is worse than no test suite.

### 7.3 Everything differentiable lives in the parameter PyTree

Never capture a differentiable value in a closure. Add a test that every declared
parameter produces a nonzero gradient on at least one objective — a silently zero column
is the failure mode this catches.

### 7.4 Checkpointing

Reverse mode over N steps naively stores N × domain. At N=500 and 34 MB that is 17 GB.
Use two-level checkpointing: a **nested scan** with √N outer segments, each checkpointed.
`jax.checkpoint` on the step function alone does **not** do this — it still stores the carry at
every one of the N steps, costing as much as the naive figure (decision §7). Both the 17 GB and
~750 MB figures above are fp32; fp64 doubles them, and the reference case is now far smaller
(`git show b5b320c:m2/reports/dense_cost_table.md`). Griewank & Walther's `revolve` is the
reference for the optimal schedule; JAX's `remat` policy is sufficient here.

**Amended (finding I4, M2.3). "√N outer segments" is wrong once k is measured, and by 7.4×.**
Peak memory for two-level checkpointing is `c + (N/c)·k` field-equivalents, where k is the number of
field-sized residuals JAX keeps per step. That is minimised at **`c = √(N·k)`**, not at `c = √N`.
The two coincide only at k = 1, the implicit "store φ only" accounting this section was written
under. k is now measured at **330** — stable in N, so it is a genuine per-step factor — and for the
M2.4 gate case (S03 3D, N = 625, 21.6 MB per field) the difference decides the gate:

| c | L = N/c | peak | 40 GB gate |
|---|---|---|---|
| √N = 25 | 25 | 179 GB | **misses** |
| √(N·k) = 454 | 2 | 24.1 GB | fits |
| 25, with step remat (three-level) | 25 | 8.21 GB | fits, 2× recompute |

At this k the residuals dominate so heavily that the optimum checkpoints nearly every step and
recomputes only pairs. **M2.4 must set the checkpoint count from the measured k, not from √N.**

**Amended again (M2.4, finding K2). This section's warning about `jax.checkpoint` is inverted at
the measured k, and the amendment above did not go far enough.** Writing the two-level peak as
`N/L + L·k` in the segment length L, the optimum is **L\* = √(N/k)** — and at N = 625, k = 349 that
is 1.3, i.e. **L = 1**, which *is* remat-on-the-step-function. The text above says that option
"does not do this — it still stores all N carries, costing as much as the naive figure". Under the
implicit k = 1 accounting this document was written with, N carries really was the naive cost. At
k = 349 the naive cost is N·k = 218,000 field-equivalents and remat-on-the-step is N + k = 955.
The thing §7.4 warns against is the thing the arithmetic now recommends.

Three-level — remat *inside* the segment as well — is the genuinely different option: `N/L + L + k`,
380 field-equivalents at the gate case, 2.5× smaller again for 2× recompute.

**And the time-for-memory trade runs backwards here** (K1). Measured warm on 2D CPU, the adjoint
ratio *improves* under checkpointing, because the unchecked tape does not fit and the machine
swaps: 22.2× → 4.9× at 128²·N=100, and 34.3× → 5.8× at 192²·N=100. This is the premise decision
I5(a) rests on, now confirmed rather than predicted.

### 7.5 Precision

fp64 for all level-set fields during M2. fp32 is a later optimisation and must be gated on
the Taylor test still passing at fp32, which it may not. Do not start in fp32 to save
memory — you will spend weeks distinguishing precision noise from a real gradient bug.

### 7.6 Stochastic velocity (forward-looking, M3)

M3's velocity is Monte Carlo. Under checkpointing the forward pass is **recomputed**
during the backward pass. If the velocity model draws fresh randomness on recompute, the
recomputed forward differs from the original and **the adjoint is inconsistent with the
forward it claims to differentiate**. This does not crash. It produces a plausible, wrong
gradient.

Requirement: the RNG key must be a deterministic function of `run_seed`, `step_index`,
`stage_index` and the stable `cell_id`, all carried in `VelocityRequest` (contract v0.3,
decisions §3 and A16).
Draws are independent per RK stage by default. Never hash it from `time`. Never use global or stateful RNG. Add a test
at M2.4 that a checkpointed recompute reproduces the forward trajectory bitwise.

---

## 8. Verification

M11 owns the *institution* — standards, cadence, the ViennaPS relationship, the public
record. The **checks themselves live here and gate M2's milestones.** A verification
result that arrives after the code it validates is archaeology, not verification.

Twenty-two checks, each with an ID (plus lettered sub-checks V1a, V12a and V14a–c; **V12a added
2026-09-16**). Milestone gates in §6 cite them by ID. Every check
writes a row to `verification_ledger.json`: id, description, cadence, last run,
git SHA, pass/fail, and the measured quantity. `verification.md` is **generated**
from that ledger, never hand-written — a hand-maintained verification record drifts from
reality within a month.

### 8.0 Cadence

| Tier | When | Runtime budget | Checks |
|---|---|---|---|
| **Fast** | every commit, in CI | < 3 min, 2D only, coarse | V1, V1a, V2, V11, V12, V12a, V14 (reduced), V14a, V14b, V14c, V15, V16, V19, V20, V21 |
| **Nightly** | scheduled, on main | < 60 min, 2D full + small 3D | V5, V6, V7, V8, V9, V10, V13, V17, V18 |
| **Gate** | at the milestone that cites it | hours | V3, V4, V22, full-resolution 3D V14 |

Fast tier is non-negotiable. If it exceeds three minutes people stop running it, and a
verification suite nobody runs is worse than none because it produces false confidence.

### 8.1 Forward verification — analytic solutions

These have exact answers. Disagreement is unambiguous.

- **V1 — Isotropic etch.** Circle (2D) / sphere (3D) under R = const. **At M2.1 this runs in
  reduced form**: 200 steps as specified, with the travel distance held inside the extension band,
  because without reinitialisation (M2.2) a band-limited velocity extension distorts φ until the
  front stalls. Full travel runs at M2.2. The constraint scales with dx — the band is a fixed number
  of *cells* — so the convergence studies (V6, V7) cannot refine the grid until M2.2 either. **Amended (decision §1,
  confirmed 2026-09-11): a positive rate removes material, so a solid disk shrinks** and the radius
  must satisfy r(t) = r₀ − R·t. The PRD's original growth form assumed the pre-decision sign
  convention. Configuration in P5 (`git show b5b320c:m2/reports/proposals.md`). Tolerance: relative error < 1% at 200 steps. Tests advection,
  reinitialisation and extension together, and is the cheapest signal that something broke.
  **Amended 2026-09-16 (finding S12.1, owner option c).** The radius is the **mean over directions**
  (72 rays), not a single axis, because the grid axis is the direction with the least error. At full
  travel on P5's geometry V1 measures −3.17 % (−0.84 % along an axis). Accepted as a **known
  discrepancy** caused by cumulative reinitialisation drift; re-measure after WENO5. The tolerance
  is unchanged. **Amended 2026-09-17: re-measured, and the discrepancy is closed** — under the
  WENO5 default the mean over 72 rays is **−0.031 %**, inside the 1 % tolerance by 32×, so the
  radius definition no longer decides the outcome. The mean is still what is asserted.
- **V2 — Plane translation.** Flat interface under constant V translates at exactly V with
  no distortion. Error < 0.1% — this one should be nearly exact, and if it is not the
  upwind scheme or the boundary condition is wrong.
- **V3 — Collimated aperture limit. ON HOLD (owner, 2026-09-17; finding S18.1).** Perfectly
  collimated directional etch must reproduce the aperture shape: vertical sidewalls, no bow, no
  faceting. Sidewall angle within 0.5° of 90°. **p = 64** (finite, stated in the config;
  decision §11), since p → ∞ is not representable.
  **Why it is on hold.** For a surface z = h(x) the level set gives the vertical descent speed as
  `h_t = −v₀·cos^(p−1)(θ)`. At p = 1 every part of the surface descends at v₀ whatever its tilt — the
  aperture translates downward, which is what "perfectly collimated" means. For p > 1 a tilted patch
  descends more slowly, so tilt grows; the mask corner supplies the first tilt and the rounding
  spreads inward. Measured at 100 nm of travel: angle error 0.195° at p = 1, 0.110° at p = 1.05,
  1.177° at p = 1.25, 4.717° at p = 2, 3.051° at p = 64, with the trench never widening (CD 100.00 nm
  throughout, since a vertical wall has n·ẑ = 0 at any p). So p is not "how collimated" the etch is;
  it is how strongly the model punishes tilt, and the law's collimated limit p = 1 is exactly the case
  §11 forbids for differentiability. A real collimated etch holds a flat floor because the ion flux is
  uniform and non-local — M3's visibility calculation. Same family as V4. Revisit at M3.
- **V4 — Facet angle. ON HOLD (decision §11).** Under a monotone cos^p law the vertical etch
  rate peaks at normal incidence, while the classical facet-angle results assume a rate peaking
  off-normal, which needs a yield curve — M5 physics. Do not spend time on V4 until someone
  derives the steady angle under this law. **V14c** covers what V4 was really guarding: a wrong
  normal-vector convention.

### 8.2 Forward verification — convergence order

Order studies are what distinguish "wrong" from "under-resolved," and without them you
cannot tell whether a discrepancy against ViennaPS is a bug or a grid.

- **V5 — Method of manufactured solutions.** Choose φ_exact(x, t), substitute into
  ∂φ/∂t + V|∇φ| = 0 to obtain a source term S, add S to the RHS, and verify the observed
  order. Requires an MMS-only code path with a source term; keep it behind a flag and test
  that the flag is off in production runs.
- **V6 — Spatial order.** Fix dt very small, refine dx over ≥4 levels, measure L¹ and L∞
  error against V1's analytic solution. Godunov upwind must show observed order ≥ 0.9.
  WENO5, if enabled, ≥ 4. **Amended 2026-09-17:** WENO5 is now the default, and this requirement is
  the one thing the switch does not satisfy — the coupled path measures ≈2.1, capped by the
  second-order velocity extension (J2), not by the Hamiltonian, whose own order is verified at 5.13.
  V6 runs both schemes explicitly; the WENO5 case is `xfail(strict=True)` and the tolerance is
  unchanged pending the owner's decision on S20.2.
- **V7 — Temporal order.** Fix dx, refine dt. TVD-RK2 must show observed order ≥ 1.9.

Report observed order, not just "error decreases." An order that is right by eye and
0.6 by measurement is a bug.

### 8.3 Forward verification — invariants and known artifacts

These have no analytic solution but strong structural constraints.

- **V8 — Zalesak's disk.** Notched disk in a rigid rotation field, one full revolution.
  Area preserved to < 2%; the notch must survive. The canonical level-set test and the
  standard against which every scheme in the literature reports.
- **V9 — Reversibility.** Advect forward N steps under V, then N steps under −V. The
  result must return to the initial shape. Symmetric-difference area < 3%. Needs no
  analytic solution and catches accumulated advection error that V1 hides.
  **Amended 2026-09-16 (finding S12.2).** Measured 7.76 % at dx = 10 nm; accepted as a known
  discrepancy with S12.1, same cause. Tolerance unchanged.
- **V10 — Grid-orientation isotropy.** Run V1 with the initial condition rotated 45° to
  the grid. Isotropic growth must remain isotropic. Measure the anisotropy as
  (r_max − r_min)/r_mean; require < 2%. Upwind schemes have real grid anisotropy and you
  need to know its size before it shows up as a spurious sidewall-angle dependence.
- **V11 — Reinitialisation restores distance.** After a reinit cycle, max||∇φ| − 1| within
  3 cells of the interface < 5%.
- **V12 — Reinitialisation does not move the interface.** Measure the enclosed area/volume
  before and after a reinit cycle with no advection. Drift per cycle < 0.1%.
  **This is the most commonly missed check in level-set codes.** Reinitialisation that
  quietly shifts the zero level set produces a systematic etch-rate bias that looks like
  physics, calibrates away into the closure parameters, and then fails to transfer.
- **V12a — Accumulated reinitialisation drift over a run. Added 2026-09-16.** V12 bounds one
  cycle; a run applies tens to hundreds, and per-cycle motion accumulates (finding S12.1: 0.066 %
  per cycle passes V12, while 40 cycles shift a disk's radius 2.15 %). Run V1's geometry twice,
  identical except that one run reinitialises and the other does not, with the rate applied
  everywhere so the run without reinitialisation does not stall at the band edge. The difference in
  final interface position is attributable to reinitialisation. Require the mean shift over
  directions < 1 % of the final radius — V1's own tolerance, so that reinitialisation alone cannot
  decide V1's verdict (finding S12.3, provisional). Currently a known discrepancy under S12.1.

### 8.4 Extraction verification

- **V13 — Analytic geometry.** Initialise φ to an *exact* trapezoid of known CD, depth and
  sidewall angle. Extract. CD error < 0.1 nm, sidewall angle < 0.2°. Sweep the trapezoid's
  position by sub-cell offsets and confirm the extracted CD varies **smoothly**, not in
  grid-sized jumps — a staircase here means extraction is not sub-cell and every gradient
  through it is garbage.

### 8.5 Gradient verification

One gradient test is not enough. These fail in different ways, which is the point.

**Amended (decision §5).** JAX builds every reverse-mode derivative by linearising with the
forward-mode JVP rules and transposing them, so `jvp` and `vjp` share those rules for **all**
primitives, not only custom ones. V15 and V16 therefore verify **transpose consistency**, not
derivative correctness — the "separate code paths" claim below is wrong. That leaves V14 as the
only check here comparing a derivative against the function itself, so three analytic-sensitivity
checks with closed-form answers join the fast tier:

- **V14a** — V1 isotropic etch: `dr/dR = −T` (the radius shrinks as the rate rises).
- **V14b** — V2 plane translation: `dz/dR = −T`, equivalently `d(depth)/dR = +T`. The magnitude is
  T either way; the sign follows from §5.1, which is the point of the check.
- **V14c** — a tilted plane under the directional law, moving along its normal at `v₀·cos^p(θ)`:
  derivatives in `v₀` and `p` are exact. Also guards the §5.1 sign convention.

**Amended 2026-09-16 (finding S13.1, owner accepted).** Tolerance for V14a, V14b and V14c: relative
error **< 1 %**. It must sit below the 5 % corruption V19 injects, and a discrete derivative cannot be
held tighter than the forward tolerance of its own configuration (V1: 1 %). V14a currently fails at
1.9 %, inheriting cumulative reinitialisation drift (S12.1); with reinitialisation off the same
derivative passes.

IDs V23 and above belong to M3; never use them here.

- **V14 — Taylor remainder.** §7.1. The primary gate.
- **V15 — Forward mode versus reverse mode.** Compare `jax.jvp` against `jax.vjp` on the
  same directional derivative. JAX implements these as **separate code paths**, so
  agreement is real evidence rather than a tautology. Forward mode is cheap when the
  parameter count is small, which it is in M2. Agreement to 1e-10 relative in fp64.
  This is the highest value-per-line check in the whole suite and it belongs in the fast tier.
- **V16 — Dot-product (transpose) test.** For random u, w: ⟨w, J u⟩ = ⟨Jᵀ w, u⟩, computed
  with `jvp` and `vjp` respectively. Catches transpose errors in any custom VJP rule. If
  no custom rules exist this is nearly free; the moment one is added it becomes essential.
- **V17 — Checkpointed equals unchecked.** At small N where both fit in memory, the
  gradient with `jax.checkpoint` must equal the gradient without it to 1e-12. Catches
  checkpointing bugs, which otherwise present as a slightly wrong gradient at large N only.
- **V18 — Checkpoint recompute.** §7.6. **Restated (decision §8):** bitwise equality of a whole
  trajectory is not something XLA guarantees, and it is not what M3 needs. Require the **RNG keys
  and sampled values to be bitwise identical** on recompute, and the **trajectory to agree to 1e-12
  relative**. CI enables deterministic GPU ops. Guards the stochastic-velocity trap before M3 can
  spring it.
- **V19 — Corrupted-gradient canary.** §7.2. Inject a 5% error into one gradient component
  and assert V14, V15 and V16 all **fail**. Runs in the fast tier. A verification suite
  that has never been shown to fail has not been verified.

### 8.6 Structural and regression

- **V20 — CFL assertion fires.** Deliberately set N too small; the run must abort with a
  clear message naming N. Tests the error path, which is otherwise never exercised.
- **V21 — Golden profile regression.** A frozen reference 2D profile, reproduced to 1e-10.
  Any intentional change requires updating the golden file in the same commit, which makes
  numerical changes visible in review rather than silent.

### 8.7 The diagnostic decomposition — run this first when anything breaks

Run V14 on **both** the smooth volumetric functional and the extracted CD:

| Volume | CD | Diagnosis |
|---|---|---|
| pass | pass | Chain is sound. |
| pass | **fail** | Bug is in extraction (§5.7). Go to V13. |
| **fail** | fail | Bug is in advection, reinitialisation or checkpointing. Go to V15, then V17. |
| **fail** | pass | Something is badly wrong with the functional itself. Stop and ask. |

This localises a gradient failure in minutes rather than days, and it is the reason §5.7
requires the smooth volumetric functional to exist at all even though no product uses it.

### 8.8 Tier 2 — ViennaPS cross-code comparison

- **V22.** One directional-etch structure and one isotropic structure. Identical initial
  geometry, identical velocity law, identical final time. Compare extracted contours by
  symmetric Hausdorff distance; **target < 2 nm, and any excess must be explained or
  logged as an open discrepancy** in the ledger — never silently accepted.

Protocol constraints, non-negotiable:

- **Separate process only.** Write geometry to VTK; compare offline in our own code.
- ViennaPS is **GPL-3.0** as of v4.3.0. Do not link, import, vendor or copy from it. The
  benchmark harness lives in its own repository with its own licence.
- Clean-room discipline: if you read its source, write a natural-language note. Do not
  transcribe.
- Run V6 and V7 **before** V22. Without a convergence study you cannot distinguish a bug
  from a grid-resolution difference, and you will burn a week arguing about it.

### 8.9 What M2 does not verify — stated honestly

M2 verifies that the geometry engine solves its equations correctly. It says nothing about
whether those are the right equations, because M2's velocity is prescribed and analytic.

- No comparison against experimental data. There is nothing physical to compare to yet.
- No ARDE, bowing, microloading or faceting-from-real-flux. All require M3.
- Grid anisotropy (V10) is measured but not eliminated; it will show up again as a
  sidewall-angle artifact once M3 supplies angle-dependent velocity.

Record this scope boundary in `verification.md` so that a reader does not mistake
a green M2 suite for a validated process model.

## 9. Deliverables

1. `fractal-m2` repo, running from clean checkout with `uv sync && pytest`.
2. `verification_ledger.json` — machine-readable record of all 22 checks: id,
   cadence, last run, git SHA, result, measured value. Written by the checks themselves.
3. `verification.md` — **generated** from the ledger. Includes the §8.9 scope
   boundary and every open discrepancy. This is the artifact that makes every downstream
   mission trustworthy, and the one a sceptical customer will ask for.
4. `gradient_verification.md` — Taylor and forward-vs-reverse plots for every
   objective, at every milestone, dated.
5. `tests/test_w_mat_study.py` (gate tier) — §5.6 diagnostic.
6. `geocore/interface.py` — the frozen, versioned velocity contract for M3. (§9 named it `m2/interface.py`; this build's package is `geocore`.)
7. `simplifications.md` — every simplification, with justification.

---

## 10. Environment

```bash
uv init fractal-m2 && cd fractal-m2
uv add "jax[cuda12]" numpy scipy matplotlib
uv add scikit-image      # V13 independent reference ONLY — not traceable by JAX, never in the
                         # differentiated path (decision A11)
uv add pyyaml omegaconf pytest hypothesis
uv add pyvista           # VTK output for ViennaPS comparison
```

Do **not** add: PyTorch, Warp, PETSc, ViennaPS (see §8.8), or any autodiff framework
besides JAX. `scipy.optimize` (L-BFGS-B) is permitted **inside `tests/` only**, for M2.8; it must
never be imported by the `m2` package, and a test asserts that (decision §12). One autodiff system, so that a gradient bug has one place to be.

---

## 11. Anti-requirements — do not do these

- **Do not accept "the optimiser converges" as evidence of gradient correctness.**
- **Do not use an adaptive or convergence-based loop count** anywhere in the differentiated
  path — timestep count, reinitialisation iterations, extension iterations. All fixed.
- **Do not apply `stop_gradient`** to reinitialisation, extension, or material blending
  to make something converge. If you believe a stop-gradient is needed, stop and ask.
- **No kinks or jumps in the differentiated path.** `jnp.sign`, `abs` and bare `maximum` on φ
  near the interface are places the derivative silently dies. Squared one-sided differences are
  allowed: the Godunov Hamiltonian is built from `max(x, 0)²`, which is C¹ (decision, §11 vs §5.2).
- **Require p > 1 in the directional law.** `max(0, n·ẑ)^p` has its kink exactly on vertical
  sidewalls, the most common surface in a trench. Use p ≥ 2 in gradient tests, and keep `0^p` out
  of the p-derivative.
- **Do not let a NaN reach the gradient through an untaken branch.** `∇φ/|∇φ|` blows up on the
  medial axis; `jnp.where` alone does not save you. Use the double-where pattern. Masking after the
  fact does not help either: `0 * NaN` is NaN, so a zero band weight will not save a NaN position.
- **Do not `stop_gradient` the band weights.** They are part of the forward map and their
  derivative is real. Something that looks like a mask is exactly what a future reader will wrap to
  make a test pass (decision B14).
- **Do not write a final time into a config.** Configs specify a target depth; T = depth / rate is
  derived, so a later rate correction changes nothing and the V21 golden profile does not break
  (decision B18).
- **Do not extract CD by cell counting or thresholding.** Sub-cell only.
- **Do not capture differentiable parameters in closures.**
- **Do not start in fp32.**
- **Do not implement narrow-band sparsity.** §4.
- **Do not implement flux, shadowing or visibility.** That is M3, and building a
  placeholder version will create a contract M3 then has to fight.
- **Do not tune `w_mat`, CFL or `n_reinit` to make a gradient test pass.** If a test only
  passes at one setting, that is the finding.
- **Do not vendor, link or copy ViennaPS.** GPL-3.0. §8.8.
- **Do not hand-write `verification.md`.** Generate it from the ledger.
- **Do not loosen a verification tolerance to make a check pass.** Raise it as a finding.
  If a tolerance is genuinely wrong, change it in a commit that says so and explains why.
- **Do not run V22 (ViennaPS) before V6 and V7.** Without convergence orders you cannot
  tell a bug from a grid difference.
- **Do not let the fast tier exceed three minutes.** People stop running slow suites, and
  a suite nobody runs produces false confidence rather than none.

---

## 12. Open questions for the human

**All seven were answered on 2026-09-11** in the 2026-09-11 decision record (`git show b5b320c:m2/decisions/`),
which is authoritative: Q1/Q2 by the coupon ladder of decision §2 (periodic single feature,
accepted); Q3/Q4 by decision §10 (mollified transitions accepted; two materials in two phases, and
M2.6 stays on the critical path); Q5/Q7 by decision §9 (2D first, endpoint-only output); Q6 by
decision §6 (Lambda H100 80GB SXM for gate runs, CPU for 2D development). The table below is kept
for the reasoning it records.

Answer before the milestone named. Do not guess; write open items into `OPEN_QUESTIONS.md` with
placeholders flagged `provisional: true`.

| # | Question | Blocks | Why it matters |
|---|---|---|---|
| **Q1** | Target feature: CD, depth, pitch, aspect ratio? | M2.1 | Sets domain size and therefore whether the §4 dense decision holds. Above ~15:1 AR at 2 nm the domain grows and sparsity returns to the table. |
| **Q2** | Is a periodic single-feature domain acceptable, or is an isolated feature / local pattern-density variation needed? | M2.1 | Periodic is far cheaper. Microloading needs neighbours in the domain and multiplies cost. |
| **Q3** | Is a mollified material transition acceptable physically, or must the interface see a sharp boundary? | M2.6 | Sharp is more physical and less differentiable. See §5.6 — this is a real trade, not a workaround. |
| **Q4** | First stack: single material, or mask + film + stop layer? | M2.6 | Single material lets M2.6 be deferred and shortens the mission by ~3 weeks. |
| **Q5** | 2D sufficient for the first demo, or is 3D required? | M2.4 | 3D roughly doubles the remaining mission. Most process-engineer intuition is 2D cross-section. |
| **Q6** | GPU available now, and which? | M2.4 | fp64 throughput varies by an order of magnitude across NVIDIA parts. §7.5 assumes fp64 is affordable. |
| **Q7** | Does anything downstream need per-timestep geometry output, or only the final profile? | M2.4 | Checkpointing is much cheaper if only the endpoint is needed. |

---

## 13. Definition of done

> Coupon case S03 evolves in 3D under a prescribed directional velocity in under 60 s **warm** on
> one H100; the Taylor-remainder test passes at second order on both a smooth functional and on
> extracted CD and sidewall angle, and the analytic sensitivities V14a–V14c agree with their
> closed-form answers; the adjoint costs under **4×** the forward pass with two-level
> checkpointing (under 3× in 2D unchecked); a checkpointed recompute reproduces the RNG stream
> bitwise and the trajectory to 1e-12 relative; and a known synthetic parameter set is recovered
> from a synthetic profile from a cold start, **with V14 asserted to pass at the recovered
> parameters** (decision §12) — recovery alone is never evidence of gradient correctness.

Plus: all 22 verification checks pass or carry a logged, explained discrepancy; the
verification ledger is generated and current; and the velocity contract is frozen and
versioned, so M3 can begin without renegotiating it.
