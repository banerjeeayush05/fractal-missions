# M2 — Differentiable Geometry Core
## Product Requirements Document (agent-facing)

**Repo:** `fractal-m2`
**Owner:** Fractal Semiconductor
**Window:** 22 weeks (Path A, weeks 0–22)
**Path:** A (recipe solve) — critical path. Also required on Path B.
**Status of this document:** authoritative. Where this PRD conflicts with a habit or a
default, follow this PRD.

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
class VelocityRequest:            # what M2 hands to a velocity model
    positions: Array              # interface-adjacent cell centres
    normals: Array
    material_fractions: Array
    time: float
    step_index: int               # REQUIRED — see §7.6 on stochastic velocity

# Velocity model signature. M2 ships analytic implementations only.
VelocityModel = Callable[[VelocityRequest, PyTree], Array]   # -> speed in nm/s
```

`PyTree` is the differentiable parameter set. **Every parameter M2 differentiates with
respect to must live in that PyTree**, never captured in a closure — a captured value is
invisible to `jax.grad` and produces a silently zero gradient column.

### 5.1 Level-set representation

- φ signed distance, **negative inside solid**, positive in the open volume. Fix this
  convention in `constants.py` and assert it in tests; sign errors here are the single
  most common source of an inverted gradient.
- Vertical extent covers initial stack plus the maximum expected etch depth plus a
  10-cell buffer.
- Lateral boundaries **periodic**. Top and bottom **Neumann** (zero normal derivative).
- 2D and 3D from the same code path, dimension as a config field. Build 2D first: it runs
  in seconds and every validation case in §8 is visualisable there.

### 5.2 Time integration

Advection of φ under normal speed V:  ∂φ/∂t + V |∇φ| = 0

- Spatial: Godunov upwind Hamiltonian for |∇φ|. WENO5 is an option behind a flag but is
  **not** required for M2 — first-order upwind with adequate resolution is sufficient to
  verify gradients, and the extra stencil complexity is a place for bugs to hide.
- Temporal: TVD-RK2 (Heun). RK3 behind a flag.

**Fixed step count. This is a hard requirement.**

The natural scheme is `dt = CFL · dx / max|V|`, integrating to a fixed final time T. But
`max|V|` depends on the parameters, so `dt` does, so the number of steps `N = ceil(T/dt)`
changes **discontinuously** with parameters. The gradient through a discontinuous step
count is wrong, and it is wrong in a way that looks like noise rather than like a bug.

Instead: fix `N` from config, set `dt = T/N`, and **assert** that
`CFL = max|V|·dt/dx ≤ 0.5` at every step. If the assertion fires, the run aborts and the
user raises `N`. Fixed `N` also gives JAX a static computation graph, which the whole
approach depends on.

### 5.3 Reinitialisation

φ drifts away from a signed distance function under advection. Re-impose it by iterating
the reinitialisation PDE  ∂φ/∂τ + sign(φ₀)(|∇φ| − 1) = 0  for a **fixed** number of
iterations `n_reinit` (config, default 5), every `reinit_every` steps (default 5).

- `sign(φ₀)` must be the **smoothed** sign, `φ₀ / sqrt(φ₀² + dx²)`. The exact sign
  function is non-differentiable at the interface and will corrupt the adjoint.
- Fixed iteration count is what makes this ordinary differentiable code. Do not implement
  a convergence-based stopping criterion — a data-dependent loop count reintroduces the
  §5.2 problem.
- Do **not** apply a stop-gradient to reinitialisation. It is a real part of the forward
  map and its derivative is real.

### 5.4 Velocity extension

Velocity is defined on the interface; advection needs it throughout the band. Extend by
solving  ∂V/∂τ + sign(φ)(∇φ/|∇φ|)·∇V = 0  for a fixed iteration count, or by the simpler
`closest-point` gather. Either is fine; both must be differentiable and use fixed counts.

### 5.5 The M3 velocity contract — freeze this early

M2 must be usable by M3 without modification. Freeze the `VelocityModel` signature above
by milestone M2.3 and version it. M3 will supply a **stochastic, expensive** velocity;
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

**Required diagnostic:** report gradient sensitivity to `w_mat` at 1, 2, 4 and 8 cells.
If the gradient depends strongly on a purely numerical smoothing width, that is a finding
and must be reported, not tuned away. See open question **Q3**.

### 5.7 Differentiable geometry extraction

Outputs an engineer cares about: `depth`, `CD` at three heights, `sidewall_angle`,
`mask_remaining`, `bow`.

- **Sub-cell extraction only.** Marching cubes / squares with linear interpolation of the
  zero crossing. Cell counting quantises the output to the grid spacing and its derivative
  is zero almost everywhere. This was the same requirement in M1 and for the same reason.
- Sidewall angle by least-squares fit to contour points in a specified depth window,
  not by finite-differencing two CD values.
- **Also expose a smooth volumetric functional**, `etched_volume = ∫(1 − H(φ)) dV` with
  mollified H. This is smooth by construction and is the diagnostic in §8.2.

---

## 6. Milestones and gates

| # | Milestone | Gate |
|---|---|---|
| **M2.0** | Repo scaffold, `CLAUDE.md`, schema, config, CI, **verification ledger** | `pytest` green; sign convention asserted; **V19** passes (the canary is caught); ledger writes rows |
| **M2.1** | 2D forward: advection, fixed-N stepping, CFL assertion | **V1, V2, V20** |
| **M2.2** | Reinitialisation + velocity extension | **V8, V9, V10, V11, V12** — V12 especially, see §8.3 |
| **M2.3** | **Reverse-mode adjoint, 2D. Velocity contract frozen.** | **V14, V15, V16** on the smooth functional; adjoint ≤3× forward cost. V5–V7 convergence orders reported |
| **M2.4** | Checkpointing + 3D | **V17, V18**; 3D **V14**; peak memory <8 GB at 512×128×128 with N=500; recompute overhead ≤2× |
| **M2.5** | Differentiable extraction | **V13** including the sub-cell smoothness sweep; **V14** on `CD_mid` and `sidewall_angle`, not only the volumetric functional |
| **M2.6** | Multi-material | Gradient survives an interface crossing a material boundary; `w_mat` sensitivity study reported |
| **M2.7** | Performance + external verification | **V3, V4, V22**; full-resolution 3D **V14**; 3D 500 nm trench <60 s on one GPU; `verification.md` generated from the ledger |
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
Use recursive (binomial) checkpointing — `jax.checkpoint` on the step function, with
√N checkpoints — giving ~750 MB and ~2× recompute. Griewank & Walther's `revolve` is the
reference for the optimal schedule; JAX's `remat` policy is sufficient here.

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

Requirement: the RNG key must be a deterministic function of `step_index` and a run-level
seed, both passed through `VelocityRequest`. Never use global or stateful RNG. Add a test
at M2.4 that a checkpointed recompute reproduces the forward trajectory bitwise.

---

## 8. Verification

M11 owns the *institution* — standards, cadence, the ViennaPS relationship, the public
record. The **checks themselves live here and gate M2's milestones.** A verification
result that arrives after the code it validates is archaeology, not verification.

Twenty-two checks, each with an ID. Milestone gates in §6 cite them by ID. Every check
writes a row to `reports/verification_ledger.json`: id, description, cadence, last run,
git SHA, pass/fail, and the measured quantity. `reports/verification.md` is **generated**
from that ledger, never hand-written — a hand-maintained verification record drifts from
reality within a month.

### 8.0 Cadence

| Tier | When | Runtime budget | Checks |
|---|---|---|---|
| **Fast** | every commit, in CI | < 3 min, 2D only, coarse | V1, V2, V11, V12, V14 (reduced), V15, V16, V19, V20, V21 |
| **Nightly** | scheduled, on main | < 60 min, 2D full + small 3D | V5, V6, V7, V8, V9, V10, V13, V17, V18 |
| **Gate** | at the milestone that cites it | hours | V3, V4, V22, full-resolution 3D V14 |

Fast tier is non-negotiable. If it exceeds three minutes people stop running it, and a
verification suite nobody runs is worse than none because it produces false confidence.

### 8.1 Forward verification — analytic solutions

These have exact answers. Disagreement is unambiguous.

- **V1 — Isotropic growth.** Circle (2D) / sphere (3D) under V = const. Radius must satisfy
  r(t) = r₀ + Vt. Tolerance: relative error < 1% at 200 steps. Tests advection,
  reinitialisation and extension together, and is the cheapest signal that something broke.
- **V2 — Plane translation.** Flat interface under constant V translates at exactly V with
  no distortion. Error < 0.1% — this one should be nearly exact, and if it is not the
  upwind scheme or the boundary condition is wrong.
- **V3 — Collimated aperture limit.** Unity yield, perfectly collimated directional etch
  (p → ∞) must reproduce the aperture shape: vertical sidewalls, no bow, no faceting.
  Sidewall angle within 0.5° of 90°. This is the sanity limit an etch engineer will check
  first.
- **V4 — Facet angle.** A mask corner under V = v₀·max(0, n·ẑ)^p reaches a steady facet
  angle set by p. Compare against the analytic steady angle. This is the test that catches
  a wrong normal-vector convention, which V1 and V2 will both miss.

### 8.2 Forward verification — convergence order

Order studies are what distinguish "wrong" from "under-resolved," and without them you
cannot tell whether a discrepancy against ViennaPS is a bug or a grid.

- **V5 — Method of manufactured solutions.** Choose φ_exact(x, t), substitute into
  ∂φ/∂t + V|∇φ| = 0 to obtain a source term S, add S to the RHS, and verify the observed
  order. Requires an MMS-only code path with a source term; keep it behind a flag and test
  that the flag is off in production runs.
- **V6 — Spatial order.** Fix dt very small, refine dx over ≥4 levels, measure L¹ and L∞
  error against V1's analytic solution. Godunov upwind must show observed order ≥ 0.9.
  WENO5, if enabled, ≥ 4.
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

### 8.4 Extraction verification

- **V13 — Analytic geometry.** Initialise φ to an *exact* trapezoid of known CD, depth and
  sidewall angle. Extract. CD error < 0.1 nm, sidewall angle < 0.2°. Sweep the trapezoid's
  position by sub-cell offsets and confirm the extracted CD varies **smoothly**, not in
  grid-sized jumps — a staircase here means extraction is not sub-cell and every gradient
  through it is garbage.

### 8.5 Gradient verification — four independent checks

One gradient test is not enough. These fail in different ways, which is the point.

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
- **V18 — Checkpoint recompute is bitwise.** §7.6. The recomputed forward trajectory must
  match the original bitwise. Guards the stochastic-velocity trap before M3 can spring it.
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

Record this scope boundary in `reports/verification.md` so that a reader does not mistake
a green M2 suite for a validated process model.

## 9. Deliverables

1. `fractal-m2` repo, running from clean checkout with `uv sync && pytest`.
2. `reports/verification_ledger.json` — machine-readable record of all 22 checks: id,
   cadence, last run, git SHA, result, measured value. Written by the checks themselves.
3. `reports/verification.md` — **generated** from the ledger. Includes the §8.9 scope
   boundary and every open discrepancy. This is the artifact that makes every downstream
   mission trustworthy, and the one a sceptical customer will ask for.
4. `reports/gradient_verification.md` — Taylor and forward-vs-reverse plots for every
   objective, at every milestone, dated.
5. `reports/w_mat_sensitivity.md` — §5.6 diagnostic.
6. `m2/interface.py` — the frozen, versioned velocity contract for M3.
7. `reports/simplifications.md` — every simplification, with justification.

---

## 10. Environment

```bash
uv init fractal-m2 && cd fractal-m2
uv add "jax[cuda12]" numpy scipy matplotlib
uv add scikit-image      # marching cubes/squares, sub-cell extraction
uv add pyyaml omegaconf pytest hypothesis
uv add pyvista           # VTK output for ViennaPS comparison
```

Do **not** add: PyTorch, Warp, PETSc, ViennaPS (see §8.3), or any autodiff framework
besides JAX. One autodiff system, so that a gradient bug has one place to be.

---

## 11. Anti-requirements — do not do these

- **Do not accept "the optimiser converges" as evidence of gradient correctness.**
- **Do not use an adaptive or convergence-based loop count** anywhere in the differentiated
  path — timestep count, reinitialisation iterations, extension iterations. All fixed.
- **Do not apply `stop_gradient`** to reinitialisation, extension, or material blending
  to make something converge. If you believe a stop-gradient is needed, stop and ask.
- **Do not use `jnp.sign`, `abs`, `maximum` on φ near the interface** without a mollified
  form. Each is a place the derivative silently dies.
- **Do not extract CD by cell counting or thresholding.** Sub-cell only.
- **Do not capture differentiable parameters in closures.**
- **Do not start in fp32.**
- **Do not implement narrow-band sparsity.** §4.
- **Do not implement flux, shadowing or visibility.** That is M3, and building a
  placeholder version will create a contract M3 then has to fight.
- **Do not tune `w_mat`, CFL or `n_reinit` to make a gradient test pass.** If a test only
  passes at one setting, that is the finding.
- **Do not vendor, link or copy ViennaPS.** GPL-3.0. §8.8.
- **Do not hand-write `reports/verification.md`.** Generate it from the ledger.
- **Do not loosen a verification tolerance to make a check pass.** Raise it as a finding.
  If a tolerance is genuinely wrong, change it in a commit that says so and explains why.
- **Do not run V22 (ViennaPS) before V6 and V7.** Without convergence orders you cannot
  tell a bug from a grid difference.
- **Do not let the fast tier exceed three minutes.** People stop running slow suites, and
  a suite nobody runs produces false confidence rather than none.

---

## 12. Open questions for the human

Answer before the milestone named. Do not guess; follow the M1 open-questions pattern and
write them into `OPEN_QUESTIONS.md` with placeholders flagged `provisional: true`.

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

> A 3D trench evolves under a prescribed directional velocity in under 60 s on one GPU;
> the Taylor-remainder test passes at second order on both a smooth functional and on
> extracted CD and sidewall angle; the adjoint costs under 3× the forward pass; a
> checkpointed recompute reproduces the forward trajectory bitwise; and a known synthetic
> parameter set is recovered from a synthetic profile from a cold start.

Plus: all 22 verification checks pass or carry a logged, explained discrepancy; the
verification ledger is generated and current; and the velocity contract is frozen and
versioned, so M3 can begin without renegotiating it.
