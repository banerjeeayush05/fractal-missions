# BUILD_ORDER.md — M2 product, staged from the capstone

Written 2026-09-14. Companion to `../m2/M2_PRD.md` (the specification) and `../m2/CLAUDE.md`
(scope, conventions, traps). This file is **not** a spec. It is a build order: what to write, in
what order, and which verification check proves each stage is done.

The reference point throughout is the capstone, [`practice/m2/m2_capstone.py`](practice/m2/m2_capstone.py)
— 143 lines that solve a directional etch in 2D and return a finite gradient. Each stage below says
what the capstone already gives you and what it is missing.

The existing `../m2/` tree implements stages 1–15 and passes 251 tests. It is deliberately left in
place. **Write each stage first, then diff against it** — reading it first reproduces its answers
without the reasoning, and the reasoning is the entire point.

---

## Scale

| | lines |
|---|---|
| capstone | 143 |
| product package, stages 1–15 only | 3,210 |
| product tests, stages 1–15 only | 4,131 |
| stages 16–19 | not yet built anywhere |

---

## What the capstone already establishes

These are correct and transfer with little change. They are also the things that are painful to
retrofit, which is why the capstone was worth writing.

- **fp64 enabled globally** ([:5](practice/m2/m2_capstone.py#L5)). V15 needs 1e-10 agreement; fp32 cannot reach it.
- **`eps` inside the `sqrt`** in the Godunov stencil ([:24](practice/m2/m2_capstone.py#L24-L44)). At a Godunov
  valley cell both branches are exactly zero and `d/dx sqrt(x)` is infinite there. One such cell
  returns NaN for every parameter. The comment in the capstone already records why.
- **Smoothed reinitialisation sign** `φ₀/√(φ₀²+dx²)` ([:46](practice/m2/m2_capstone.py#L46-L53)). The exact
  sign function is non-differentiable at the interface and corrupts the adjoint.
- **Fixed loop counts everywhere** — `N`, `REINIT_EVERY`, `REINIT_ITERS` are configuration, never
  computed from the data. A data-dependent loop count makes the step count discontinuous in θ, and
  the gradient through it is wrong in a way that looks like noise.
- **Parameters in a traced array, not a closure** ([:103](practice/m2/m2_capstone.py#L103)). A captured value is
  invisible to `jax.grad` and produces a silently zero gradient column.
- **Sub-cell zero-crossing extraction** ([:66](practice/m2/m2_capstone.py#L66-L79)) with `argmax` for a
  fixed-shape first/last crossing. Cell counting quantises to the grid and its derivative is zero
  almost everywhere.
- **Interpolating the row as well as the crossing** ([:88](practice/m2/m2_capstone.py#L88)) — extracting CD at a
  height that falls between grid rows requires interpolating both.

What the capstone does **not** have is any evidence the gradient is right. It prints `isfinite`,
which is the one property a wrong gradient reliably has.

---

## What it leaves out

| # | capstone | product requires | consequence of the shortcut |
|---|---|---|---|
| 1 | axis 0 lateral, axis 1 vertical ([:19-20](practice/m2/m2_capstone.py#L19-L20)) | axis 0 vertical, increasing toward the plasma; ẑ = +e₀ | every M3 velocity model reads normals against the wrong ẑ |
| 2 | module globals, `N = 300` hand-picked ([:8-14](practice/m2/m2_capstone.py#L8-L14)) | YAML + schema; `N = ceil(D/(cfl_target·dx))` derived from target **depth**, never a final time | a rate correction silently invalidates N, CFL and the golden profile at once |
| 3 | velocity evaluated on all 65,536 cells ([:56](practice/m2/m2_capstone.py#L56)) | two bands — evaluation 1.5 cells (where the model is *called*), extension 8 cells (where the stencils need a value) — with closest-point gather between them | ~80× wasted M3 Monte-Carlo cost; and without a band there is no `VelocityRequest`, so M3 cannot attach at all |
| 4 | `params` is a bare 3-vector | frozen `VelocityRequest`: padded capacity K, `weights`, `cell_id`, `n_active`, `step_index`, `stage_index`, `run_seed` | J becomes discontinuous in θ the moment a cell enters the band; stochastic M3 velocity is unreproducible |
| 5 | CD at **relative** mid-depth ([:85-88](practice/m2/m2_capstone.py#L85-L88)) | CD at fixed **absolute** heights; sidewall angle by least-squares over a fixed absolute depth window | a depth-relative window changes its number of fit points discretely, producing a gradient artifact |
| 6 | Python `for` over N ([:113](practice/m2/m2_capstone.py#L113)) with `@jax.checkpoint` per step ([:102](practice/m2/m2_capstone.py#L102)) | `lax.scan`; two-level nested scan with L derived from **measured** k | a 300×-unrolled graph; and L becomes a knob that can be tuned until a memory gate passes |
| 7 | CFL asserted once from an analytic `VMAX` ([:134](practice/m2/m2_capstone.py#L134)) | per-step CFL returned as a scan aux and checked on the host; the abort message names N | works only because this velocity law has a closed-form maximum. M3's will not |

**And one that is easy to miss.** The capstone's mask is solid in φ with no material field
([:94-99](practice/m2/m2_capstone.py#L94-L99)), so the directional law etches its top face at `v_iso + v_dir`.
The mask erodes. With nothing protecting the field, a directional etch removes the flat surface at
the same rate it deepens the floor, and the trench **translates downward instead of deepening**.
This is finding H2, and it silently affected a real gate case (finding L2).

---

## The ordering correction: the harness comes before the physics

PRD §6 places **V19 in the M2.0 gate** — before any solver exists. `m2/tests/test_v19_canary.py`
states how: *"Built against a trivial analytic function (no solver exists yet)."*

So the Taylor / jvp / vjp harness and its corrupted-gradient canary are **stages 6 and 7**, not
something bolted on after the solver works. This is the opposite of the natural instinct, and the
reason is worth internalising:

> If the harness is first built against the solver, a passing check is ambiguous — it could mean the
> gradient is right, or it could mean the harness is blind. Building it against a function whose
> gradient is known in closed form, and *proving it can fail*, removes that ambiguity permanently.

A verification suite that has never been shown to fail has not been verified.

---

# Stages

## Phase 0 — ground rules

### Stage 1 — Project skeleton
`uv init`; dependencies from PRD §10; `.python-version`; fp64 enabled at package import.

**Capstone:** [:5](practice/m2/m2_capstone.py#L5) does the fp64 line and nothing else.
**Size:** ~20 lines. **Gate:** `uv run pytest` collects.

### Stage 2 — `constants.py`
φ < 0 inside solid. Axis 0 vertical, increasing toward the plasma, so ẑ = +e₀. Velocity models
return an etch rate **R in nm/s, positive removes material**; M2 advects `φ_t − R|∇φ| = 0` so the
sign flip lives in M2 alone. `dτ = 0.5·dx`. Mollified-Heaviside width `1.5·dx`. `w_mat` default
2 cells. `V19_CORRUPTION_FACTOR = 1.05`.

**Capstone:** folds the sign into the velocity return ([:62](practice/m2/m2_capstone.py#L62), returns a negative
number) and uses the opposite axis convention. Both need inverting.
**Gate:** every constant and every sign asserted in a test. Cheapest stage in the list, and the one
that inverts everything downstream if it is wrong.

---

## M2.0 — scaffold, schema, harness, ledger

### Stage 3 — `schema.py`
Frozen dataclasses: `Grid`, `Material`, `Geometry`, `VelocityRequest`, and the `VelocityModel`
callable type. `VelocityRequest` carries `positions`, `normals`, `material_fractions`, `weights`,
`cell_id`, `n_active`, `time`, `step_index`, `stage_index`, `run_seed`. No physics in this file.

**Capstone:** no equivalent — φ is a bare array passed between functions.
**Gate:** shape and dtype contract tests.

### Stage 4 — `config.py`
YAML load. `N = ceil(D / (cfl_target · dx))` derived from target **depth**; an explicit `n_steps`
below the derived minimum is rejected at load time. Cases carry a `role`, and a fit may consume only
`calibrate` cases. `configs/cases/*` refuse to load until the rate has been measured;
`configs/dev/*` carry a nominal rate marked `provisional` and may never certify a check that claims
agreement with the coupon.

**Capstone:** [:8-14](practice/m2/m2_capstone.py#L8-L14) — `T` and `N` are chosen by hand, and nothing prevents a
CFL-violating pair.
**Gate:** load-time rejection tests for each rule above.

### Stage 5 — `verification/registry.py`, `ledger.py`, `pytest_plugin.py`
Pre-registered check IDs (V1–V22, plus V1a and V14a–c), each with a cadence and a tolerance. Ledger
rows record id, description, cadence, last run, git SHA, pass/fail and the measured quantity. A test
claims an ID with `@pytest.mark.check("V14")`. Local runs write an untracked ledger; the ledger of
record is written only by CI on a clean checkout, and `--ledger-of-record` refuses on a dirty tree.

**Capstone:** no equivalent. Results are `print` statements.
**Gate:** a test claiming an unregistered ID fails; rows are actually written.

### Stage 6 — `verification/gradcheck.py`
The intellectual core of the mission, ~500 lines, and it is built against a toy function.

- **Taylor remainder (V14).** Perturbation scales are required, not optional: `δᵢ ∝ max(|θᵢ|, scaleᵢ)`.
  A parameter sitting at 1e-12 would otherwise receive a 1e-12 perturbation and report a pass having
  probed nothing.
- **Anchored window.** Sweep 8 decades of h, record the whole curve, fit the 2 decades immediately
  above the **measured** noise floor. The solver's map is piecewise smooth, so the top of the range
  is not quadratic and says nothing about the gradient; the bottom is where a correct
  `½h²δᵀHδ` and a wrong `|ε·g·δ|·h` separate most.
- **Cancellation ceiling.** |R| must grow with h. Where it does not, two terms are cancelling and the
  log-log slope is meaningless — it reads ~1.6 on a gradient that is exactly right. Cap the window
  below the first notch. This is safe only because the window is anchored at the floor: the cut
  removes points *above* the notch, never below, and below is where a first-order error dominates.
- **Two-zone scoring** over ≥20 directions, each passing on its own. Fewer than 3 points in the
  window is FAIL. Slope < 1.8 is FAIL. [1.8, 2.2] passes as `clean_quadratic`; above 2.2 passes as
  `degenerate_direction`. There is no upper bound to enforce — a wrong gradient leaves the
  first-order term uncancelled and tends to slope 1.
- **V15** — `jax.jvp` against `jax.vjp` on the same directional derivative, 1e-10 relative.
- **V16** — dot-product transpose test: `⟨w, Ju⟩ = ⟨Jᵀw, u⟩`.

**Capstone:** [:143](practice/m2/m2_capstone.py#L143) checks `isfinite`. That is the whole of its gradient evidence.
**Gate:** a correct gradient passes all three on the toy function.

### Stage 7 — `verification/analytic.py` and `canary.py` — V19
A trivial analytic objective with a closed-form gradient. Then the canary: corrupt one gradient
component by ×1.05 and assert V14, V15 and V16 **all fail**, with an unmutated control passing.
Then the canary's own mutation guard — five sabotaged harnesses (V14/V15/V16 always passes; the
whole harness always fails; the corruption is a no-op) must each turn V19 red.

**Gate: this closes M2.0.**

**Two findings you inherit here — do not spend a week rediscovering them:**

- **A9.** V15 and V16 are **transpose checks, not derivative checks**. JAX builds reverse mode by
  linearising with the JVP rules and transposing them, so both share those rules. Corrupt a
  `custom_jvp` rule and jvp and vjp agree on the *wrong* derivative: V15 and V16 pass, only V14
  fails. The PRD's "separate code paths" claim is wrong and is amended.
- **A10.** The [1.8, 2.2] band **rejects correct gradients along directions of zero curvature**.
  If `δᵀ∇²J δ = 0` the remainder is O(h³) or pure roundoff. V2's own setup — a flat front under
  isotropic V, displacement linear in V — is exactly that case.

---

## M2.1 — 2D forward and velocity-request assembly

### Stage 8 — `stencils.py`
Godunov upwind Hamiltonian for |∇φ|; one-sided differences; `eps` inside the sqrt; periodic lateral
and Neumann vertical boundaries.

**Capstone:** [`grad_mag`](practice/m2/m2_capstone.py#L24-L44) is correct and transfers directly, with the axes
swapped per stage 2. Note it already takes `w` and uses only `sign(w)` — V for advection, S for
reinitialisation — which is the right factoring.

### Stage 9 — `solver.py`
TVD-RK2 (Heun) inside `lax.scan`, fixed N, `dt = T/N`. Per-step CFL returned as a **scan auxiliary
output** and asserted on the host after the call — not with `checkify`, which avoids any question
about how `checkify` composes with `grad`. The abort message names N.

**Capstone:** [`step`](practice/m2/m2_capstone.py#L102-L109) has the RK2 right, including recomputing V at stage 2.
The loop ([:113](practice/m2/m2_capstone.py#L113)) is Python and unrolls; the CFL check
([:134](practice/m2/m2_capstone.py#L134)) fires once, from an analytic bound.

### Stage 10 — `initial.py`
Exact analytic φ for circle, plane, trench and trapezoid cases. Vertical extent covers the initial
stack plus maximum expected etch depth plus a 10-cell buffer.

**Capstone:** [`initial_phi`](practice/m2/m2_capstone.py#L94-L99) builds a trench by `min`/`max` of half-spaces,
then reinitialises — a reasonable pattern, but V13 will later need an *exact* trapezoid SDF rather
than a reinitialised approximation.

### Stage 11 — `band.py`, `interface.py`, `velocity.py`, `rng.py`
The largest single jump from the capstone.

- **Two bands.** Extension band 8 cells, weight tapering to zero over the outer 2: the interface
  moves up to `CFL·reinit_every ≈ 2` cells between reinitialisations, reinitialisation propagates
  about `n_reinit = 5` cells, and the upwind stencil reaches 1 more. Evaluation band 1.5 cells,
  where the velocity model is actually called; it stays thin because deeper cells project to nearly
  the same surface point and M3 would pay for the same Monte Carlo estimate several times.
- **Closest-point projection and gather** — M3 evaluates on the thin set, M2 gathers outward across
  the full band at negligible cost.
- **Fixed-capacity padded set** of size K, sized from the **worst** step, not the first: the
  interface lengthens as the trench deepens and overflow must not fire at step 900 of 1000. Peak
  occupancy is reported every run; overflow aborts like the CFL assertion.
- **`cell_id` and the smooth weight are two halves of one mechanism, not two safeguards.** The id
  stops a cell's arrival in the band from shifting everyone else's random draws; the weight reaching
  exactly zero at the band edge stops the arriving cell's own draw from entering discontinuously.
  Remove either and J is discontinuous in θ. `cell_id` is opaque: seed with it, never index geometry.
- **RNG key** = f(run_seed, step_index, stage_index, cell_id). Never global, never stateful, never
  hashed from `time`.
- **Two analytic models:** isotropic (R constant) and directional (`R = v₀·max(0, n·ẑ)^p`).
- **NaN discipline.** `weights = 0` does not protect against NaN — `0 * NaN` is NaN — so padded
  entries need a *benign position*, and normals use the double-where pattern regardless of weight.

**Capstone:** [`velocity`](practice/m2/m2_capstone.py#L56-L62) computes normals on every cell with `eps` inside the
sqrt, which is the right guard but the wrong architecture. There is no band, no request object and
no RNG.

**Gate (closes M2.1): V1 in reduced form, V2, V1a, V20.**
V1a is the only guard against a sign inversion — V14 through V16 cannot catch one. V1 runs reduced
here (travel held inside the extension band) because without reinitialisation a band-limited
extension distorts φ until the front stalls.

---

## M2.2 — reinitialisation and extension

### Stage 12 — `reinit.py`
Iterate `∂φ/∂τ + sign(φ₀)(|∇φ| − 1) = 0` for a **fixed** `n_reinit` (default 5) every
`reinit_every` steps (default 5), with the smoothed sign. Plus velocity extension across the 8-cell
band, either by the PDE `∂V/∂τ + sign(φ)(∇φ/|∇φ|)·∇V = 0` or by closest-point gather. No
stop-gradient on either — they are real parts of the forward map and their derivatives are real.

**Capstone:** [`reinit`](practice/m2/m2_capstone.py#L46-L53) is essentially the product version already, including
`dtau = 0.5*dx` and the fixed count. What is missing is the extension.

**Gate (closes M2.2): V8, V9, V10, V11, V12.**
V12 — reinitialisation does not move the interface, drift < 0.1 % per cycle — is the most commonly
missed check in level-set codes. Reinitialisation that quietly shifts the zero level set produces a
systematic etch-rate bias that looks like physics, calibrates away into the closure parameters, and
then fails to transfer.

---

## M2.3 — reverse-mode adjoint, 2D

### Stage 13 — point the stage-6 harness at the solver
No new harness code. That is the payoff of building it first.

- `verification/params.py` — the solver's parameter scales.
- **V14a** — V1 isotropic etch: `dr/dR = −T`.
- **V14b** — V2 plane translation: `d(depth)/dR = +T`. The magnitude is T either way; the sign
  follows from the convention, which is the point of the check.
- **V14c** — a tilted plane under the directional law, moving along its normal at `v₀·cos^p(θ)`;
  derivatives in `v₀` and `p` exact. Also guards the normal-vector convention.
- Measure the residual factor k and the adjoint ratio. **Report, do not gate.**
- Report V5, V6, V7 convergence orders.

**Capstone:** [`objective`](practice/m2/m2_capstone.py#L126-L128) and the `jax.grad` call are the shape of this, but
targets are taken from the model's own simulation ([:123](practice/m2/m2_capstone.py#L123)) and the evidence is
`isfinite`.

**Gate (closes M2.3): V14, V15, V16 on the smooth volumetric functional; V14a, V14b, V14c.**

**Two traps specific to this stage.**
The directional law has a kink on vertical sidewalls (`n·ẑ = 0`): require `p > 1`, use `p ≥ 2` in
gradient tests, and keep `0^p` out of the p-derivative. And sub-cell extraction is continuous but
not smooth — the derivative jumps when a crossing passes a grid node — so V14 on CD is touchier than
V14 on volume. Check that before hunting a bug.

---

## Stages 14–19

| stage | milestone | delivers | gate |
|---|---|---|---|
| **14** | interlude | WENO5 behind a flag. Only D⁻ and D⁺ change; the Godunov upwind selection on top is untouched; thread the flag into reinitialisation as well as advection. **ε fixed, not the published stencil `max`** — that max is a kink in the differentiated path. Falls back to the two-point stencil within 3 cells of a non-periodic boundary, or it reads the Neumann-replicated edge value as smooth data and invents a phantom interface | reconstruction order verified; V14/V15/V16 re-run |
| **15** | M2.4 | `checkpoint.py` as a **nested scan** parametrised by segment length L: peak = N/L + L·k, optimal at L\* = √(N/k). Derive L from **measured** k, never hand-pick it, or it becomes a knob that makes a memory gate pass. 3D through the same code path, dimension a config field | V17 (checkpointed = unchecked to 1e-12), V18 (RNG keys and sampled values bitwise identical on recompute, trajectory to 1e-12), 3D V14. Peak memory < 40 GB and warm adjoint ratio ≤ 4× need an H100 |
| **16** | M2.5 | `functionals.py` — the smooth `solid_volume = ∫(1−H(φ))dV` with mollified H at `1.5·dx`, which exists only to power the §8.7 diagnostic. Then extraction: depth, CD at **fixed absolute** heights, sidewall angle by least-squares over a fixed absolute depth window, bow, mask_remaining | V13 including the sub-cell smoothness sweep — a staircase there means every gradient through extraction is garbage. V14 on `CD_mid` and `sidewall_angle`, not only on the volumetric functional |
| **17** | M2.6 | Per-material **fraction** field summing to 1, never an integer ID field — an integer ID makes the rate jump a step function whose derivative is zero everywhere and undefined at one point. Mollify over `w_mat`. The mask becomes a solid body in φ with an etch-rate multiplier of **zero**: infinite selectivity preserved, pattern transfer restored, undercut beneath the mask edge allowed, the mask corner a real corner | gradient survives an interface crossing a material boundary; `w_mat` sensitivity study at 1, 2, 4 and 8 cells, reported as a fraction of the marker thickness, **reported honestly whatever it shows** |
| **18** | M2.7 | V3 (collimated aperture, p = 64), V22 (ViennaPS cross-code — separate process, VTK exchange, GPL clean room, and V6/V7 must run **first** or you cannot distinguish a bug from a grid), and `verification.md` **generated** from the ledger including the §8.9 scope boundary | case S03 in 3D < 60 s warm on one H100, compile time reported separately |
| **19** | M2.8 | Inverse sanity: recover a known synthetic parameter set from a synthetic profile **with noise**, from a cold start | recovery to within the noise floor. Note the capstone takes its targets from its own simulation, which is an inverse crime; this stage adds noise and a cold start |

---

## Check index

| stage | checks it closes |
|---|---|
| 2 | sign convention asserted |
| 5 | ledger writes rows |
| 7 | **V19** (+ its mutation guard) |
| 11 | **V1** (reduced), **V1a**, **V2**, **V20** |
| 12 | **V8**, **V9**, **V10**, **V11**, **V12** |
| 13 | **V14**, **V14a**, **V14b**, **V14c**, **V15**, **V16**; V5/V6/V7 reported |
| 15 | **V17**, **V18**, 3D V14 |
| 16 | **V13**, V14 on CD and sidewall angle |
| 17 | `w_mat` study |
| 18 | **V3**, **V22**, V21 golden profile |
| 19 | inverse sanity |

V4 is on hold: under a monotone cos^p law the vertical etch rate peaks at normal incidence, while
classical facet-angle results assume a rate peaking off-normal, which needs M5 yield curves. V14c
covers what V4 was really guarding.

---

## Rules that hold at every stage

- **Never loosen a tolerance to make a check pass.** A failure is a finding. If a tolerance is
  genuinely wrong, change it in its own commit, with the reason.
- **Never resolve an ambiguity by guessing.** It goes in `OPEN_QUESTIONS.md`, classified A (PRD
  error), B (placeholder, with the mechanical rule that produced it), C (gap blocking a later
  milestone), D (minor). Assume the PRD contains arithmetic and cross-reference errors — it does,
  and several have been found.
- **Stop at every gate.** Run the checks, report, wait. Never build ahead.
- **No `stop_gradient`** on reinitialisation, extension or material blending. The band weights look
  like a mask, which is exactly what a future reader will wrap to make a test pass. They participate
  in the gradient.
- **Do not tune `w_mat`, CFL, band width or `n_reinit`** to make a gradient test pass. That is a
  finding, not a fix.
- **Answers supersede the PRD.** When a decision lands, amend the PRD in the same commit that
  implements it.
- **Diagnostics that justify a decision must be tracked.** One that lived in an untracked scratch
  directory is gone, and the measurement that justified building WENO5 is no longer reproducible
  (finding J6).

## The governing constraint

**A gradient that is wrong but plausible is worse than no gradient.** An optimiser fed a subtly
wrong gradient still converges — to the wrong recipe, confidently, with nothing in the output
announcing it. Almost every unusual rule above exists because of that sentence.
