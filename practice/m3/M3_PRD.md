# M3 — Feature-Scale Transport and Flux
## Product Requirements Document (agent-facing)

**Repo:** `fractal-m3`
**Owner:** Fractal Semiconductor
**Window:** 22 weeks (Path A, weeks 6–28)
**Path:** A (recipe solve) — critical path. Required at reduced fidelity on Path B.
**Depends on:** M2 (frozen `VelocityModel` contract, M2 PRD §5.5)
**Status of this document:** authoritative. Where this PRD conflicts with a habit or a
default, follow this PRD.

---

## 0. How to use this document

Work **milestone by milestone** in the order in §6. Each has acceptance tests. **Stop at
each gate, run the tests, report results, and wait for review.** Do not build ahead.

If a requirement is ambiguous or you believe it is wrong, say so and stop. Do not resolve
ambiguity by guessing. Write `OPEN_QUESTIONS.md` following the M1/M2 pattern: class **A**
contradictions that block work, class **B** placeholders with the mechanical rule that
produced them, class **C** gaps blocking a later milestone, class **D** minor.

Write `CLAUDE.md` at repo root as your first action, summarising §3 (non-goals), §4 (the
sphere-tracing decision), §7 (numerical requirements) and §11 (anti-requirements).

**Assume this document contains errors.** Check its arithmetic and cross-references.

---

## 1. What this mission is for

M2 evolves a surface given a velocity field. M3 supplies that field. Formally M3
implements

> Γ : (S, θ) ↦ {Γ_s(x)}_s for all x ∈ S

the per-species particle flux arriving at each point of the surface S, given the surface
itself and the parameters θ.

**The structural fact that makes this hard:** Γ_s(x) depends on the *entire* surface, not
on a neighbourhood of x. A point deep in a trench receives less flux because the rest of
the trench is in the way. Consequently ∂Γ_s(x)/∂θ has contributions from geometry changes
arbitrarily far from x, and those contributions arrive through a **discontinuous**
visibility function.

M3 does **not** own surface chemistry. It consumes scattering distributions and sticking
coefficients through a contract (§5.6) and produces flux. M5 owns what those
distributions are.

---

## 2. The single most important design constraint

**Automatic differentiation applied naively to a Monte Carlo ray tracer produces a
biased gradient, and the bias is silent.**

The proof is one line. Let

> I(θ) = ∫₀¹ ℋ(x − θ) dx = 1 − θ,  so dI/dθ = −1

Form the standard unbiased estimator: draw x ~ U(0,1), take Î = ℋ(x − θ). Differentiate
pathwise, which is what AD does:

> ∂Î/∂θ = −δ(x − θ) = 0 for almost every x,  hence 𝔼[∂Î/∂θ] = 0 ≠ −1

Nothing is infinite. Nothing raises. The variance is zero. The estimator confidently
returns the wrong answer, and it returns it precisely at mask edges, corners and
silhouettes — the geometric configurations that determine the profile.

A ray tracer built with `jax.grad` and nothing further will pass a smoke test on a flat
surface and be wrong on every real feature. **§5.4 is not an optimisation. It is the
mission.**

---

## 3. Scope and non-goals

### In scope
- Ray casting against the M2 level set by sphere tracing.
- Multi-species path-traced transport with next-event estimation and Russian roulette.
- Unbiased gradients through visibility discontinuities (warped-area sampling).
- Variance reduction sufficient to meet the gradient noise floor of §7.4.
- Conformance to the frozen `VelocityModel` contract from M2 PRD §5.5.

### Explicitly NOT in scope — do not build these
- **Surface chemistry.** Yield curves, sticking values, polymer kinetics, selectivity.
  All M5. M3 consumes them through §5.6 and ships only analytic test distributions.
- **Geometry evolution.** M2 owns the level set. M3 reads φ and never writes it.
- **Reactor-scale anything.** The source distribution arrives through the M7 interface.
- **Charging.** Deferred. Leave the hook in §5.1 and nothing else.
- **Gas-phase collisions inside the feature.** Forbidden by §4.1, not merely unnecessary.
- **Product redeposition in v0.** Schema support required (§5.7); implementation gated.
- **Calibration, optimisers, posteriors.** M8.

If you want to add any of the above, stop and ask.

---

## 4. Architectural decisions

### 4.1 The transport regime — why this is a rendering problem

Mean free path of a hard-sphere gas: λ = k_B T / (√2 π d² p). At p = 20 mTorr = 2.67 Pa,
T = 300 K, d ≈ 4×10⁻¹⁰ m:

> λ ≈ (1.38×10⁻²³ × 300) / (1.414 × 3.1416 × 1.6×10⁻¹⁹ × 2.67) ≈ 2.2 × 10⁻³ m

against a feature of order 10⁻⁶ m, giving **Kn = λ/L ≈ 2×10³**.

Kn ≫ 1 is the free-molecular limit: particles travel ballistically between surface
encounters. The Boltzmann collision integral vanishes and the governing equation
degenerates to free streaming with surface boundary conditions — which is the rendering
equation. **This licenses the entire architecture and must be checked, not assumed, for
the target process** (open question Q1).

### 4.2 Ray casting: sphere tracing, not mesh extraction

| Option | Verdict | Reasoning |
|---|---|---|
| **Sphere tracing on φ** | **Selected** | M2 maintains φ as a signed distance function — that is what reinitialisation is for — so a ray may advance by \|φ(x)\| without risk of crossing the surface (Hart 1996). No mesh, no acceleration structure, composed entirely of differentiable operations on φ, and the hit-point derivative follows from the implicit function theorem (§5.2). |
| Marching cubes + BVH | Rejected | The BVH and the triangle connectivity are discrete structures that change combinatorially as geometry moves. This adds a second, independent family of discontinuities on top of the physical silhouette discontinuities that §5.4 already has to handle. |
| Voxel traversal (DDA) | Rejected | Robust and simple, but the intersection point is quantised to cell boundaries, which reintroduces the extraction-quantisation failure of M2 PRD §5.7. |

### 4.3 Estimator: path tracing, not radiosity

A deterministic alternative exists. Under purely diffuse re-emission the problem reduces
to a dense linear system in surface flux with view-factor coefficients, differentiable by
implicit differentiation and free of the §2 problem entirely.

**Rejected as primary, retained as a verification reference (V33).** The decisive
argument is angular dependence: ion specularity and angle-dependent sticking are
first-order effects in the profiles we care about and radiosity cannot express them at
any cost. Secondary: view factors are O(N²) to build and degrade in deep concavity.

### 4.4 Differentiation: warped-area sampling

Three published methods handle the §2 problem. Selected: **warped-area sampling**
(Bangaru, Li & Durand, SIGGRAPH Asia 2020).

| Method | Verdict |
|---|---|
| Edge sampling (Li et al. 2018) | Reference implementation only, for V35. Correct and easiest to understand, but requires explicit silhouette detection — a codimension-one set that must be located numerically on a level set — and carries its own sampling variance. |
| Reparameterisation (Loubet et al. 2019) | Rejected. Avoids silhouette detection but the original formulation carries a small bias, and constructing the map is delicate. |
| **Warped-area sampling** | **Selected.** Converts the boundary integral to an interior integral by the divergence theorem, so one ordinary area-sampling process estimates both terms. Unbiased, no silhouette detection, constant-factor cost. |
| Score function / REINFORCE | **Forbidden** (§11). Unbiased and discontinuity-tolerant, but variance does not meet §7.4 at feasible sample counts. |

---

## 5. Technical specification

### 5.0 Data contracts — build these first, before any physics

```python
# m3/schema.py
@dataclass(frozen=True)
class Species:
    name: str
    index: int
    mass_amu: float
    is_ion: bool
    charge: int = 0          # hook for charging; unused in v0

@dataclass(frozen=True)
class SourceSpec:
    """Arrives from M6 across the M7 interface. In v0, analytic."""
    species: Species
    kind: str                # "cosine" | "gaussian_cone" | "tabulated_ieadf"
    params: PyTree           # differentiable

@dataclass(frozen=True)
class TransportConfig:
    n_primary: int           # samples per surface point. FIXED, see §7.1
    max_bounces: int         # hard cap; RR handles termination, §5.3
    rr_start_bounce: int
    sphere_trace_iters: int  # FIXED, see §5.2
    use_nee: bool            # must be True in production, §5.3
    warp_enabled: bool       # must be True in production, §5.4
    flux_lag_steps: int      # §5.8
```

The velocity-side contract is **already frozen by M2 PRD §5.5** and must not be
renegotiated:

```python
VelocityModel = Callable[[VelocityRequest, PyTree], Array]
# VelocityRequest carries: positions, normals, material_fractions, time, step_index
```

`step_index` is load-bearing: it seeds the RNG deterministically (§7.5). Every
differentiable parameter must live in the `PyTree`, never in a closure — a captured value
is invisible to `jax.grad` and yields a silently zero gradient column.

### 5.1 Source distributions

Analytic in v0; the M7 interface replaces them later without changing the signature.

- **Cosine (neutrals).** L(ω) ∝ μ. Sample by inverse CDF; derive the sampler rather than
  copying it.
- **Gaussian cone (ions).** L(ω) ∝ exp(−θ²/2σ²) with σ from the sheath: σ ≈ √(T_i/2eV_s),
  of order 1° for T_i ~ 0.05 eV and V_s ~ 200 V. σ is a **differentiable parameter**.
- **Tabulated IEADF.** Bilinear interpolation over (E, θ) with differentiable table
  values. This is the shape M6 will deliver.

Energy is carried alongside direction for ions, because M5's yield depends on it. M3 does
not interpret energy; it transports it.

### 5.2 Sphere tracing

```
t₀ = 0,  t_{k+1} = t_k + |φ(x₀ + t_k ω)|
```

- φ sampled by **trilinear interpolation** — differentiable, and consistent with M2's
  sub-cell convention. Do not nearest-neighbour sample.
- **Fixed iteration count** from `sphere_trace_iters`. A convergence-based loop makes the
  computational graph depend on θ and corrupts the gradient exactly as M2 PRD §5.2
  describes for timesteps. This rule has no exceptions.
- Grazing rays converge slowly. Record the fraction of rays that exhaust the iteration
  budget without converging; if it exceeds 1%, raise the budget and report it.

**Hit-point derivative.** The intersection satisfies φ(x₀ + tω; θ) = 0. Differentiating,

> ∇φ·ω (∂t/∂θ) + ∂φ/∂θ = 0  ⟹  **∂t/∂θ = −(∂φ/∂θ)/(∇φ·ω)**

Exact, and free given quantities already computed. The denominator vanishes at tangency —
which is the silhouette, which is where visibility jumps. **The implicit-function
singularity and the visibility discontinuity are the same phenomenon**, and §5.4 is what
handles it. Do not attempt to regularise the denominator; that trades a correct
singular term for a smooth wrong one.

### 5.3 The estimator

Flux at x is Γ_s(x) = ∫_{H(x)} L_s(x ← ω) μ dω, with L given by the transport equation

> L_s(x ← ω) = V(x,ω) L_s^sky(ω) + (1−V(x,ω)) ∫_{H(y)} f_s(y, ω_i → −ω) L_s(y ← ω_i) μ_i dω_i,  y = r(x,ω)

Estimated by path tracing. Three requirements:

**Next-event estimation is mandatory.** At each vertex, evaluate the direct contribution
by explicitly sampling the visible sky, and exclude it from the recursive term to avoid
double counting. For a trench of aspect ratio A the solid angle subtended by the mouth
from the bottom scales as A⁻², so naive sampling wastes a fraction 1 − O(A⁻²) of paths at
the bottom — exactly where accuracy matters. NEE is the difference between a usable and
an unusable estimator, not a refinement.

**Russian roulette for termination.** Continue with probability q, divide the accumulated
weight by q. Unbiased because 𝔼 = q·(W/q) + (1−q)·0 = W. Set q = 1 − s so paths terminate
as particles do. `max_bounces` is a hard cap and a diagnostic, not the termination
mechanism; if it binds more than rarely, report it.

**Bounce budget.** With uniform sticking s ≥ s_min, ‖𝒦‖∞ ≤ 1 − s_min and the Neumann
series converges geometrically, requiring k > log ε / log(1 − s_min) bounces for relative
accuracy ε. For s = 0.05 and ε = 10⁻³ that is ≈ 135. **Low-sticking species are both the
expensive case and the physically interesting case** — they penetrate deep features — so
do not size the budget from the ion, which typically has s ≈ 1.

### 5.4 Differentiation — warped-area sampling

Required reading before implementing: §2 of this document, then Bangaru et al. 2020.

The derivative of Γ(θ) = ∫_H g(ω,θ) V(ω,θ) dω has two terms by the Leibniz rule for a
moving domain:

> dΓ/dθ = ∫_H V ∂g/∂θ dω + ∮_∂ Δg · v_n dℓ

The first is what AD computes; the second — over the silhouette ∂, with Δg the jump and
v_n the silhouette's normal velocity — is what it misses.

Choose a vector field 𝒱(ω) on the hemisphere satisfying **𝒱·n_∂ = v_n on ∂**, free
elsewhere subject to differentiability. Applying the divergence theorem on each region of
constant V and summing, the interior boundaries pair into the jump, giving

> **dΓ/dθ = ∫_H [ V ∂g/∂θ + ∇·(g 𝒱) ] dω**

No boundary term, no delta function, estimable by the same sampling process as the
forward pass, unbiased.

Implementation requirements:

- 𝒱 is built as a kernel-weighted average of nearby boundary velocities using auxiliary
  rays, per Bangaru et al. The kernel width is a config parameter; **report gradient
  sensitivity to it** at four values, in the manner of M2 PRD §5.6's `w_mat` study. A
  gradient that depends strongly on a purely numerical parameter is a finding.
- `warp_enabled = False` must remain available. It is not a fallback; it exists so that
  V35 can demonstrate the bias of §2 experimentally.

### 5.5 Variance reduction and sampling

- **Cosine-weighted importance sampling** for diffuse scattering; source-matched sampling
  for the direct term.
- **Stratification** over the hemisphere, and **scrambled Sobol** sequences. Scrambling
  preserves unbiasedness; unscrambled Sobol does not, under reuse.
- **Sample counts are fixed**, from config. A variance-adaptive sample count makes the
  computational graph depend on θ. If spatially varying sample counts are wanted, they
  must come from a precomputed schedule that is not a function of θ.

### 5.6 The M5 boundary — scattering distributions

M3 defines the interface; M5 supplies the content. Ship only analytic test distributions.

```python
ScatterModel = Callable[[ScatterRequest, PyTree], ScatterSample]
# ScatterRequest: species, incident direction, energy, normal, material_fractions
# ScatterSample:  outgoing direction, weight, sticking probability, energy out
```

Energy conservation constrains any admissible model:

> ∫_{H(y)} f_s(y, ω_i → ω_o) μ_o dω_o = 1 − s_s(y, ω_i) ≤ 1

**Add a conformance test asserting this to 10⁻⁶ for every supplied model.** A model
violating it makes ρ(𝒦) ≥ 1 and the Neumann series will not converge; the symptom is a
run that never terminates or a flux that grows without bound, and the cause will not be
obvious.

Analytic models shipped by M3 for its own testing:

- `diffuse(s)` — f = (1−s)/π · μ_o. Memoryless.
- `specular(s, sharpness)` — mollified about the mirror direction ℛ(ω_i). **Mollified,
  not a delta**: a delta reflection is a discontinuity in ω_o and reintroduces §2 inside
  the scattering model. Report sensitivity to `sharpness`.
- `absorb()` — s = 1. The ion default in v0.

### 5.7 Redeposition and the fixed-point problem

Etch products leave a surface point at a rate determined by M5's chemistry, which depends
on the flux M3 computes. Enabling redeposition therefore closes a loop: Γ → etch rate →
product source → Γ.

**v0 excludes it.** The schema must carry it from the start — a per-point product
emission term in `ScatterSample` — because retrofitting a source term into a shipped
transport interface is a rewrite, exactly as M2 PRD §5.6 argues for the byproduct return
channel.

When enabled (M3.8, gated): solve by Picard fixed-point iteration, two or three passes,
and **take gradients by implicit differentiation, not by unrolling**. Unrolling costs
memory linearly in iterations and is numerically fragile. See Christianson (1994) and the
deep-equilibrium formulation of Bai, Kolter & Koltun (NeurIPS 2019).

### 5.8 The performance problem, and flux lagging

**This section revises a gate in the M2 PRD. Read it before planning M3.4.**

M2's performance gate is a 3D 500-step forward solve in under 60 s. M3 is called once per
Runge–Kutta stage, so TVD-RK2 over 500 steps means **1000 velocity evaluations**.

Order-of-magnitude budget for a 512×128×128 domain:

| Quantity | Estimate |
|---|---|
| Interface-adjacent cells | ~10⁵ |
| Primary samples per point | 64 |
| Mean path length at s ≈ 0.5 | ~2 vertices |
| Rays per velocity call | ~1.3 × 10⁷ |
| Sphere-trace iterations per ray | ~64, each a trilinear fetch |
| Forward solve at 1000 calls | **10² – 10³ s** |
| Adjoint at ~3× | **10³ s and up** |

So M3 exceeds M2's 60 s gate by one to two orders of magnitude. This is a real finding,
not a pessimistic assumption, and it must be resolved by design rather than by
optimisation alone.

**Flux lagging.** Surface flux changes slowly relative to geometry over a single
timestep. Recompute flux every `flux_lag_steps` and hold it fixed between recomputes. A
fixed lag schedule does not depend on θ, so it is gradient-safe; it changes the physical
model, and the size of that change must be measured.

**Required study (M3.6, check V38):** profile error versus `flux_lag_steps` ∈ {1, 2, 5,
10, 20}, reported as CD and sidewall-angle deviation from the `flux_lag_steps = 1`
reference. Choose the largest lag whose error is below the M4 metrology repeatability
floor — beyond that point the error is unmeasurable and the compute is wasted.

**Memory.** Reverse-mode through 1.3×10⁷ rays × 64 sphere-trace iterations would require a
tape of order 10¹⁰ values. This is infeasible. **`jax.checkpoint` on the whole velocity
evaluation is mandatory, not optional** — recompute the transport during the backward
pass, bounding memory at ~2× forward cost. This only works if the recomputed transport is
bitwise identical to the original, which is precisely what M2 PRD §7.6 and check V18
exist to guarantee.

---

## 6. Milestones and gates

| # | Milestone | Gate |
|---|---|---|
| **M3.0** | Scaffold, `CLAUDE.md`, schema, config, ledger, CI | `pytest` green; V23 (energy-conservation conformance) catches a deliberately non-conserving scatter model; `VelocityModel` signature matches M2 §5.5 exactly |
| **M3.1** | Sphere tracing, 2D | V24, V25; non-convergent ray fraction reported and under 1% |
| **M3.2** | Direct (single-bounce) flux, 2D, no scattering | **V26, V27** — analytic view factors recovered, error falling as N^(−1/2) |
| **M3.3** | Full path tracing: NEE, Russian roulette, scattering | **V28, V29, V30** — furnace test uniform; closure and reciprocity hold |
| **M3.4** | **Warped-area sampling. Gradients, 2D.** | **V31, V32, V35** — Taylor remainder O(h²) with CRN fixed; **and the same test fails with `warp_enabled=False`** |
| **M3.5** | 3D, checkpointed | V34, V36; memory bounded; recompute bitwise identical (inherits V18) |
| **M3.6** | Performance: lagging, variance budget | **V38** lag study; **V37** noise floor met at production sample count; revised performance gate agreed with M2 owner |
| **M3.7** | Coupled to M2, physical validation | **V39** ARDE curve shape correct and steepening with sticking; **V40** ViennaRay comparison |
| **M3.8** | Redeposition, gated | Fixed point converges; gradients by implicit differentiation pass V31 |

---

## 7. Numerical requirements

### 7.1 No data-dependent loop counts in the differentiated path
Sphere-trace iterations, sample counts, bounce caps, and fixed-point iterations are all
**fixed from config**. Russian roulette is the single permitted stochastic termination,
and it is permitted because path length is not a discontinuous function of a parameter
value. Same rule and same reasoning as M2 PRD §5.2.

### 7.2 Do not regularise the hit-point singularity
∂t/∂θ = −(∂φ/∂θ)/(∇φ·ω) is singular at tangency by construction. Adding ε to the
denominator produces a smooth wrong gradient. The singularity is handled by §5.4, not by
softening.

### 7.3 Bounce budget sized from the lowest-sticking species
Per §5.3. Sizing from the ion under-resolves every neutral.

### 7.4 Gradient noise floor
M8 optimises to ~1 nm CD. Require, per component,

> SNR_i = |∂J/∂θ_i| / √(Var[∂Ĵ/∂θ_i]) > 1

Since variance falls as 1/N, required samples scale as the inverse square of the signal.
**Produce this budget at M3.6 and report the implied wall-clock cost**, rather than
discovering it at M8 integration.

Related open risk: M8 specifies L-BFGS, which builds curvature from y_k = ∇J_{k+1} − ∇J_k
and requires s_kᵀy_k > 0. If gradient noise is comparable to the gradient change between
iterates, the curvature condition fails or is spuriously satisfied and the Hessian
approximation degrades. Quasi-Newton methods are not noise-robust. **M3 must therefore
expose a sample-count knob and a per-component variance estimate alongside every
gradient.** See Q6.

### 7.5 Common random numbers
The RNG key is a deterministic function of `step_index`, surface-point index, and a
run-level seed. Never global, never stateful.

Without this, J(θ) is not a function — two evaluations at the same θ differ — and the
Taylor-remainder test cannot be run at all, because noise dominates the O(h²) residual.
CRN is a **prerequisite for verification**, not an optimisation.

### 7.6 Precision
fp64 for transport accumulation during M3. fp32 for ray traversal may be evaluated at
M3.6 and is gated on V31 still passing.

---

## 8. Verification (V23–V40, continuing the M11 ledger)

Checks write to the shared `verification_ledger.json`; `verification.md` is **generated**,
never hand-written. Cadence follows M2 PRD §8.0: fast tier under three minutes on every
commit, nightly, gate-only.

### 8.1 Structural
- **V23 — Energy conservation conformance.** ∫ f μ_o dω_o = 1 − s to 10⁻⁶ for every
  scatter model. Deliberately non-conserving model must be rejected. *Fast tier.*
- **V24 — Sphere trace against analytic surfaces.** Plane and sphere; hit distance exact
  to 10⁻⁶ relative. *Fast tier.*
- **V25 — Non-convergent ray fraction.** Reported every run; gate at 1%.

### 8.2 Transport against exact answers
- **V26 — Analytic view factors.** Two infinite parallel strips (crossed-strings
  formula); coaxial parallel discs; point above an infinite plane (F = 1). Relative error
  must fall as N^(−1/2) with the expected constant. **A plateau indicates bias.**
- **V27 — Cosine-law flux.** Γ = (2π/3)L₀ for a flat surface under a cosine source.
  Analytic, exact.
- **V28 — Furnace test.** Fully enclose the geometry, set s = 0, purely diffuse
  re-emission, uniform injection. Steady-state flux must be **exactly uniform** on every
  surface regardless of geometric complexity. The single best whole-system transport test.
- **V29 — Closure.** Σ_j F_ij = 1 for an enclosure.
- **V30 — Reciprocity.** A_i F_ij = A_j F_ji.

### 8.3 Gradients
- **V31 — Taylor remainder**, CRN fixed, slope in [1.8, 2.2] over ≥20 random directions.
  Run on both a smooth functional (total flux over the surface) and a localised one (flux
  at a single sidewall point), in the manner of M2 PRD §8.7.
- **V32 — Forward vs reverse mode.** `jax.jvp` against `jax.vjp`, agreement to 1e-10 in
  fp64. Separate code paths, so agreement is evidence rather than tautology.
- **V34 — Checkpointed equals unchecked** at small sample counts.
- **V35 — The bias demonstration.** Run V31 with `warp_enabled=False` and assert it
  **fails**. This is the experimental confirmation of §2 and the M3 analogue of M2's V19
  canary. **Ship both Taylor plots side by side in the report.**

### 8.4 Cross-method and physical
- **V33 — Radiosity cross-check.** On a shallow, purely diffuse geometry where radiosity
  is valid, the path-traced flux and the deterministic linear solve must agree within
  estimator noise, and their gradients must agree to 1e-6. Two independent methods, two
  independent differentiation routes, one answer.
- **V36 — Golden regression.** Frozen reference flux field, reproduced to 1e-10 under CRN.
- **V37 — Noise floor.** §7.4 SNR > 1 at the production sample count, per component.
- **V38 — Flux lag study.** §5.8. CD and sidewall-angle error versus `flux_lag_steps`.
- **V39 — ARDE.** Coupled to M2, etch depth versus aspect ratio at fixed chemistry. The
  curve must have the correct shape and **steepen with increasing sticking coefficient**.
  The model was never tuned to this; ARDE emerges from transport alone, so agreement is
  real evidence rather than a fit.
- **V40 — ViennaRay comparison.** Separate process, VTK exchange, symmetric Hausdorff on
  flux profiles. **GPL-3.0: do not link, import, vendor or copy.** Clean-room discipline
  applies. Run V26 and V38 first — without them you cannot distinguish a bug from a
  sampling or lag difference.

### 8.5 What M3 does not verify
M3 verifies that transport is computed correctly given a scattering model. It says
nothing about whether the scattering model is right — that is M5, and it is where the
chamber-specific physics lives. Record this boundary in `verification.md` so a green M3
suite is not mistaken for a validated process model.

---

## 9. Deliverables

1. `fractal-m3`, running from clean checkout with `uv sync && pytest`.
2. `reports/verification.md` — generated from the ledger, including §8.5.
3. **`reports/warp_bias_demonstration.md`** — the V35 plot pair. The single most
   important artifact M3 produces.
4. `reports/noise_floor_budget.md` — §7.4, with wall-clock implications.
5. `reports/flux_lag_study.md` — §5.8 / V38.
6. `reports/warp_kernel_sensitivity.md` — §5.4.
7. `m3/interface.py` — the frozen `ScatterModel` contract for M5.
8. `reports/simplifications.md`.

---

## 10. Environment

```bash
uv init fractal-m3 && cd fractal-m3
uv add "jax[cuda12]" numpy scipy matplotlib
uv add pyyaml omegaconf pytest hypothesis
uv add pyvista            # VTK exchange for V40
uv add fractal-m2         # path dependency on the M2 package
```

Do **not** add: PyTorch, Warp, Mitsuba, OptiX bindings, Embree bindings, ViennaRay. One
autodiff system, so a gradient bug has one place to be.

---

## 11. Anti-requirements — do not do these

- **Do not ship a gradient path without warped-area sampling.** §2. This is the mission.
- **Do not use a score-function / REINFORCE estimator** for the geometric gradient. §4.4.
- **Do not regularise the ∇φ·ω denominator.** §7.2.
- **Do not use a delta-function specular model.** Mollify. §5.6.
- **Do not use adaptive sample counts, convergence-based sphere tracing, or any
  data-dependent loop count** in the differentiated path. §7.1.
- **Do not size the bounce budget from the ion.** §7.3.
- **Do not implement surface chemistry.** §3. Analytic test distributions only.
- **Do not write to φ.** M2 owns the geometry.
- **Do not omit next-event estimation** "for simplicity". §5.3. Without it the estimator
  is unusable in exactly the deep features the product exists to model.
- **Do not run without checkpointing on the velocity evaluation.** §5.8.
- **Do not hand-write `reports/verification.md`.** Generate it.
- **Do not loosen a verification tolerance to make a check pass.** Raise it as a finding.
- **Do not link, import, vendor or copy ViennaRay or ViennaPS.** GPL-3.0. §8.4.

---

## 12. Open questions for the human

| # | Question | Blocks | Why it matters |
|---|---|---|---|
| **Q1** | Target chemistry and operating pressure? | M3.0 | Fixes the species set, and §4.1's Kn must be checked at the actual pressure. Above ~200 mTorr in a small feature the free-molecular assumption weakens. |
| **Q2** | Target aspect ratio for the first demo? | M3.3 | Sets the NEE requirement and the sample budget. A⁻² scaling means a 20:1 feature is 100× harder than 2:1. |
| **Q3** | Is specular ion reflection required in v0, or is `absorb()` sufficient? | M3.3 | Specularity drives bowing and twisting. Including it roughly doubles M3.3 and adds the `sharpness` mollification study. |
| **Q4** | Is product redeposition required for the first demo? | M3.8 | If yes, M3.8 moves onto the critical path and the fixed-point machinery is needed. If no, schema-only. |
| **Q5** | What is the acceptable forward-solve wall clock, given §5.8? | M3.6 | M2's 60 s gate is unachievable with real transport. This number must be renegotiated deliberately, not discovered. |
| **Q6** | Does M8 commit to L-BFGS, or will it accept a noise-tolerant optimiser? | M3.6 | Determines whether M3 must hit a hard deterministic-gradient bar (§7.4) or can trade accuracy for speed. Settle before M3.6 freezes the sampling interface. |
| **Q7** | Which GPU, and is fp64 throughput acceptable? | M3.5 | Consumer parts run fp64 at 1/64 of fp32. §7.6 assumes fp64 accumulation is affordable. |

---

## 13. Repository layout

```
fractal-m3/
  CLAUDE.md
  OPEN_QUESTIONS.md
  config/
    transport.yaml          # TransportConfig
    sources.yaml            # analytic source specs
  m3/
    schema.py               # Species, SourceSpec, TransportConfig
    interface.py            # frozen ScatterModel contract for M5
    sphere_trace.py         # §5.2
    sampling.py             # §5.5 stratification, Sobol, CRN
    sources.py              # §5.1
    scatter_analytic.py     # diffuse, specular, absorb — test models only
    transport.py            # §5.3 path tracing, NEE, RR
    warp.py                 # §5.4 warped-area sampling
    radiosity.py            # §4.3 deterministic reference, for V33
    velocity.py             # the M2 VelocityModel adapter
    verify/                 # V23–V40
  reports/
  runs/
  tests/
```

---

## 14. Conventions

- ω always points **away** from the point under discussion; μ = ω·n > 0 on the visible
  hemisphere. Assert this once and never vary it — half of all errors in this material
  are sign errors from mixed conventions.
- φ < 0 inside solid, inherited from M2 PRD §5.1. Assert on import.
- Every transport function docstring cites its source with equation or section number.
- Commit at each gate. Never commit a flux field without its config hash and seed.

---

## 15. Definition of done

> Coupled to M2, a 3D trench evolves under path-traced multi-species transport within the
> renegotiated wall-clock budget of Q5; the Taylor-remainder test passes at second order
> on both a global and a localised flux objective with common random numbers fixed; the
> same test demonstrably **fails** with warped-area sampling disabled; the furnace test is
> uniform and analytic view factors are recovered at the expected convergence rate; the
> ARDE curve has the correct shape and steepens with sticking coefficient; and the
> `ScatterModel` contract is frozen and versioned so M5 can begin without renegotiating it.
