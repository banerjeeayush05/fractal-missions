# CLAUDE.md — M2, differentiable geometry core

Shared repo rules: `../CLAUDE.md`. Spec: `M2_PRD.md`. Decisions that supersede it: `decisions/`
(2026-09-11 response r3, then the second-round response; later files win). Open items: `OPEN_QUESTIONS.md`.
Cross-mission impacts: `../CROSS_MISSION.md`. Run: `uv sync && uv run pytest`.

## Non-goals (§3) — if you want one of these, stop and ask
- Flux, visibility, shadowing, re-emission, redeposition (M3). Prescribed analytic velocity only.
- Surface chemistry, yield curves, sticking (M5). Reactor-scale anything (M6). Calibration,
  optimisers, posteriors (M8) — except `scipy.optimize` inside `tests/` for M2.8, never in the package.
- Sparse / narrow-band data structures. Charging, pulsing, stress, wet etch, crystal anisotropy.
- Wafer-scale or multi-feature domains. The mask is not modelled: infinitely selective, pre-cut into φ.

## Architecture (§4): dense storage, masked compute, JAX
Static shapes let plain JAX reverse mode work; a fixed-count Hamilton–Jacobi reinit is just more
differentiable PDE (no hand-derived fast-marching adjoint); no active set that jumps with
parameters. Cost accepted: memory scales with volume, compute is wasted far from the interface.
Revisit only for multi-feature or wafer-scale domains. The reference case is small (S03 3D is
21.6 MB per fp64 field), so the decision holds with a wide margin.

## Conventions — a sign error here inverts every gradient
- φ < 0 inside solid, > 0 in open volume. n = ∇φ/|∇φ| points from solid into open.
- Velocity models return an **etch rate R** in nm/s: positive removes material. M2 advects
  φ_t − R|∇φ| = 0, so the sign flip lives in M2 alone, not in every M3/M5 model.
- Axis 0 is vertical and **increases toward the plasma**, so ẑ = +e₀ and R = v₀·max(0, n·ẑ)^p
  etches up-facing surfaces. A trench floor recedes away from the plasma; an overhang underside
  does not move. V14–V16 cannot catch a sign inversion — V1a is the only guard.
- Cases carry a `role`: a fit may only consume `calibrate` cases. Enforced by `require_calibration_case`.

## Numerical requirements (§7, decisions §4, §7)
- **V14 (Taylor) is the gate**, scored in two zones over ≥20 directions, each passing on its own:
  fewer than 3 points in the window is FAIL ("insufficient signal"); slope < 1.8 is FAIL;
  [1.8, 2.2] passes as `clean_quadratic`; above 2.2 passes as `degenerate_direction`. There is no
  upper bound to enforce — a wrong gradient leaves the first-order term uncancelled and tends to
  slope 1. The ledger records the classification per direction and the degenerate fraction per run:
  a jump in that fraction is a finding.
- **The slope is fitted in an ANCHORED WINDOW** (decision I7): sweep 8 decades, record the whole
  curve, fit the 2 decades immediately above the **measured** floor. The solver's map is piecewise
  smooth, so the top of the range is not quadratic (slope 1.29 at 50 steps) and says nothing about
  the gradient; the bottom is where correct (½h²δᵀHδ) and wrong (|ε·g·δ|·h) separate most. The
  floor is measured at h ≈ 1e-10, not modelled — B19's C·√N·eps·|J| reads 51–65× high and each
  factor of 10 in a discard threshold costs a decade of window. Two decades is a condition on the
  *clean_quadratic* label, not on pass/fail. **Obligation: any change to the window must be shown
  to leave the V19 canary catching 5 % corruptions** — that test is the reason it is not blinding.
- **Parameter scales are required**, not optional: δ_i ∝ max(|θ_i|, scale_i). A parameter sitting
  at 1e-12 would otherwise get a 1e-12 perturbation and report a pass having probed nothing.
- **V15 and V16 are transpose checks, not derivative checks.** JAX builds reverse mode by
  linearising with JVP rules and transposing, so both share those rules. Only V14 compares a
  derivative against the function. V14a/b/c (analytic sensitivities) carry the rest.
- **V19**: a gradient with one component ×1.05 must make V14, V15 and V16 fail, with an unmutated
  control passing. Its own mutation guard is `tests/test_v19_mutants.py`.
- **Two spatial schemes** (§5.2): `godunov` (first order, still the DEFAULT) and `weno5`
  (fifth-order HJ-WENO, built post-M2.3). WENO5 cuts V10 anisotropy 125× and V1 radius error 120×,
  but its scheme-level order is ~2.08 not ≥4 (finding J2: the velocity-extension path is second
  order), V8's notch still fails (J4), and making it the default is an open owner decision (J5).
  Its ε is fixed where the literature scales it by a stencil `max` — that max would be a kink.
  Within 3 cells of a hard boundary it falls back to Godunov, or it extrapolates from the Neumann
  clamp and invents a phantom interface (J3).
- **The Taylor remainder can notch.** |R| must grow with h; where it does not, two terms are
  cancelling and the log-log slope is meaningless — it reads ~1.6 on a gradient that is exactly
  right. `_cancellation_ceiling` caps the window below the first notch. This is safe only because
  the window is anchored at the floor: the cut removes points ABOVE the notch, never below, and
  below is where a first-order error dominates. Do not "fix" a failing V14 by widening the band.
- Every differentiable parameter lives in the params PyTree, never in a closure.
- Fixed counts everywhere: N is derived as ceil(D/(cfl_target·dx)), never hand-written; fixed
  n_reinit, fixed extension iterations. CFL ≤ 0.5 asserted every step, returned as a scan output
  and checked on the host (not `checkify`).
- fp64 everywhere. Two-level checkpointing is a **nested scan** (`m2/checkpoint.py`), parametrised
  by segment length L: peak = N/L + L·k, optimal at **L\* = √(N/k)**. At the measured k (349
  Godunov, 1050 WENO5) that is **L = 1**, which is remat-on-the-step-function — so §7.4's warning
  against it is inverted at real k and the PRD is amended (finding K2). Three-level is N/L + L + k.
  Never hand-pick L: derive it from measured k, or it becomes a knob that makes a memory gate pass.
- **Checkpointing makes the adjoint FASTER here, not slower** (K1): the unchecked tape does not fit,
  so the usual time-for-memory trade runs backwards. 22.2× → 4.9× at 128²·N=100. Decision I5(a)
  rests on this.
- RNG key = f(run_seed, step_index, stage_index, **cell_id**), all carried in VelocityRequest.
  Never global, never stateful, never hashed from `time`.
- **The stable `cell_id` and the smooth band weight are two halves of one mechanism, not two
  safeguards.** The id stops a cell's arrival in the band from shifting everyone else's draws; the
  weight going to zero at the band edge stops the arriving cell's own draw from entering
  discontinuously. Remove either and J is discontinuous in θ. `cell_id` is opaque: seed with it,
  never index geometry with it. Common random numbers do not survive a change of grid spacing.
- **Two bands**: extension (8 cells, tapering over the outer 2) is where a valid velocity must
  exist for the stencils; evaluation (1.5 cells) is where the velocity model is *called*, and it
  stays thin because deeper cells project to the same surface point and M3 would pay twice. K is
  sized from the evaluation band at the **worst** step, and peak occupancy is reported every run.
- **Configs specify a target depth, never a final time.** T = depth / rate is derived, so a rate
  correction rescales time and changes nothing else — N, CFL and the golden profile all hold.
  `configs/cases/*` refuse to load until the rate is measured; `configs/dev/*` carry a nominal rate
  marked provisional and may never certify a check that claims agreement with the coupon.
- Endpoint-only output from the differentiated solve; debug frames are dumped outside it.

## Traps that have already bitten
- **NaN poisoning.** ∇φ/|∇φ| blows up where |∇φ| → 0 (the trench centreline, a circle's centre).
  `jnp.where` does not save you: a NaN in the untaken branch still poisons the gradient. Use the
  double-where pattern. Masking afterwards does not save you either — `0 * NaN` is NaN — so padded
  request entries need a benign position, not just a zero weight.
- **The band weights participate in the gradient.** Never `stop_gradient` them. They look like a
  mask, which is exactly what a future reader will wrap to make a test pass.
- **The directional law has a kink on vertical sidewalls** (n·ẑ = 0). Require p > 1, use p ≥ 2 in
  gradient tests, and keep 0^p out of the p-derivative.
- **Sub-cell extraction is continuous but not smooth**: the derivative jumps when a crossing passes
  a grid node, so V14 on CD is touchier than on volume. Check that before hunting a bug.
- Squared one-sided differences (Godunov) are fine: `max(x,0)²` is C¹. The rule is no kinks or
  jumps in the differentiated path, not "never write max".

## Anti-requirements — literal
- No "the optimiser converges" as gradient evidence. No adaptive or convergence-based loop counts.
- No `stop_gradient` on reinitialisation, extension or material blending. If you think one is
  needed, stop and ask.
- No CD by cell counting or thresholding. No fp32. No narrow-band sparsity. No flux or visibility.
- Do not tune `w_mat`, CFL, band width or `n_reinit` to make a gradient test pass — that is a finding.
- Do not run V22 before V6 and V7. Do not let the fast tier exceed 3 minutes.
