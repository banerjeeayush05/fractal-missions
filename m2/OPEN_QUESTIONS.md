# M2 — Open questions

Status: written at M2.0 (week 1); **answered 2026-09-11** by
`decisions/2026-09-11-open-questions-response-r3.md`, which is authoritative over the PRD.
The findings below are kept verbatim as the record of what was asked; the index says what each
one became. Items raised *while implementing* those answers are at the end of this file.
Classes: **A** contradiction or error in the PRD · **B** value the PRD needs but never supplies, so a
placeholder is in use (`provisional: true`, with the mechanical rule that produced it) · **C** gap
that blocks a later milestone · **D** minor, noted, no action.
The M1 open-questions file is not in this repo, so this follows the A–D scheme given at kickoff.

---

## Resolution index (2026-09-11)

| Item | Status | What it became |
|---|---|---|
| A1 sign/orientation | **resolved** §1 | Etch rate R > 0 removes material; φ_t − R\|∇φ\| = 0; axis 0 toward the plasma, ẑ = +e₀. In `constants.py`, asserted in `tests/test_sign_convention.py`. New check **V1a** guards it. |
| A2 etched_volume sign | **resolved** A2 | Renamed `solid_volume`; H width fixed at 1.5·dx in `constants.py`. Sign is irrelevant for its §8.7 job. |
| A3 fp32 vs fp64 figures | **resolved** §7 | The 8 GB gate was fp32-sized and is withdrawn; now <40 GB on one H100 against case S03. |
| A4 checkpointing recipe | **resolved** §7 | Two-level means a nested scan; `jax.checkpoint` on the step function alone is not it. Confirmed as an error in the PRD and the lesson plan. |
| A5 N=500 vs CFL | **resolved** §2 | Bare step counts are gone: N = ceil(D/(cfl_target·dx)), cfl_target 0.4. N ≥ 625 at dx = 10 nm. Enforced by the config loader. |
| A6 domain height | **resolved** §2 | 10-cell headroom + 2500 nm etch + 10-cell buffer = 270 cells at dx = 10 nm. Mask is not modelled. |
| A7 seed in VelocityRequest | **resolved** §3 | Contract v0.2 adds `run_seed` and `stage_index`; independent draws per RK stage by default. Still provisional pending M3 sign-off. |
| A8 interface-adjacent set | **resolved** §3 | Fixed-capacity padded set (K from config), smooth membership weights, closest-point projections as positions. |
| A9 V15/V16 blind spot | **resolved** §5, broadened | V15/V16 are transpose checks for *all* primitives, not just custom rules. New **V14a/V14b/V14c** analytic sensitivities carry derivative correctness. |
| A10 Taylor degenerate case | **resolved** §4 | Noise floor with point exclusion; "insufficient signal" fails; remainder at the floor or slope ≥ 2.8 passes as `degenerate_direction`. |
| A11 scikit-image | **resolved** A11 | CD at fixed heights needs only 1D sub-cell crossings, JAX-native; scikit-image is the V13 reference only. |
| A12 M2.8 optimiser | **resolved** §12 | `scipy.optimize` allowed in `tests/` only, never in the package; asserted by `tests/test_package_boundaries.py`. M2.8 must also assert V14 at the recovered parameters. |
| A13 adjoint cost gates | **resolved** §7 | Priority: memory > recompute > ratio. <40 GB, ≤2× recompute, ≤4× adjoint (3× retained for 2D unchecked). With case S03 the conflict dissolves: two-level fits with k ≤ ~1372. |
| A15 cross-mission | **superseded** | Tracked entry by entry in `../CROSS_MISSION.md` (X1–X16). |
| B1 layout | accepted | CLAUDE.md now split three ways (decision §3): root shared rules, `m2/CLAUDE.md`, `m3/CLAUDE.md`. |
| B2 jax[cuda12] markers | accepted | Unchanged. |
| B3 Taylor protocol | **superseded** §4 | h-range 1e-6..1e-1 and 11 points remain provisional; the pass rules are now the decision's. |
| B4, B5 normalisation | **amended** | Normalise by ‖∇J‖·‖δ‖, never by the directional derivative. |
| B6 "V14 reduced" | accepted | Reduce the problem, never the protocol. |
| B7 ledger mechanics | accepted + §13 | CI on a clean checkout is the only writer of the ledger of record; local runs write `reports/local/` (untracked). |
| B8 array layouts | accepted | Plus `weights` (B14) and the padded-set shape (K, d). |
| B9 contract version | accepted | Now v0.2, still `provisional`. |
| B10 config rules | accepted | Plus the `case` block, `role`, and the derived step count. |
| B11 liveness test | accepted | Unchanged. |
| B12 cost-table inputs | **rerun** | Now against S03 at dx = 10 and 5 nm, plus phase-2 M01. |
| B13 canary precondition | accepted | Now stated on the relative sensitivity \|θ_k·∂J/∂θ_k\|, matching relative-unit directions. |
| C1–C9 | answered | See decision "Smaller items" and §11; each is now a config value, a constant, or an approved test-only tool (rigid-rotation model for V8/V9). |
| D1–D12 | noted | D8 resolved: CFL is a scan output asserted on the host, not `checkify`. A14 was a numbering mistake and is withdrawn. |

---

## A — Contradictions and errors

**A1. Velocity sign: under the PRD's own conventions, the directional "etch" deposits.** Blocks M2.1.
§5.1 puts φ < 0 in solid. §5.2 advects with φ_t + V|∇φ| = 0. So V > 0 moves the interface along
n = ∇φ/|∇φ|, which points out of the solid: the solid **grows**. V1 ("isotropic growth, r = r₀ + Vt")
agrees with that. But §5.5's directional model V = v₀·max(0, n·ẑ)^p is ≥ 0 for v₀ > 0. On a trench
floor (open volume above, so n = +ẑ if ẑ points out of the wafer) it therefore **fills the trench**.
If ẑ points into the wafer, max(0, n·ẑ) instead picks out downward-facing surfaces, which is also wrong.
The PRD never says which end of grid axis 0 is the wafer top, or what sign an etch rate has.
V14–V16 cannot catch this: they check the derivative of the function as written, not the function
as intended. Needed: (i) the etch sign convention (for example V > 0 = etch with φ_t − V|∇φ| = 0, or keep
§5.2 and pass −rate), and (ii) the orientation of axis 0 and ẑ. Nothing is assumed in the code:
`constants.py` fixes only the φ sign and the normal direction.

**A2. `etched_volume = ∫(1 − H(φ)) dV` measures remaining solid, not etched volume.** Blocks M2.3.
With φ < 0 in solid and the usual H(φ) = 1 for φ > 0, the quantity 1 − H(φ) is the solid indicator.
The integral therefore falls as etching proceeds, and its gradient has the opposite sign to etched volume.
Either H means H(−φ), or the intended quantity is ∫H(φ) dV minus the initial open volume. This is the
diagnostic functional for §8.7, so its sign has to be settled first. The mollification width of H is
also unspecified (see C3).

**A3. §4 and §7.4 use fp32 bytes, but §7.5 mandates fp64.** Blocks the M2.4 gate definition.
"34 MB per field", "17 GB" and "~750 MB" are all fp32 figures. At fp64 they are 67.1 MB, 33.6 GB and
1.54 GB. The M2.4 "<8 GB" gate appears to have been sized against the fp32 numbers
(`reports/dense_cost_table.md` §1).

**A4. §7.4's checkpointing recipe does not give √N memory, and its estimate omits the main term.** Blocks M2.4.
(a) `jax.checkpoint` on the step function alone rematerialises work inside each step, but it still
stores the carry at every one of the N steps: N × state, the same as the "naive" figure. √N storage
needs a nested scan (outer √N segments, each checkpointed).
(b) "Recursive (binomial)" and "√N checkpoints" are different schedules. The "~2× recompute" figure
belongs to two-level √N.
(c) "~750 MB" counts only stored states. The backward pass also keeps every per-step residual for the
segment being recomputed: L × k × field, where k is the number of field-sized residuals per step.
If k ≳ 7, two-level √N misses the 8 GB gate at 512×128×128 in fp64. Three-level fits up to
k ≈ 74 but costs 2× recompute (dense_cost_table §2–3). The lesson plan repeats (a).

**A5. N = 500 cannot etch the §4 reference trench within CFL ≤ 0.5.** Blocks the M2.4 gate and M2.1 configs.
The trench floor moves at max|V|, so max|V|·T ≥ D, where D is the etch depth. Then
CFL = max|V|·T/(N·dx) ≥ D/(N·dx). For D = 1000 nm, dx = 2 nm and N = 500, that gives CFL ≥ 1.0 > 0.5.
The §5.2 assertion would fire on exactly the configuration the M2.4 memory gate is written for.
It needs N ≥ 1000, and that is exactly at the limit with no margin. (A 500 nm etch needs N ≥ 500.)

**A6. 512 vertical cells cannot hold a 1 µm trench under §5.1.** Blocks M2.1 domain setup (see Q1).
512 × 2 nm = 1024 nm. §5.1 requires initial stack + maximum etch depth + a 10-cell (20 nm) buffer.
That leaves ≤ 4 nm for the mask and any open headroom above it. §5.1 also does not say whether the
buffer sits above the stack (headroom) or below the maximum etch depth.

**A7. §7.6 needs a run-level seed in `VelocityRequest`; §5.0 has none.** Blocks the contract freeze (M2.3) and V18 (M2.4).
§7.6: "RNG key … deterministic function of step_index and a run-level seed, both passed through
VelocityRequest". §5.0's `VelocityRequest` has `step_index` only. Separately: Heun calls the model
twice per step (§5.5 allows one call per stage). A key built from (seed, step_index) alone gives both
stages identical Monte Carlo draws, so the stage needs to be part of the key. `time` differs between the
two stages, but hashing a float time into a key is fragile. The schema is built verbatim per §5.0,
without a seed, and the contract is marked provisional (B9).

**A8. `positions: interface-adjacent cell centres` implies a data-dependent set.** Blocks M2.1 design and the M2.3 freeze.
(a) The number of interface-adjacent cells changes with φ. Inside `jit`/`scan` that is a dynamic
shape, which §4's static-shape rationale rules out.
(b) Picking "adjacent" cells means thresholding |φ|, and a cell joins or leaves the set discontinuously
as parameters change. That is the active-set problem §4 says the dense design removes.
Options: evaluate on every cell with a smooth weight (static, but M3's velocity is "expensive" per
§5.5), or use a fixed-capacity padded set with a mollified membership weight. Either way it changes the
contract M3 inherits. For now the schema only checks that leading (batch) shapes agree (B8).

**A9. V19's "V14, V15 and V16 all fail" holds for only one of three injection points.** Affects M2.3.
§8.5 says JAX's jvp and vjp are "separate code paths". That is not true for custom rules:
- Corruption in a `jax.custom_jvp` rule: JAX derives the VJP by transposing that JVP, so jvp and vjp
  agree on the wrong derivative. V15 and V16 **pass** and only V14 fails.
- Corruption in a `jax.custom_vjp` rule: `jax.jvp` raises, so V15 and V16 cannot run at all
  (the harness reports FAIL, not skip).
- Corruption of the returned reverse-mode gradient: all three fail. This is §7.2's literal wording,
  and it is the M2.0 V19 gate.
All three behaviours are executable tests (`tests/test_v19_blind_spots.py`) against the installed JAX.
Consequence: V15 and V16 catch transpose inconsistencies, not wrong derivatives. The first
`custom_vjp` added to M2 will need a reference JVP for V15/V16 to mean anything.

**A10. The §7.1 gate rejects correct gradients along directions of zero curvature.** Blocks M2.3.
If δᵀ∇²J δ = 0, the remainder R(h) is O(h³) (slope 3) or pure roundoff, and a correct gradient fails
the [1.8, 2.2] band. M2's own V2 setup is exactly this case: under isotropic V, a flat front's
displacement is V·T, linear in V. The PRD needs a rule for this case. Until then the M2.0 harness
reports it as FAIL with a diagnostic ("remainder at roundoff floor" or "slope > 2.2") rather than
passing it.

**A11. scikit-image cannot be the §5.7 differentiable extraction.** Blocks M2.5.
§10 lists scikit-image for "marching cubes/squares, sub-cell extraction". Its routines run on NumPy
and cannot be traced by JAX: calling them on a tracer raises, and wrapping them in `pure_callback`
gives no gradient. The M2.5 gate (V14 on CD_mid and sidewall_angle) needs a JAX-native sub-cell
extraction. scikit-image can serve only as an independent reference for V13.

**A12. M2.8 needs an optimiser, and §3 excludes optimisers.** Blocks M2.8.
"Recover a known synthetic parameter set … from a cold start" needs a fitter. §3: "Calibration,
optimisers, posteriors. M8." — out of scope. Needed: permission to use an off-the-shelf optimiser
(for example `scipy.optimize`) as a test-only tool, or move M2.8 to M8.

**A13. The "≤3× forward" adjoint target conflicts with the checkpointing the memory gate forces.** Blocks M2.4 and §13.
M2.3 says "adjoint ≤3× forward cost". §13 and the mission plan say "under 3×". M2.4 says
"recompute overhead ≤2×". Two-level checkpointing adds one forward pass; three-level (needed if k ≳ 7,
A4) adds two. The reverse sweep of a stencil code typically costs 1–3× forward on top of that. The
PRD also does not say whether "adjoint" includes the original forward, or whether "recompute ≤2×" is
relative to the forward or to the unchecked adjoint.

**A15. Cross-mission: `m3/M3_PRD.md` (added during M2.0) assumes things M2 has not done.** Blocks the M2.3 freeze and the M3.0 gate.
The full, maintained list is `../CROSS_MISSION.md` (X1–X13); this entry keeps the summary.
Only the contract passages were read (M3 §5 lines ~180–195, §7.5, the ~345–356 budget table).
(a) M3 says the `VelocityModel` contract is "already frozen by M2 PRD §5.5" and gates M3.0 on
matching it "exactly". M2 §5.5 freezes it at M2.3, and it is provisional now (A7, A8).
(b) M3 §7.5 keys its RNG on (step_index, surface-point index, run-level seed), and its field list
for `VelocityRequest` has no seed. That points to the seed living in M3's own config, which would
contradict M2 §7.6 ("both passed through VelocityRequest"), so A7 needs one answer covering both PRDs.
Keying on step_index alone means both Heun stages of a step share the same draws, which also
needs a deliberate decision.
(c) M3 budgets "~10⁵ interface-adjacent cells" per call, i.e. it is designed around the
data-dependent point set of A8.
(d) M3 restates M2's performance gate as "3D 500-step forward solve", which inherits A5.
A root-level CLAUDE.md is loaded by M3 sessions too. The current one is scoped to M2 in its first
lines, and should be split per mission before M3 work starts here (B1).

---

## B — Placeholders in use (all `provisional: true`)

**B1. Project location and CLAUDE.md placement.** `provisional: true`
The PRD says repo `fractal-m2`; the actual repo is the `fractal-missions` monorepo, and the humans put
the PRD in `m2/`. *Rule:* PRD paths are read as relative to the directory that holds the PRD. So the
project root is `m2/` (pyproject, `m2/m2/` package, `m2/reports/`, `m2/tests/`). CLAUDE.md is at the git
root, as instructed, so it is always loaded and survives compaction. It carries a header scoping it to
M2, because a root CLAUDE.md is read by every mission and M2's non-goals are M3's goals. CI has to live
in `.github/workflows/` at the git root (GitHub only reads it there) and runs with `working-directory: m2`.

**B2. `jax[cuda12]` is not installable on macOS.** `provisional: true`
Verified: the literal §10 `uv add "jax[cuda12]"` fails on this machine (Apple M1): "jax-cuda12-plugin
only has wheels for manylinux". That breaks §9's "`uv sync && pytest` from clean checkout". *Rule:* keep
the literal requirement where it can install (`jax[cuda12]; sys_platform == 'linux'`) and use plain `jax`
(CPU) elsewhere. Apple's GPU backend has no fp64, so under §7.5 macOS runs are CPU-only. Linux CI
runners without a GPU download the CUDA wheels and fall back to CPU (heavy install; not yet exercised).

**B3. Taylor-harness protocol (§7.1 leaves these open).** `provisional: true`
h = 11 log-spaced points on [1e-6, 1e-1] (exactly 5 decades); δ ~ N(0, I) on the flattened parameter
vector, normalised to unit 2-norm; 20 directions; fixed PRNG seed; slope = least-squares fit of log R
against log h over **all** points, with none dropped; **every** δ must pass on its own.
*Rules:* the minimum counts §7.1 allows; the strictest reading of "over ≥20 random δ"; never exclude
data. The top of the h range (1e-1) keeps h well below an O(1) parameter scale. The bottom (1e-6) keeps
R ≈ h² ≈ 1e-12 far above fp64 roundoff for |J| ≈ 1. The harness refuses looser arguments
(fewer directions or decades, a wider band). Parameter scaling in physical units is C3.

**B4. V15 "1e-10 relative": relative to what?** `provisional: true`
|a − b| ≤ 1e-10 · max(|a|, |b|). Both exactly zero counts as a pass. Same 20 directions as V14.
*Rule:* symmetric relative error, reusing V14's direction count.

**B5. V16 has no tolerance in the PRD.** `provisional: true`
1e-10, with the same normalisation as B4, over 20 random (u, w) pairs. *Rule:* inherit from the sibling
check that uses the same primitives (jvp/vjp) at the same precision.

**B6. "V14 (reduced)" in the fast tier is undefined.** `provisional: true`
Reduce only the problem (grid, N, domain), never §7.1's protocol (≥20 δ, ≥5 decades, the slope band).
*Rule:* §7 is labelled "correctness requirements"; §8.0 is a runtime budget; the requirement wins.

**B7. Ledger mechanics.** `provisional: true`
- One row per check ID, updated in place, because the "last run" field implies latest state. All 22 IDs
  are pre-registered as `not_implemented` so a missing check is visible.
- Fields: the union of §8 and §9 (id, description, cadence, last_run, git_sha, result, measured), plus
  `milestones`, `tests`, `git_dirty` (a SHA of a dirty tree does not identify the code that ran; `reports/`
  is excluded from the dirty test), and `environment` (python/jax/x64).
- A row is written only when every test tied to that ID was collected and none was deselected. A partial
  run (`-k`, `-m`, `--lf`) must not certify a check. A skipped test counts as fail. Rows are written only
  when collection covered the default `testpaths`: `pytest tests/one_file.py` may omit some of a check's
  tests, so it writes nothing (`--ledger-any-scope` exists for the plugin's own temp-ledger tests).
- Every run writes to the tracked ledger (§8: "every check writes a row"). The ledger is never committed
  automatically.

**B8. Array layouts.** `provisional: true`
`Geometry.material` has shape (n_materials, *grid.shape), fp64, and the leading axis is `Material.index`.
`VelocityRequest`: positions (..., d), normals (..., d), material_fractions (..., n_mat); speed returned
with shape (...). Only the batch-shape agreement is enforced, because the batch layout is A8.
Sum-to-1 is checked on concrete arrays at 1e-12 absolute (`MATERIAL_SUM_ATOL`).
*Rule:* the index field in §5.0 names a material axis; nothing else is imposed.

**B9. Contract location and version.** `provisional: true`
It is defined once in `m2/schema.py` (§5.0's code block) and re-exported from `m2/interface.py` (§9.6)
together with `CONTRACT_VERSION = "0.1.0"` and `CONTRACT_STATUS = "provisional"`. A fingerprint test fails
if the fields change without a version bump. The freeze itself happens at M2.3 (§5.5), after A7 and A8.

**B10. Config rules.** `provisional: true`
A default exists only where the PRD states one (n_reinit = 5, reinit_every = 5, w_mat = 2 cells,
godunov, tvd_rk2, mms off). Everything else is required and marked `???` in `configs/reference.yaml`
(N, T, grid, seed, materials, extension method and count). Unknown keys are rejected. CFL_max and the
Taylor protocol live in `constants.py` and cannot be set from config (§11: do not tune CFL). Precision
must be float64. `mms: true` is refused unless the caller passes `allow_mms=True`. n_reinit ≥ 0 is
accepted because the PRD does not forbid disabling reinit.

**B11. §7.3 "nonzero gradient".** `provisional: true`
Checked per scalar component (a Jacobian column), and exactly ≠ 0. *Rule:* a closure-captured parameter
gives an exact zero, not a small one. A tolerance would only hide it.

**B12. Cost table inputs.** `provisional: true`
The 2D size is 512×128 (the x–z slice of the §4 reference). k ∈ {1, 20, 50} residual fields per step
(1 is the PRD's implicit accounting; 20 and 50 come from an op count of Godunov + Heun + one amortised
reinit iteration). 10–30 field passes per step. 200 flop/cell/step. All are replaced by measurements at M2.3.

**B13. V19 canary procedure.** `provisional: true`
Each gradient component is corrupted in turn (×1.05), at a θ₀ where every |∂J/∂θ_k| ≥ 0.1. A 5% error on
a near-zero component is undetectable by any check, so the test asserts this precondition. V19 passes
only if the uncorrupted control passes V14, V15 and V16 **and** every corrupted case fails all three.
A harness that always failed would otherwise "pass" V19. V19 also has its own mutation guard
(`tests/test_v19_mutants.py`, part of the V19 row). Five sabotaged harnesses (V14, V15 or V16 always
passes; the whole harness always fails; the corruption is a no-op) must each turn V19 red, and an
unmutated control in the same inner session must stay green.

---

## C — Gaps that block later milestones (no placeholder chosen)

**§12 questions, all unanswered:** Q1 target feature and Q2 periodic single feature (M2.1); Q3 mollified
vs sharp and Q4 first stack (M2.6); Q5 2D vs 3D, Q6 which GPU and Q7 per-step output (M2.4).
`reports/dense_cost_table.md` shows where dense fp64 breaks. The §4 reference fits the 8 GB gate only
with three-level checkpointing. 512³ fits no single GPU below 80 GB, and fits 80 GB only if k ≲ 30.
For Q6: this dev machine is an 8 GB M1 with no CUDA. It can run 2D and small 3D on CPU, but not the M2.4 gate.

**C1 (M2.1).** V1: r₀, V, dx, grid and T are unspecified. V2: "error < 0.1 %" of what (position relative to
distance travelled?). Origin and units of `positions`. The V21 golden-profile configuration. Axis
orientation (A1).

**C2 (M2.2).** Reinit pseudo-timestep dτ is not given. Five iterations at dτ = 0.5 dx carry information
at most 2.5 cells, but V11 checks a 3-cell band "after a reinit cycle", so V11's starting φ needs
specifying. Extension method and iteration count are not given. V8 and V9 need a position-dependent
velocity (rigid rotation, V = u(x)·n), which is neither of §5.5's two models: may a test-only third model
be added? V8 resolution and rotation period are not given, and there is a risk that < 2 % area loss is
unattainable with the mandated first-order Godunov (level-set-only schemes lose notch area even at high
order, e.g. Enright et al., JCP 183, 2002). V12's shape and resolution are not given.

**C3 (M2.3).** Direction scaling for Taylor, V15 and V16 when parameters carry mixed units (v₀ in nm/s,
p dimensionless). V5's manufactured solution. V6 error location: the SDF of a growing circle has a kink
at the centre, which caps the L∞ order domain-wide; should it be measured in a band? V7's reference: with
dx fixed, error against the analytic solution plateaus at the O(dx) spatial error and the measured order
→ 0. A same-dx, fine-dt reference is needed, and order 2 holds only where the Godunov switch does not
flip. The width of H in `etched_volume`. How adjoint cost is measured (wall-clock after compile?).

**C4 (M2.4).** V17: is 1e-12 relative or absolute? V18: how is the recomputed trajectory observed
(it lives inside the backward pass)? And bitwise equality is not guaranteed by XLA even for a
deterministic velocity: recompute can fuse differently (FMA contraction), and GPU reductions can be
nondeterministic without `--xla_gpu_deterministic_ops`. How is peak memory measured?

**C5 (M2.5).** Which three heights for CD? The sidewall-angle depth window? Definitions of `bow` and
`mask_remaining`? See also A11.

**C6 (M2.6).** Q3 and Q4. What material fractions hold in the open region above the stack? How is mask
etch rate (selectivity) parameterised?

**C7 (M2.7).** V3 needs a finite p for "p → ∞". V4's analytic facet angle is neither given nor cited.
Needs a reviewed derivation: for p = 1, V/cos θ is constant for |θ| < 90°, which may make the test
degenerate. "3D 500 nm trench": which dimension is 500 nm? §4 uses a 1 µm depth. Is the 60 s cold
(with JIT) or warm? V22 needs a separate repo, owner and licence. The V22 target of < 2 nm is one cell
at dx = 2 nm.

**C8 (M2.8).** Noise model, definition of "noise floor", which parameters. Also A12.

**C9 (CI).** No nightly check exists until M2.2, so the nightly workflow is not created yet: a scheduled
job running zero tests would report green on nothing. The gate tier needs a GPU runner (Q6). The fast
workflow has not yet run on GitHub.

---

## D — Minor, noted, no action

- D1. Cross-reference errors: §5.7 "diagnostic in §8.2" should be §8.7. §10 "ViennaPS (see §8.3)" should
  be §8.8. The kickoff's "§13 layout": §13 is the definition of done; the layout used is the union of
  every path the PRD names (§5.0, §5.1, §8, §9).
- D2. §8.5 is headed "four independent checks" but lists six (V14–V19).
- D3. §4 says dense "removes three of the four" hard problems; its table resolves all four.
- D4. The ledger fields differ between §8 (includes description) and §9 (no description). B7 uses the union.
- D5. "≤3×" (M2.3) vs "under 3×" (§13).
- D6. Mission plan v0.3 vs PRD (the PRD wins; noted because the PRD announces only the sparsity
  revision): TVD-RK3 became RK2; "leave the charging hook in" became charging out of scope (§3); window
  w2–24 vs weeks 0–22; the plan preferred avoiding reinit or an exact adjoint, and said "reinitialise
  rarely", while the PRD reinitialises every 5 steps.
- D7. The lesson plan (not authoritative) uses a combined model V = v_iso + v_dir·max(0, n·ẑ)^p, not
  §5.5's two separate models, and repeats A4(a).
- D8. The §5.2 CFL "assert at every step" cannot be a Python `assert` on traced values inside `scan`.
  It needs `checkify` or a host callback, and the abort surfaces at the end of the jitted call.
- D9. `VelocityRequest` holds arrays and must cross `jit`/`scan`, so it is registered as a pytree.
  `step_index: int` and `time: float` will be traced arrays at runtime; the annotations are kept verbatim.
- D10. `Grid.periodic` is a field, but §5.1 fixes it (lateral periodic, vertical not). It is validated to
  §5.1 now; Q2 may change it.
- D11. The multi-dimensional CFL: max|V|·dt/dx ≤ 0.5 implies Σ|V nᵢ|·dt/dx ≤ 0.5√d ≈ 0.87 in 3D, which is
  inside the first-order Godunov / TVD-RK2 bound. Consistent.
- D12. V3 says "unity yield"; yields are M5 vocabulary. For M2 it can only mean v₀.

---

## New items raised while implementing the 2026-09-11 decisions

**A16. Does `point_index` mean the row in the padded array, or a stable cell id?** Blocks the contract freeze (M2.3).
Decision §3 makes `positions` a fixed-capacity padded set built with `jnp.nonzero`, and §7.5 of the
M3 PRD keys the RNG on `(run_seed, step_index, stage_index, point_index)`. If `point_index` is the
row, then adding or dropping one band cell shifts every later row, so most points get different
random numbers for an arbitrarily small parameter change. Common random numbers break, and with
them the Taylor test that M3's V31 depends on. A stable identifier (the flattened grid-cell index)
fixes it at no cost, but it changes the contract, so it needs the M3 owner. Implemented as v0.2
*without* a stable id, and flagged here and in CROSS_MISSION X3. **Do not freeze the contract until
this is answered.**

**B14. `weights` was added to `VelocityRequest`; the decision did not list it.** `provisional: true`
§3 requires a fixed-capacity padded set and a smooth membership weight, but does not say how a
velocity model learns which entries are padding. `jnp.nonzero(..., size=K)` pads with index 0, so
padding is otherwise indistinguishable from a real point at cell 0 — M3 would spend rays on it.
*Rule:* what the padding decision forces must be carried in the request. Weight 0 marks both
padding and cells beyond the band. For the M3 owner to confirm with A16.

**B15. A slope in (2.2, 2.8) is unspecified.** `provisional: true`
Decision §4 passes [1.8, 2.2] and passes ≥ 2.8, and does not say what happens between. *Rule:*
strictest reading — FAIL.

**B16. New check IDs use letter suffixes.** `provisional: true`
V23+ belong to M3 (decision §5), so the sign check is **V1a** and the analytic sensitivities are
**V14a/V14b/V14c**, following the decision's own naming. The registry asserts no numeric ID exceeds V22.

**B17. `Material.is_void`.** `provisional: true`
Decision §10 makes the open volume an explicit material. Identifying it by name would be fragile,
so it is a flag, validated as exactly one per material set.

**B18. Case configs ship with `???` for values the coupon has not supplied.** `provisional: true`
`time.final_time_s` (needs the blanket rate from S00) and `velocity_band.width_cells` (C10). They
load only when those are supplied, which is deliberate: a case config cannot silently acquire a
guessed etch rate. Tests supply them explicitly.

**B19. Noise-floor fallback for a deterministic J.** `provisional: true`
Repeated evaluations of a deterministic J differ by exactly zero, which would exclude nothing.
*Rule:* floor = max(measured spread, 100·eps·max(|J|, 1)). The multiple is arbitrary and should be
revisited at M2.3 against the real solver's roundoff.

**B20. A parameter that is exactly zero is never probed.** `provisional: true`
Relative-unit directions scale δ_i by |θ_i| (decision §4), so a parameter sitting at 0 gets no
perturbation and its gradient column is untested. The harness raises only if *every* parameter is
zero. Whether a zero-valued parameter should fall back to an absolute scale is open; it matters the
first time a parameter's natural value is 0.

**B21. Ordering of "insufficient signal" against the degenerate rule.** `provisional: true`
A remainder that is genuinely O(h³) falls below the floor sooner and can never span five decades,
so span-first scoring would fail a correct gradient. *Rule:* fit the surviving points (≥3 needed);
a slope ≥ 2.8 is degeneracy and passes; anything else with too few decades is insufficient signal.

**B22. Lateral extent of the blanket case S00.** `provisional: true`
A blanket film is laterally uniform, so the width is arbitrary; 20 cells is used.

**B23. `seed: 0` in the case configs.** `provisional: true`
Any run-level seed is valid; 0 is used so runs are reproducible by default. It is recorded in the ledger.

**C10 (M2.1). `velocity_band.width_cells` has no value.** Decision §3 makes band width a config
parameter with a sensitivity study "in the manner of `w_mat`", but supplies no starting value. It
is `???` in every shipped config.

**C11 (M2.1). `time.final_time_s` needs the blanket etch rate.** It is depth / rate, and the rate
comes from S00. Until the coupon is etched, every case config needs it supplied explicitly.

**C12. Proposals owed by M2, per the decision's smaller items.** Not blockers now, but M2 must
propose and raise before the milestone, not during: V1 and V21 configurations built on the §2 case
grid (M2.1); the specific CD heights and the sidewall depth window for 2500 nm (M2.5); definitions
of `bow` and `mask_remaining` (M2.5); what to extract at a material crossing, a sidewall residual
against the phase-1 prediction being the obvious candidate (before M2.6); M2.8's noise model from
the coupon's metrology repeatability floor (M2.8).

---

## Second round — resolutions (2026-09-11, `decisions/2026-09-11-second-round-response.md`)

| Item | Status | What it became |
|---|---|---|
| A16 point index | **resolved** | The RNG key uses a stable, opaque `cell_id` (flattened grid index). Contract v0.3. The id and the smooth weight are two halves of one mechanism. CRN does not survive a change of grid spacing — recorded in the contract notes. |
| B14 weights | **approved, extended** | Plus `n_active` as a separate diagnostic, the `0 * NaN` warning, and a ban on `stop_gradient` for weights (now in PRD §11 and `CLAUDE.md`). |
| B15, B21 scoring | **replaced** | One two-zone rule: <3 points above the floor → FAIL; slope < 1.8 → FAIL; [1.8, 2.2] → PASS `clean_quadratic`; > 2.2 → PASS `degenerate_direction`. Five decades conditions the label, not the verdict. V19 gained a degeneracy case and a slope-six mutant; the ledger records per-direction classifications and the degenerate fraction. |
| B16 IDs | **approved, extended** | Plus a `mission` field on every ledger row. |
| B17 is_void | approved | Unchanged. Raise it rather than relaxing the exactly-one rule. |
| B18, C11 | **resolved** | Two config trees (`cases/` measured, `dev/` provisional); a provisional config may not certify a coupon-agreement check. Configs specify **depth, never time**: T = depth/rate, which makes a later rate correction free. |
| B19 noise floor | **amended** | floor = max(spread, C·√N·eps·max(|J|,1)), C = 10. The √N scaling is the point. |
| B20 zero parameter | **replaced** | Every declared parameter carries a **required** `scale`; perturbations use max(\|θ_i\|, scale_i). Canary precondition restated on the scaled gradient. Contract version bumped; M3 to adopt the same convention. |
| B22, B23 | approved | Plus: a nightly second-seed run of the gradient checks once M3 makes the seed load-bearing (C14). |
| C10 band width | **resolved** | Two bands: extension 8 cells (taper 2), evaluation 1.5 cells. K sized from the evaluation band at the worst step; peak occupancy reported. Sweep the evaluation band at 1, 1.5, 2, 3. |
| C12 proposals | **actioned** | Delivered as proposals with recommendations in `reports/proposals.md`, not as questions. |

## New items from the second round

**B24. The `narrow_span` label.** `provisional: true`
The rule says five decades conditions the `clean_quadratic` classification. A direction whose slope
lands in the band on fewer than five decades is therefore not `clean_quadratic`, but it still
passes; it is labelled `narrow_span`. Flagged because the decision names only three labels.

**B25. `n_active` is carried in `VelocityRequest`, not only in M2's diagnostics.** `provisional: true`
"Report `n_active` separately" could mean a run diagnostic. It is in the request as well, so a
velocity model can skip padded entries rather than inferring the count from weights. One more field
for the M3 owner to confirm.

**C13 (M2.2/M2.3). The evaluation band width of 1.5 cells is empirical.** The owner flagged it as
the least certain number in the decision. The sweep at 1, 1.5, 2 and 3 cells settles it, and the
result is reported whatever it says. It also sets how many velocity evaluations M3 is asked for per
step, so it belongs in the M3 conversation (CROSS_MISSION X17).

**C14 (M3.4-era). Second-seed nightly run.** Once M3 makes the run seed load-bearing, a nightly job
should run the gradient checks at a second seed while the golden regression stays at seed 0, to
catch anything accidentally seed-specific. Worth little today; noted so it is not lost.

---

## Scope decisions

**M2.1 builds the contract-shaped velocity path** (owner, 2026-09-11). M2.1's checks all need a
velocity field, and the contract routes velocity through the thin evaluation band as a padded set
that is then gathered outward — but that gather is §5.4, scheduled at M2.2. Rather than write a
temporary evaluate-everywhere path that M2.2 would delete, M2.1 builds the real one: band masks,
padded request (`weights`, `cell_id`, `n_active`), K sizing from the worst step, closest-point
projection and gather. V1 and V2 then become the first exercise of the request assembly, with
closed-form answers.

Nothing about the gates changes: same checks, same tolerances, same meaning. What changes is when
one slice of §5.4 is built. **Exposure:** the assembly is built against provisional contract v0.3.
If the M3 conversation changes who sets K (item 3) or the `cell_id` convention (item 5), this is the
module that reworks — so it stays in one file behind one function, and the contract fingerprint test
fails loudly if the contract shifts. A change to the evaluation band width (item 6) is only a config
value. PRD §6 and §5.4 are amended to match.

---

## M2.1 findings

**F1. Without reinitialisation, a band-limited velocity extension stalls the front.** Resolved for
M2.1 by owner decision (option A, 2026-09-11); the cause is removed at M2.2.
φ advances inside the extension band and not outside it, so the field distorts a little more each
step. While the interface has travelled less than the band is wide, the distortion never reaches it
and the scheme is exact. Past that point |∇φ| at the interface leaves 1 (measured 5.012), the
closest-point projection stops landing on the surface, the gathered weight falls below the floor and
the rate becomes exactly zero. Measured on plane translation, 200 steps, 150 nm of travel:

| extension band | moved (expect 149.831 nm) | \|∇φ\| at interface |
|---|---|---|
| 8 cells = 80 nm | 71.133 | 5.012 |
| 20 cells = 200 nm | 149.831 | 1.000 |
| 64 cells = 640 nm | 149.831 | 1.000 |

V1 and V2 therefore run at M2.1 with the protocol intact (200 steps, §8.1 tolerances) and the
problem reduced (30 nm of travel, half the untapered band). Widening the band instead would be
tuning a numerical parameter to make a check pass, which §11 forbids. Both limitation and mechanism
are executable tests in `tests/test_forward_2d.py`, so the full-travel versions cannot be restored
to the gate by accident.

**F2. The reduced-form constraint scales with dx, which blocks grid refinement until M2.2.**
The band is a fixed number of *cells*, so at dx = 2.5 nm it is only 20 nm wide. The same 30 nm
travel then exceeds it and V1's error jumps from 8.6e-3 to 4.5e-2:

| dx | V1 relative error |
|---|---|
| 10 nm | 8.6e-3 (passes, tolerance 1e-2) |
| 5 nm | 2.0e-3 |
| 2.5 nm | 4.5e-2 — travel exceeds the band, front stalls |

Consequence: **V6 and V7 (convergence order, nightly, M2.3) cannot run before M2.2.** A refinement
study on a scheme whose valid travel shrinks with dx measures the band limit, not the order.

**F3. V1 passes at dx = 10 nm with only 16 % margin** (8.63e-3 against a 1e-2 tolerance). At
dx = 5 nm the same check measures 2.0e-3. The accepted P5 configuration (dx = 10 nm, 64×64,
r₀ = 300 nm) is kept as approved rather than refined after seeing the result; flagging the margin
rather than quietly moving the grid. If you would prefer V1 to run at dx = 5 nm, say so and it is a
one-line change.

**F4. A NaN reached the gradient through `jnp.sqrt`, and the guard tests caught it.**
`∇φ/|∇φ|` was double-where guarded at the division, but `|∇φ| = sqrt(Σ(∂φ)²)` has an infinite
derivative at zero, and `0 * inf` is NaN — so the *masked* branch still poisoned the backward pass
at the medial axis. Both square roots in `stencils.py` now guard the sqrt itself, not only the
division that follows. Pinned by `test_no_nan_reaches_the_gradient_through_the_medial_axis`.

---

## M2.2 findings

**G0. F1 and F2 are resolved.** Reinitialisation removes the band limit: with it on, plane
translation over 150 nm (≈ 2× the extension band) lands within 7.4e-6 nm — 4.9e-8 of the travel,
against a 1e-3 tolerance. With it off, the same run still stalls at 47 %. Both halves are pinned by
`test_reinitialisation_is_what_removes_the_band_limit`, so the reason reinitialisation exists cannot
become folklore. V1 and V2 are back at full travel, and V1's margin improved on its own (F3):
relative error 3.8e-3 against a 1e-2 tolerance, where the reduced form measured 8.6e-3.

**G1. V8 (Zalesak) fails, and it is the scheme, not a bug.** **Needs an owner decision.**
After one revolution: **43 % area loss, and the notch fills completely.** The failure converges away
at roughly first order, which is what the mandated first-order Godunov scheme (§5.2) should do:

| grid | disk radius | area loss | notch filled |
|---|---|---|---|
| 100² (the canonical setup) | 15 cells | 43.0 % | 100 % |
| 200² | 30 cells | 21.0 % | 90 % |
| 300² | 45 cells | 13.3 % | 67 % |

Reaching 2 % would need roughly a 2000² grid, which no tier can run. The PRD anticipated this:
record it as a finding **with the WENO5 result alongside**, and do not change the tolerance (§11,
decision §11). WENO5 exists as a §5.2 flag but is not implemented, so the finding is currently
half-complete. See the report for options; M2.2's gate cites V8, so the gate is not met until this
is decided.

**G2. "Area" is unspecified in V8, V9 and V12, and the candidate measures differ by an order of
magnitude.** One answer is needed, not three. For V9's reversibility run:

| measure | reads | what it actually measures |
|---|---|---|
| zero-contour XOR (sub-cell) | **0.74 %** | the symmetric difference of the two regions — exact for disks |
| sharp cell indicator | 4.1 % | the same, quantised to whole cells; floor is one cell ring = 10 % of the area |
| mollified-Heaviside XOR | 5.3 % | not the region difference: it also responds to how φ is reshaped near the interface |

V9 and V12 currently assert the contour measure and record the others. V12's own numbers: the
interface does not move at all (0.000 pm), while the mollified measure reads 0.115 % and grows with
the smoothing width. **Awaiting confirmation that the zero contour is the intended measure.**

**G3. V10's verdict depends on the resolution it runs at, which the PRD does not specify.**
The anisotropy is real and first-order in dx: 4.2 %, 1.6 %, 0.67 % at dx = 10, 5, 2.5 nm. It passes
the 2 % tolerance from about 30 cells across the feature onward. V10 now runs at dx = 5 nm on the
stated rule that verification should run at a production-representative resolution — the coupon's
smallest feature is 50 cells across, while the old setup was 15. Flagged rather than assumed.

**G4. A periodic lateral boundary requires a periodic initial condition.** A φ that increases
monotonically across the lateral axis jumps by the domain width at the seam, and reinitialisation
correctly repairs that seam — which looked like a solver regression but was a bad test geometry. The
vertical-sidewall case now uses a slab, whose distance to the nearer periodic image is genuinely
periodic and comes back bit-for-bit unchanged. Worth remembering when the coupon geometries grow.

**G1 status: LOGGED AS A KNOWN DISCREPANCY** (owner accepted 2026-09-11). The V8 test is marked
`xfail(strict=True)` so the ledger row reads **fail** with the measured numbers and the suite does
not go green on a tolerated failure — and if V8 ever starts passing, the strict marker fails too, so
a fix cannot land unnoticed. The tolerance is unchanged (§11). M2.2 is accepted with this exception.

---

## The half-day diagnostic: does the diffusion matter for the product?

Asked and answered on the coupon geometry rather than on Zalesak. Two measurements:

**Isotropic etch, where an exact answer exists.** Under a constant isotropic rate a signed-distance
field simply shifts, so the exact interface at time T is the −R·T level set of the initial field —
corners included. Error against that exact answer, on the coupon trench:

| dx | depth error | CD error | sidewall angle error |
|---|---|---|---|
| 10 nm | 0.004 nm | 0.057 nm | **0.72°** |
| 5 nm | 0.006 nm | 0.011 nm | 0.17° |
| 2.5 nm | 0.007 nm | 0.014 nm | 0.03° |

**Directional etch through a mask window — the production model.** Self-convergence against dx = 2.5 nm,
800 nm of etching:

| dx | depth | CD @100 nm | CD @400 nm | sidewall angle |
|---|---|---|---|---|
| 10 nm | +0.093 nm | +0.282 nm | +0.127 nm | **+1.42°** |
| 5 nm | +0.030 nm | +0.090 nm | +0.006 nm | +0.49° |

**H1. The finding: CD and depth are fine, sidewall angle is not.** At the production resolution of
dx = 10 nm, discretisation costs ~0.3 nm of CD and ~0.1 nm of depth — comfortably inside any
metrology floor — but **1.4° of sidewall angle**. That exceeds V3's requirement (within 0.5° of 90°)
and is about 7× the metrology repeatability proposed in P8 (0.2°). It falls to ~0.5° at dx = 5 nm and
below 0.2° at dx = 2.5 nm, converging between first and second order.

So the case for a higher-order scheme rests on sidewall angle, not on Zalesak. **This needs an owner
decision** — see the report for the options (WENO5 at dx = 10, or run the coupon cases at finer dx).

**H2. "Infinitely selective mask, not modelled" is under-specified, and it bites now.**
Decision §2 says the trench is pre-cut into φ and the mask is infinitely selective. But with nothing
representing the mask, a directional etch removes the flat field at exactly the rate it removes the
trench floor, so the trench never deepens — it translates downward, preserving its depth. The
diagnostic above had to emulate the mask as a test-only lateral window on the rate.
Needed: how is the mask's footprint represented? A static lateral mask field multiplying the rate is
the obvious candidate — time-invariant, differentiable, static shapes — but it is not specified, and
M2.5's extraction and M2.8's recovery both depend on it. **Blocks a meaningful coupon simulation.**

**H1 decision (owner, 2026-09-13): WENO5 lands immediately after M2.3's gradient harness is green.**
Reason recorded so the ordering is not re-litigated: new numerics should arrive where V14/V15/V16
can verify them at once, rather than being checked by eye. The cost of deferring is near zero
because the adjoint is automatic — changing the forward scheme later means re-running the gradient
checks, not rewriting an adjoint. Until then the coupon runs carry ~1.4° of sidewall-angle
discretisation error at dx = 10 nm (H1), which must be quoted alongside any sidewall-angle result
produced before WENO5 lands. V8 is revisited at the same time.

**H2 RESOLVED (owner, 2026-09-13): the mask is modelled as geometry at M2.6**, as an accepted
amendment to decision §2. A solid body in φ, marked as mask material, with an etch-rate multiplier
of zero: infinite selectivity preserved, pattern transfer restored, undercut beneath the mask edge
allowed, a real mask corner, and geometry M3 can trace rays against. It adds a material *contact*,
not a *crossing*, so the `w_mat` study still concerns exactly one crossing — the SiGe marker — which
is what §2 was protecting. M2.6 becomes "multi-material, including the mask". PRD §5.6, §5.7 and the
§6 milestone table are amended; `reports/simplifications.md` S1 is rewritten so it no longer claims
the mask is absent. Coupon-shaped runs before M2.6 remain illustrative, not physical.

**H3 (M2.6/M2.8). Selectivity becomes a fittable parameter, and mask loss is the measurement that
tests it.** The owner confirms the coupon metrology can report mask loss. At infinite selectivity
the model predicts exactly zero, so any measured loss falsifies the zero multiplier and its size
sets the value to fit. Two things follow: (a) `mask_remaining` gains a physical reference case,
which P4 previously said it lacked; (b) if the measured loss is non-negligible, the mask stops being
a *contact* and becomes a second material *crossing*, which the `w_mat` study would then have to
cover. Worth deciding before M2.6 freezes its study design. Note also that the fit must respect the
pre-registration rule: selectivity is fitted on `calibrate` cases only.

---

## M2.3 findings

**I1. The Taylor h-range set at M2.0 overshoots the quadratic regime on the real solver.**
**CLOSED by decision I7 (owner, 2026-09-13).** The `h_max` cap proposed below was implemented as
`gradcheck.h_max_for_interface_motion` and is retained as a reusable rule, but it is *not* what
scores V14: the anchored window of I7 supersedes it, because it locates the same regime by
measurement instead of by a model of the interface motion. V14 now passes at N = 50.

V14 passes on a 13-step run and **fails on a 50-step run** — on a gradient that is demonstrably
correct. Local slopes over 3-point windows, from large h to small:

    reinit on,  N=50:  0.47  1.50  1.60  1.64  2.00  2.02  2.01  2.01
    reinit off, N=50:  0.89  2.15  0.54  3.18  4.64  0.53  3.78      (a non-smooth map)

The small-h decades are a clean O(h²). The top of B3's range is simply outside the regime where the
quadratic model holds: at h = 0.1 a 10 % change in the etch rate moves the interface about two
cells, which is a large geometric change, and the remainder is then governed by higher-order terms
rather than by the gradient. A least-squares fit across both regimes lands near 1.7 and fails.

**The gradient is fine, and we can show it:** the V19 canary on this same solver objective catches a
5 % corruption in every component, with slopes collapsing to ~1.0. So the check has power; at large
h it lacks validity.

**Proposed rule, for approval.** Cap the largest perturbation by how far it moves the interface, not
by a fixed number: `h_max` such that the induced motion is below ~0.1 cell, i.e.
`h_max ≈ 0.1 / (travel in cells)`. It is derived from what the test needs rather than fitted to the
data, and it predicts the observed transition (N = 50, 20 cells of travel → h_max ≈ 5e-3; the data
turns quadratic at h ≈ 1e-2). With the floor-based exclusion below, the usable window is then about
3–4 decades, which the decision §4 rules already pass as `narrow_span`.

**I2. Reinitialisation regularises the map, not just the forward solve.** With it off, the remainder
is erratic at every scale (0.23 → 2.5 → 1.5e-4 for successive h) — the functional is not smooth.
With it on, the small-h remainder is clean to two decimals. Worth knowing before anyone proposes
reinitialising less often to save time.

**I3. The usable h-window narrows as N grows, and the noise floor is why.** The floor scales as
√N·eps·|J|, so at production N = 625 it is ~7× higher than at N = 13. Combined with I1's upper cap,
the window at production scale may be under three decades. This needs measuring before the M2.4 and
M2.7 gates depend on it — not discovering there.

*Status after I7:* partly defused, not closed. The anchored window rides the floor wherever it is,
so a floor that rises with N moves the window rather than shrinking it — the failure mode is now the
window colliding with the kink scale from below, not with the floor. Both bounds still need
measuring at N = 625 before M2.4. The harness reports `h_window_min`/`h_window_max` and both floor
figures on every run, so the collision will be visible in the ledger before it is fatal.

**I4. k is measured: ≈ 330 field-equivalents per step, not the 1–50 bracketed at M2.0.**
Measured directly rather than estimated: `jax.linearize` partially evaluates the function and the
residuals are exactly the constants the linearised part closes over. The value is stable in N
(349.2, 349.2, 348.7 at N = 25, 50, 100), so it is a genuine per-step factor. Reinitialisation is
about 40 % of it — k falls to 210 with it switched off, which is the memory price of the smoothing
finding I2 describes.

What it implies for the M2.4 gate case (S03 in 3D, 21.6 MB per fp64 field, N = 625), from the
regenerated `reports/dense_cost_table.md`:

| scheme | checkpoints c, segment L | memory | vs the 40 GB gate |
|---|---|---|---|
| no checkpointing | — | 4.46 TB | impossible |
| remat step fn only (§7.4 literal) | — | 20.6 GB | fits, but stores all N carries |
| **two-level, c = √N = 25** | c=25, L=25 | **179 GB** | **does not fit** |
| two-level, c tuned to k | c=454, L=2 | **24.1 GB** | fits |
| three-level | c=25, L=25, step remat | 8.21 GB | fits, at 2× recompute |

**Correction to my earlier report, and it matters for M2.4.** I wrote "~20 GB under two-level
checkpointing". That was the *continuous optimum* 2√(N·k)·field = 19.6 GB; the honest discrete
number is 24.1 GB. More importantly it is the optimum for a checkpoint count **tuned to k**, which
is not what "√N checkpointing" means. Peak memory for two-level is `c + (N/c)·k` field-equivalents,
minimised at `c = √(N·k)`, not at `c = √N`. At k = 1 those coincide, which is why the distinction
never surfaced at M2.0 — the PRD's §7.4 phrasing was written under the implicit k = 1 accounting.

At the measured k = 330 they differ by 7.4×, and the naive √N split **misses the 40 GB gate at
179 GB**. The tuned optimum puts c at 454 of 625 steps: at this k the residuals dominate so heavily
that you checkpoint nearly every step and recompute only pairs.

So the gate is still reachable, but **M2.4 must choose the checkpoint count from measured k, not
from √N**, and the cheapest compliant option is three-level at 8.21 GB if the 2× recompute is
acceptable. Flagged for the owner as a change in what M2.4 has to build.

**I5. The adjoint cost ratio misses the M2.3 target, and the reason is memory.**
**Needs an owner decision.** Measured warm, compile excluded, 2D unchecked:

| case | forward | adjoint | ratio |
|---|---|---|---|
| 64×64, N=50 | 0.036 s | 0.177 s | **4.96×** (target ≤ 3×) |
| 128×128, N=100 | 0.113 s | 3.30 s | 29× |
| 192×192, N=100 | 0.118 s | 8.51 s | 72× |
| 64×64, reinit off | 0.036 s | 0.135 s | 3.78× |

The blow-up with size is not algorithmic: at k ≈ 330 the unchecked tape is ~4.3 GB at 128×100 and
~9.5 GB at 192×100, so this 8 GB machine is swapping. The 64×64 figure of 4.96× is the honest
unchecked number, and it exceeds the ≤3× target the M2.3 gate carries.

Three things follow. First, **checkpointing is likely to make the adjoint faster, not slower**, on
any machine where the unchecked tape does not fit — the usual time-for-memory trade runs backwards
here. Second, the target should probably be re-set against a checkpointed run on the real gate
hardware rather than against an unchecked 2D run on a laptop. Third, if the ratio must come down
structurally, k is the lever: the closest-point gather's bilinear sampling and the reinitialisation
sweep are the two largest contributors, and both have cheaper formulations.

Options: (a) accept 4.96× as the 2D unchecked measurement, and set the real target at M2.4 against a
checkpointed run on the H100; (b) reduce k first; (c) keep the 3× target and treat this as a failure
to be fixed before M2.4. Recommendation: (a), because the number the product cares about is the
checkpointed ratio on the gate hardware, and this laptop measurement cannot stand in for it.

**RESOLVED — option (a) (owner, 2026-09-13).** The M2.3 gate's "adjoint ≤3× forward in 2D unchecked"
is withdrawn and replaced by "adjoint ratio measured and reported, not gated"; M2.4's ≤4× becomes the
first ratio that is actually gated, and it is measured on a **checkpointed** run on the gate
hardware. PRD §6 amended at both rows.

Two consequences to carry into M2.4, so this does not read later as a target quietly dropped:

- **M2.4's ≤4× is now load-bearing for two claims, not one.** It is the only remaining cost gate on
  the adjoint, and it is also the first measurement that can confirm the prediction this decision
  rests on — that checkpointing makes the adjoint *faster* here, because the unchecked tape does not
  fit in RAM and the machine is swapping. If the checkpointed H100 ratio comes in above 4×, that
  prediction was wrong and option (b) (reduce k) returns immediately, with the closest-point
  gather's bilinear sampling and the reinitialisation sweep as the two named levers.
- **`tests/test_adjoint_cost.py` keeps measuring and asserts nothing about the ratio.** It records
  `adjoint_ratio` and `target_met` in the ledger every run. That is deliberate: a number that is
  reported but not gated still shows a regression, whereas deleting the measurement would make M2.4
  the first time anyone looks.

**I6. The noise-floor model is ~65× conservative, and that costs two decades. RESOLVED — the floor
is now measured (decision I7, owner 2026-09-13).**
B19 sets the floor at `10·√N·eps·|J|`, which for the M2.3 case gives 1.66e-9. Measured — by
evaluating the remainder at a step far below any real signal — the floor is **2.55e-11**. The model
is a safe upper bound, but paying 65× in a quantity that enters as a *threshold for discarding
data* costs about two decades of usable window. Measuring it costs three extra evaluations.

*Implemented* as `gradcheck.measure_noise_floor`, per direction, at h = 1e-10 where the true
remainder is ~1e-20·|H|. B19's model is still computed on every run and recorded beside the
measurement (`noise_floor_b19_model`, `noise_floor_model_over_measured`), so the two stay
comparable — on the M2.3 case the ratio is 51×. The measured floor is never allowed below one
rounding of J itself, so a remainder that cancels to zero by luck cannot admit noise as signal.

**I7. Five decades of clean quadratic behaviour is not attainable on this solver, for a structural
reason.** **APPROVED AND IMPLEMENTED (owner, 2026-09-13: "I will go with your call"); supersedes
the h_max proposal in I1.** PRD §7.1 amended; `CLAUDE.md` numerical requirements updated.

Sweeping one direction from h = 1e-2 down to 1e-10 on the N = 50 case, local slopes are:

    h:      1e-2  3e-3  1e-3  3e-4  1e-4  3e-5  1e-5  3e-6  1e-6  3e-7  1e-7  3e-8 ...
    slope:        0.56  2.19  0.72  2.17  2.00  2.00  2.00  1.99  2.02  1.26  0.74

Three regimes. Above ~1e-4 the slopes are **erratic, not merely wrong** — the discrete map is
piecewise smooth, and a step that straddles one of its kinks (upwind max/min switching, a band cell
entering or leaving, a bilinear sample crossing a cell boundary) picks up a first-order jump.
Between ~1e-4 and ~3e-7 the remainder is cleanly quadratic. Below that it is at the floor.

So the clean window is about **2.5–3 decades**, bounded above by the kink spacing and below by fp64.
No choice of a fixed range recovers five, and the kink spacing shrinks as N grows.

**Proposal: anchor the window to the measured floor.** Fit the ~2 decades immediately above it,
rather than a fixed h-range. The justification is not convenience: as h → 0 a correct gradient's
remainder goes as ½h²δᵀHδ while a wrong one's goes as |ε·g·δ|·h, so the decades just above the floor
are exactly where the two differ most. Measured on the M2.3 case:

| gradient | slopes in the anchored window |
|---|---|
| correct | 1.992 – 2.288 |
| 5 % corrupted (each component in turn) | **1.000, 1.000, 1.000** |

Compare the fixed window, which scored a *correct* gradient at 1.68 and failed it. The anchored
window has more discrimination, not less, which is the test that separates this from narrowing a
window until everything passes.

Everything else stays: ≥20 directions, every direction passing on its own, the two-zone scoring of
B15/B21. What changes is where the window sits and that the floor is measured rather than modelled.

### I7 as implemented, and what it cost to check

The sweep was widened to 8 decades / 17 points so the floor is inside the swept range rather than
assumed; the anchor is the smallest swept h whose remainder clears **100×** the measured floor, and
the fit runs 2 decades up from there. The margin is derived, not tuned: at 100× the weakest scored
point carries at most 1 % noise, biasing the slope by ~log10(1.01)/span ≈ 0.003. Calibrated against
an exact quadratic, whose slope is 2 by construction — 1.9996–2.0016 at 100×, 1.983–2.005 at 10×.
That calibration is now a permanent test (`test_exact_quadratic_has_slope_two`) with the bound
*computed from the constants* rather than written down, so loosening the margin fails it.

The full remainder curve on the M2.3 case, at N = 50, with the three regimes visible at once:

| h | 1e-1 … 1e-3 | 1e-3 … 3e-7 | < 1e-7 |
|---|---|---|---|
| slope | **1.29** | **2.00** | 0 (floor ≈ 1.4e-11) |
| what it is | kinks, higher-order geometry | the quadratic regime | fp64 |

The window landed at **[3.2e-7, 3.2e-4]** — inside the quadratic regime, with a decade to spare at
the top. V14 at N = 50 now reads 17 `clean_quadratic` + 3 `degenerate_direction`, all passing.

**The anti-gaming check, which is the part that matters.** A narrower window is a weaker test unless
it is shown not to be, so `test_the_anchored_window_finds_the_quadratic_regime_at_a_long_run` asserts
three things at the hardest case rather than the easiest: the anchored scoring passes; the top two
decades fitted alone score 1.29, i.e. **below** the band, which is the same signature the scoring
reads as "first-order term present" — so the finding is real and reproduces on every nightly run;
and a 5 % corruption of **each** gradient component in turn is still caught. If a future change
narrows the window into blindness, that third assertion fails before the gate turns green.

Cost: 20 evaluations of J per direction instead of 11, i.e. V14 roughly doubled. Nothing was moved
between tiers to pay for it.

Note this supersedes I1's h_max rule, which was derived for the wrong bound: it caps interface
motion at 0.1 cell, but the kinks turn out to be far finer than that — the clean region begins near
0.002 cells of motion. `h_max_for_interface_motion` is implemented and available, but the anchored
window is the better mechanism and I recommend it instead.

**I8. V5, V6 and V7 are implemented and reported; the orders are comfortable, with one caveat and
one structural finding.** (§8.2, M2.3 gate: "V5–V7 convergence orders reported".)

| check | what it measures | requirement | observed | pairwise |
|---|---|---|---|---|
| V5 (MMS) | manufactured source, directional law | ≥ 0.9 | **1.73** L¹, 1.62 L∞ | 1.86, 1.70, 1.62 |
| V6 (spatial) | dx refined, vs V1's analytic disk | ≥ 0.9 | **1.60** L¹, 1.59 L∞ | 1.69, 1.63, 1.48 |
| V7 (temporal) | dt refined, self-convergence | ≥ 1.9 | **1.990** | 1.97, 1.99, 2.01 |

**The caveat: V5 and V6 exceeding first order is not good news, it is a soft measurement.** A circle
is the friendliest interface an upwind scheme can be given — smooth, and the exact solution stays a
signed distance, which reinitialisation keeps restoring. The downward drift in both pairwise
sequences is superconvergence decaying toward the asymptotic rate. The margin over 0.9 will shrink
on real geometry, and the half-day diagnostic already showed where: **sidewall angle 1.4° at
dx = 10 against V3's 0.5°.** V6 passing at 1.60 and V3 failing at dx = 10 are consistent, not
contradictory, and WENO5 is still the fix. The pairwise sequence is asserted and recorded so the
shrinkage shows up as a number rather than as a surprise at M2.7.

V7 landing on 1.990 with no drift is the control that makes the caveat credible: the same norm over
the same region reports exactly the nominal order when the nominal order is what is there.

**The structural finding: a dx-refinement study of this solver is invalid without reinitialisation,
and the reason is the band.** The extension band is 8 *cells*, so halving dx halves it in nanometres
while the travel stays fixed. Past some refinement the interface leaves the band it started in;
outside the band the rate is tapered to zero, so those cells never moved, and φ near the interface is
assembled from stale values. Measured observed order with reinitialisation off: **−1.96**, i.e. the
error grows about 4× per refinement. This is the M2.1 decision (owner, option A — "travel beyond the
extension band needs reinitialisation") reappearing as a convergence result, and it is pinned in
`test_the_order_study_needs_reinitialisation` so V6's configuration cannot later look like a
convenient default. V7 is the exception and must run with reinitialisation off — its schedule is
fixed in *steps*, so refining dt would change how often it fires — so V7 holds the travel to 3 cells,
a bound taken from the band width rather than chosen after seeing a result.

**V5 required a solver change: an MMS source path** (§8.2 asks for one explicitly). `solver.step`
and `solver.solve` take an optional `source`; `final_phi` — the differentiated production entry
point — does not, and `M2Config` has no source field, so no configuration can switch it on.
`test_the_mms_source_path_is_off_in_production` asserts all three, plus that a non-zero source does
change the answer, so the path cannot rot into dead code while the order study reports on nothing.

The manufactured solution is worth recording because two details decide whether it is valid rather
than decorative. φ_exact = |x − c| − r(t) is an **exact signed distance for any r(t)**, so the
closest-point projection the gather performs lands on the true interface; a manufactured φ with
|∇φ| ≠ 1 would be projected elsewhere and the solver's rate would not be the R the source assumes.
And the source **carries the band taper** — S = −r′(t) − w(φ_exact)·R_exact — because the solver's
rate is `R(projection)·extension_weight(φ)`, which is zero outside the band; an untapered source
would leave the far field drifting by ∫R dt, and that error propagates inward one cell per step and
contaminates the measurement region. r(t) = 200 − 3t + 4·sin(2πt/T) oscillates, which no positive
etch rate produces, so the check cannot be satisfied by the physics being right.

## WENO5 findings (post-M2.3, owner decision H1)

**J0. WENO5 is implemented, verified at order 5.13, and wired to the existing `spatial_scheme`
flag.** It is **not** the default — `godunov` still is. **Needs an owner decision** (see J5).
The reconstruction is Jiang & Peng's HJ-WENO; only D⁻ and D⁺ change, the Godunov upwind selection
on top of them is untouched, and the flag is threaded into reinitialisation as well as advection
(a first-order Hamiltonian in the repair would put back the smearing the advection just removed).

What it buys, measured on product-shaped cases rather than benchmarks:

| check | godunov | weno5 | requirement |
|---|---|---|---|
| V10 grid anisotropy (dx = 5) | 1.625 % | **0.013 %** | < 2 % |
| V1 disk radius error (dx = 5) | 1.081 % | **0.009 %** | < 1 % |
| V6 L¹ error at dx = 20 | 6.64 | **0.043** (155× lower) | — |
| V14 worst slope, directional | 1.802 | **1.996** | ≥ 1.8 |

The V14 line is worth noting: WENO5 makes the *gradient* check cleaner too, not just the forward
solve. And V15/V16 hold (3.5e-16 and 4.9e-13 against a 1e-10 tolerance).

**One deliberate deviation from the published scheme, for differentiability.** Jiang & Peng set the
weight regulariser to ε = 1e-6·max(v₁²…v₅²). That `max` over the stencil is a kink in the
differentiated path, which §11 forbids, so ε is fixed instead. This is safe because the smoothness
indicators are built from v = Δφ/dx, which is |∇φ| — order 1 for a signed distance, whatever the
units. `test_the_fixed_epsilon_is_not_load_bearing` moves ε by ±100× and requires the observed order
not to move, so the deviation is shown inert rather than asserted to be.

**J1. The Taylor remainder develops a cancellation notch under WENO5, and the harness mis-scored it
as a gradient error. FIXED.** This was the serious-looking result of the day and it turned out to be
a harness defect, not a gradient defect.

V14 failed in 2 of 20 directions at slope ~1.59 — the "first-order term present" signature. The
discriminators said otherwise immediately: V15 and V16 passed, and V14 on the one-parameter
isotropic model was perfect at 2.000–2.000. Reading the signed remainder for direction 19:

    h        1e-3      3.2e-4    1e-4      3.2e-5    1e-5
    R       -2.3e-7   -9.4e-7   -9.4e-8   -9.4e-9   -9.4e-10
             ^ 40x smaller than the h^2 line predicts; |R| went DOWN as h went UP

From 3.2e-4 downward each half-decade divides |R| by exactly 10 — slope 2.000. The gradient is
right. At h ≈ 1e-3 the h² and h³ terms nearly cancel, |R| dips, and a least-squares fit through the
dip reads shallow.

**Two hypotheses were tried and the first was wrong, which is worth recording.** The first guess was
the C^⌊p⌋ smoothness of max(0, n·ẑ)^p: at p = 2 the second derivative jumps, and a Taylor remainder
is a second-order probe. It made three predictions; p = 3 passing was one of them, but the other two
failed — the failing directions had no excess p-component (0.18 and 0.49 against a median of 0.32),
and p = 1.5, which is *less* smooth, passed cleanly at 1.921. Non-monotonic in smoothness, so not a
smoothness effect. The second guess, a sign change in the signed remainder, also failed: the guard
found none, because the terms only *nearly* cancel and R never crosses zero.

The invariant that does hold is **monotonicity**: for a remainder governed by a single leading term,
|R| grows with h. `_cancellation_ceiling` caps the scored window below the first point where it does
not. **It cannot hide a wrong gradient**, and that is the property that makes it admissible rather
than convenient: a pure first-order error gives R ∝ h, strictly monotone, so nothing is cut; and
when a wrong gradient does notch, the cut keeps every point *below* it, which is exactly where the
first-order term dominates. Pinned by two unit tests on synthetic remainders and by the canary,
which still catches 5 % corruptions in every component under both schemes.

WENO5 is what surfaced this: a first-order scheme's leading error is large enough that the
cancellation sits above the scored window. Making the scheme accurate moved it inside.

**J2. WENO5's scheme-level order is ~2.08, not the ≥4 the registry requires. Needs an owner
decision.** Logged as `xfail(strict)` with the evidence, tolerance unchanged (§11).

The reconstruction itself is verified at **5.13** on a field with an analytic derivative, so WENO5 is
not broken. The leading hypothesis is that the scheme-level order is capped by the **velocity
extension path, which is second order by construction**: `normals` uses central differences and
`gather_to_band` uses bilinear interpolation. A fifth-order Hamiltonian fed a second-order rate field
cannot be better than second order overall.

*This is a hypothesis, not a measurement.* The experiment that would confirm it: raise the normals
to a fourth-order central stencil and the gather to a cubic sample, and re-run — the order should
move toward 4 if the cap is real, and stay at 2 if it is not. That is half a day, and it is the
right next step if ≥4 matters. The error constant is 155× better either way, which is what V3 and
V10 actually care about.

**J3. WENO5's wide stencil manufactured a spurious interface at the domain boundary. FIXED, and the
CFL assertion is what caught it.** `shift` implements the Neumann condition by replicating the edge
value; WENO reads that flat run as perfectly smooth, gives it maximum weight and extrapolates from
an artefact of the boundary. On Zalesak at 100², φ at the bottom-edge corner cell (99, 4) drifted
from +67 to **−2.96** by step 588 — a phantom solid blob 67 cells from anything real. It entered the
velocity band and, under the rotation model whose rate grows with radius, pushed the CFL from 0.34 to
0.70 and tripped the assertion.

Fixed by falling back to the two-point stencil within 3 cells of a non-periodic edge. The mask is a
function of the grid index alone, so it is a compile-time constant: no data-dependent branch, no
kink. Worth noting that the CFL assertion did its job here — it turned a silent far-field corruption
into a hard stop.

**J4. V8 (Zalesak) under WENO5: the area criterion is met, the notch criterion is not. Needs an
owner decision.**

| | area loss | notch filled | CFL |
|---|---|---|---|
| godunov, dt = 1.0 (the logged case) | 43.0 % | 100 % | 0.339 |
| weno5, dt = 1.0 | — | — | **0.693, trips** |
| weno5, dt = 0.5 (same physical time) | **1.23 %** | **31.4 %** | 0.169 |

So WENO5 takes the area loss from 43 % to 1.23 %, inside V8's 2 %. But V8 also requires the notch to
survive and the test asserts `filled < 0.25`; 31.4 % fails it. **The check still fails, on a
different criterion.** This is the honest end state for a pure level-set method and it matches what
I said before building it: the notch is a corner, and WENO5's high order applies in smooth regions.

The remaining CFL trip at dt = 1.0 is *after* the J3 fix and is not yet explained. It is not a
phantom blob — at the spike φ_min is −6.00 and the field is healthy. The likely cause is the
rotation model itself: it is a **test-only** velocity whose rate grows without bound with distance
from the rotation centre, so a band cell far out gets an arbitrarily large rate. No production model
has that property — `isotropic` and `directional` return rates independent of position. That makes
this a test-harness interaction rather than a solver defect, but **it is unconfirmed** and should not
be written up as resolved.

Options for V8: (a) leave it xfail with the new numbers recorded, which is strictly more informative
than the 43 % figure; (b) spend the time to resolve the notch, which means the corner methods
discussed — subcell-fix reinitialisation is the differentiability-safe one; (c) accept that a pure
level-set method does not pass Zalesak's notch criterion and record that as the mission's position.
Recommendation: (a) now, and treat (b) as a scheduled piece of work only if the notch matters for a
product feature — which, on the coupon geometry, it has not so far.

**J5. Should WENO5 become the default? Needs an owner decision — I have not changed it.**
`spatial_scheme` still defaults to `godunov`, so nothing in `configs/` changes behaviour and every
existing check result stands. Arguments for switching: every product-shaped measurement improves by
two orders of magnitude, V14 gets *cleaner*, and V3's sidewall requirement is the reason the scheme
was built. Arguments for waiting: J2 is unresolved, V8 still fails on the notch, and the cost has
not been measured — WENO5 is a 7-point stencil against 2, so both runtime and the residual factor k
will rise, and k feeds directly into the M2.4 memory gate (finding I4). **That last point is the
one I would want measured before switching**, because M2.4's checkpoint schedule is derived from k.

**J6. The 1.4° sidewall figure in H1 is not currently reproducible, and I could not check what
WENO5 does to it.** The half-day diagnostic lived in a scratch script under `reports/local/`, which
is untracked, and it is gone. My attempted reconstruction reads ~5° for *both* schemes at *both*
dx = 10 and dx = 5 — it does not converge with dx, so it is dominated by my own hard-edged mask
emulation rather than by discretisation, and it is not a valid V3 proxy. I have not reported a
sidewall number for WENO5 as a result.

This matters because **1.4° against V3's 0.5° is the stated justification for building WENO5 at
all.** V3 is an M2.7 gate check and does not exist yet. The right fix is to build V3 properly rather
than to reconstruct a diagnostic, and that is now blocked behind H2's resolution — the mask becomes
real geometry at M2.6, and a collimated-etch sidewall measurement wants the real mask, not a lateral
rate window. Recorded so the justification is not treated as settled evidence.

## M2.4 findings

**K0. Checkpointing, 3D, V17 and V18 are implemented and passing.** `m2/checkpoint.py` provides the
schedule; `final_phi` takes `levels` (0 none, 2 two-level, 3 three-level) and a segment length.
V17 holds at 1e-12 across every schedule tried, in 2D and 3D. 3D needed essentially no new code —
`shift`, the Hamiltonian, the band and the gather are all written over `grid.ndim` — which is the
§4 dense/static-shape decision being repaid.

**K1. Checkpointing makes the adjoint FASTER, confirming the premise of decision I5(a).**
Measured warm, compile excluded, 2D on CPU:

| case | unchecked | two-level L=1 | three-level L=25 |
|---|---|---|---|
| 64², N=50 | 4.75× | **3.19×** | 4.15× |
| 128², N=100 | 22.22× | **4.91×** | 6.11× |
| 192², N=100 | 34.30× | **5.77×** | 6.74× |

The usual time-for-memory trade runs backwards because the unchecked tape does not fit — 4.3 GB at
128² on an 8 GB machine — so the unchecked adjoint swaps and replaying a segment from cache is
cheaper than faulting residuals back in. **This was a prediction when I5(a) was decided and it is
now a measurement.** It does not establish M2.4's ≤4× gate, which is a checkpointed 3D run on the
H100; these are 2D on CPU.

**K2. §7.4's warning about `jax.checkpoint` is inverted at the measured k. PRD amended.**
Writing two-level peak as `N/L + L·k`, the optimum is **L\* = √(N/k)**. At N = 625, k = 349 that is
1.3, i.e. L = 1 — which *is* remat-on-the-step-function, the thing §7.4 warns against. The warning
was correct under the implicit k = 1 accounting the document was written with, where N carries is
the naive cost. At k = 349, naive is N·k = 218,000 field-equivalents and remat-on-the-step is
N + k = 955. Not a small correction: it changes what M2.4 should build.

**K3. Peak memory is only half measurable on this machine, and the gate depends on the half that is
not.** The distinction had to be made explicit in the code or it would have become a false claim:

- *persistent* storage — segment boundaries kept for the whole backward pass — is measured directly
  by counting the constants the linearised function closes over, and equals ceil(N/L) exactly;
- *transient* storage — residuals live while one segment is replayed, the `L·k` term — is recomputed
  and freed, so it is never a jaxpr constant and is invisible to that method. It is **modelled**.

The transient term dominates: at N = 625, L = 1, k = 349 the split is 625 persistent to 349
transient, and at L = 5 it is 125 to 1746. So reading the measured number as peak would understate a
checkpointed run by more than 10×. `persistent_residuals` now says so in its docstring, and the
structural test asserts it equals ceil(N/L) rather than pretending it is a memory figure.

**The 40 GB gate therefore cannot be closed here.** It needs a device-memory profile on the H100
(decision X13). Projection from the measured k, for S03 3D (270×100×100, N = 625, 21.6 MB/field):

| schedule | Godunov (k=349) | WENO5 (k=1050) |
|---|---|---|
| none | 4.5 TB | 14.2 TB |
| two-level, L = 1 | **20.6 GB** | **36.2 GB** |
| three-level, L = 25 | 8.6 GB | 23.8 GB |

**K4. WENO5 triples k, and that is the answer to J5.** k = 349.2 (Godunov) → **1049.5** (WENO5), a
7-point stencil against 2. Forward time rises 2.45× as well. The consequence for the M2.4 gate is in
the table above: two-level goes from 20.6 GB to **36.2 GB against a 40 GB budget — 90 % utilised**.
That is not a comfortable margin for a projection whose dominant term is modelled rather than
measured.

So J5 (should WENO5 be the default?) now has a number attached, and the options are no longer
symmetric: (a) keep Godunov as the default and enable WENO5 per-case where accuracy needs it;
(b) make WENO5 the default and run three-level checkpointing at the gate (23.8 GB, 2× recompute);
(c) make WENO5 the default and accept the 90 % margin on two-level. **Recommendation: (a) until the
H100 profile exists**, because the margin in (c) rests on a modelled term and (b) trades a measured
2× recompute for accuracy whose scheme-level order is itself an open finding (J2). The accuracy case
for WENO5 is real (V10 125× better) but it is not yet tied to a product requirement that Godunov
fails — J6 removed that evidence.

**K5. `m2/rng.py` now implements the contract's RNG keying, which nothing did before.** V18 needs it
and M3 consumes it (CROSS_MISSION X2, X3): `fold_in(fold_in(PRNGKey(run_seed), step), stage)` then
`fold_in(cell_id)` per entry. `fold_in` rather than arithmetic on the seed, or (seed+1, step)
collides with (seed, step+1) and two runs that should be independent share a stream. This adds no
field to `VelocityRequest`, so the contract fingerprint is unchanged and v0.3 still stands.

**K6. Two M2.4 gate items cannot be closed on this hardware.** Peak memory < 40 GB, and the adjoint
ratio ≤ 4× warm — both are specified on one H100 and both need the Lambda run decision X13 already
anticipated. Everything else in the gate is done: V17, V18, 3D V14/V15/V16/V17, and the checkpoint
schedule derived from measured k. **The milestone cannot be declared complete without that run**, and
I am not going to report it as complete on projections.

