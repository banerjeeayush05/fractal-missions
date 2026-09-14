# Handoff — fractal-missions

For a new contributor joining M2. Written 2026-09-14, at commit `89e5e9a`.

## What this is

Everything Fractal sells reduces to one capability: **the derivative of a wafer feature with
respect to a process parameter.** Recipe solve is that derivative fed to an optimiser; calibration
is it fed to a fitter; drift inversion is it pointed at chamber state.

**M2 builds the piece that computes it**: level-set interface evolution in 2D and 3D, in JAX, with
a gradient that has been *verified* rather than assumed. M2 does not compute the surface velocity —
that is M3. M2 consumes a velocity field through a frozen contract so the two can fail independently.

The governing constraint, from PRD §2: **a gradient that is wrong but plausible is worse than no
gradient.** An optimiser fed a subtly wrong gradient still converges — to the wrong recipe,
confidently, with nothing in the output announcing it. Almost every unusual rule in this repo exists
because of that sentence.

## Read these first, in this order

| file | what it is | size |
|---|---|---|
| `CLAUDE.md` (root) | rules every mission shares | 33 lines |
| `WORKING_AGREEMENT.md` | how to report problems and when to stop and ask | 56 lines |
| `m2/CLAUDE.md` | M2 scope, conventions, numerics, traps | 113 lines |
| `m2/M2_PRD.md` | the specification. Authoritative, and amended in place | 846 lines |
| `m2/OPEN_QUESTIONS.md` | every open item and every finding, with evidence | 1173 lines |
| `CROSS_MISSION.md` | M2 findings that change M3's assumptions (X1–X19) | 196 lines |

`m2/decisions/` holds verbatim owner decision records. **Decisions supersede the PRD**, and the PRD
text is amended in the same commit that implements one — so the PRD and the code should never
disagree. If you find them disagreeing, that is a bug worth reporting.

## Where things stand

| milestone | state |
|---|---|
| M2.0 scaffold, schema, ledger, CI | done |
| M2.1 2D forward + velocity-request assembly | done |
| M2.2 reinitialisation | done, one accepted discrepancy (V8) |
| M2.3 reverse-mode adjoint 2D | done |
| WENO5 (scheduled between M2.3 and M2.4) | done, three open findings |
| M2.4 checkpointing + 3D | **built, gate not closed — needs an H100** |
| M2.5 differentiable extraction | not started |
| M2.6 multi-material incl. mask as geometry | not started |
| M2.7 performance + ViennaPS cross-check | not started |
| M2.8 inverse sanity | not started |

**251 tests pass, 2 xfail.** Fast tier 87 s against a 180 s budget; full suite ~8 min.

## Running it

```bash
cd m2
uv sync
uv run pytest                      # everything, including nightly
uv run pytest -m "not nightly"     # the fast tier only
uv run pytest -m nightly
```

Checks write rows to an untracked local ledger. **CI on a clean checkout is the only writer of the
ledger of record** (`m2/reports/verification_ledger.json`) — local runs cannot update it, and that
is deliberate. `uv run pytest --ledger-of-record` refuses on a dirty tree.

Check IDs are pre-registered in `m2/verification/registry.py` (V1–V22 plus V1a, V14a–c). A test
claims one with `@pytest.mark.check("V14")`.

## Rules that are not negotiable

These are the ones most likely to be violated by accident, because each looks like ordinary good
practice pointing the other way.

- **Never loosen a tolerance to make a check pass.** A failure is a finding. If a tolerance is
  genuinely wrong, change it in its own commit with the reason. Two checks currently fail and are
  recorded as `xfail(strict)` rather than tolerated — see below.
- **Never resolve an ambiguity by guessing.** It goes in `OPEN_QUESTIONS.md`, classified A (PRD
  error), B (placeholder, with the mechanical rule that produced it), C (gap blocking a later
  milestone), D (minor). Assume the PRD contains arithmetic and cross-reference errors — it has, and
  several have been found.
- **Stop at every milestone gate.** Run the acceptance tests, report, wait. Never build ahead.
- **No `stop_gradient` on reinitialisation, extension or material blending.** The band weights look
  like a mask, which is exactly what a future reader will wrap to make a test pass. They participate
  in the gradient.
- **Do not tune `w_mat`, CFL, band width or `n_reinit` to make a gradient test pass.** That is a
  finding, not a fix.
- **One autodiff system: JAX.** Never add PyTorch, Warp, PETSc, Mitsuba, OptiX or Embree. A gradient
  bug must have one place to be.
- **GPL clean room.** ViennaPS and ViennaRay are GPL-3.0. Never link, import, vendor or copy them.
  Cross-code comparison runs in a separate process and repo, exchanging VTK files. If you read their
  source, write a natural-language note; do not transcribe.
- **Verification records are generated, never hand-written.**

## Conventions — a sign error here inverts every gradient

- φ < 0 inside solid, > 0 in open volume. `n = ∇φ/|∇φ|` points from solid into open.
- Velocity models return an **etch rate R in nm/s: positive removes material.** M2 advects
  `φ_t − R|∇φ| = 0`, so the sign flip lives in M2 alone, not in every M3/M5 model.
- **Axis 0 is vertical and increases toward the plasma**, so `ẑ = +e₀`.
- V14–V16 cannot catch a sign inversion. **V1a is the only guard.**
- Configs specify a target **depth, never a final time**. `T = depth / rate` is derived, so a rate
  correction rescales time and changes nothing else.

## Two known discrepancies, both recorded with evidence

Both are `xfail(strict=True)` — so if either starts passing, CI goes red and forces a revisit. They
cannot quietly become folklore.

**V8 (Zalesak's disk).** Under WENO5 the area loss is 1.23 %, inside the 2 % tolerance, but the
notch criterion still fails (31.4 % filled against < 25 % required). This is the honest end state
for a pure level-set method: the notch is a corner, and WENO5's high order applies in smooth
regions. Hybrid particle/level-set methods fix it and would break differentiability in three
separate places — see finding J4 and the discussion above it.

**V6 under WENO5.** The registry requires observed order ≥ 4; we measure ~2.08. The reconstruction
itself is verified at 5.13 in `tests/test_weno.py`, so WENO5 is not broken. The leading hypothesis
is that the *velocity-extension path* caps it — `normals` uses central differences and the gather
uses bilinear interpolation, both second order, and a fifth-order Hamiltonian fed a second-order
rate field cannot beat second order. **That is a hypothesis, not a measurement**, and finding J2
names the experiment that would confirm it.

## Open decisions waiting on the owner

Findings are lettered by milestone: F (M2.1), G (M2.2), H, I (M2.3), J (WENO5), K (M2.4), L
(pre-flight). All are in `m2/OPEN_QUESTIONS.md` with the evidence.

- **J2** — WENO5's order shortfall above. Confirm the cap, or accept it?
- **J4** — V8's notch. Leave recorded, or spend effort on corner methods?
- **J5** — should WENO5 be the *default*? It triples the residual factor k (349 → 1050), which moves
  the M2.4 memory projection from 20.6 GB to 36.2 GB against a 40 GB budget. **Interim decision:
  Godunov stays the default** until the H100 profile exists.
- **J6** — the 1.4° sidewall figure that justified building WENO5 **is not currently reproducible**.
  The diagnostic lived in an untracked scratch directory and is gone. WENO5's case now rests on V10
  (125× better) and V1 (120× better), not on a product requirement Godunov fails. Worth knowing
  before anyone cites that 1.4°.

## What is blocked, and on what

**M2.4's gate cannot close on the available hardware.** Two of its items are specified on one H100:

- peak memory < 40 GB
- warm adjoint ratio ≤ 4×

Everything else in that gate is done: V17 (checkpointed gradient equals unchecked to 1e-12, 2D and
3D), V18, 3D V14/V15/V16/V17, and a checkpoint schedule derived from measured k.

There is a subtlety worth understanding before reading any memory number here. **Peak memory is only
half measurable without a GPU.** *Persistent* storage — the segment boundaries kept for the whole
backward pass — is measured directly and equals `ceil(N/L)`. *Transient* storage — the residuals
live while one segment is replayed — is recomputed and freed, never becomes a jaxpr constant, and is
**modelled**. The transient term dominates: at L = 5 the split is 125 persistent to 1746 transient.
Reading the measured number as peak understates by more than 10×.

`m2/scripts/gate_m2_4.py` runs the gate and writes a dated report. `--forward-only` validates the run
plan on CPU. That pre-flight already caught two defects that would each have wasted paid instance
time (findings L1, L2).

Hardware-gated numbers go into a dated report under `m2/reports/` — **not** into the ledger of
record. `reports/verification.md`, which is generated from the ledger and will cite that report as
an externally-produced measurement, is an M2.7 deliverable and does not exist yet.

## Traps that have already bitten

- **NaN poisoning.** `∇φ/|∇φ|` blows up where `|∇φ| → 0` — the trench centreline, a circle's centre.
  `jnp.where` does not save you: a NaN in the untaken branch still poisons the gradient. Use the
  double-where pattern. Masking afterwards does not save you either, because `0 * NaN` is NaN.
  This bit once through `jnp.sqrt`, whose derivative is infinite at zero: the division was guarded
  and the sqrt was not.
- **The Taylor remainder can notch.** `|R|` must grow with h; where it does not, two terms are
  cancelling and the log–log slope is meaningless — it reads ~1.6 on a gradient that is exactly
  right. Do not "fix" a failing V14 by widening the band.
- **The directional law has a kink on vertical sidewalls** (`n·ẑ = 0`). Require p > 1, use p ≥ 2 in
  gradient tests, and keep `0^p` out of the p-derivative.
- **The mask is not modelled until M2.6**, and it matters more than it sounds. A directional etch
  removes the flat field at exactly the rate it removes the trench floor, so a trench does not
  deepen — it translates, and erodes. This silently affected the M2.4 gate case (finding L2).
- **Diagnostics that justify a decision must be tracked.** J6 happened because one lived in
  `reports/local/`, which is untracked. Put them in `tests/` or a tracked report.
- Squared one-sided differences are fine: `max(x,0)²` is C¹. The rule is no kinks in the
  differentiated path, not "never write `max`".

## A question this handoff does not answer

The working agreement is written for a single owner who makes the calls on open findings. With a
second contributor, **who decides** — and does the "stop at every gate and wait for review" rule mean
waiting on either of you, or both? Worth settling explicitly rather than discovering it at the next
gate.

## Untracked files

`m3/M3_PRD.md`, `m3/M3_Lesson_Plan.docx` and `real world/` are present but not committed, and were
deliberately left alone. The repo has a public remote, so committing them would publish them.
