# Cross-mission change log

Findings or decisions in one mission that change what another mission's PRD assumes.
The PRDs are human-owned: agents record entries here and never edit another mission's PRD.

**Rules**
- A mission's first action (before its CLAUDE.md, before M3.0) is to read this file and copy
  every entry that targets it into its own `OPEN_QUESTIONS.md`.
- When a human answers an item that has an entry here, update its **Status** in the same change:
  `open` → `decided: <decision, date>` → `applied: <where the target mission absorbed it>`.
- Line references are to `m3/M3_PRD.md` as first read on 2026-09-11. Re-check them if that file changes.

Statuses below were last updated 2026-09-11, after the M2 decision records
(`m2/decisions/2026-09-11-open-questions-response-r3.md`, then the second-round response). "decided" means M2 has an answer and has
implemented it; **no entry is `applied` yet, because M3 has not started.**

---

## M2 → M3

**X1. The velocity contract is not frozen, and is now at v0.3.** Status: **decided (M2 §3), M3 sign-off pending.**
M3 says the contract is "already frozen by M2 PRD §5.5" (L182) and gates M3.0 on matching it
"exactly" (L387). It is at **v0.3** and stays `provisional` until the M3 owner signs off; M2.3 must
not freeze it before then. M3's window starts week 6 (L6), likely before M2.3, so M3.0 cannot gate
on a frozen contract. **Action for M3:** review v0.3 (`m2/m2/interface.py`, fingerprint pinned by
test) and either sign off or negotiate the six items listed at the end of this section.

**X2. `run_seed` and `stage_index` now exist.** Status: **decided (M2 §3), implemented in M2.**
The RNG key is a deterministic function of `(run_seed, step_index, stage_index, cell_id)` (X3),
never global, never stateful, never hashed from `time`. Draws are **independent per RK stage** by
default; passing `stage_index` lets a model choose to share them, and M2 does not make that choice.
M3 §7.5 (L432) must be updated to name `stage_index`.

**X3. The RNG key uses a stable, opaque `cell_id`.** Status: **decided (M2 A16), implemented in v0.3.**
Keyed on the array row, one cell entering the band shifted every later row and broke common random
numbers. The key is now `(run_seed, step_index, stage_index, cell_id)`, where `cell_id` is the
flattened grid index — stable for the life of the grid. Three things M3 must adopt:
(a) the id is **opaque**: seed with it, never index geometry with it, or M3 acquires a dependency on
M2's grid layout; (b) the id and the smooth band weight are **two halves of one mechanism** — the id
stops an arrival from disturbing other cells' draws, the weight stops the arriving cell's own draw
from entering discontinuously; (c) **common random numbers do not survive a change of grid
spacing**, which is a real constraint on M3's convergence and cross-resolution studies.

**X4. Step counts are derived now, and the reference case is much smaller.** Status: **decided (M2 §2).**
Bare step counts are gone: N = ceil(D / (cfl_target·dx)) at cfl_target 0.4. The reference is coupon
case S03 (CD 500 nm, pitch 1000 nm, depth 2500 nm), N ≥ 625 at dx = 10 nm, so a TVD-RK2 run is
**1250 velocity evaluations**, not the 1000 in M3 §5.8 (L346). M3's budget table (L348–357) is
built on a 512×128×128 domain with ~10⁵ interface-adjacent cells; S03 is far smaller, so the ray
budget and the 10²–10³ s forward estimate should be recomputed before M3.6.

**X5. Checkpointing a whole velocity call does not bound its memory.** Status: **open (M3 side).**
M3 L374–379: the backward pass of a single checkpointed call still records its full tape, ~10¹⁰
values ≈ 80 GB in fp64 by M3's own count. It needs ray batching with remat, or the analytic
hit-point derivative of §5.2 (see X7). Mitigating news from M2: with case S03 two-level
checkpointing fits the memory gate comfortably, so M2 recomputes the forward **once**, not twice —
M3's velocity model is therefore evaluated twice per gradient, not three times.

**X6. V18 restated: bitwise applies to the RNG, not the trajectory.** Status: **decided (M2 §8).**
RNG keys and sampled values must be bitwise identical on recompute; the trajectory must agree to
**1e-12 relative**; CI enables deterministic GPU ops. Bitwise trajectory equality is not something
XLA guarantees. M3's V34/V36 and its reliance on V18 (L378) should adopt the same wording.

**X7. M3's V32 repeats a claim M2 has shown to be wrong.** Status: **decided (M2 §5).**
JAX builds every reverse-mode derivative by linearising with JVP rules and transposing, so `jvp`
and `vjp` share those rules for **all** primitives, not only custom ones. V15/V16 — and M3's V32
(L474) — verify **transpose consistency**, not derivative correctness. M2 added analytic-sensitivity
checks (V14a/b/c) to cover the gap; M3 needs its own equivalent, and this matters more there
because the §5.2 hit-point derivative will likely be a custom rule. Executable evidence:
`m2/tests/test_v19_blind_spots.py`.

**X8. Taylor scoring is now a two-zone rule, and M3's V31 adopts it.** Status: **decided (M2 B15/B21).**
Fewer than 3 points above the noise floor → FAIL "insufficient signal"; slope < 1.8 → FAIL;
[1.8, 2.2] → PASS `clean_quadratic`; > 2.2 → PASS `degenerate_direction`. There is no upper bound to
enforce: a wrong gradient tends to slope 1. Five decades conditions the label, not the verdict. The
floor is max(measured spread, 10·√N·eps·max(|J|,1)) — and M3's spread is real Monte Carlo noise,
which is what the mechanism was built for. Concrete case in M3: V27's Γ = (2π/3)L₀ (L462) is linear
in L₀. **M2 passing what M3 fails would waste a day on bookkeeping**, so raise this with the M3
owner. M3's V35 should also gain the degeneracy case M2's V19 did.

**X9. Etch sign and orientation are fixed.** Status: **decided (M2 §1), implemented in M2.**
Velocity models return an **etch rate R** in nm/s: positive removes material. M2 advects
φ_t − R|∇φ| = 0, so M3's `velocity.py` adapter returns a positive rate for etching and never
applies the sign itself. Axis 0 increases toward the plasma, ẑ = +e₀.

**X10. Flux lagging must not be hidden state.** Status: **open.**
If M3's adapter caches flux between calls (L365), that is stateful: under checkpoint recompute the
backward pass replays steps in a different order, so the cache would serve the wrong step's flux,
silently. The lag schedule has to live in M2's time loop or be explicit carried state. A contract
question for the X1 freeze.

**X11. Check-ID ranges settled; the shared ledger is not.** Status: **partly decided (M2 §5, §13).**
V23–V40 belong to M3; M2 must never use them, and its registry asserts no numeric ID above V22
(M2 added the letter-suffixed V1a, V14a, V14b, V14c). Still open: M3 L447 writes to "the shared
`verification_ledger.json`", but M2's loader rejects unknown IDs
(`m2/m2/verification/ledger.py`). Either the registry becomes per-mission and extensible, or each
mission keeps its own ledger. M2 also decided (§13) that CI on a clean checkout is the only writer
of the ledger of record, with local runs writing an untracked one; M3 should match.

**X12. Monorepo layout and CLAUDE.md split.** Status: **decided (M2 §3), implemented.**
`CLAUDE.md` at the repo root now holds only the rules every mission shares; `m2/CLAUDE.md` holds
M2's scope and non-goals; `m3/CLAUDE.md` is a placeholder telling the M3 agent to read this file
first and then write the real one (M3 PRD §0). M2 is a path dependency for M3 (`../m2`,
distribution `fractal-m2`, import `m2`).

**X13. GPU decision, shared.** Status: **decided (M2 §6).**
Lambda **H100 80GB SXM** for gate runs; A100 80GB and B200 acceptable alternates; avoid RTX 6000,
A6000 and B300-class parts, where fp64 runs at a fraction of fp32. Development runs on CPU. This
answers M3's Q7 as well as M2's Q6. M3's fp32-traversal option (§7.6) stays a later optimisation,
gated on V31 still passing.

**X14. M3 can reuse M2's gradient harness rather than reimplementing it.** Status: **suggestion.**
`m2.verification.gradcheck` implements the decided scoring rules (Taylor with noise floor and
degenerate directions, transpose checks, protocol that refuses to be loosened), and
`m2.verification.ledger` plus the pytest plugin implement check registration. M3 already takes
`fractal-m2` as a dependency (L527). Reimplementing them would put the same rules in two places.

**X15. A canary needs a control and a mutation guard.** Status: **suggestion, from M2.0 experience.**
M3's V35 (disable warped-area sampling and require V31 to fail) is the analogue of M2's V19. Two
things M2 learned the hard way: assert that the **unmutated** case passes in the same run, or a
harness that always fails would "pass" the canary; and add a guard that deliberately sabotages the
harness and requires the canary to go red (`m2/tests/test_v19_mutants.py`), or nothing tests the
test of the test.

**X16. M2 is 2D-first and endpoint-only.** Status: **decided (M2 §9).**
The first demo is 2D — every coupon case is a long trench, exactly represented in 2D — and the
differentiated solve returns only the final φ, with debug frames dumped outside the gradient path.
M3's coupling work (M3.7) should assume that, and 3D is off the demo's critical path.

**X17. The evaluation band width sets how many velocity calls M3 gets.** Status: **decided (M2 C10), value provisional.**
M2 now has two bands: extension (8 cells) for its own stencils, and **evaluation (1.5 cells)**,
which is the only set the velocity model is called on. Deeper cells project to nearly the same
surface point, so an 8-cell evaluation band would buy the same expensive Monte Carlo estimate
several times over and overshoot M3's §5.8 budget by a factor of several. At 1.5 cells the shell is
about 3 cells thick, giving two to three evaluation points per surface location, with §5.4's
closest-point gather bridging to the full band. The owner flags 1.5 as the least certain number;
a sweep at 1, 1.5, 2 and 3 settles it (M2 OPEN_QUESTIONS C13). M3 should agree the value, since it
sets the per-step call count.

**X18. Declared parameter scales are a shared convention.** Status: **decided (M2 B20).**
Every differentiable parameter declares a **required** `scale`, its typical magnitude, and
perturbations use max(|θ_i|, scale_i). Without it a parameter sitting at 1e-12 gets a 1e-12
perturbation and reports a pass having probed nothing. Scales are a property of the parameter, not
a harness detail — M8 needs the same numbers when it steps across mixed units. M3 should declare
its own parameters the same way.

**X19. The mask will exist as geometry, which M3 needs for shadowing.** Status: **decided (M2, owner 2026-09-13).**
From M2.6 the mask is a solid body in φ with a zero etch-rate multiplier, not a rate mask. This
matters to M3 directly: a ray tracer can intersect geometry but cannot see a multiplier, so a mask
represented only as "rate = 0 here" would let rays pass straight through it and overestimate flux at
the trench bottom. M3 should expect a material-fraction field that includes a mask material, and
should treat it as an opaque obstacle. Until M2.6 lands there is no mask in φ at all, so any
M2 profile M3 consumes before then is illustrative rather than physical.

---

## The M3 conversation: six items, and the freeze depends on four

1. Where the run-level seed lives (M2 §7.6 says the request; M3 §7.5 implies M3's config). **X2**
2. Whether the two Heun stages share draws. M2 passes `stage_index`; M3 decides. **X2**
3. Who sets K, and that M3 must accept a padded array with a `weights` mask. **X3, B14**
4. N is about 625 per solve, so roughly 1250 velocity calls, not the 1000 in M3 §5.8. **X4**
5. The RNG key uses a stable, opaque `cell_id`; CRN does not cross grid spacings. **X3**
6. The evaluation band width, which sets how many velocity evaluations M3 is asked for. **X17**

**Do not freeze the contract until 1, 2, 5 and 6 are answered.**
