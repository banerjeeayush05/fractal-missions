# Decision record — second round of open questions

Verbatim copy of the mission owner's response supplied on 2026-09-11, answering the items raised
while implementing `2026-09-11-open-questions-response-r3.md`. **Authoritative: supersedes both the
PRD and the first-round decisions where they conflict.** Implementation status is tracked in
`../OPEN_QUESTIONS.md`; cross-mission consequences in `../../CROSS_MISSION.md`.

---

All items answered. As before, these are authoritative and supersede both the PRD and my
11 September decisions where they conflict; update the affected text in the same commit that
implements each one.

Three of these are corrections to my own decisions rather than to the PRD. A16 in particular found
a hole in the contract I specified, before it shipped. That is the file working.

---

## A16 — the RNG key uses a stable cell id

**Your analysis is correct and the defect is mine.** Keyed on the array row, one cell entering the
band shifts every later row, so nearly every point draws different randomness for an arbitrarily
small parameter change. Common random numbers break and the Taylor test cannot be run at all.

The key is

    (run_seed, step_index, stage_index, cell_id)

where `cell_id` is the **flattened grid index** of the cell a surface point was derived from. It is
stable for the life of the grid: cell 12345 is always cell 12345, and what varies is only whether it
is in the band. `jnp.nonzero` already returns these, so the fix is free.

Three things to record in the contract text and in `m2/CLAUDE.md`:

- **The stable id and the smooth membership weight are two halves of one mechanism, not two
  safeguards.** The id stops a cell's arrival from disturbing everyone else's draws. The weight
  reaching zero at the band edge stops the arriving cell's own draw from entering discontinuously.
  Remove either and J is discontinuous in theta. Say this in a comment, because the next person to
  read this code will be tempted to simplify one of them away.
- **The id is opaque to the velocity model.** It seeds an RNG and is never used to index geometry.
  Otherwise M3 acquires a dependency on M2's grid layout and the two missions stop being
  independently replaceable.
- **Common random numbers do not survive a change of grid spacing.** Ids mean different things at
  different dx. Harmless in M2 where the velocity is deterministic; a real constraint on M3's
  convergence and cross-resolution studies. Write it into the contract notes now.

Keep the contract provisional until the M3 owner confirms. This item joins that conversation rather
than starting a new one.

## B14 — `weights` approved

Correct catch and the right mechanism: `jnp.nonzero` pads with index 0, a real cell, so without
weights M3 would spend rays on the corner of the domain. Weight 0 marking both padding and
beyond-band is better than two separate flags.

Three additions:

- **Report `n_active` separately** as a diagnostic. The weights array cannot distinguish padding
  from a real band-edge cell, and you need that number to size K and to see overflow coming.
- **Zero weight does not protect against NaN.** `0 * NaN` is NaN, and it will poison the whole
  gradient. Padded entries need a benign position, and normals need the double-where guard
  regardless of weight.
- **No `stop_gradient` on the weights.** They are part of the forward map and their derivative is
  real. PRD §11 already forbids this move for reinitialisation and material blending; weights belong
  on the same list, because something that looks like a mask is exactly what a future reader will
  wrap to make a test pass.

## B15 and B21 — one two-zone rule, replacing my three-zone one

My rule was badly shaped and you were right to notice the gap. The fix is to remove the gap, not
fill it.

A wrong gradient leaves the first-order term uncancelled, so the remainder goes as h and the slope
tends to 1. There is no mechanism by which an incorrect gradient produces a slope above 2 — where
such an error could hide it sits below the noise floor, and those points are excluded anyway. The
upper bound of 2.2 therefore discriminates nothing.

Scoring, in this order:

    fewer than 3 points survive the floor   -> FAIL, "insufficient signal"
    slope < 1.8                             -> FAIL
    slope in [1.8, 2.2]                     -> PASS, logged "clean_quadratic"
    slope > 2.2                             -> PASS, logged "degenerate_direction"

The five-decade span is a condition on the `clean_quadratic` classification, not on pass or fail.
That resolves B21 as you reasoned: an O(h³) remainder reaches the floor sooner and can never span
five decades, so span-first scoring would fail a correct gradient.

Three consequences, all of which must land in the same commit:

- **V19 gains a case.** The loosening opens one new blind spot: a harness returning garbage with a
  slope of 6 now passes as degenerate. Add a mutant that is both genuinely degenerate and corrupted,
  and assert it still fails.
- **The ledger records the classification per direction, and the degenerate fraction per run.** That
  population is the thing to watch. If it moves from two directions in twenty to fifteen, something
  changed even though everything is green — treat a sudden jump as a finding.
- **M3's V31 adopts the same rule.** Same test, different objective. Raise it with the M3 owner;
  M2 passing what M3 fails would waste a day on pure bookkeeping.

PRD §7.1 currently states the band as a hard requirement. Update that text.

## B16 — approved, plus a mission field

Letter suffixes are right and the registry assertion that no numeric ID exceeds V22 is a good
instinct: it makes the M2/M3 boundary machine-enforced rather than remembered.

Add a `mission` field to every ledger row. Suffixes prevent a collision today; a mission field
prevents one when a later mission invents its own.

## B17 — approved

A flag rather than a name is correct; matching on the string would break the first time someone
writes "Void". Keep the exactly-one validation. If a synthetic test ever needs a config with no void
at all, raise it rather than relaxing the rule quietly.

## B18 and C11 — a development config, and the invariance that makes it cheap

The discipline is right and refusing to load beats defaulting. But taken literally it means nothing
runs end to end until a wafer has been etched, and M2.1 through M2.5 all need runnable configs now.

**Two config trees:**

- `configs/cases/*` keep `???` and refuse to load without measured values.
- `configs/dev/*` carry a nominal rate marked `provisional: true`, and are **forbidden from writing
  a ledger row for any check that claims agreement with the coupon**. Gate and development work
  proceeds; nothing can silently certify against a guessed rate.

**The nominal rate is 5.83 nm/s (350 nm/min).** Reasoning, so it can be checked rather than trusted:
a published measurement of chlorine ICP silicon etching reports 0.35 um/min, against 1 um/min for an
SF6-based fluorine RIE on the same samples. Same chemistry class and same tool class as lampoly,
which is a transformer-coupled high-density source built for polysilicon. Mid-range for the tool,
and a measured number rather than a handbook figure.

The arithmetic closes:

    T   = 2500 nm / 5.83 nm/s              = 429 s, use 430
    N   = 2500 / (0.4 * 10 nm)             = 625 steps
    dt  = 430 / 625                        = 0.688 s
    CFL = 5.83 * 0.688 / 10                = 0.40   OK

Channel split: 10% isotropic, 90% directional — so `v_iso` = 0.58 nm/s and `v_dir` = 5.25 nm/s at
normal incidence, `p` = 2. This split is a weaker guess than the total and is exactly what the coupon
is meant to determine. Mark it provisional.

**The rule that makes a later correction free: configs specify target depth, never final time.**
Derive `T = depth / rate`. The level-set equation is invariant under scaling the rate by lambda and
the time by 1/lambda — the profile is bit-for-bit identical, N comes from depth and dx so it does not
move, CFL has both factors scaling oppositely so it does not move, and the reinitialisation
pseudo-timestep is in length units so it does not move either. With depth-based configs a rate
correction propagates automatically and **the V21 golden profile does not break** on a number with no
physical consequence. With time-based configs it would.

That also dissolves C11: what the config is missing is not the etch time, it is the rate, and the
rate is a parameter of the velocity model rather than a property of the run.

One thing to record in `reports/simplifications.md`: the scaling invariance holds because nothing in
M2 or M3 has an intrinsic time constant. A chemistry with one — polymer accumulation, for instance —
would break it. Flag it for whoever builds M5.

## B19 — noise floor, with the constant made principled

Correct problem: a deterministic J evaluated twice differs by exactly zero, so measured spread
excludes nothing.

The 100 can stop being arbitrary. Roundoff accumulates over N steps, and under a random-walk model
the accumulated error scales as sqrt(N) * eps * |J|. At N = 625 that is about 25 eps, so your 100 eps
is roughly four-fold conservative — defensible, but the scaling is the useful part, because it keeps
the floor sensible when N changes by a factor of three between grid refinements. Use

    floor = max(measured_spread, C * sqrt(N) * eps * max(|J|, 1)),  C = 10

Keep your plan to revisit it against the real solver at M2.3.

## B20 — declared parameter scales

Correct, and the hazard is wider than you stated. Guarding only exact zero catches the case that
eventually announces itself and misses the one that does not: a parameter sitting at 1e-12 gets a
perturbation of 1e-12, which probes nothing while reporting a pass.

Every declared parameter carries a **required** `scale`, its typical magnitude. Perturbations use

    delta_i  proportional to  max(|theta_i|, scale_i)

Required rather than optional, because an optional field will be omitted exactly where it matters.
For the reference velocity model: `v_iso` scale 0.58 nm/s, `v_dir` scale 5.25 nm/s, `p` scale 2.

Two knock-ons. B13's canary precondition should be stated on the **scaled** gradient, not the raw
one. And M8 will need these same numbers when it steps across mixed units, so declare them as a
property of the parameter rather than as a test-harness detail.

This is a schema change: land it before M2.3, and bump the contract version even though `scale` lives
on M2's parameter declarations rather than inside `VelocityRequest`. Tell the M3 owner the
convention so both missions declare scales the same way.

## B22 and B23 — approved

Twenty cells for a laterally uniform film costs nothing; the only requirement is that it comfortably
exceeds the stencil and reinitialisation reach so nothing wraps. Seed 0 recorded in the ledger is
right.

For later, when M3 makes the seed load-bearing: have a nightly job run the gradient checks at a
second seed, with the golden regression staying at seed 0. It catches anything accidentally
seed-specific. Worth little today and a lot at M3.4.

## C10 — two bands, not one

This is not a missing number, it is a missing decision, and it has a cost attached. Asking for one
band width assumes there is one band.

    extension_band_cells = 8     weight tapering to zero over the outer 2
    eval_band_cells      = 1.5   weight zero beyond

**Extension band, 8 cells.** Advection and reinitialisation need a valid velocity wherever their
stencils reach. The interface moves up to CFL * reinit_every = 2 cells between reinitialisations,
reinitialisation propagates about n_reinit = 5 cells, and the upwind stencil reaches 1 more.

**Evaluation band, 1.5 cells.** This is where M3 is actually called, and it must be thin. Cells
deeper in the band project to nearly the same point on the surface, so calling M3 for all of them
buys the same expensive Monte Carlo estimate several times. M3 budgets around 1e5
interface-adjacent cells and its §5.8 cost analysis is already its hardest constraint; an 8-cell
evaluation band would overshoot it by a factor of several. At 1.5 cells the shell is about 3 cells
thick, giving two to three evaluation points per surface location — ample coverage for the gather.

The bridge is already sanctioned: PRD §5.4 permits closest-point gather as the extension mechanism,
so M3 evaluates on the thin set and M2 gathers outward across the full band at negligible cost.

**The sensitivity study sweeps the evaluation band** at 1, 1.5, 2 and 3 cells, because that is the
one with a cost consequence. The extension band only needs a check that 8 is sufficient — if a
narrower band changes the answer, it was too narrow.

Note that 1.5 is the number I am least sure of; the structure is clear and the value is empirical.
Let the sweep settle it and report what it says.

**Capacity K is sized from the evaluation band, at the worst step, not the first.** The interface
lengthens as the trench deepens, so the final-step band count substantially exceeds the initial one.
Size K from the deepest expected geometry with margin, or the overflow assertion will fire at step
900 of 1000 after most of a gate run has been spent. Report peak occupancy every run.

Raise the evaluation band width with the M3 owner alongside A16: it is the parameter that sets how
many velocity evaluations M3 is asked for per step.

## C12 — proposals, not questions

The list is accurate and naming what you owe is right. One instruction: those should arrive as
**proposals with a recommendation**, not as open questions. You have the context to choose CD heights
for a 2500 nm trench. Choose, justify, mark provisional, and let the reply be a review rather than a
research task.

---

## Five things the file did not raise

- **`0 * NaN` is NaN.** Covered under B14, repeated because it is the failure that costs a day.
- **CRN does not survive a grid change.** Covered under A16. Harmless in M2, a real constraint on M3.
- **The cell id must be opaque to M3.** Covered under A16. Seeding only, never indexing.
- **K depends on the trajectory, not the initial geometry.** Covered under C10.
- **The weights participate in the gradient.** Covered under B14. It belongs in `m2/CLAUDE.md` next
  to the stop-gradient prohibition.

## The M3 conversation now has six items

1. Where the run-level seed lives (M2 §7.6 says the request; M3 §7.5 implies M3's config).
2. Whether the two Heun stages share draws. M2 passes `stage_index`; M3 decides.
3. Who sets K, and that M3 must accept a padded array with a `weights` mask.
4. N is about 625 per solve, so roughly 1250 velocity calls, not the 1000 in M3 §5.8.
5. **New:** the RNG key uses a stable, opaque `cell_id`; CRN does not cross grid spacings.
6. **New:** the evaluation band width, which sets how many velocity evaluations M3 is asked for.

Do not freeze the contract until 1, 2, 5 and 6 are answered.

## Proceed

Land A16, B14 and the B15/B21 scoring rule first, since they change what the existing tests mean.
Then B20, which is a schema change and should precede M2.3. Then C10 and the dev configs, which
unblock everything downstream. Re-run and report.
