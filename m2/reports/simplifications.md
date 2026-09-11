# Simplifications

Every simplification M2 makes, with its justification and its cost (PRD §9 item 7).
Hand-maintained, unlike the verification record, which is generated. Add an entry when you make a
simplification, not afterwards.

## S1 — The mask is not modelled

**What.** The 500 nm SiO₂ mask is treated as infinitely selective and is not represented. The
trench is pre-cut through it into the initial φ, in both coupon phases.

**Why.** Decision §2. It keeps phase 1 strictly single-material, and keeps phase 2 to exactly
**one** material transition — the SiGe marker being studied — rather than three. Modelling the
mask would add two more transitions and their erosion physics to a mission whose subject is the
geometry engine, not selectivity.

**Cost.** `mask_remaining` has no physical reference case. It is still implemented and tested
against synthetic geometry at M2.5, but it is never validated against the coupon. If mask erosion
later matters, this is the first simplification to revisit.

## S2 — First demo is 2D, and 3D is not physically meaningful for these cases

**What.** The demo path is 2D. 3D exists to exercise the 3D code path (case `S03_3d`).

**Why.** Decision §9 and §2. Every coupon case is a long line trench, so it is translationally
invariant and exactly represented in 2D. 3D would compute the same answer at 100× the cost.

**Cost.** None for these cases, but a 3D run of them must not be reported as a physical prediction.

## S3 — Prescribed analytic velocity, not transport

**What.** M2's velocity models are analytic (isotropic, and directional `R = v₀·max(0, n·ẑ)^p`).

**Why.** PRD §1 and §3: M3 owns flux. M2 and M3 must be able to fail independently.

**Cost.** M2 cannot produce ARDE, bowing, microloading or faceting from real flux, and nothing in
M2 validates the physics of the rate law. Recorded again in the §8.9 scope boundary.

## S4 — Test-only velocity models beyond the two shipped ones

**What.** A rigid-rotation velocity model exists for V8 (Zalesak) and V9 (reversibility).

**Why.** Decision C2 approved it: it is a test harness, not a flux model, so §3's ban on transport
does not bite. Those two checks need a position-dependent field that neither shipped model provides.

**Cost.** It must never leave `tests/`, and it is not a physical model of anything.

## S5 — Fixed material-transition width, fixed reinit pseudo-timestep

**What.** `w_mat` defaults to 2 cells; the mollified Heaviside width is fixed at 1.5·dx in code;
reinitialisation uses dτ = 0.5·dx.

**Why.** Decisions §10, A2 and C2. Fixing them in code rather than config keeps them out of reach
of anyone trying to tune a gradient test into passing (§11).

**Cost.** `w_mat` is a real physical approximation at a material boundary, which is why its
sensitivity study (M2.6) reports the gradient's dependence on it **and** `w_mat` as a fraction of
the marker thickness. If the gradient depends strongly on it, that is a finding, not a knob.

## S6 — Time is derived from depth, which assumes no intrinsic time constant

**What.** Configs specify a target depth; the final time is derived as T = depth / rate. Scaling the
rate by λ and the time by 1/λ leaves the profile bit-for-bit identical: N comes from depth and dx,
CFL has both factors scaling oppositely, and the reinitialisation pseudo-timestep is in length units.

**Why.** Decision B18. It makes a later correction to the etch rate free — the V21 golden profile
does not break on a number with no physical consequence — and it is what lets development proceed on
a nominal rate before the coupon is etched.

**Cost, and who needs to know.** The invariance holds only because nothing in M2 or M3 has an
intrinsic time constant. A chemistry that does — polymer accumulation, surface-species residence,
anything with a memory — breaks it, and then time is no longer a free rescaling. **Flagged for
whoever builds M5.**
