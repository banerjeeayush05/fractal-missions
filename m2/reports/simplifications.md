# Simplifications

Every simplification M2 makes, with its justification and its cost (PRD §9 item 7).
Hand-maintained, unlike the verification record, which is generated. Add an entry when you make a
simplification, not afterwards.

## S1 — The mask is modelled as geometry, with infinite selectivity

**What.** From M2.6 the 500 nm SiO2 mask is a solid body in φ, marked as mask material, with an
etch-rate multiplier of **zero**. It never erodes. Before M2.6 it is absent entirely, so any
coupon-shaped run in M2.1–M2.5 is illustrative, not physical.

**Why.** Amendment to decision §2, owner 2026-09-13. The original decision left the mask out to keep
phase 1 single-material, but finding H2 showed that a patterned etch cannot be simulated without it:
with nothing covering the field, a directional etch removes the flat surface at exactly the rate it
deepens the floor, so the trench translates downward instead of deepening. Modelling it as geometry
rather than as a rate multiplier keeps undercut beneath the mask edge, gives a real mask corner, and
lets M3 trace rays against it — a multiplier is invisible to a ray tracer.

**What is still simplified: infinite selectivity.** The mask never erodes or facets. It adds a
material *contact*, not a *crossing* — the front slides under the mask but never passes into it — so
the `w_mat` study still concerns exactly one crossing, the SiGe marker.

**When erosion arrives.** Infinite selectivity is the zero value of the per-material rate multiplier
that decision §10 already puts in the parameter PyTree, so switching it on is a number, not a
rebuild. What is missing is physics, not code: a selectivity value, and the angular yield curve that
makes a mask corner facet rather than merely shorten (M5, fed by M3 — and the subject of V4,
currently on hold for exactly that reason).

**The risk this carries, and how it is checked.** Silicon-to-oxide selectivity in a chlorine etch is
typically in the tens, so a 2500 nm silicon etch may remove a meaningful fraction of a 500 nm mask.
If it does, the mask corner recedes, the opening widens and the top CD drifts — an error that looks
like physics, would be absorbed into a fitted closure parameter, and would then fail to transfer to
a different mask thickness. **The coupon metrology can report mask loss** (owner, 2026-09-13), so
this is checkable: the model predicts exactly zero loss, and any measured loss is evidence the
multiplier must become non-zero.

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
