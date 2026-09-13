# Proposals owed by M2

Decision C12: these arrive as **proposals with a recommendation**, not as open questions. Each is
marked `provisional: true` and needs a review, not research. All heights and depths are measured
**below the original wafer surface** (the top of the silicon at t = 0), since the mask is not
modelled; axis 0 increases toward the plasma, so depth = (z_surface − z).

---

## P1 — CD heights (M2.5) `provisional: true`

**Recommendation.** Report CD at a fixed absolute ladder of depths:

    100, 400, 800, 1250, 1700, 2400 nm

with **100 / 1250 / 2400 nm** as the three headline heights for a completed 2500 nm etch.

**Why.** 100 nm sits just below the mask edge, where undercut and top rounding show first; 1250 nm
is mid-depth, where bowing is largest; 2400 nm is 100 nm clear of the nominal floor, so it stays on
the sidewall rather than sampling floor curvature. Absolute, not fractional, because a
depth-relative window changes the number of fit points discretely and produces a gradient artifact —
and because the metrology reports absolute heights.

**The trajectory wrinkle.** At the 1/3 and 2/3 etch times the trench is only ~830 and ~1670 nm deep,
so the deeper heights do not exist yet. The ladder handles this: report every height that lies above
the current floor and mark the rest as not-applicable. Never silently substitute a fraction of the
current depth, which would reintroduce exactly the artifact this avoids.

## P2 — Sidewall-angle window (M2.5) `provisional: true`

**Recommendation.** Least-squares fit over the fixed absolute window **400–1600 nm**.

**Why.** It excludes the top ~400 nm, where mask-edge rounding dominates, and the bottom ~900 nm,
where the floor's curvature bends the wall. It spans 1.2 µm — 120 cells at dx = 10 nm — so the fit
has ample points and is insensitive to any single crossing. For the partial etch times, fit the
portion of the window that exists and report the window actually used.

## P3 — `bow` (M2.5) `provisional: true`

**Recommendation.** `bow = max over 200–2300 nm of half_width(z) − half_width(100 nm)`, in nm,
positive outward, reported together with `bow_depth`, the depth at which the maximum occurs.

**Why.** Bow is a bulge relative to the opening, so it needs a reference, and the top CD height is
the reference an engineer already has. Reporting the location as well as the magnitude distinguishes
a bow from a general CD offset, which is the distinction the profile actually cares about.

## P4 — `mask_remaining` (M2.5) `provisional: true`

**Recommendation.** Defined only when a material carries `is_mask`: the vertical extent, at the
lateral position of the mask centre, over which that material's fraction is ≥ 0.5, with the
fraction-weighted sub-cell crossing at both ends. It returns `None` when no mask material exists,
which is the case for the whole coupon (the mask is not modelled, S1 in `simplifications.md`).

**Why.** Keeping it a real sub-cell measurement rather than a cell count means it stays
differentiable.

**Updated 2026-09-13.** The mask is modelled as geometry from M2.6, and the coupon metrology can
report mask loss, so `mask_remaining` now **has a physical reference case**. At infinite selectivity
the model predicts exactly zero loss, which makes this the sharpest test of that assumption we have:
any measured loss falsifies the zero multiplier directly, and its size sets the selectivity to fit.

## P5 — V1 configuration (M2.1) **ACCEPTED 2026-09-11**

**Recommendation.** A solid disk, **etched** isotropically, on the case grid vocabulary:
dx = 10 nm, 64 × 64 cells, r₀ = 300 nm centred, isotropic model with `v_iso` = 5.83 nm/s,
T = 25.7 s, 200 steps. Analytic answer **r(t) = r₀ − R·t**, ending at r = 150 nm; tolerance as the
PRD states, relative error < 1 % at 200 steps.

**The sign, settled.** PRD §8.1 wrote V1 as *growth*, r(t) = r₀ + Vt, which assumed the
pre-decision sign convention. The owner confirmed on 2026-09-11 that **positive rate removes
material is ground truth**, so the disk shrinks: r(t) = r₀ − R·t. PRD §8.1 and the check registry
are amended, and V14a/V14b's analytic answers carry the matching sign (dr/dR = −T).

## P6 — V21 golden profile (M2.1) `provisional: true`

**Recommendation.** The S03 2D geometry coarsened to dx = 40 nm (68 × 25 cells), depth 2500 nm,
cfl_target 0.4 → N = 157, `configs/dev/S03.yaml` rates, seed 0. Freeze the final φ field hash plus
the extracted depth and the three headline CDs, to 1e-10.

**Why.** Coarse enough for the fast tier (seconds), while still exercising advection,
reinitialisation, extension and extraction on the real case geometry. Because time is derived from
depth, a later correction to the etch rate does **not** invalidate this golden file (decision B18).

## P7 — Phase-2 observable at the material crossing (before M2.6) `provisional: true`

**Recommendation.** A **sidewall half-width residual** against the phase-1 prediction, over the
fixed window 350–500 nm (the marker sits at ~400–430 nm): report `max |Δhalf_width|`, the depth at
which it occurs, and the change in instantaneous vertical etch rate across the crossing.

**Why.** The observable at a crossing is a local perturbation of the sidewall and a change in rate,
not a change in final depth — the front returns to plain silicon afterwards and the endpoint depth
is nearly unaffected. Referencing the phase-1 prediction is what isolates the marker's effect from
everything else the profile is doing.

## P8 — M2.8 noise model `provisional: true`

**Recommendation.** Independent Gaussian noise on each extracted observable, with 1σ taken from the
coupon's metrology repeatability floor once measured. Until then: **CD 1.0 nm, depth 5 nm, sidewall
angle 0.2°**. Recovery counts as successful when each parameter lands inside the interval implied by
that noise, and M2.8 additionally asserts that **V14 passes at the recovered parameters** — recovery
is never itself evidence of gradient correctness.

**Why.** These are placeholder magnitudes chosen to be the right order for CD-SEM and
cross-section metrology at this feature size, not measurements. The real numbers come from repeated
measurements of the same coupon feature, which is also what makes the M2.8 tolerance defensible.
