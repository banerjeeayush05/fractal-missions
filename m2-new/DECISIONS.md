# DECISIONS — M2 (m2-new build)

Settled design choices, newest first. An item arrives here from `OPEN_QUESTIONS.md` once it is decided,
and is deleted from that file. Owner words are quoted verbatim before they are interpreted.

Decisions supersede the PRD, and `M2_PRD.md` in this folder is amended in place by each one; amended
passages are marked with the date. The original `m2/` tree was deleted on 2026-09-17; its own records —
the 2026-09-11 owner decisions, the lettered findings A1–L2, the proposals — are in git history at
commit `b5b320c`, e.g. `git show b5b320c:m2/OPEN_QUESTIONS.md`.

---

## 2026-09-17 — Five items settled in one pass (S12.3, S14.2, S17.5, S3.1, S4.1)

> Skip S14.2, 1% is good for S12.3, keep 200 nm for S17.5, I don't care about the version name figure
> that out, the roles names are fine

**S12.3 — V12a's tolerance is 1 %.** Accepted. The mean interface shift attributable to
reinitialisation alone, over a full run, must stay below V1's own forward tolerance: if
reinitialisation may move the surface further than the check it runs inside allows, that check's verdict
is decided by reinitialisation rather than by the physics. V12a currently measures 2.15 % and stays
`xfail(strict=True)` under S12.1.

**S14.2 — the WENO5 re-measurement is skipped.** WENO5's recorded numbers (V1 0.034 %, V9 0.009 %,
V10 0.009 %, V12a 0.002 %, V14a 0.36 %, V6 order ~2) were measured BEFORE the SSP-RK3 fix of S14.1 and
are not re-confirmed. Consequence, stated plainly: the revisit of S12.1 promised at stage 12 — "accept
the drift now, re-measure after WENO5" — does not happen. The reinitialisation drift stands as accepted
on the Godunov numbers alone, and WENO5's figures should be treated as indicative rather than measured
until someone re-runs them. Godunov remains the default, so nothing in the shipped path depends on them.

**S17.5 — mask thickness stays 200 nm.** Still `provisional`: no coupon metrology stands behind it. The
grid height (290 cells) is derived from it, so a measured thickness would change the grid. S00 keeps
100 nm on the same rule.

**S3.1 — the velocity contract is v0.3.** Delegated to the build ("I don't care about the version name
figure that out"). §5.0's field list is the one implemented and it includes `cell_id` and `n_active`,
which §5.5's description of v0.2 does not mention; v0.3 is therefore the version that matches the code.
`CONTRACT_PROVISIONAL` stays `True` — the version number is settled, the FREEZE is not, because PRD §5.5
requires the M3 owner's sign-off and that has not happened.

**S4.1 — the role vocabulary is `calibrate`, `validate`, `dev`.** Accepted as declared.

## 2026-09-17 — Adjoint ratio of 4.65x accepted (S15.4)

> Just accept 4.65 adjoint ratio.

The M2.4 gate asks for a warm adjoint ratio at or under 4x. Measured on one H100: **4.65x**, a 16 %
miss, with the forward at 4.2 s against a 60 s budget and peak memory at 20.7 GB against 40 GB.

Accepted as the measured figure rather than chased. The 4x target came from decision I5(a), set at M2.4
on a different machine before this build existed, and nothing downstream is waiting on the difference.
The ratio is flat in run length (4.28x at N = 5 to 4.63x at N = 625) and L = 1 is measured optimal, so
this is the scheme's cost, not a schedule mistake.

If it is ever worth chasing, the lead is `top_k`, which is half the forward step (S18.2) and would lift
the forward time and the ratio together.

## 2026-09-17 — V3 (collimated aperture) put on hold (S18.1)

> Ok so shelve V3 as well.

V3 is marked `ON_HOLD` in the check registry, like V4, and the plugin skips it. The reason is the
velocity law, not the solver: for a surface `z = h(x)` the vertical descent speed is
`h_t = -v0 cos^(p-1)(theta)`, so only p = 1 descends independently of tilt. For p > 1 a tilted patch
descends more slowly and the tilt feeds on itself; the mask corner supplies the first tilt.

Measured over 100 nm of travel — angle error 0.195 deg (p = 1), 0.110 deg (1.05), 1.177 deg (1.25),
4.717 deg (2), 3.051 deg (64), with CD exactly 100.00 nm at every p, so the trench never widens. V3's
0.5 deg is met only near p = 1, which PRD §11 forbids because `max(0, c)^p` has a kink at c = 0 there.

A real collimated etch holds a flat floor because the ion flux is uniform and non-local, which is M3.
Revisit when M3 supplies flux. The supporting measurements stay as tests in `tests/test_aperture.py`
so the decision can be revisited against numbers rather than memory. PRD §8.1 amended.

## 2026-09-17 — SiGe marker case stays at dx = 2 nm (S17.1)

> Just keep 17.1 at dx = 2 nm.

The `w_mat` study's reference case runs at dx = 2 nm with the default `w_mat` of 2 cells, so the
mollification is 4 nm, about 13 % of the 30 nm marker. No move to a finer grid.

**`w_mat` is never fitted** — fitting a numerical smoothing width is the knob-tuning PRD §11 forbids.
The standing obligation is that any physical parameter later fitted from marker-crossing data records
the `w_mat` it was fitted under, because `d(depth)/d(film rate)` moves about 5 % between 1-cell and
2-cell smoothing and 38 % from 1 to 8 cells. The sensitivity itself stays open as S17.1.

No code change: PRD §5.6 already specifies dx = 2 nm. Reproduced by `tests/test_w_mat_study.py`.

## 2026-09-17 — Mask thickness 200 nm, and the grid derived from it (S17.4)

> For S17.4, there is a mask, the film (Si wafer where the trench is created), and the SiGe substrate
> below it? Grid height is 2700 nm with 2500 unmasked etch, so shouldn't mask height be 200 nm to fil
> the grid? ... Choose a reasonable mask thickness and increase the total grid height accordingly

Mask thickness **200 nm**, recorded `provisional` (S17.5): a conventional hard-mask-to-etch ratio for a
2.5 um silicon trench is order 10 %, and 200 nm is 20 cells at dx = 10 nm, so the mask is resolved rather
than a two-cell sliver.

The grid follows: 100 (bottom buffer) + 2500 (etch) + 200 (mask) + 100 (top buffer) = 2900 nm =
**290 cells**. Sizing the mask to fill the old 2700 nm grid was not taken: that grid is exactly full
before any mask, and choosing physics to fit a grid inverts PRD §5.1's dependency.

**Stack, corrected:** S03 is mask on silicon film; there is no SiGe in it. The SiGe marker is a thin
(~30 nm) layer buried INSIDE the film in the phase-2 case, not a substrate.

Changes: `mask_thickness_nm` config field, required whenever a material is `is_mask`; the vertical-extent
check needs depth + mask + both buffers; `configs/dev/S03.yaml` 270 -> 290 cells; `configs/dev/S00.yaml`
100 nm mask. `geocore/verification/gate.py` builds the masked geometry, so finding **L2 is withdrawn**
for the M2.4 gate case.

## 2026-09-17 — Bow definition (S16.1)

> The definition for bow for 16.1 is fine.

`bow = CD(z_mid) − (CD(z_low) + CD(z_high)) / 2` over a fixed absolute depth window: zero for a straight
taper, positive when the trench is wider in the middle. The common "maximum CD in the window" definition
is not used because its `max` is a kink in the differentiated path (§11). PRD §5.7 amended.

## 2026-09-16 — Add a check bounding accumulated reinitialisation drift (V12a)

> Add the check to bound drift over a run and change the PRD in the m2-new folder

V12 bounds ONE reinitialisation cycle; a run applies tens to hundreds. **V12a** was added: two runs of
V1's geometry differing only in whether they reinitialise, with the rate applied everywhere so the
unreinitialised run does not stall at the band edge. The difference in final radius is what
reinitialisation is responsible for. Its tolerance is still provisional (S12.3).

PRD amended: §6 M2.2 gate, §8 check count, §8.0 fast tier, §8.1 V1, §8.3 V9 and the new V12a paragraph,
§8.5 V14a–c.

## 2026-09-16 — V14a–V14c tolerance is 1 % (S13.1)

> 1c, 2 accept

The PRD gives these analytic sensitivities no tolerance, and the registry's earlier 1e-6 was invented at
stage 5 with no basis. **1 % relative error** for all three: below the 5 % corruption V19 injects, and no
tighter than the forward tolerance of the configuration it runs in (V1: 1 %). Measured under it: V14b
3.5e-5, V14c 1.6e-3 and 2.1e-3, V14a 1.9e-2 — so V14a fails, which is how the rule was shown not to have
been chosen to make everything pass.

## 2026-09-16 — Cumulative reinitialisation drift accepted as a known discrepancy (S12.1, option c)

> 1c, 2 accept

Reinitialisation drift is recorded as a known discrepancy rather than fixed now. V1 (full travel), V9,
V12a and V14a stay `xfail(strict=True)`; tolerances unchanged. They are to be re-measured after WENO5
and the decision revisited — which has not happened yet, because the re-measurement is incomplete
(S14.2). The alternative options were: investigate against the original tree first, or build a
subcell-fix reinitialisation.

## 2026-09-16 — Work continued past the M2.2 gate

The owner directed stage 13 (M2.3) to start while S12.1 was unresolved, and accepted S12.1 the same day.
The M2.2 gate is therefore accepted with that exception.

---

## Structural decisions taken in the build, not by the owner

These are Class A changes announced when made. They are here because they shape the code, not because
anyone had to choose between options.

| date | decision |
|---|---|
| 2026-09-17 | One `OPEN_QUESTIONS.md` and one `DECISIONS.md`; no `decisions/`, `reports/` or `scripts/` directories. Diagnostics that used to write report files are gate-tier tests (`tests/test_gate_m2_4.py`, `tests/test_w_mat_study.py`) whose numbers land in this file or in OPEN_QUESTIONS. The ledger writes to `verification_ledger.json` (of record) and `.verification_ledger_local.json` (untracked) at the project root |
| 2026-09-17 | The M2.4 gate sizes the request capacity from a PROBE run's worst step, and `forward_plan` asserts capacity as well as CFL. The first H100 run sized K at 3x the first step's occupancy (96,000) while the interface grew to 131,400 as the trench deepened, and nothing aborted: the run silently truncated part of the interface, which PRD §5.0 forbids |
| 2026-09-17 | The ledger MERGES rows across runs (finding S13.4, now closed). A fast-tier run knew nothing about V8 and used to delete its row, so a record generated from that file was partial while looking complete. Rows from a run replace their namesakes; the rest carry forward with their own provenance. A check claimed by several tests keeps the worst result and the union of their measurements |
| 2026-09-16 | WENO5 reinitialisation steps with SSP-RK3, not forward Euler: WENO5's eigenvalues sit near the imaginary axis, outside Euler's stability region, and with Euler a 150-cell disk opened 1268 cells of phantom holes and Zalesak's disk tripped the CFL assertion. Godunov keeps Euler, so every recorded Godunov number stands |
| 2026-09-16 | A test may declare itself into a SLOWER tier than its check's registry cadence (never a faster one). PRD §8.0 puts small 3D in nightly; without this the fast tier ran 245 s against a 180 s budget |
| 2026-09-16 | `functionals.py` (the smooth volumetric functional) moved from stage 16 to stage 13, because the M2.3 gate needs V14 on it |
| 2026-09-15 | Checkpoint segment length L is derived from measured k, never from sqrt(N) and never written by hand (finding I4). At k = 344 and N = 625, L = 1 |
| 2026-09-14 | Material fractions are read at the closest SURFACE point, not at the band cell. The cell lookup was the step function PRD §5.6 warns about |
