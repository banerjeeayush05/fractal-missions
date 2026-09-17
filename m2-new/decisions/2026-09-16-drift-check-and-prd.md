# Owner decisions — 2026-09-16 (second)

## Verbatim

> Add the check to bound drift over a run and change the PRD in the m2-new folder

## Interpretation

**Add the check.** Answers the side question left open in S12.1. A new check, **V12a**, bounds the
interface displacement attributable to reinitialisation over a full run. Its ID follows the lettered
sub-check pattern (V1a, V14a) because V23 and above belong to M3 (PRD §8.5).

The owner did not set V12a's tolerance. The rule used is recorded as finding **S12.3**,
`provisional: true`: the mean shift must stay below V1's forward tolerance, 1 %. Measured 2.15 % at
dx = 10 nm, so V12a is recorded `xfail(strict=True)` under the S12.1 known discrepancy.

**Change the PRD in m2-new.** `M2_PRD.md` in this folder (identical to the committed original when
amended) now carries the 2026-09-16 decisions:

| PRD passage | amendment |
|---|---|
| header | this copy belongs to `m2-new`; lists the amending decision files |
| §6, M2.2 gate | adds V12a; records acceptance with the S12.1 known discrepancy |
| §8, check count | notes V12a |
| §8.0, fast tier | adds V12a |
| §8.1, V1 | radius is the mean over directions; −3.17 % accepted as known discrepancy |
| §8.3, V9 | 7.76 % accepted as known discrepancy (S12.2) |
| §8.3 | new V12a paragraph |
| §8.5, V14a–c | tolerance < 1 % (S13.1, owner accepted) |

Not amended, because no owner decision covers them: S13.2 (which refinement pair defines observed
order), S13.4 (local ledger overwrite), S3.1 (contract version), S4.1 (role vocabulary), S1.1 (JAX
dependency marker).
