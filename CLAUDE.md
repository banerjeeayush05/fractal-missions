# CLAUDE.md — fractal-missions

**How to report and when to ask: `WORKING_AGREEMENT.md`.** Problems get explained twice (precisely,
then in plain English) with compared alternatives; requirement-level changes may be made but must be
announced and flagged as the less serious class; anything that changes or compromises the project
stops and is flagged as serious so the owner can probe it.

Rules every mission in this repo shares. Mission-specific scope, non-goals and numerics live in
that mission's own file: `m2/CLAUDE.md`, `m3/CLAUDE.md`. Read the mission file before working in
its directory; M2's non-goals are M3's goals, so never apply one mission's scope to another.

- **One autodiff system: JAX.** Never add PyTorch, Warp, PETSc, Mitsuba, OptiX or Embree bindings.
  A gradient bug must have one place to be.
- **GPL clean room.** ViennaPS and ViennaRay are GPL-3.0. Never link, import, vendor or copy them.
  Cross-code comparison runs in a separate process and repo, exchanging VTK files. If you read
  their source, write a natural-language note; do not transcribe.
- **Verification records are generated, never hand-written.** Checks write ledger rows;
  `verification.md` is generated from the ledger. CI on a clean checkout is the only writer of the
  ledger of record; local runs write an untracked ledger.
- **Never loosen a tolerance to make a check pass.** A failure is a finding. If a tolerance is
  genuinely wrong, change it in its own commit, with the reason.
- **Never resolve an ambiguity by guessing.** Write it into the mission's `OPEN_QUESTIONS.md`,
  classified A (PRD error), B (placeholder, with `provisional: true` and the mechanical rule that
  produced it), C (gap blocking a later milestone), D (minor). Assume PRDs contain arithmetic and
  cross-reference errors, and check them.
- **Stop at every milestone gate.** Run the acceptance tests, report, and wait for review. Never
  build ahead.
- **Answers supersede the PRD.** When a decision lands, update the PRD text in the same commit that
  implements it, and say in the commit message which decision it implements. Never leave the PRD
  and the code disagreeing. Decision records live in `<mission>/decisions/`.
- **Cross-mission impacts: `CROSS_MISSION.md`.** A finding in one mission that changes another
  mission's assumptions goes there, with a status. Update that status whenever an answer touches an
  entry. A mission's first action is to read it.
