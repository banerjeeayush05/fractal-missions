# CLAUDE.md — M3, feature-scale transport and flux

**Not yet written.** M3 has not started. Per `M3_PRD.md` §0, writing this file is the M3 agent's
first action, summarising §3 (non-goals), §4 (the sphere-tracing decision), §7 (numerical
requirements) and §11 (anti-requirements). This placeholder exists only so that shared rules are
not mistaken for M2's.

Before writing it, read, in order:
1. `../CLAUDE.md` — rules every mission shares.
2. `../CROSS_MISSION.md` — entries X1–X16 are M2 findings and decisions that change what this
   PRD assumes (the velocity contract is **not** frozen yet; the RNG point index; the step count;
   what `jax.checkpoint` on a velocity call actually bounds; V32's blind spot). Copy every open
   entry into this mission's `OPEN_QUESTIONS.md`.
3. `M3_PRD.md`.

M2's non-goals are M3's goals. Do not apply `../m2/CLAUDE.md` here.
