# Working agreement

How the agent communicates with the mission owner. Set by the owner on 2026-09-11, and it applies
to every mission in this repo. Read it with `CLAUDE.md`; when in doubt, this file governs how to
*report*, and `CLAUDE.md` governs what to *do*.

## 1. When you hit a problem, explain it twice

First the precise version, in the language of the spec. Then the **same thing in plain English**,
assuming the reader has not been staring at the code.

Then give **alternatives, compared against each other**: what each one costs, what it buys, what it
risks, and which one you recommend and why. Not a menu with no recommendation, and not a single
option presented as the only possibility.

A useful shape:

> **What broke.** V14 fails on CD but passes on volume.
> **In plain terms.** The gradient is right for the smooth measure of how much material was removed,
> but wrong for the width of the trench, so the bug is in how the width is measured, not in the physics.
> **Options.** (a) Fix the extraction to be sub-cell everywhere — a day, removes the cause, no risk
> to anything else. (b) Smooth the extraction — an hour, hides the symptom, biases every width we
> report. (c) Report the finding and stop. **Recommend (a)**, because (b) trades a correct number
> for a plausible one, which is the failure this mission exists to prevent.

## 2. Two classes of change, handled differently

**Class A — adjusts a requirement. Make it, but say so first.**
Examples: a tolerance's normalisation, a naming convention, a label for a test classification,
where a value lives (config vs constant), a config field the decisions imply but do not name.

- Tell the owner what you are changing and why **before** or in the same message as the change.
- **Flag it explicitly as the less serious class** — the owner should not have to work out which
  kind it is.
- Make it persist: record it in `OPEN_QUESTIONS.md` (with `provisional: true` and the mechanical
  rule behind it), amend the PRD text in the same commit, and update `CROSS_MISSION.md` if another
  mission inherits it. A change nobody can find later is a change that will be made again, differently.

**Class B — changes or compromises the project. Stop and say so plainly.**
Examples: anything that alters what a gate proves; a sign or convention change; a new dependency or
a second autodiff path; loosening a check; a scope move between milestones; a decision that another
mission inherits; anything that would make a green suite mean less than it did.

- Do not proceed on your own judgement. Say clearly that **this is the serious class**, so the owner
  knows to probe it.
- Give the owner what they need to probe: what it changes, what it would cost to avoid, and what the
  consequence is if the judgement turns out wrong.

If you cannot tell which class a change is in, treat it as Class B.

## 3. Standing rules this sits on top of

- Never resolve an ambiguity by guessing (`CLAUDE.md`).
- Stop at every milestone gate, report, and wait.
- Report outcomes faithfully: a failing test is reported as failing, with the output.
- Numbers that change between reports get called out, not quietly replaced.
