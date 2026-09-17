"""Verification machinery: what the checks are, what they measured, and how to score a gradient.

Built BEFORE the physics, against functions whose answers are known in closed form. PRD §6 puts
V19 — the corrupted-gradient canary — in the M2.0 gate, before a solver exists. The reason is
that a harness first built against the solver gives an ambiguous pass: it could mean the gradient
is right, or it could mean the harness is blind, and nothing distinguishes the two.
"""
