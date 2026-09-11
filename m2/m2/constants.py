"""Conventions and fixed numerical constants.

Nothing here is configurable. PRD §11 forbids tuning CFL (or tolerances) to make a test pass,
so these values live in code, where changing one is a reviewed commit, not a config edit.

Decisions of 2026-09-11 (`decisions/2026-09-11-open-questions-response-r3.md`) are authoritative
over the PRD; the section they come from is cited at each block.
"""

# --- Level-set representation (PRD §5.1) -------------------------------------------------
# φ is a signed distance: NEGATIVE inside solid, POSITIVE in the open volume.
PHI_SOLID_SIGN = -1
PHI_OPEN_SIGN = +1
# Consequence: n = ∇φ/|∇φ| points from solid into the open volume.
NORMAL_POINTS_FROM = "solid"
NORMAL_POINTS_TO = "open"

# --- Etch-rate sign and orientation (decision §1, resolving OQ A1) -----------------------
# Velocity models return an ETCH RATE R in nm/s: positive removes material, negative deposits
# (legal, unused in M2). M2 advects  φ_t − R|∇φ| = 0,  i.e.
#     dφ/dt = DPHI_DT_RATE_SIGN * R * |∇φ|
# so the sign flip lives in one place in M2 rather than in every M3/M5 velocity model.
DPHI_DT_RATE_SIGN = +1
# Axis 0 is vertical and INCREASES TOWARD THE PLASMA, so ẑ = +e₀.
Z_AXIS = 0
Z_HAT_SIGN = +1
AXIS0_INCREASES_TOWARD = "plasma"
# Therefore R = v₀·max(0, n·ẑ)^p etches up-facing surfaces, a trench floor recedes away from the
# plasma (toward decreasing axis 0), and a downward-facing overhang does not move. Checked by V1a.

# --- Precision (PRD §7.5) ---------------------------------------------------------------
DTYPE = "float64"

# --- Time integration (PRD §5.2; decision §2) --------------------------------------------
CFL_MAX = 0.5  # hard assertion bound, never configurable
CFL_TARGET_DEFAULT = 0.4  # config default; N = ceil(D / (CFL_target · dx))

# --- Reinitialisation (decision, C2) -----------------------------------------------------
REINIT_DTAU_CELLS = 0.5  # dτ = 0.5·dx

# --- Mollified Heaviside for the volumetric functional (decision, A2) --------------------
# `solid_volume = ∫(1 − H(φ)) dV`. Width fixed in code, not config.
HEAVISIDE_WIDTH_CELLS = 1.5

# --- Directional velocity model (decision, "four things the file missed") ----------------
# R = v₀·max(0, n·ẑ)^p has a kink at n·ẑ = 0 — i.e. on vertical sidewalls — unless p > 1.
DIRECTIONAL_MIN_P = 1.0  # exclusive lower bound
DIRECTIONAL_MIN_P_FOR_GRADIENT_TESTS = 2.0

# --- Taylor-remainder protocol, V14 (PRD §7.1; decision §4) ------------------------------
TAYLOR_MIN_DIRECTIONS = 20
TAYLOR_MIN_DECADES = 5
TAYLOR_SLOPE_BAND = (1.8, 2.2)
# Decision B15/B21 replaced the three-zone rule with two zones: a wrong gradient leaves the
# first-order term uncancelled and tends to slope 1, and no incorrect gradient produces a slope
# above 2, so there is no upper bound to enforce. Above the band is degenerate, and passes.
TAYLOR_DEGENERATE_SLOPE = 2.2  # > band upper bound: pass, logged as degenerate_direction
TAYLOR_MIN_POINTS_ABOVE_FLOOR = 3  # fewer than this above the floor is "insufficient signal"
TAYLOR_H_MAX = 1e-1  # provisional: OPEN_QUESTIONS B3
TAYLOR_N_H = 11  # provisional: B3 (2 points per decade over 5 decades)
# Noise floor: estimated from repeated evaluations of J at fixed θ, with a roundoff-scaled
# fallback for a perfectly deterministic J. Points below the floor are excluded (decision §4).
# floor = max(measured spread, C·√N·eps·max(|J|, 1)): roundoff accumulates over N solver steps as
# a random walk, and the √N keeps the floor sensible when N changes between grid refinements.
TAYLOR_NOISE_FLOOR_REPEATS = 8
TAYLOR_NOISE_FLOOR_C = 10.0  # decision B19; revisit against the real solver at M2.3

# --- Forward vs reverse, V15; dot-product test, V16 (PRD §8.5; decision B4/B5) ------------
# Both are TRANSPOSE-consistency checks, not derivative checks: JAX builds reverse mode by
# linearising with JVP rules and transposing, so the two share those rules. See OQ A9.
V15_RTOL = 1e-10
V15_MIN_DIRECTIONS = 20
V16_RTOL = 1e-10  # provisional: B5 (PRD gives no V16 tolerance)
V16_MIN_PAIRS = 20
# Relative error is normalised by ‖∇J‖·‖δ‖, never by the directional derivative, which can be
# tiny for a δ nearly orthogonal to the gradient (decision B4/B5).

# --- Corrupted-gradient canary, V19 (PRD §7.2) -------------------------------------------
V19_CORRUPTION_FACTOR = 1.05
# Precondition on the SCALED sensitivity max(|θ_k|, scale_k)·|∂J/∂θ_k| (decisions B13, B20).
V19_MIN_GRAD_COMPONENT = 0.1

# --- Material fractions (PRD §5.6) -------------------------------------------------------
MATERIAL_SUM_ATOL = 1e-12  # provisional: B8
