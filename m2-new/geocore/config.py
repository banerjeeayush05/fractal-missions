"""Per-case configuration, and the rules that make a config refuse to load.

`constants.py` holds what is fixed for the whole product. This module holds what legitimately
differs between cases — grid, target depth, band widths, `n_reinit`, `w_mat`, the CFL target —
and enforces the constraints that keep those from becoming knobs.

Four rules do most of the work here. Each exists because of a specific way a config file can be
quietly wrong:

**1. A config names a target DEPTH, never a final time.** `T = depth / rate` is derived, so a
later rate correction rescales time and changes nothing else: N, the CFL number and the V21
golden profile all still hold. Write `final_time_s` into a file instead and a rate correction
silently changes what the run simulates.

**2. N is derived, never written by hand.** `N = ceil(depth / (cfl_target * dx))` — the front
advances at most `cfl_target * dx` per step, so that is the fewest steps that can cover the
depth. An explicit `n_steps` is allowed as an override but is REJECTED AT LOAD TIME if it is
below the derived minimum, because a too-small N does not fail visibly; it fails as a CFL
violation partway through a run you have already paid for.

**3. A case may not certify a coupon claim on a rate nobody measured.** Configs carry the rate's
provenance, not just its value. `configs/dev/*` carry a nominal rate marked provisional; they may
run anything, but `require_measured_rate` refuses to let them back a claim of agreement with the
coupon.

**4. A fit may only consume `calibrate` cases.** Enforced by `require_calibration_case`. Without
it, fitting against a validation case is a one-line mistake that produces excellent numbers and
no held-out evidence at all.

---

**S4.1 — the role vocabulary is not specified anywhere.** Class B, `provisional: true`.
`m2/CLAUDE.md` line 28 names exactly one role, `calibrate`, and states the rule that a fit may
only consume those. It does not enumerate the others, and the PRD does not mention roles at all.
*Mechanical rule applied:* declare the smallest vocabulary that makes the stated rule meaningful
— `calibrate` (a fit may consume), `validate` (held out; a fit may not), `dev` (nominal rate,
provisional, never certifies a coupon claim). `dev` is forced by the `configs/dev/*` rule on line
89. Adding a role later is cheap; the rule being unenforceable is not. Needs an owner ruling.

**S4.2 — the directory rule and the role field are two spellings of one constraint.** Class A.
`m2/CLAUDE.md` line 89 phrases the provenance rule by directory (`configs/cases/*` versus
`configs/dev/*`); this module keys the rule off the `role` FIELD, which travels with the file.
`load_case` additionally checks that the two agree, so a `dev` config copied into `configs/cases/`
is rejected rather than silently promoted. Behaviour is a superset of the literal rule.

**S4.3 — `omegaconf` is a PRD §10 dependency and is unused.** Class D.
Loading uses `yaml.safe_load` plus explicit construction, so that an unknown key is an error
rather than a silently ignored typo — which is the failure mode that matters for a file full of
numbers people are not supposed to tune. OmegaConf's interpolation and merging would make a
config's effective value depend on resolution order. Left unused rather than removed; revisit if
a real need appears.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import pathlib
from typing import Any, Final, Mapping

import yaml

from geocore.constants import CFL_MAX
from geocore.schema import Grid, Material

SPATIAL_SCHEMES: Final[frozenset[str]] = frozenset({"godunov", "weno5"})
TEMPORAL_SCHEMES: Final[frozenset[str]] = frozenset({"rk2", "rk3"})
VELOCITY_MODELS: Final[frozenset[str]] = frozenset({"isotropic", "directional"})


class Role(enum.Enum):
    """What a case is allowed to be used for. See S4.1 — provisional vocabulary."""

    CALIBRATE = "calibrate"
    VALIDATE = "validate"
    DEV = "dev"


class ConfigError(ValueError):
    """Raised at load time. A config that violates a rule does not load in a degraded state."""


# ------------------------------------------------------------------------------ components


@dataclasses.dataclass(frozen=True)
class EtchRate:
    """A rate in nm/s, carried WITH its provenance.

    The value alone is not enough: the same 4.0 nm/s means something different if it came from
    coupon metrology than if someone picked it to make a dev case run in a reasonable time.
    `measured=False` is not a lesser number, it is a number that may not back a claim.
    """

    nm_per_s: float
    measured: bool
    source: str

    def __post_init__(self) -> None:
        if not (self.nm_per_s > 0.0 and math.isfinite(self.nm_per_s)):
            raise ConfigError(f"rate must be finite and positive, got {self.nm_per_s}")
        if not self.source:
            raise ConfigError("rate.source must say where the number came from")

    @property
    def provisional(self) -> bool:
        return not self.measured


@dataclasses.dataclass(frozen=True)
class ReinitConfig:
    """PRD §5.3. Fixed counts, never a convergence criterion: a data-dependent loop count makes
    the graph shape depend on the parameters, and the gradient through that is wrong."""

    n_reinit: int = 5
    every: int = 5

    def __post_init__(self) -> None:
        if self.n_reinit < 1:
            raise ConfigError(f"n_reinit must be at least 1, got {self.n_reinit}")
        if self.every < 1:
            raise ConfigError(f"reinit.every must be at least 1, got {self.every}")


@dataclasses.dataclass(frozen=True)
class BandConfig:
    """PRD §5.4, decision C10. TWO bands, with different jobs.

    `extension_cells` is where a valid velocity must EXIST for the stencils: the interface moves
    up to `cfl * reinit.every` cells between reinitialisations, reinitialisation propagates about
    `n_reinit` cells, and the upwind stencil reaches one more.

    `evaluation_cells` is where the velocity model is CALLED, and it stays thin because cells
    deeper in the band project to nearly the same surface point — calling M3 for them buys the
    same expensive Monte Carlo estimate several times over.
    """

    extension_cells: float = 8.0
    taper_cells: float = 2.0
    evaluation_cells: float = 1.5

    def __post_init__(self) -> None:
        if self.evaluation_cells <= 0.0:
            raise ConfigError("evaluation_cells must be positive")
        if self.taper_cells <= 0.0 or self.taper_cells >= self.extension_cells:
            raise ConfigError(
                f"taper_cells ({self.taper_cells}) must be positive and inside "
                f"extension_cells ({self.extension_cells})"
            )
        if self.evaluation_cells >= self.extension_cells:
            raise ConfigError(
                f"the evaluation band ({self.evaluation_cells} cells) must sit inside the "
                f"extension band ({self.extension_cells} cells): the velocity is evaluated on "
                f"the thin set and gathered outward across the wide one"
            )


@dataclasses.dataclass(frozen=True)
class VelocitySpec:
    """Which analytic model, and its INITIAL parameter values.

    These are starting values only. At run time every differentiable parameter lives in the
    params PyTree (PRD §7.3); a value captured from a config object in a closure is invisible to
    `jax.grad`, which returns a silently zero column for it rather than an error.
    """

    model: str
    params: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model not in VELOCITY_MODELS:
            raise ConfigError(f"unknown velocity model {self.model!r}, expected {VELOCITY_MODELS}")
        # PRD §11: the directional law max(0, n.z)^p has its kink exactly on vertical sidewalls,
        # the most common surface in a trench. p > 1 moves the kink out of the differentiated path.
        if self.model == "directional":
            p = self.params.get("p")
            if p is None:
                raise ConfigError("the directional model needs an exponent p")
            if p <= 1.0:
                raise ConfigError(
                    f"directional p must be > 1 (PRD §11: p = 1 puts a kink on every vertical "
                    f"sidewall), got {p}"
                )
        object.__setattr__(self, "params", dict(self.params))


# ----------------------------------------------------------------------------------- case


@dataclasses.dataclass(frozen=True)
class CaseConfig:
    """One runnable case. Everything derived is a property, so it cannot drift from its inputs."""

    name: str
    role: Role
    grid: Grid
    target_depth_nm: float
    rate: EtchRate
    velocity: VelocitySpec
    materials: tuple[Material, ...]
    cfl_target: float = 0.4
    n_steps_override: int | None = None
    reinit: ReinitConfig = dataclasses.field(default_factory=ReinitConfig)
    bands: BandConfig = dataclasses.field(default_factory=BandConfig)
    w_mat_cells: float = 2.0
    # Stage 17. Thickness of the mask body sitting on the film top, nm. Required when a material is
    # marked `is_mask`, because the grid must hold the mask as well as the etch (PRD §5.1) and the
    # vertical extent is DERIVED from it -- sizing the mask to fit a grid already chosen would be
    # choosing physics to suit a number.
    mask_thickness_nm: float | None = None
    spatial_scheme: str = "godunov"
    temporal_scheme: str = "rk2"

    def __post_init__(self) -> None:
        if not (0.0 < self.cfl_target <= CFL_MAX):
            raise ConfigError(
                f"cfl_target must be in (0, {CFL_MAX}], got {self.cfl_target}: {CFL_MAX} is the "
                f"hard ceiling asserted every step, so a target at or above it cannot hold"
            )
        if not (self.target_depth_nm > 0.0 and math.isfinite(self.target_depth_nm)):
            raise ConfigError(f"target_depth_nm must be finite and positive, got "
                              f"{self.target_depth_nm}")
        if self.spatial_scheme not in SPATIAL_SCHEMES:
            raise ConfigError(f"spatial_scheme must be one of {sorted(SPATIAL_SCHEMES)}")
        if self.temporal_scheme not in TEMPORAL_SCHEMES:
            raise ConfigError(f"temporal_scheme must be one of {sorted(TEMPORAL_SCHEMES)}")
        if self.w_mat_cells <= 0.0:
            raise ConfigError(f"w_mat_cells must be positive, got {self.w_mat_cells}")

        # Rule 3. A case that is not `dev` is a case whose numbers are claimed to mean something.
        if self.role is not Role.DEV and self.rate.provisional:
            raise ConfigError(
                f"case {self.name!r} has role {self.role.value!r} but its rate is provisional "
                f"(source: {self.rate.source!r}). configs/cases/* refuse to load until the rate "
                f"is measured; use role 'dev' for a nominal rate"
            )

        self._check_materials()
        self._check_depth_fits()

        # Rule 2, checked LAST so its message can quote the derived minimum.
        if self.n_steps_override is not None:
            if self.n_steps_override < self.n_steps_min:
                raise ConfigError(
                    f"n_steps={self.n_steps_override} is below the derived minimum "
                    f"{self.n_steps_min} = ceil({self.target_depth_nm} / ({self.cfl_target} * "
                    f"{self.grid.spacing_nm})). A too-small N does not fail at load; it fails as "
                    f"a CFL violation partway through the run"
                )

    def _check_materials(self) -> None:
        if not self.materials:
            raise ConfigError("a case needs at least one material")
        indices = [m.index for m in self.materials]
        if sorted(indices) != list(range(len(indices))):
            raise ConfigError(
                f"material indices must be 0..n-1 with no gaps or repeats, got {sorted(indices)}: "
                f"they are columns in the fraction field"
            )
        if len({m.name for m in self.materials}) != len(self.materials):
            raise ConfigError("material names must be unique")
        if sum(m.is_mask for m in self.materials) > 1:
            raise ConfigError("at most one material may be the mask")

    def _check_depth_fits(self) -> None:
        """PRD §5.1: the domain covers the stack plus the etch depth plus a buffer. Catching this
        here is worth a lot — the alternative is discovering it as an interface that reaches the
        Neumann boundary and stalls, which looks like a physics result.

        With a mask the stack is `mask + etch depth`, and the buffer is needed at BOTH ends: 10 cells
        of clearance under the final floor, and 10 above the mask top.
        """
        if self.mask is not None and self.mask_thickness_nm is None:
            raise ConfigError(
                f"case {self.name!r} declares material {self.mask.name!r} as the mask but gives no "
                f"mask_thickness_nm. The grid's vertical extent is derived from it (PRD §5.1)."
            )
        if self.mask_thickness_nm is not None and self.mask_thickness_nm <= 0.0:
            raise ConfigError(f"mask_thickness_nm must be positive, got {self.mask_thickness_nm}")
        from geocore.constants import VERTICAL_BUFFER_CELLS, VERTICAL_AXIS

        vertical_nm = self.grid.shape[VERTICAL_AXIS] * self.grid.spacing_nm
        buffer_nm = VERTICAL_BUFFER_CELLS * self.grid.spacing_nm
        mask_nm = self.mask_thickness_nm or 0.0
        needed_nm = self.target_depth_nm + mask_nm + (2.0 if mask_nm else 1.0) * buffer_nm
        if vertical_nm < needed_nm:
            raise ConfigError(
                f"vertical extent {vertical_nm:g} nm cannot hold a {self.target_depth_nm:g} nm etch"
                + (f" under a {mask_nm:g} nm mask" if mask_nm else "")
                + f" plus {VERTICAL_BUFFER_CELLS}-cell buffer(s) ({needed_nm:g} nm needed). "
                  f"The grid follows the stack, not the other way round."
            )

    # --------------------------------------------------------------------------- derived

    @property
    def n_steps_min(self) -> int:
        """PRD §5.2, decision §2. The front advances at most `cfl_target * dx` per step."""
        return math.ceil(self.target_depth_nm / (self.cfl_target * self.grid.spacing_nm))

    @property
    def n_steps(self) -> int:
        """Fixed N. PRD §5.2: `dt = T/N` with N static, never `dt = CFL*dx/max|V|` — max|V|
        depends on the parameters, so N would change DISCONTINUOUSLY with them, and a gradient
        through a discontinuous step count is wrong in a way that looks like noise."""
        return self.n_steps_override if self.n_steps_override is not None else self.n_steps_min

    @property
    def final_time_s(self) -> float:
        """Rule 1. Derived, never written into the file."""
        return self.target_depth_nm / self.rate.nm_per_s

    @property
    def dt_s(self) -> float:
        return self.final_time_s / self.n_steps

    @property
    def mask(self) -> Material | None:
        return next((m for m in self.materials if m.is_mask), None)

    @property
    def n_materials(self) -> int:
        return len(self.materials)

    def summary(self) -> str:
        return (
            f"{self.name} [{self.role.value}] {self.grid.ndim}D {self.grid.shape} @ "
            f"{self.grid.spacing_nm:g} nm | depth {self.target_depth_nm:g} nm | "
            f"rate {self.rate.nm_per_s:g} nm/s "
            f"({'measured' if self.rate.measured else 'PROVISIONAL'}) | "
            + (f"mask {self.mask_thickness_nm:g} nm | " if self.mask_thickness_nm else "")
            + f"T {self.final_time_s:g} s | N {self.n_steps} "
            f"(min {self.n_steps_min}) | dt {self.dt_s:g} s | {self.spatial_scheme}"
        )


# ---------------------------------------------------------------------------------- guards


def require_calibration_case(case: CaseConfig) -> CaseConfig:
    """`m2/CLAUDE.md`: a fit may only consume `calibrate` cases.

    Called by anything that fits parameters to data. Fitting against a validation case is a
    one-character mistake that yields excellent numbers and destroys the held-out evidence —
    and nothing downstream can detect that it happened.
    """
    if case.role is not Role.CALIBRATE:
        raise ConfigError(
            f"case {case.name!r} has role {case.role.value!r}; a fit may only consume "
            f"'calibrate' cases"
        )
    return case


def require_measured_rate(case: CaseConfig, claim: str) -> CaseConfig:
    """`m2/CLAUDE.md`: a provisional rate may never certify a claim of agreement with the coupon.

    `claim` names what is being certified, so the refusal says what was about to be asserted on
    a number nobody measured.
    """
    if case.rate.provisional:
        raise ConfigError(
            f"case {case.name!r} carries a provisional rate ({case.rate.source!r}) and may not "
            f"certify {claim!r}"
        )
    return case


# ---------------------------------------------------------------------------------- loading


def _require_keys(mapping: Mapping[str, Any], allowed: set[str], required: set[str],
                  where: str) -> None:
    """Unknown keys are an ERROR, not ignored.

    This is the single most valuable line in the loader. In a file of numbers people are told not
    to tune, a typo like `cfl_taget: 0.9` that silently falls back to the default is undetectable:
    the run succeeds, the number is wrong, and the file on disk says otherwise.
    """
    keys = set(mapping)
    unknown = keys - allowed
    if unknown:
        raise ConfigError(f"unknown key(s) in {where}: {sorted(unknown)}; allowed: {sorted(allowed)}")
    missing = required - keys
    if missing:
        raise ConfigError(f"missing key(s) in {where}: {sorted(missing)}")


def case_from_dict(raw: Mapping[str, Any], name_hint: str = "<dict>") -> CaseConfig:
    """Build a CaseConfig from plain data, validating strictly."""
    _require_keys(
        raw,
        allowed={"name", "role", "grid", "target_depth_nm", "rate", "velocity", "materials",
                 "cfl_target", "n_steps", "reinit", "bands", "w_mat_cells", "spatial_scheme",
                 "temporal_scheme", "mask_thickness_nm"},
        required={"name", "role", "grid", "target_depth_nm", "rate", "velocity", "materials"},
        where=name_hint,
    )

    grid_raw = raw["grid"]
    _require_keys(grid_raw, {"shape", "spacing_nm", "periodic"},
                  {"shape", "spacing_nm", "periodic"}, f"{name_hint}.grid")
    grid = Grid(
        shape=tuple(int(n) for n in grid_raw["shape"]),
        spacing_nm=float(grid_raw["spacing_nm"]),
        periodic=tuple(bool(flag) for flag in grid_raw["periodic"]),
    )

    rate_raw = raw["rate"]
    _require_keys(rate_raw, {"nm_per_s", "measured", "source"},
                  {"nm_per_s", "measured", "source"}, f"{name_hint}.rate")
    rate = EtchRate(float(rate_raw["nm_per_s"]), bool(rate_raw["measured"]),
                    str(rate_raw["source"]))

    vel_raw = raw["velocity"]
    _require_keys(vel_raw, {"model", "params"}, {"model"}, f"{name_hint}.velocity")
    velocity = VelocitySpec(str(vel_raw["model"]),
                            {k: float(v) for k, v in (vel_raw.get("params") or {}).items()})

    materials = []
    for i, m in enumerate(raw["materials"]):
        _require_keys(m, {"name", "index", "is_mask"}, {"name", "index"},
                      f"{name_hint}.materials[{i}]")
        materials.append(Material(str(m["name"]), int(m["index"]), bool(m.get("is_mask", False))))

    reinit_raw = raw.get("reinit") or {}
    _require_keys(reinit_raw, {"n_reinit", "every"}, set(), f"{name_hint}.reinit")
    bands_raw = raw.get("bands") or {}
    _require_keys(bands_raw, {"extension_cells", "taper_cells", "evaluation_cells"}, set(),
                  f"{name_hint}.bands")

    try:
        role = Role(str(raw["role"]))
    except ValueError:
        raise ConfigError(
            f"unknown role {raw['role']!r}; expected one of {[r.value for r in Role]}"
        ) from None

    return CaseConfig(
        name=str(raw["name"]),
        role=role,
        grid=grid,
        target_depth_nm=float(raw["target_depth_nm"]),
        rate=rate,
        velocity=velocity,
        materials=tuple(materials),
        cfl_target=float(raw.get("cfl_target", 0.4)),
        n_steps_override=(int(raw["n_steps"]) if raw.get("n_steps") is not None else None),
        reinit=ReinitConfig(**{k: int(v) for k, v in reinit_raw.items()}),
        bands=BandConfig(**{k: float(v) for k, v in bands_raw.items()}),
        w_mat_cells=float(raw.get("w_mat_cells", 2.0)),
        mask_thickness_nm=(float(raw["mask_thickness_nm"])
                           if raw.get("mask_thickness_nm") is not None else None),
        spatial_scheme=str(raw.get("spatial_scheme", "godunov")),
        temporal_scheme=str(raw.get("temporal_scheme", "rk2")),
    )


def load_case(path: str | pathlib.Path) -> CaseConfig:
    """Load and validate a case from YAML.

    Also enforces S4.2: the directory and the `role` field must agree, so a dev config copied
    into `configs/cases/` is rejected rather than silently promoted to something that can
    certify a coupon claim.
    """
    path = pathlib.Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path} does not contain a mapping")

    case = case_from_dict(raw, name_hint=str(path))

    parent = path.parent.name
    if parent == "dev" and case.role is not Role.DEV:
        raise ConfigError(
            f"{path} is in configs/dev/ but declares role {case.role.value!r}; dev configs carry "
            f"a nominal rate and must say so"
        )
    if parent == "cases" and case.role is Role.DEV:
        raise ConfigError(
            f"{path} is in configs/cases/ but declares role 'dev'; a dev case copied into "
            f"cases/ would be treated as evidence"
        )
    return case
