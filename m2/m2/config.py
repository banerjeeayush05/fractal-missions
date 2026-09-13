"""Run configuration: YAML (OmegaConf) → validated, frozen dataclasses.

Rules (OPEN_QUESTIONS B10; decisions §2, B18, B20, C10):
- A default exists only where the PRD or a decision states one. Everything else is required and
  written as ``???``, and loading fails while any ``???`` remains.
- Unknown keys are rejected, so nothing can be tuned from config that the PRD fixes in code
  (CFL_MAX, the Taylor protocol, tolerances: see ``m2.constants``).
- **Configs specify a target depth, never a final time.** T = depth / rate is derived, and the
  level-set equation is invariant under scaling the rate by λ and the time by 1/λ: N comes from
  depth and dx, CFL has both factors scaling oppositely, and the reinitialisation pseudo-timestep
  is in length units. So a later correction to the rate changes nothing and the V21 golden profile
  does not break (decision B18).
- N is derived too: N = ceil(D / (cfl_target · dx)).
- Every declared parameter carries a **required** ``scale``, its typical magnitude (decision B20).
- Two config trees: ``configs/cases/*`` refuse to load without measured values;
  ``configs/dev/*`` carry a nominal rate marked provisional and may never certify a check that
  claims agreement with the coupon (``assert_may_certify``).
All errors are collected and reported together.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from m2.constants import CFL_MAX, CFL_TARGET_DEFAULT, DIRECTIONAL_MIN_P, DTYPE
from m2.schema import Grid, Material, SchemaError, validate_materials

SPATIAL_SCHEMES = ("godunov", "weno5")  # §5.2: WENO5 behind a flag
TIME_INTEGRATORS = ("tvd_rk2", "tvd_rk3")  # §5.2: RK3 behind a flag
EXTENSION_METHODS = ("pde", "closest_point")  # §5.4
ROLES = ("calibrate", "predict")  # decision §2
VELOCITY_MODELS = ("isotropic", "directional")  # §5.5


class ConfigError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("invalid M2 config:\n  - " + "\n  - ".join(self.errors))


@dataclass(frozen=True)
class Parameter:
    """One differentiable parameter: a value, its declared typical magnitude, and provenance."""

    value: float
    scale: float  # REQUIRED (decision B20): perturbations use max(|value|, scale)
    provisional: bool = False  # true = a guess, not a measurement


@dataclass(frozen=True)
class VelocityConfig:
    model: str
    params: dict[str, Parameter]

    @property
    def values(self) -> dict[str, float]:
        return {k: p.value for k, p in self.params.items()}

    @property
    def scales(self) -> dict[str, float]:
        return {k: p.scale for k, p in self.params.items()}

    @property
    def provisional(self) -> bool:
        return any(p.provisional for p in self.params.values())

    @property
    def max_rate_nm_s(self) -> float:
        """Fastest surface speed: normal incidence for the directional law."""
        v = self.values
        return v["v_iso"] + v.get("v_dir", 0.0)


@dataclass(frozen=True)
class BandConfig:
    """Two bands, not one (decision C10).

    The extension band is where a valid velocity must exist for the advection and reinitialisation
    stencils. The evaluation band is where the velocity model is actually called, and it must be
    thin: cells deeper in the band project to nearly the same surface point, so calling M3 for them
    buys the same expensive Monte Carlo estimate several times.
    """

    extension_cells: float
    extension_taper_cells: float  # weight tapers to zero over the outer cells of the extension band
    evaluation_cells: float  # weight zero beyond this; M3 is called here only
    capacity: int | None  # K, sized from the evaluation band at the WORST step, not the first


@dataclass(frozen=True)
class CaseSpec:
    id: str
    phase: int  # 1 = single material, 2 = SiGe marker layer
    role: str  # calibrate | predict — pre-registration, enforced in code
    target_depth_nm: float
    cd_nm: float | None  # None for the open-field blanket case S00
    pitch_nm: float | None
    etch_time_fractions: tuple[float, ...]


@dataclass(frozen=True)
class ExtensionConfig:
    method: str
    n_iterations: int | None


@dataclass(frozen=True)
class M2Config:
    case: CaseSpec
    dimension: int
    grid: Grid
    n_steps: int
    cfl_target: float
    seed: int
    materials: tuple[Material, ...]
    extension: ExtensionConfig
    velocity: VelocityConfig
    bands: BandConfig
    n_reinit: int = 5
    reinit_every: int = 5
    w_mat_cells: float = 2.0
    spatial_scheme: str = "godunov"
    time_integrator: str = "tvd_rk2"
    precision: str = DTYPE
    mms: bool = False

    @property
    def final_time_s(self) -> float:
        """Derived, never configured: T = depth / rate (decision B18)."""
        return self.case.target_depth_nm / self.velocity.max_rate_nm_s

    @property
    def dt_s(self) -> float:
        return self.final_time_s / self.n_steps

    @property
    def cfl(self) -> float:
        return self.velocity.max_rate_nm_s * self.dt_s / self.grid.spacing_nm

    @property
    def provisional(self) -> bool:
        """True when any input is a guess rather than a measurement (decision B18)."""
        return self.velocity.provisional

    @property
    def solid_materials(self) -> tuple[Material, ...]:
        return tuple(m for m in self.materials if not m.is_void)


def derived_n_steps(target_depth_nm: float, spacing_nm: float, cfl_target: float) -> int:
    """N = ceil(D / (CFL_target · dx)) (decision §2). The front travels D at the fastest rate."""
    return math.ceil(target_depth_nm / (cfl_target * spacing_nm))


def require_calibration_case(cfg: M2Config) -> M2Config:
    """Gate for any fitting routine. Decision §2: a fit must never touch a `predict` case."""
    if cfg.case.role != "calibrate":
        raise ConfigError([
            f"case {cfg.case.id} has role {cfg.case.role!r}: fitting may only consume "
            f"'calibrate' cases (decision §2, pre-registration)"
        ])
    return cfg


def assert_may_certify(cfg: M2Config, check_id: str) -> None:
    """A provisional config may not write a ledger row for a check claiming coupon agreement.

    Decision B18: development proceeds on a nominal rate, but nothing certifies against a guess.
    """
    from m2.verification.registry import CHECKS

    spec = CHECKS.get(check_id)
    if spec is None:
        raise KeyError(f"unknown check ID {check_id!r}")
    if cfg.provisional and spec.claims_coupon_agreement:
        raise ConfigError([
            f"check {check_id} claims agreement with the coupon, but case {cfg.case.id} uses "
            f"provisional inputs (a nominal rate, not a measurement). Development configs cannot "
            f"certify coupon agreement (decision B18)."
        ])


_TOP_KEYS = {"case", "dimension", "grid", "time", "seed", "materials", "reinit", "extension",
             "velocity", "bands", "material_transition", "numerics"}
_SUB_KEYS = {
    "case": {"id", "phase", "role", "target_depth_nm", "cd_nm", "pitch_nm", "etch_time_fractions"},
    "grid": {"shape", "spacing_nm", "periodic"},
    "time": {"cfl_target", "n_steps"},
    "reinit": {"n_iterations", "every"},
    "extension": {"method", "n_iterations"},
    "velocity": {"model", "params"},
    "bands": {"extension_cells", "extension_taper_cells", "evaluation_cells", "capacity"},
    "material_transition": {"w_mat_cells"},
    "numerics": {"spatial_scheme", "time_integrator", "precision", "mms"},
}
_MATERIAL_KEYS = {"name", "index", "is_mask", "is_void"}
_PARAM_KEYS = {"value", "scale", "provisional"}


def load_config(path: str | Path, overrides: Sequence[str] = (), *, allow_mms: bool = False) -> M2Config:
    """Load YAML, apply ``key=value`` dotlist overrides, validate."""
    cfg = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides)))
    missing = sorted(OmegaConf.missing_keys(cfg))
    if missing:
        raise ConfigError([f"{k}: required value not set ('???'); see OPEN_QUESTIONS.md" for k in missing])
    raw = OmegaConf.to_container(cfg, resolve=True)
    return config_from_dict(raw, allow_mms=allow_mms)


def config_from_dict(raw: Mapping[str, Any], *, allow_mms: bool = False) -> M2Config:
    errs: list[str] = []
    required = {"case", "dimension", "grid", "seed", "materials", "extension", "velocity", "bands"}
    _keys(raw, _TOP_KEYS, "", errs, required=required)
    sub = {k: raw.get(k) or {} for k in _SUB_KEYS}
    for k, allowed in _SUB_KEYS.items():
        if not isinstance(sub[k], Mapping):
            errs.append(f"{k}: must be a mapping")
            sub[k] = {}
        else:
            _keys(sub[k], allowed, f"{k}.", errs)

    case = _case(sub["case"], errs)
    dimension = _int(raw.get("dimension"), "dimension", errs, lo=2)
    if dimension is not None and dimension not in (2, 3):
        errs.append(f"dimension: must be 2 or 3, got {dimension}")

    g = sub["grid"]
    shape = g.get("shape")
    if not isinstance(shape, list) or not shape:
        errs.append(f"grid.shape: must be a non-empty list, got {shape!r}")
    elif dimension is not None and len(shape) != dimension:
        errs.append(f"grid.shape: has {len(shape)} axes but dimension is {dimension}")
    grid = None
    try:
        if isinstance(shape, list) and isinstance(g.get("periodic"), list):
            grid = Grid(tuple(shape), g.get("spacing_nm"), tuple(g["periodic"]))
        else:
            errs.append("grid.periodic: must be a list of bools, one per axis")
    except SchemaError as e:
        errs.append(f"grid: {e}")

    t = sub["time"]
    cfl_target = _float(t.get("cfl_target", CFL_TARGET_DEFAULT), "time.cfl_target", errs, positive=True)
    if cfl_target is not None and cfl_target > CFL_MAX:
        errs.append(f"time.cfl_target: {cfl_target} exceeds the hard bound CFL_MAX={CFL_MAX} (§5.2, §11)")
    n_steps = None
    if grid is not None and case is not None and cfl_target is not None:
        minimum = derived_n_steps(case.target_depth_nm, grid.spacing_nm, cfl_target)
        given = t.get("n_steps")
        if given is None:
            n_steps = minimum
        else:
            n_steps = _int(given, "time.n_steps", errs, lo=1)
            if n_steps is not None and n_steps < minimum:
                errs.append(f"time.n_steps: {n_steps} is below the CFL-derived minimum {minimum} "
                            f"= ceil({case.target_depth_nm} / ({cfl_target} * {grid.spacing_nm})); "
                            f"leave it unset to derive it (decision §2)")
    seed = _int(raw.get("seed"), "seed", errs)
    materials = _materials(raw.get("materials"), errs)
    velocity = _velocity(sub["velocity"], errs)
    bands = _bands(sub["bands"], errs)

    r = sub["reinit"]
    n_reinit = _int(r.get("n_iterations", 5), "reinit.n_iterations", errs, lo=0)
    reinit_every = _int(r.get("every", 5), "reinit.every", errs, lo=1)

    e = sub["extension"]
    method, n_ext = e.get("method"), e.get("n_iterations")
    if method not in EXTENSION_METHODS:
        errs.append(f"extension.method: must be one of {EXTENSION_METHODS}, got {method!r}")
    elif method == "pde":
        n_ext = _int(n_ext, "extension.n_iterations", errs, lo=1)
    elif n_ext is not None:
        errs.append("extension.n_iterations: must be null for closest_point")

    w_mat = _float(sub["material_transition"].get("w_mat_cells", 2.0), "material_transition.w_mat_cells",
                   errs, positive=True)

    nu = sub["numerics"]
    spatial = _choice(nu.get("spatial_scheme", "godunov"), SPATIAL_SCHEMES, "numerics.spatial_scheme", errs)
    integ = _choice(nu.get("time_integrator", "tvd_rk2"), TIME_INTEGRATORS, "numerics.time_integrator", errs)
    precision = nu.get("precision", DTYPE)
    if precision != DTYPE:
        errs.append(f"numerics.precision: must be {DTYPE!r} (PRD §7.5, §11), got {precision!r}")
    mms = nu.get("mms", False)
    if not isinstance(mms, bool):
        errs.append(f"numerics.mms: must be a bool, got {mms!r}")
    elif mms and not allow_mms:
        errs.append("numerics.mms: the MMS source term is verification-only (PRD §8.2 V5); "
                    "production loads refuse it")

    if errs:
        raise ConfigError(errs)
    cfg = M2Config(
        case=case, dimension=dimension, grid=grid, n_steps=n_steps, cfl_target=cfl_target, seed=seed,
        materials=materials, extension=ExtensionConfig(method, n_ext), velocity=velocity, bands=bands,
        n_reinit=n_reinit, reinit_every=reinit_every, w_mat_cells=w_mat, spatial_scheme=spatial,
        time_integrator=integ, precision=precision, mms=mms,
    )
    if cfg.cfl > CFL_MAX:
        raise ConfigError([f"derived CFL {cfg.cfl:.3f} exceeds CFL_MAX={CFL_MAX}: "
                           f"raise n_steps or lower the rate (§5.2)"])
    return cfg


def _velocity(v: Mapping[str, Any], errs: list[str]) -> VelocityConfig | None:
    model = _choice(v.get("model"), VELOCITY_MODELS, "velocity.model", errs)
    raw_params = v.get("params")
    if not isinstance(raw_params, Mapping) or not raw_params:
        errs.append("velocity.params: must be a non-empty mapping of parameter -> {value, scale}")
        return None
    expected = {"isotropic": {"v_iso"}, "directional": {"v_iso", "v_dir", "p"}}.get(model)
    params: dict[str, Parameter] = {}
    for name, spec in raw_params.items():
        if not isinstance(spec, Mapping):
            errs.append(f"velocity.params.{name}: must be a mapping with value and scale")
            continue
        _keys(spec, _PARAM_KEYS, f"velocity.params.{name}.", errs, required={"value", "scale"})
        value = _float(spec.get("value"), f"velocity.params.{name}.value", errs)
        scale = _float(spec.get("scale"), f"velocity.params.{name}.scale", errs, positive=True)
        prov = spec.get("provisional", False)
        if not isinstance(prov, bool):
            errs.append(f"velocity.params.{name}.provisional: must be a bool, got {prov!r}")
            prov = False
        if value is not None and scale is not None:
            params[name] = Parameter(value, scale, prov)
    if expected is not None and set(params) != expected:
        errs.append(f"velocity.params: model {model!r} needs exactly {sorted(expected)}, "
                    f"got {sorted(params)}")
        return None
    if "p" in params and params["p"].value <= DIRECTIONAL_MIN_P:
        errs.append(f"velocity.params.p.value: must be > {DIRECTIONAL_MIN_P} — at p = 1 the law has a "
                    f"kink on vertical sidewalls, where the derivative dies (decision, §11)")
    for name in ("v_iso", "v_dir"):
        if name in params and params[name].value < 0:
            errs.append(f"velocity.params.{name}.value: must be >= 0 (an etch rate; negative deposits)")
    if params and model and not errs:
        if VelocityConfig(model, params).max_rate_nm_s <= 0:
            errs.append("velocity.params: the total rate at normal incidence must be > 0")
    return VelocityConfig(model, params) if model and params else None


def _bands(b: Mapping[str, Any], errs: list[str]) -> BandConfig | None:
    ext = _float(b.get("extension_cells"), "bands.extension_cells", errs, positive=True)
    taper = _float(b.get("extension_taper_cells"), "bands.extension_taper_cells", errs, positive=True)
    ev = _float(b.get("evaluation_cells"), "bands.evaluation_cells", errs, positive=True)
    cap = b.get("capacity")
    if cap is not None:
        cap = _int(cap, "bands.capacity", errs, lo=1)
    if None in (ext, taper, ev):
        return None
    if ev > ext:
        errs.append(f"bands.evaluation_cells ({ev}) must not exceed bands.extension_cells ({ext}): "
                    f"the velocity model is called on the thin set and gathered outward")
    if taper > ext:
        errs.append(f"bands.extension_taper_cells ({taper}) must not exceed extension_cells ({ext})")
    return BandConfig(ext, taper, ev, cap)


def _case(c: Mapping[str, Any], errs: list[str]) -> CaseSpec | None:
    _keys(c, _SUB_KEYS["case"], "case.", errs, required={"id", "phase", "role", "target_depth_nm"})
    cid = c.get("id")
    if not isinstance(cid, str) or not cid:
        errs.append(f"case.id: must be a non-empty str, got {cid!r}")
    phase = _int(c.get("phase"), "case.phase", errs, lo=1)
    if phase is not None and phase not in (1, 2):
        errs.append(f"case.phase: must be 1 or 2, got {phase}")
    role = _choice(c.get("role"), ROLES, "case.role", errs)
    depth = _float(c.get("target_depth_nm"), "case.target_depth_nm", errs, positive=True)
    cd, pitch = c.get("cd_nm"), c.get("pitch_nm")
    if cd is not None:
        cd = _float(cd, "case.cd_nm", errs, positive=True)
    if pitch is not None:
        pitch = _float(pitch, "case.pitch_nm", errs, positive=True)
    if (cd is None) != (pitch is None):
        errs.append("case.cd_nm and case.pitch_nm must both be set, or both null (open field)")
    if cd is not None and pitch is not None and cd >= pitch:
        errs.append(f"case.cd_nm ({cd}) must be smaller than case.pitch_nm ({pitch})")
    fracs = c.get("etch_time_fractions", [1.0])
    if not isinstance(fracs, list) or not fracs or not all(
        isinstance(f, (int, float)) and not isinstance(f, bool) and 0 < f <= 1 for f in fracs
    ):
        errs.append(f"case.etch_time_fractions: must be a non-empty list in (0, 1], got {fracs!r}")
        fracs = [1.0]
    if None in (cid, phase, role, depth):
        return None
    return CaseSpec(cid, phase, role, depth, cd, pitch, tuple(float(f) for f in fracs))


def _materials(mats_raw: Any, errs: list[str]) -> tuple[Material, ...]:
    if not isinstance(mats_raw, list) or not mats_raw:
        errs.append("materials: must be a non-empty list")
        return ()
    parsed = []
    for i, m in enumerate(mats_raw):
        if not isinstance(m, Mapping):
            errs.append(f"materials[{i}]: must be a mapping")
            continue
        _keys(m, _MATERIAL_KEYS, f"materials[{i}].", errs, required={"name", "index", "is_mask"})
        try:
            parsed.append(Material(m.get("name"), m.get("index"), m.get("is_mask"), m.get("is_void", False)))
        except SchemaError as e:
            errs.append(f"materials[{i}]: {e}")
    if len(parsed) != len(mats_raw):
        return ()
    try:
        return validate_materials(parsed)
    except SchemaError as e:
        errs.append(f"materials: {e}")
        return ()


def _keys(d: Mapping, allowed: set[str], prefix: str, errs: list[str], required: set[str] = frozenset()) -> None:
    for k in sorted(set(d) - allowed):
        errs.append(f"{prefix}{k}: unknown key (allowed: {sorted(allowed)})")
    for k in sorted(required - set(d)):
        errs.append(f"{prefix}{k}: required")


def _int(v: Any, name: str, errs: list[str], lo: int | None = None) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int):
        errs.append(f"{name}: must be an int, got {v!r}")
        return None
    if lo is not None and v < lo:
        errs.append(f"{name}: must be >= {lo}, got {v}")
        return None
    return v


def _float(v: Any, name: str, errs: list[str], positive: bool = False) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        errs.append(f"{name}: must be a finite number, got {v!r}")
        return None
    if positive and v <= 0:
        errs.append(f"{name}: must be > 0, got {v}")
        return None
    return float(v)


def _choice(v: Any, options: tuple[str, ...], name: str, errs: list[str]) -> str | None:
    if v not in options:
        errs.append(f"{name}: must be one of {options}, got {v!r}")
        return None
    return v


def parameter_tree(cfg: M2Config) -> tuple[dict[str, float], dict[str, float]]:
    """The differentiable parameter PyTree and its declared scales (decision B20).

    Everything M2 differentiates with respect to lives in this tree and is never captured in a
    closure — a captured value is invisible to `jax.grad` and produces a silently zero gradient
    column (§5.0, §7.3). The scales travel with the values because the gradient harness needs them:
    perturbations use max(|value|, scale), so a parameter sitting near zero is still probed.
    """
    return dict(cfg.velocity.values), dict(cfg.velocity.scales)
