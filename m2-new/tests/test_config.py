"""Case configuration: what it derives, and what it refuses to load.

Stage 4 built `config.py` without a test file of its own; its rules were exercised only indirectly.
This file covers the derivations and the load-time refusals directly, and the stage-17 mask rules.
"""

import pathlib

import pytest
import yaml

from geocore.config import (
    CaseConfig, ConfigError, Role, case_from_dict, load_case, require_calibration_case,
    require_measured_rate,
)
from geocore.schema import Grid
from geocore.solver import SolvePlan



# ---------------------------------------------------------------------------- derivations


def test_n_and_time_are_derived_from_the_target_depth():
    """PRD §5.2: configs carry a target DEPTH; N and T follow. A rate correction then rescales time
    and changes nothing else."""
    case = load_case("configs/dev/S03.yaml")
    assert case.n_steps == case.n_steps_min == 625
    assert case.final_time_s == pytest.approx(case.target_depth_nm / case.rate.nm_per_s)
    assert case.dt_s == pytest.approx(case.final_time_s / case.n_steps)


def test_an_explicit_step_count_below_the_derived_minimum_is_refused():
    raw = _raw("configs/dev/S03.yaml")
    raw["n_steps"] = 600
    with pytest.raises(ConfigError, match="derived minimum"):
        case_from_dict(raw, "test")


# ------------------------------------------------------------------------------ refusals


def test_an_unknown_key_is_an_error_not_a_silent_default():
    """The most valuable rule in the loader: `cfl_taget: 0.9` falling back to the default would be
    undetectable -- the run succeeds and the file on disk disagrees with what ran."""
    raw = _raw("configs/dev/S03.yaml")
    raw["cfl_taget"] = 0.9
    with pytest.raises(ConfigError, match="unknown key"):
        case_from_dict(raw, "test")


def test_a_provisional_rate_cannot_carry_a_non_dev_role():
    raw = _raw("configs/dev/S03.yaml")
    raw["role"] = "calibrate"
    with pytest.raises(ConfigError, match="provisional"):
        case_from_dict(raw, "test")


def test_the_guards_refuse_a_dev_case():
    case = load_case("configs/dev/S03.yaml")
    assert case.role is Role.DEV
    with pytest.raises(ConfigError, match="calibrate"):
        require_calibration_case(case)
    with pytest.raises(ConfigError, match="provisional"):
        require_measured_rate(case, "agreement with the coupon")


def test_a_directional_case_needs_p_above_one():
    """PRD §11: p = 1 puts the law's kink on every vertical sidewall."""
    raw = _raw("configs/dev/S03.yaml")
    raw["velocity"]["params"]["p"] = 1.0
    with pytest.raises(ConfigError, match="p must be > 1"):
        case_from_dict(raw, "test")


# ------------------------------------------------------------------- stage 17: the mask


def test_a_mask_material_requires_a_thickness():
    """The grid's vertical extent is DERIVED from the stack (PRD §5.1), so a case that declares a mask
    without saying how thick it is cannot be sized."""
    raw = _raw("configs/dev/S03.yaml")
    raw.pop("mask_thickness_nm")
    with pytest.raises(ConfigError, match="mask_thickness_nm"):
        case_from_dict(raw, "test")


def test_the_grid_must_hold_the_mask_the_etch_and_both_buffers():
    """100 (bottom buffer) + 2500 (etch) + 200 (mask) + 100 (top buffer) = 2900 nm = 290 cells. The
    same grid with a 400 nm mask needs 3100 nm and is refused."""
    case = load_case("configs/dev/S03.yaml")
    dx = case.grid.spacing_nm
    assert case.grid.shape[0] * dx == 2 * 10 * dx + case.target_depth_nm + case.mask_thickness_nm

    raw = _raw("configs/dev/S03.yaml")
    raw["mask_thickness_nm"] = 400.0
    with pytest.raises(ConfigError, match="cannot hold"):
        case_from_dict(raw, "test")


def test_an_unmasked_case_needs_only_one_buffer():
    """Without a mask there is nothing above the surface to protect, so the old rule still applies."""
    raw = _raw("configs/dev/S03.yaml")
    raw["mask_thickness_nm"] = None
    raw["materials"] = [{"name": "film", "index": 0}]
    raw["grid"]["shape"] = [270, 100, 100]
    assert case_from_dict(raw, "test").mask is None


def test_the_default_spatial_scheme_is_defined_in_exactly_one_place():
    """S20.7. A config that does not name a scheme must get the SOLVER's default, not a literal
    repeated in the loader.

    `load_case` used to read `raw.get("spatial_scheme", "godunov")`. When the default moved to
    weno5 on 2026-09-17 that line was missed, so `CaseConfig()` said weno5 while every case loaded
    from YAML said godunov -- including S03, and therefore the entire M2.4 gate. It was invisible
    because both values are legal and the solver runs happily either way.

    Asserted against the dataclass field rather than against the string "weno5", so this test keeps
    holding when the default next moves, which is the only way it can catch the same bug twice.
    """
    default = CaseConfig.__dataclass_fields__["spatial_scheme"].default
    for name in ("S00", "S03"):
        raw = _raw(f"configs/dev/{name}.yaml")
        assert "spatial_scheme" not in raw, f"{name} now names a scheme; this test assumes it does not"
        assert case_from_dict(raw, name).spatial_scheme == default
    # and an explicit value still wins
    raw = _raw("configs/dev/S03.yaml")
    raw["spatial_scheme"] = "godunov"
    assert case_from_dict(raw, "test").spatial_scheme == "godunov"
    # the solver's own default must agree with the config's, or "the default" means two things
    assert SolvePlan(Grid((8, 8), 1.0, (False, True)), 1.0, 1).spatial_scheme == default


def _raw(path: str) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text())
