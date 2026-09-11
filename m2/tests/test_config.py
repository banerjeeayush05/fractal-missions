"""Config loading and validation (OPEN_QUESTIONS B10; decisions §2, §10, B18, B20, C10)."""

from pathlib import Path

import pytest
import yaml

from m2.config import (
    ConfigError,
    assert_may_certify,
    config_from_dict,
    derived_n_steps,
    load_config,
    require_calibration_case,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
REFERENCE = CONFIG_DIR / "reference.yaml"
CASES = sorted((CONFIG_DIR / "cases").glob("*.yaml"))
DEV = sorted((CONFIG_DIR / "dev").glob("*.yaml"))

# The coupon ladder refuses to load until S00 has been etched (decision B18, OPEN_QUESTIONS C11).
MEASURED = ["velocity.params.v_iso.value=0.58", "velocity.params.v_dir.value=5.25",
            "velocity.params.p.value=2.0"]


def _raw():
    return {
        "case": {"id": "T01", "phase": 1, "role": "calibrate", "target_depth_nm": 2500.0,
                 "cd_nm": 500.0, "pitch_nm": 1000.0, "etch_time_fractions": [0.5, 1.0]},
        "dimension": 2,
        "grid": {"shape": [270, 100], "spacing_nm": 10.0, "periodic": [False, True]},
        "time": {"cfl_target": 0.4},
        "seed": 0,
        "materials": [{"name": "void", "index": 0, "is_mask": False, "is_void": True},
                      {"name": "silicon", "index": 1, "is_mask": False}],
        "extension": {"method": "closest_point", "n_iterations": None},
        "velocity": {"model": "directional",
                     "params": {"v_iso": {"value": 0.58, "scale": 0.58},
                                "v_dir": {"value": 5.25, "scale": 5.25},
                                "p": {"value": 2.0, "scale": 2.0}}},
        "bands": {"extension_cells": 8.0, "extension_taper_cells": 2.0, "evaluation_cells": 1.5,
                  "capacity": None},
    }


# --- the shipped configs -------------------------------------------------------------------


def test_both_config_trees_cover_the_decision_ladder():
    assert {p.stem for p in CASES} == {"S00", "S01", "S02", "S03", "S03_3d"}
    assert {p.stem for p in DEV} == {"S00", "S01", "S02", "S03", "S03_3d"}


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_coupon_configs_refuse_to_load_without_a_measured_rate(path):
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    missing = {e.split(":")[0] for e in exc.value.errors}
    assert {"velocity.params.v_iso.value", "velocity.params.v_dir.value"} <= missing


@pytest.mark.parametrize("path", DEV, ids=lambda p: p.stem)
def test_dev_configs_load_as_shipped_and_are_marked_provisional(path):
    cfg = load_config(path)
    assert cfg.provisional, "a nominal rate must be marked provisional (decision B18)"
    assert cfg.velocity.max_rate_nm_s == pytest.approx(5.83)


@pytest.mark.parametrize(
    "stem,cd,pitch,role,shape",
    [
        ("S00", None, None, "calibrate", (270, 20)),
        ("S01", 2000.0, 4000.0, "calibrate", (270, 400)),
        ("S02", 1000.0, 2000.0, "predict", (270, 200)),
        ("S03", 500.0, 1000.0, "predict", (270, 100)),
        ("S03_3d", 500.0, 1000.0, "predict", (270, 100, 100)),
    ],
)
def test_case_geometry_matches_the_decision_tables(stem, cd, pitch, role, shape):
    coupon = load_config(CONFIG_DIR / "cases" / f"{stem}.yaml", MEASURED)
    dev = load_config(CONFIG_DIR / "dev" / f"{stem}.yaml")
    for cfg in (coupon, dev):
        assert (cfg.case.cd_nm, cfg.case.pitch_nm, cfg.case.role) == (cd, pitch, role)
        assert cfg.grid.shape == shape
        assert (cfg.case.target_depth_nm, cfg.grid.spacing_nm) == (2500.0, 10.0)
        assert shape[0] == 10 + int(2500 / 10) + 10
        assert [m.name for m in cfg.materials] == ["void", "silicon"]
        assert not any(m.is_mask for m in cfg.materials)
    if pitch is not None:
        assert shape[1] * coupon.grid.spacing_nm == pitch
    # Only the rate differs between trees: geometry is identical.
    assert coupon.grid == dev.grid and coupon.n_steps == dev.n_steps


def test_time_is_derived_from_depth_and_rate_and_closes_the_cfl_arithmetic():
    """Decision B18: T = depth / rate, N from depth and dx, so CFL lands on the target exactly."""
    cfg = load_config(CONFIG_DIR / "dev" / "S03.yaml")
    assert cfg.n_steps == derived_n_steps(2500.0, 10.0, 0.4) == 625
    assert cfg.final_time_s == pytest.approx(2500.0 / 5.83, rel=1e-12)
    assert cfg.dt_s == pytest.approx(cfg.final_time_s / 625)
    assert cfg.cfl == pytest.approx(0.4, rel=1e-12)


def test_rate_correction_leaves_the_discretisation_untouched():
    """The scaling invariance the depth-based rule buys: rate * lambda, time / lambda, same profile."""
    raw = _raw()
    base = config_from_dict(raw)
    for name in ("v_iso", "v_dir"):
        raw["velocity"]["params"][name]["value"] *= 1.37
    scaled = config_from_dict(raw)
    assert scaled.n_steps == base.n_steps
    assert scaled.cfl == pytest.approx(base.cfl, rel=1e-12)
    assert scaled.final_time_s == pytest.approx(base.final_time_s / 1.37, rel=1e-12)


def test_provisional_configs_cannot_certify_a_coupon_agreement_check(monkeypatch):
    import dataclasses

    from m2.verification import registry

    dev = load_config(CONFIG_DIR / "dev" / "S01.yaml")
    coupon = load_config(CONFIG_DIR / "cases" / "S01.yaml", MEASURED)
    assert_may_certify(dev, "V19")  # V19 claims nothing about the coupon
    # No shipped check claims coupon agreement yet, so pretend one does.
    patched = dict(registry.CHECKS)
    patched["V21"] = dataclasses.replace(registry.CHECKS["V21"], claims_coupon_agreement=True)
    monkeypatch.setattr(registry, "CHECKS", patched)
    with pytest.raises(ConfigError, match="provisional inputs"):
        assert_may_certify(dev, "V21")
    assert_may_certify(coupon, "V21")  # measured inputs may certify
    assert not coupon.provisional


def test_roles_gate_fitting_routines():
    """Decision §2: a fit must never consume a `predict` case. Pre-registration in code."""
    calib = load_config(CONFIG_DIR / "dev" / "S01.yaml")
    predict = load_config(CONFIG_DIR / "dev" / "S03.yaml")
    assert require_calibration_case(calib) is calib
    with pytest.raises(ConfigError, match="may only consume 'calibrate' cases"):
        require_calibration_case(predict)


def test_bands_are_two_and_the_evaluation_band_is_the_thin_one():
    cfg = load_config(CONFIG_DIR / "dev" / "S03.yaml")
    assert cfg.bands.extension_cells == 8.0
    assert cfg.bands.evaluation_cells == 1.5
    assert cfg.bands.evaluation_cells < cfg.bands.extension_cells
    assert cfg.bands.capacity is None  # derived from the worst step at M2.1


def test_parameter_scales_are_declared_and_required():
    cfg = load_config(CONFIG_DIR / "dev" / "S03.yaml")
    assert set(cfg.velocity.scales) == {"v_iso", "v_dir", "p"}
    assert all(s > 0 for s in cfg.velocity.scales.values())
    raw = _raw()
    del raw["velocity"]["params"]["p"]["scale"]
    with pytest.raises(ConfigError, match="scale"):
        config_from_dict(raw)


# --- schema rules ---------------------------------------------------------------------------


def test_reference_yaml_refuses_to_load_with_unset_values():
    with pytest.raises(ConfigError) as exc:
        load_config(REFERENCE)
    missing = {e.split(":")[0] for e in exc.value.errors}
    assert {"case.id", "dimension", "grid.shape", "seed", "materials", "velocity.model"} <= missing


def test_reference_yaml_lists_every_accepted_section():
    doc = yaml.safe_load(REFERENCE.read_text())
    assert set(doc) == {"case", "dimension", "grid", "time", "seed", "materials", "velocity",
                        "bands", "reinit", "extension", "material_transition", "numerics"}


def test_defaults_only_where_a_decision_states_them():
    cfg = config_from_dict(_raw())
    assert (cfg.n_reinit, cfg.reinit_every, cfg.w_mat_cells) == (5, 5, 2.0)
    assert (cfg.spatial_scheme, cfg.time_integrator, cfg.precision, cfg.mms) == (
        "godunov", "tvd_rk2", "float64", False)
    assert cfg.cfl_target == 0.4
    assert cfg.solid_materials == (cfg.materials[1],)


def _errors(mutate, **kw):
    raw = _raw()
    mutate(raw)
    with pytest.raises(ConfigError) as exc:
        config_from_dict(raw, **kw)
    return "\n".join(exc.value.errors)


def test_rejects_unknown_keys_including_cfl_and_tolerances():
    assert "unknown key" in _errors(lambda r: r.update(cfl_max=0.9))
    assert "unknown key" in _errors(lambda r: r.update(taylor={"n_directions": 5}))
    assert "unknown key" in _errors(lambda r: r.update(numerics={"slope_band": [1.0, 3.0]}))
    assert "unknown key" in _errors(lambda r: r.update(time={"final_time_s": 430.0}))


def test_rejects_fp32():
    assert "float64" in _errors(lambda r: r.update(numerics={"precision": "float32"}))


def test_mms_refused_unless_explicitly_allowed():
    raw = _raw()
    raw["numerics"] = {"mms": True}
    with pytest.raises(ConfigError, match="verification-only"):
        config_from_dict(raw)
    assert config_from_dict(raw, allow_mms=True).mms is True


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r.update(dimension=3), "dimension is 3"),
        (lambda r: r["time"].update(n_steps=500), "below the CFL-derived minimum 625"),
        (lambda r: r["time"].update(cfl_target=0.9), "exceeds the hard bound"),
        (lambda r: r["grid"].update(periodic=[True, True]), "vertical axis"),
        (lambda r: r.update(reinit={"n_iterations": 2.5}), "reinit.n_iterations"),
        (lambda r: r["extension"].update(method="pde"), "extension.n_iterations"),
        (lambda r: r["velocity"].update(model="isotropic"), "needs exactly"),
        (lambda r: r["velocity"]["params"]["p"].update(value=1.0), "kink on vertical sidewalls"),
        (lambda r: r["velocity"]["params"]["v_dir"].update(value=-1.0), "must be >= 0"),
        (lambda r: r["velocity"]["params"]["v_iso"].update(scale=0.0), "scale"),
        (lambda r: r["bands"].update(evaluation_cells=9.0), "must not exceed"),
        (lambda r: r["bands"].update(extension_cells=0.0), "extension_cells"),
        (lambda r: r["bands"].update(capacity=0), "capacity"),
        (lambda r: r.update(material_transition={"w_mat_cells": 0}), "w_mat_cells"),
        (lambda r: r["case"].update(role="fit"), "case.role"),
        (lambda r: r["case"].update(cd_nm=2000.0), "smaller than"),
        (lambda r: r["materials"][0].update(is_void=False), "exactly one material must be the void"),
        (lambda r: r.pop("seed"), "seed"),
        (lambda r: r.pop("velocity"), "velocity"),
        (lambda r: r.pop("bands"), "bands"),
    ],
)
def test_rejects_invalid(mutate, match):
    assert match in _errors(mutate)


def test_all_errors_reported_together():
    raw = _raw()
    raw["seed"] = "x"
    raw["case"]["role"] = "fit"
    raw["bands"]["evaluation_cells"] = -1.0
    with pytest.raises(ConfigError) as exc:
        config_from_dict(raw)
    assert len(exc.value.errors) >= 3


def test_phase_two_materials_use_the_same_interface():
    """Decision §10: phase 1 is a one-solid-material instance of the general code, not a branch."""
    raw = _raw()
    raw["case"]["phase"] = 2
    raw["materials"].append({"name": "sige", "index": 2, "is_mask": False})
    cfg = config_from_dict(raw)
    assert len(cfg.solid_materials) == 2
