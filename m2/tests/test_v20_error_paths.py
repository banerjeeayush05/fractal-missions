"""V20 — the CFL assertion fires (PRD §8.6), M2.1 gate, plus the capacity error path.

These test the paths that are otherwise never exercised. A guard nobody has seen fire is a guard
nobody knows works — the same reasoning as V19.

The CFL bound is checked as a scan output asserted on the host (decision D8), not with `checkify`,
which avoids any question about how it composes with `grad`. There are two lines of defence: the
config loader refuses an `n_steps` below the CFL-derived minimum at load time, and the solver
refuses the run itself, which also catches a rate that exceeds what the config assumed.
"""

import numpy as np
import pytest
from helpers import MATERIALS, synthetic_config

from m2 import initial, solver
from m2.config import ConfigError, config_from_dict
from m2.constants import CFL_MAX

RATE = 5.83


def _run(cfg, *, n_steps, dt):
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi0 = initial.half_space(cfg.grid, 320.0)
    return solver.solve(cfg, phi0, material, {"v_iso": RATE}, n_steps=n_steps, dt=dt)


@pytest.mark.check("V20")
def test_cfl_assertion_fires_and_names_n(ledger_measure):
    """Deliberately too few steps: the run must abort with a message naming N."""
    cfg = synthetic_config(n_steps=2)
    travel, n = 30.0, 2
    dt = travel / RATE / n  # CFL = R*dt/dx = 1.5 at dx = 10 nm

    with pytest.raises(solver.CFLViolation) as exc:
        _run(cfg, n_steps=n, dt=dt)

    message = str(exc.value)
    assert f"N = {n}" in message, f"the abort must name N; got: {message}"
    assert "discarded" in message
    ledger_measure.update({"n_steps": n, "dt_s": dt, "cfl_bound": CFL_MAX,
                           "expected_cfl": RATE * dt / cfg.grid.spacing_nm, "message": message})


@pytest.mark.check("V20")
def test_a_run_just_inside_the_bound_is_accepted(ledger_measure):
    """The guard must not be a blanket refusal: just inside the bound still runs."""
    cfg = synthetic_config(n_steps=8)
    dx = cfg.grid.spacing_nm
    dt = 0.45 * dx / RATE  # CFL = 0.45 < 0.5
    result = _run(cfg, n_steps=8, dt=dt)
    assert result.max_cfl == pytest.approx(0.45, rel=1e-9)
    assert result.max_cfl <= CFL_MAX
    ledger_measure["accepted_cfl"] = result.max_cfl


@pytest.mark.check("V20")
def test_the_config_loader_refuses_the_same_thing_earlier(ledger_measure):
    """Second line of defence: N below the CFL-derived minimum never reaches the solver."""
    raw = {
        "case": {"id": "T", "phase": 1, "role": "calibrate", "target_depth_nm": 2500.0,
                 "cd_nm": 500.0, "pitch_nm": 1000.0},
        "dimension": 2, "grid": {"shape": [270, 100], "spacing_nm": 10.0, "periodic": [False, True]},
        "time": {"cfl_target": 0.4, "n_steps": 100},
        "seed": 0,
        "materials": [{"name": "void", "index": 0, "is_mask": False, "is_void": True},
                      {"name": "silicon", "index": 1, "is_mask": False}],
        "extension": {"method": "closest_point", "n_iterations": None},
        "velocity": {"model": "isotropic", "params": {"v_iso": {"value": RATE, "scale": RATE}}},
        "bands": {"extension_cells": 8.0, "extension_taper_cells": 2.0, "evaluation_cells": 1.5,
                  "capacity": None},
    }
    with pytest.raises(ConfigError, match="below the CFL-derived minimum 625"):
        config_from_dict(raw)
    ledger_measure["loader_refuses_n_below_minimum"] = True


def test_capacity_overflow_aborts_rather_than_dropping_points():
    """If the padded request fills, points are silently dropped — so the run must abort instead.

    Sized from the worst step, this should never fire in production (decision C10); it fires here
    because the capacity is set absurdly low.
    """
    cfg = synthetic_config(n_steps=4)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi0 = initial.half_space(cfg.grid, 320.0)
    with pytest.raises(solver.CapacityOverflow, match="peak occupancy"):
        solver.solve(cfg, phi0, material, {"v_iso": RATE}, n_steps=4, dt=0.05, capacity=8)


def test_reported_occupancy_stays_below_the_estimated_capacity():
    """The estimate of decision C10 must actually hold for a real geometry."""
    cfg = synthetic_config(n_steps=20)
    material = initial.uniform_material(cfg.grid, MATERIALS)
    phi0 = initial.trench(cfg.grid, surface_nm=480.0, cd_nm=200.0, depth_nm=200.0)
    result = solver.solve(cfg, phi0, material, {"v_iso": RATE}, n_steps=20, dt=0.05)
    assert 0 < result.peak_occupancy < result.capacity
    assert np.isfinite(result.max_cfl)
