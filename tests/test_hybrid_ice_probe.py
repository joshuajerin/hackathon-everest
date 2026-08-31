from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from hackathon_everest.hybrid_ice_probe import run_hybrid_ice_probe
from hackathon_everest.models import SENSOR_CHANNELS

try:
    _mujoco = importlib.import_module("mujoco")
    _HAS_MUJOCO = hasattr(_mujoco, "MjModel")
except ImportError:
    _HAS_MUJOCO = False

pytestmark = pytest.mark.skipif(not _HAS_MUJOCO, reason="MuJoCo extra not installed")


def test_hybrid_ice_probe_keeps_material_truth_outside_packet() -> None:
    root = Path(__file__).parents[1]
    run = run_hybrid_ice_probe(root / "mujoco/crampon_probe.xml", duration_s=0.9, seed=41)
    values = run.sensor_matrix

    assert values.shape == (90, SENSOR_CHANNELS)
    assert run.ice_penetration_m.shape == (90, 4)
    assert run.ice_shear_capacity_n.shape == (90, 4)
    assert run.ice_contact_force_n.shape == (90, 4, 3)
    assert np.isfinite(values).all()
    assert np.all(run.ice_penetration_m >= 0.0)
    assert abs(values[-20:, :4].sum(axis=1).mean() - 150.0) < 2.0
    assert not np.shares_memory(values, run.ice_penetration_m)
    assert not np.shares_memory(values, run.ice_contact_force_n)


def test_hybrid_ice_slope_lateral_sweep_changes_direction_without_leaking_truth() -> None:
    root = Path(__file__).parents[1]
    uphill = run_hybrid_ice_probe(
        root / "mujoco/crampon_probe.xml",
        duration_s=0.8,
        slope_deg=5.0,
        lateral_drive_force_n=-15.0,
        seed=41,
    )
    downhill = run_hybrid_ice_probe(
        root / "mujoco/crampon_probe.xml",
        duration_s=0.8,
        slope_deg=5.0,
        lateral_drive_force_n=15.0,
        seed=41,
    )

    assert uphill.sensor_matrix.shape == downhill.sensor_matrix.shape == (80, SENSOR_CHANNELS)
    assert np.isfinite(uphill.sensor_matrix).all()
    assert np.isfinite(downhill.sensor_matrix).all()
    assert uphill.report()["lateral_travel_m"] < -0.05
    assert downhill.report()["lateral_travel_m"] > 0.05
    assert uphill.ice_shear_capacity_n.shape == (80, 4)
    assert not np.shares_memory(uphill.sensor_matrix, uphill.ice_shear_capacity_n)
