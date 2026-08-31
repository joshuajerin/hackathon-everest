from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from hackathon_everest.models import SENSOR_CHANNELS
from hackathon_everest.mujoco_probe import run_mujoco_probe

try:
    _mujoco = importlib.import_module("mujoco")
    _HAS_MUJOCO = hasattr(_mujoco, "MjModel")
except ImportError:
    _HAS_MUJOCO = False

pytestmark = pytest.mark.skipif(not _HAS_MUJOCO, reason="MuJoCo extra not installed")


def test_single_foot_probe_compiles_and_respects_sensor_contract() -> None:
    root = Path(__file__).parents[1]
    run = run_mujoco_probe(root / "mujoco/crampon_probe.xml", duration_s=0.8)
    values = run.sensor_matrix

    assert values.shape == (80, SENSOR_CHANNELS)
    assert np.isfinite(values).all()
    assert all(packet.valid_mask.all() for packet in run.packets)
    assert set(run.contact_geom_pairs) == {
        ("ice_plane", "probe_0_geom"),
        ("ice_plane", "probe_1_geom"),
        ("ice_plane", "probe_2_geom"),
        ("ice_plane", "probe_3_geom"),
    }

    report = run.report()
    assert report["steady_load_error_percent"] < 1.0
    assert report["steady_force_balance_cv"] < 0.01
    assert np.allclose(values[-1, :4], 37.5, atol=0.5)
    assert np.all((values[-1, 4:8] > 0.0) & (values[-1, 4:8] < 0.020))


def test_inclined_rigid_plane_exposes_downhill_drift_and_axial_projection() -> None:
    root = Path(__file__).parents[1]
    run = run_mujoco_probe(root / "mujoco/crampon_probe.xml", duration_s=0.8, slope_deg=10.0)

    values = run.sensor_matrix
    assert values.shape == (80, SENSOR_CHANNELS)
    assert np.isfinite(values).all()
    assert run.report()["slope_deg"] == 10.0
    # The condim=1 plane has no traction. The carriage reaches its downhill
    # travel limit and the uphill pair unloads; this is the explicit control.
    assert run.report()["lateral_travel_m"] > 0.10
    assert np.allclose(values[-1, :2], 0.0, atol=0.1)
    assert np.all(values[-1, 2:4] > 70.0)
