from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SOURCE = Path(__file__).parents[2] / "source/hackathon_everest_isaaclab"
sys.path.insert(0, str(SOURCE))

from hackathon_everest_isaaclab.sensors.faults import (
    SENSOR_FAULT_MODES,
    apply_sensor_faults,
    balanced_fault_codes_by_group,
)


def tensors():
    return (
        torch.full((6, 2, 4), 100.0),
        torch.full((6, 2, 4), 0.01),
        torch.zeros((6, 2, 3)),
        torch.zeros((6, 2, 3)),
        torch.tensor([0.04, 0.20, 0.8, 0.2, 0.05]).view(1, 1, 5).expand(6, 2, 5).clone(),
        torch.zeros(6, dtype=torch.long),
    )


def test_all_six_sensor_fault_modes_preserve_bilateral_packet_boundary() -> None:
    assert SENSOR_FAULT_MODES == (
        "nominal",
        "one_force_stale",
        "one_probe_saturated",
        "imu_bias_burst",
        "radar_interface_merge",
        "packet_latency_burst",
    )
    force, penetration, acceleration, gyro, radar, mask = apply_sensor_faults(
        *tensors(), sample_index=25
    )
    assert force.shape == (6, 2, 4)
    assert mask.shape == (6, 2, 19)
    assert not mask[1, 0, 0]
    assert mask[1, 1].all()
    assert force[2, 0, 1] == 700.0
    assert penetration[2, 0, 1] == pytest.approx(0.055)
    torch.testing.assert_close(acceleration[3, 0], torch.tensor([0.8, -0.5, 0.3]))
    torch.testing.assert_close(gyro[3, 0], torch.tensor([0.05, -0.04, 0.03]))
    assert radar[4, 0, 1] == radar[4, 0, 0]
    assert mask[5].all()  # latency burst is inactive at sample 25


def test_latency_burst_and_reset_cycle_rotate_fault_modes() -> None:
    values = list(tensors())
    values[-1] = torch.ones(6, dtype=torch.long)
    force, _, _, _, _, mask = apply_sensor_faults(*values, sample_index=5)
    assert not mask[0, 0, 0]  # nominal environment rotated to one-force-stale
    assert not mask[4].any()  # code 5 after rotation
    assert torch.equal(force[5], values[0][5])  # code 0 after rotation is unchanged


def test_fault_codes_are_balanced_independently_within_contact_modes() -> None:
    groups = [mode for _ in range(12) for mode in ("flat", "hybrid", "front")]
    codes = balanced_fault_codes_by_group(groups, device="cpu")
    for mode in ("flat", "hybrid", "front"):
        selected = codes[torch.tensor([value == mode for value in groups])]
        assert torch.bincount(selected, minlength=len(SENSOR_FAULT_MODES)).tolist() == [2] * 6
