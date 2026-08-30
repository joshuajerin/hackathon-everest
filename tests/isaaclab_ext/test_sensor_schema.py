from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

EXTENSION_SOURCE = Path(__file__).parents[2] / "isaaclab_ext/source/hackathon_everest_isaaclab"
sys.path.insert(0, str(EXTENSION_SOURCE))

from hackathon_everest_isaaclab.data.schema import (
    PrivilegedSensorBatch,
    VisibleSensorBatch,
    assert_visible_field_names,
)
from hackathon_everest_isaaclab.sensors.crampon_sensor import (
    CramponSensorAdapter,
    CramponSensorConfig,
    pack_channel_groups,
)

from hackathon_everest.models import Proprioception, SynchronizedSensorPacket


def _channel_groups(*, offset: float = 0.0) -> tuple[np.ndarray, ...]:
    def bilateral(start: int, width: int) -> np.ndarray:
        left = np.arange(start, start + width, dtype=np.float64) + offset
        right = left + 100.0
        return np.stack([left, right], axis=0)[None, ...]

    return (
        bilateral(0, 4),
        bilateral(4, 4),
        bilateral(8, 3),
        bilateral(11, 3),
        bilateral(14, 5),
    )


def _observe(
    adapter: CramponSensorAdapter,
    *,
    timestamp_s: float,
    offset: float = 0.0,
    fresh_mask: np.ndarray | None = None,
) -> VisibleSensorBatch:
    force, depth, accel, gyro, radar = _channel_groups(offset=offset)
    return adapter.observe(
        axial_force_n=force,
        penetration_m=depth,
        accelerometer_mps2=accel,
        gyroscope_rps=gyro,
        radar_frontend=radar,
        timestamp_s=timestamp_s,
        fresh_mask=fresh_mask,
        context={"commanded_probe_load_n": np.array([[132.0, 132.0]])},
        commands={"probe_load_n": np.array([[132.0, 132.0]])},
    )


def test_packet_order_matches_existing_abi_and_keeps_foot_axis() -> None:
    groups = _channel_groups()
    packed = pack_channel_groups(*groups)

    assert packed.shape == (1, 2, 19)
    np.testing.assert_array_equal(packed[0, 0], np.arange(19, dtype=float))
    np.testing.assert_array_equal(packed[0, 1], np.arange(19, dtype=float) + 100.0)

    canonical = SynchronizedSensorPacket(
        timestamp_s=0.0,
        axial_force_n=groups[0][0, 0],
        penetration_m=groups[1][0, 0],
        accelerometer_mps2=groups[2][0, 0],
        gyroscope_rps=groups[3][0, 0],
        radar_frontend=groups[4][0, 0],
        valid_mask=np.ones(19, dtype=bool),
        proprioception=Proprioception(
            foot_position_xyz_m=np.zeros(3),
            foot_velocity_xyz_mps=np.zeros(3),
            pelvis_roll_pitch_yaw_rad=np.zeros(3),
            commanded_probe_load_n=132.0,
            commanded_foot_speed_mps=0.2,
            body_weight_on_foot_n=120.0,
        ),
    )
    np.testing.assert_array_equal(packed[0, 0], canonical.vector())

    with pytest.raises(ValueError, match="must not be flattened to 38"):
        VisibleSensorBatch(
            packet_values=packed.reshape(1, 38),
            valid_mask=np.ones((1, 38), dtype=bool),
            timestamp_s=np.zeros(1),
            sample_age_s=np.zeros((1, 38)),
        )


def test_visible_plane_is_allowlisted_and_truth_canary_cannot_influence_it() -> None:
    first_truth = PrivilegedSensorBatch(
        truth_canary=np.array([1.0]), payload={"contact_force_world_n": np.ones((1, 2, 4, 3))}
    )
    second_truth = PrivilegedSensorBatch(
        truth_canary=np.array([-999.0]),
        payload={"contact_force_world_n": np.full((1, 2, 4, 3), 8_000.0)},
    )
    first = _observe(
        CramponSensorAdapter(CramponSensorConfig(sample_drop_probability=0.0)),
        timestamp_s=0.0,
    )
    second = _observe(
        CramponSensorAdapter(CramponSensorConfig(sample_drop_probability=0.0)),
        timestamp_s=0.0,
    )

    assert not hasattr(first, "truth_canary")
    assert first_truth.truth_canary.item() != second_truth.truth_canary.item()
    assert set(first.as_dict()) == set(second.as_dict())
    np.testing.assert_array_equal(first.packet_values, second.packet_values)
    np.testing.assert_array_equal(first.valid_mask, second.valid_mask)

    with pytest.raises(ValueError, match="non-deployable fields"):
        VisibleSensorBatch(
            packet_values=first.packet_values,
            valid_mask=first.valid_mask,
            timestamp_s=first.timestamp_s,
            sample_age_s=first.sample_age_s,
            context={"contact_force_world_n": np.zeros((1, 2, 4, 3))},
        )
    with pytest.raises(ValueError, match="non-allowlisted fields"):
        assert_visible_field_names({"packet_values", "truth_canary"})


def test_cadence_and_dropout_are_explicit_metadata() -> None:
    adapter = CramponSensorAdapter(
        CramponSensorConfig(
            packet_rate_hz=100.0,
            sample_drop_probability=0.0,
            cadence_tolerance_s=1.0e-9,
        )
    )
    first = _observe(adapter, timestamp_s=0.0)
    assert first.valid_mask.all()
    assert np.count_nonzero(first.sample_age_s) == 0

    fresh = np.ones((1, 2, 19), dtype=bool)
    fresh[..., :8] = False
    second = _observe(adapter, timestamp_s=0.01, offset=1_000.0, fresh_mask=fresh)

    np.testing.assert_array_equal(second.packet_values[..., :8], first.packet_values[..., :8])
    np.testing.assert_array_equal(
        second.packet_values[..., 8:],
        pack_channel_groups(*_channel_groups(offset=1_000.0))[..., 8:],
    )
    assert not second.valid_mask[..., :8].any()
    assert second.valid_mask[..., 8:].all()
    np.testing.assert_allclose(second.sample_age_s[..., :8], 0.01)
    np.testing.assert_array_equal(second.sample_age_s[..., 8:], 0.0)
    np.testing.assert_allclose(second.timestamp_s, 0.01)

    with pytest.raises(ValueError, match="configured period"):
        _observe(adapter, timestamp_s=0.025, offset=2_000.0)


def test_first_packet_cannot_claim_held_values() -> None:
    fresh = np.ones((1, 2, 19), dtype=bool)
    fresh[..., 0] = False
    with pytest.raises(ValueError, match="first packet"):
        _observe(CramponSensorAdapter(), timestamp_s=0.0, fresh_mask=fresh)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch extra not installed")
def test_torch_path_preserves_device_and_never_round_trips_through_numpy() -> None:
    import torch

    groups = tuple(torch.as_tensor(value, dtype=torch.float32) for value in _channel_groups())
    adapter = CramponSensorAdapter(CramponSensorConfig(sample_drop_probability=0.0))
    result = adapter.observe(
        axial_force_n=groups[0],
        penetration_m=groups[1],
        accelerometer_mps2=groups[2],
        gyroscope_rps=groups[3],
        radar_frontend=groups[4],
        timestamp_s=0.0,
    )
    assert isinstance(result.packet_values, torch.Tensor)
    assert isinstance(result.valid_mask, torch.Tensor)
    assert result.packet_values.device == groups[0].device
    assert tuple(result.packet_values.shape) == (1, 2, 19)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch extra not installed")
def test_float32_packets_keep_exact_cadence_beyond_sixteen_seconds() -> None:
    import torch

    groups = tuple(torch.as_tensor(value, dtype=torch.float32) for value in _channel_groups())
    adapter = CramponSensorAdapter(
        CramponSensorConfig(sample_drop_probability=0.0, cadence_tolerance_s=1.0e-9)
    )
    result = None
    for sample in range(2_001):
        timestamp = torch.full((1, 2), sample * 0.01, dtype=torch.float64)
        result = adapter.observe(
            axial_force_n=groups[0],
            penetration_m=groups[1],
            accelerometer_mps2=groups[2],
            gyroscope_rps=groups[3],
            radar_frontend=groups[4],
            timestamp_s=timestamp,
        )
    assert result is not None
    assert result.timestamp_s.dtype == torch.float64
    assert result.sample_age_s.dtype == torch.float32
    torch.testing.assert_close(result.timestamp_s, torch.full((1, 2), 20.0, dtype=torch.float64))


def test_vector_environment_reset_forces_fresh_packet_without_breaking_global_cadence() -> None:
    adapter = CramponSensorAdapter(CramponSensorConfig(sample_drop_probability=0.0))
    groups = tuple(np.repeat(value, 2, axis=0) for value in _channel_groups())
    first = adapter.observe(
        axial_force_n=groups[0],
        penetration_m=groups[1],
        accelerometer_mps2=groups[2],
        gyroscope_rps=groups[3],
        radar_frontend=groups[4],
        timestamp_s=0.0,
    )
    adapter.mark_environment_reset(np.array([1]))
    changed = tuple(value + 500.0 for value in groups)
    second = adapter.observe(
        axial_force_n=changed[0],
        penetration_m=changed[1],
        accelerometer_mps2=changed[2],
        gyroscope_rps=changed[3],
        radar_frontend=changed[4],
        timestamp_s=0.01,
        fresh_mask=np.zeros((2, 2, 19), dtype=bool),
    )
    np.testing.assert_array_equal(second.packet_values[0], first.packet_values[0])
    np.testing.assert_array_equal(second.packet_values[1], pack_channel_groups(*changed)[1])
    assert not second.valid_mask[0].any()
    assert second.valid_mask[1].all()
    assert np.all(second.sample_age_s[1] == 0.0)

    adapter.reset()
    restarted = adapter.observe(
        axial_force_n=groups[0],
        penetration_m=groups[1],
        accelerometer_mps2=groups[2],
        gyroscope_rps=groups[3],
        radar_frontend=groups[4],
        timestamp_s=0.0,
    )
    assert restarted.valid_mask.all()
