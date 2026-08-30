from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import Any

import numpy as np

from hackathon_everest.models import SENSOR_CHANNELS as CANONICAL_SENSOR_CHANNELS

SENSOR_CHANNELS = CANONICAL_SENSOR_CHANNELS
FOOT_COUNT = 2


class FootAxis(IntEnum):
    """Stable bilateral tensor axis. Each entry still owns a 19-channel packet."""

    LEFT = 0
    RIGHT = 1


AXIAL_FORCE_SLICE = slice(0, 4)
PENETRATION_SLICE = slice(4, 8)
ACCELEROMETER_SLICE = slice(8, 11)
GYROSCOPE_SLICE = slice(11, 14)
RADAR_FRONTEND_SLICE = slice(14, 19)

CHANNEL_SLICES = MappingProxyType(
    {
        "axial_force_n": AXIAL_FORCE_SLICE,
        "penetration_m": PENETRATION_SLICE,
        "accelerometer_mps2": ACCELEROMETER_SLICE,
        "gyroscope_rps": GYROSCOPE_SLICE,
        "radar_frontend": RADAR_FRONTEND_SLICE,
    }
)

VISIBLE_FIELD_ALLOWLIST = frozenset(
    {
        "packet_values",
        "valid_mask",
        "timestamp_s",
        "sample_age_s",
        "context",
        "commands",
    }
)
DEPLOYABLE_CONTEXT_ALLOWLIST = frozenset(
    {
        "foot_position_xyz_m",
        "foot_velocity_xyz_mps",
        "pelvis_roll_pitch_yaw_rad",
        "commanded_probe_load_n",
        "commanded_foot_speed_mps",
        "body_weight_on_foot_n",
    }
)
DEPLOYABLE_COMMAND_ALLOWLIST = frozenset(
    {
        "requested_vx_mps",
        "requested_vy_mps",
        "requested_wz_rps",
        "mode",
        "probe_load_n",
        "approach_speed_mps",
    }
)


def _shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(value.shape)
    except AttributeError as exc:
        raise TypeError("Sensor plane arrays must expose a shape") from exc


def _is_boolean_array(value: Any) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return False
    if dtype == np.dtype(bool):
        return True
    # Avoid importing Torch in the schema module. torch.bool stringifies this way.
    return str(dtype) == "torch.bool"


def _validate_named_fields(
    values: Mapping[str, Any], allowlist: frozenset[str], *, plane_name: str
) -> None:
    unexpected = set(values).difference(allowlist)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(f"{plane_name} contains non-deployable fields: {names}")


def assert_visible_field_names(field_names: set[str] | frozenset[str]) -> None:
    """Reject observation or writer schemas that add non-deployable top-level fields."""

    unexpected = set(field_names).difference(VISIBLE_FIELD_ALLOWLIST)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(f"Visible plane contains non-allowlisted fields: {names}")


@dataclass(frozen=True)
class VisibleSensorBatch:
    """Deployable bilateral packets with an explicit foot axis and metadata.

    Shapes are ``(..., 2, 19)`` for packet values, masks, and sample ages and
    ``(..., 2)`` for timestamps. A history dimension may appear before the foot
    axis. The two feet are never flattened into a 38-channel packet.
    """

    packet_values: Any
    valid_mask: Any
    timestamp_s: Any
    sample_age_s: Any
    context: Mapping[str, Any] = field(default_factory=dict)
    commands: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values_shape = _shape(self.packet_values)
        if len(values_shape) < 3 or values_shape[-2:] != (FOOT_COUNT, SENSOR_CHANNELS):
            raise ValueError(
                "packet_values must have shape (..., 2, 19); bilateral packets "
                "must not be flattened to 38 channels"
            )
        if _shape(self.valid_mask) != values_shape:
            raise ValueError("valid_mask must match packet_values shape")
        if not _is_boolean_array(self.valid_mask):
            raise TypeError("valid_mask must have boolean dtype")
        if _shape(self.sample_age_s) != values_shape:
            raise ValueError("sample_age_s must match packet_values shape")
        if _shape(self.timestamp_s) != values_shape[:-1]:
            raise ValueError("timestamp_s must have shape packet_values.shape[:-1]")
        _validate_named_fields(
            self.context, DEPLOYABLE_CONTEXT_ALLOWLIST, plane_name="Visible context"
        )
        _validate_named_fields(
            self.commands, DEPLOYABLE_COMMAND_ALLOWLIST, plane_name="Visible commands"
        )

    def as_dict(self) -> dict[str, Any]:
        result = {
            "packet_values": self.packet_values,
            "valid_mask": self.valid_mask,
            "timestamp_s": self.timestamp_s,
            "sample_age_s": self.sample_age_s,
            "context": dict(self.context),
            "commands": dict(self.commands),
        }
        assert_visible_field_names(set(result))
        return result


@dataclass(frozen=True)
class PrivilegedSensorBatch:
    """Simulator-only plane. It is deliberately not accepted by sensor adapters."""

    truth_canary: Any
    payload: Mapping[str, Any] = field(default_factory=dict)
