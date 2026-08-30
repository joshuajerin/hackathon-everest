from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.schema import FOOT_COUNT, SENSOR_CHANNELS, VisibleSensorBatch

try:  # Isaac Lab provides Torch; the lightweight CPU test environment may not.
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only without the optional extra
    torch = None


@dataclass(frozen=True)
class CramponSensorConfig:
    packet_rate_hz: float = 100.0
    sample_drop_probability: float = 0.003
    cadence_tolerance_s: float = 1.0e-6
    seed: int = 7

    def __post_init__(self) -> None:
        if self.packet_rate_hz <= 0.0:
            raise ValueError("packet_rate_hz must be positive")
        if not 0.0 <= self.sample_drop_probability <= 1.0:
            raise ValueError("sample_drop_probability must be in [0, 1]")
        if self.cadence_tolerance_s < 0.0:
            raise ValueError("cadence_tolerance_s must be non-negative")

    @property
    def packet_period_s(self) -> float:
        return 1.0 / self.packet_rate_hz


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _validate_group_shape(value: Any, width: int, name: str) -> tuple[int, ...]:
    try:
        shape = tuple(value.shape)
    except AttributeError as exc:
        raise TypeError(f"{name} must expose a shape") from exc
    if len(shape) < 3 or shape[-2:] != (FOOT_COUNT, width):
        raise ValueError(f"{name} must have shape (..., 2, {width}), got {shape}")
    return shape


def pack_channel_groups(
    axial_force_n: Any,
    penetration_m: Any,
    accelerometer_mps2: Any,
    gyroscope_rps: Any,
    radar_frontend: Any,
) -> Any:
    """Pack the canonical per-foot ABI without flattening the bilateral axis."""

    groups = (
        (axial_force_n, 4, "axial_force_n"),
        (penetration_m, 4, "penetration_m"),
        (accelerometer_mps2, 3, "accelerometer_mps2"),
        (gyroscope_rps, 3, "gyroscope_rps"),
        (radar_frontend, 5, "radar_frontend"),
    )
    shapes = [_validate_group_shape(value, width, name) for value, width, name in groups]
    prefixes = {shape[:-1] for shape in shapes}
    if len(prefixes) != 1:
        raise ValueError("All sensor channel groups must share the same leading shape")

    torch_groups = [_is_torch_tensor(value) for value, _, _ in groups]
    if any(torch_groups):
        if not all(torch_groups):
            raise TypeError("Do not mix Torch and NumPy arrays in one packet")
        devices = {value.device for value, _, _ in groups}
        dtypes = {value.dtype for value, _, _ in groups}
        if len(devices) != 1 or len(dtypes) != 1:
            raise TypeError("All Torch sensor groups must share device and dtype")
        if not all(value.is_floating_point() for value, _, _ in groups):
            raise TypeError("Torch sensor groups must use floating-point dtype")
        packed = torch.cat([value for value, _, _ in groups], dim=-1)
    else:
        arrays = [np.asarray(value) for value, _, _ in groups]
        if not all(np.issubdtype(value.dtype, np.floating) for value in arrays):
            raise TypeError("NumPy sensor groups must use floating-point dtype")
        packed = np.concatenate(arrays, axis=-1)

    if tuple(packed.shape[-2:]) != (FOOT_COUNT, SENSOR_CHANNELS):
        raise AssertionError("Internal error: canonical packet shape was not produced")
    return packed


class CramponSensorAdapter:
    """Stateful 100 Hz packetizer with held-value dropout metadata.

    The adapter consumes only hardware-shaped channel groups. Simulator material
    parameters, contact vectors, labels, and truth canaries are intentionally not
    accepted by this API.
    """

    def __init__(self, config: CramponSensorConfig | None = None):
        self.config = config or CramponSensorConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._previous_values: Any | None = None
        self._sample_age_s: Any | None = None
        self._last_timestamp_s: Any | None = None

    @staticmethod
    def _copy(value: Any) -> Any:
        return value.clone() if _is_torch_tensor(value) else value.copy()

    @staticmethod
    def _timestamp_array(timestamp_s: Any, packet_values: Any) -> Any:
        expected_shape = tuple(packet_values.shape[:-1])
        if _is_torch_tensor(packet_values):
            if _is_torch_tensor(timestamp_s):
                result = timestamp_s.to(device=packet_values.device, dtype=packet_values.dtype)
            else:
                result = torch.as_tensor(
                    timestamp_s, device=packet_values.device, dtype=packet_values.dtype
                )
            if result.ndim == 0:
                result = result.expand(expected_shape)
            if tuple(result.shape) != expected_shape:
                raise ValueError(f"timestamp_s must have shape {expected_shape}")
            return result

        result = np.asarray(timestamp_s, dtype=packet_values.dtype)
        if result.ndim == 0:
            result = np.broadcast_to(result, expected_shape).copy()
        if tuple(result.shape) != expected_shape:
            raise ValueError(f"timestamp_s must have shape {expected_shape}")
        return result

    @staticmethod
    def _fresh_mask_array(fresh_mask: Any, packet_values: Any) -> Any:
        expected_shape = tuple(packet_values.shape)
        if _is_torch_tensor(packet_values):
            if not _is_torch_tensor(fresh_mask):
                raise TypeError("fresh_mask must be a Torch tensor for Torch packets")
            if fresh_mask.dtype != torch.bool:
                raise TypeError("fresh_mask must have boolean dtype")
            result = fresh_mask.to(device=packet_values.device)
        else:
            result = np.asarray(fresh_mask)
            if result.dtype != np.dtype(bool):
                raise TypeError("fresh_mask must have boolean dtype")
        if tuple(result.shape) != expected_shape:
            raise ValueError(f"fresh_mask must have shape {expected_shape}")
        return result

    def _random_fresh_mask(self, packet_values: Any) -> Any:
        shape = tuple(packet_values.shape)
        if _is_torch_tensor(packet_values):
            fresh = torch.ones(shape, dtype=torch.bool, device=packet_values.device)
            dropped = torch.rand(shape[:-1], device=packet_values.device) < (
                self.config.sample_drop_probability
            )
            fresh[..., :8] &= ~dropped.unsqueeze(-1)
            return fresh

        fresh = np.ones(shape, dtype=bool)
        dropped = self._rng.random(shape[:-1]) < self.config.sample_drop_probability
        fresh[..., :8] &= ~dropped[..., None]
        return fresh

    def _check_cadence(self, timestamp_s: Any) -> Any:
        delta = timestamp_s - self._last_timestamp_s
        period = self.config.packet_period_s
        tolerance = self.config.cadence_tolerance_s
        if _is_torch_tensor(timestamp_s):
            expected = torch.full_like(delta, period)
            matches = torch.isclose(delta, expected, rtol=0.0, atol=tolerance)
            if not bool(torch.all(matches).item()):
                raise ValueError("Packet timestamps must advance by exactly the configured period")
            return delta

        if not np.allclose(delta, period, rtol=0.0, atol=tolerance):
            raise ValueError("Packet timestamps must advance by exactly the configured period")
        return delta

    def observe(
        self,
        *,
        axial_force_n: Any,
        penetration_m: Any,
        accelerometer_mps2: Any,
        gyroscope_rps: Any,
        radar_frontend: Any,
        timestamp_s: Any,
        fresh_mask: Any | None = None,
        context: Mapping[str, Any] | None = None,
        commands: Mapping[str, Any] | None = None,
    ) -> VisibleSensorBatch:
        packet_values = pack_channel_groups(
            axial_force_n,
            penetration_m,
            accelerometer_mps2,
            gyroscope_rps,
            radar_frontend,
        )
        timestamp = self._timestamp_array(timestamp_s, packet_values)

        if self._previous_values is None:
            if fresh_mask is not None:
                first_mask = self._fresh_mask_array(fresh_mask, packet_values)
                if _is_torch_tensor(first_mask):
                    all_fresh = bool(torch.all(first_mask).item())
                else:
                    all_fresh = bool(np.all(first_mask))
                if not all_fresh:
                    raise ValueError("The first packet cannot hold values before a sample exists")
            if _is_torch_tensor(packet_values):
                valid_mask = torch.ones_like(packet_values, dtype=torch.bool)
                sample_age = torch.zeros_like(packet_values)
            else:
                valid_mask = np.ones_like(packet_values, dtype=bool)
                sample_age = np.zeros_like(packet_values)
            visible_values = packet_values
        else:
            if tuple(packet_values.shape) != tuple(self._previous_values.shape):
                raise ValueError("Packet batch shape cannot change without resetting the adapter")
            if _is_torch_tensor(packet_values) != _is_torch_tensor(self._previous_values):
                raise TypeError("Packet backend cannot change without resetting the adapter")
            delta = self._check_cadence(timestamp)
            valid_mask = (
                self._fresh_mask_array(fresh_mask, packet_values)
                if fresh_mask is not None
                else self._random_fresh_mask(packet_values)
            )
            if _is_torch_tensor(packet_values):
                visible_values = torch.where(valid_mask, packet_values, self._previous_values)
                age_increment = delta.unsqueeze(-1)
                sample_age = torch.where(
                    valid_mask,
                    torch.zeros_like(packet_values),
                    self._sample_age_s + age_increment,
                )
            else:
                visible_values = np.where(valid_mask, packet_values, self._previous_values)
                age_increment = delta[..., None]
                sample_age = np.where(
                    valid_mask,
                    np.zeros_like(packet_values),
                    self._sample_age_s + age_increment,
                )

        result = VisibleSensorBatch(
            packet_values=self._copy(visible_values),
            valid_mask=self._copy(valid_mask),
            timestamp_s=self._copy(timestamp),
            sample_age_s=self._copy(sample_age),
            context=context or {},
            commands=commands or {},
        )
        self._previous_values = self._copy(visible_values)
        self._sample_age_s = self._copy(sample_age)
        self._last_timestamp_s = self._copy(timestamp)
        return result
