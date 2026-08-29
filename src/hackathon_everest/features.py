from __future__ import annotations

import numpy as np

from .models import SynchronizedSensorPacket

FORCE_STATS = ("final", "max", "mean", "std", "slope")
DEPTH_STATS = ("final", "max", "mean", "std", "slope")
IMU_STATS = ("mean", "std", "max_abs")


def _linear_slope(values: np.ndarray, times: np.ndarray) -> float:
    if len(times) < 2 or times[-1] <= times[0]:
        return 0.0
    return float(np.polyfit(times, values, 1)[0])


def feature_names() -> list[str]:
    names: list[str] = []
    for spike in range(4):
        names.extend(f"force_{spike}_{stat}" for stat in FORCE_STATS)
    for spike in range(4):
        names.extend(f"depth_{spike}_{stat}" for stat in DEPTH_STATS)
    for axis in ("ax", "ay", "az", "gx", "gy", "gz"):
        names.extend(f"imu_{axis}_{stat}" for stat in IMU_STATS)
    names.extend(
        [
            "radar_support_depth",
            "radar_second_interface_depth",
            "radar_return_strength",
            "radar_void_probability",
            "radar_uncertainty",
            "commanded_load_final",
            "commanded_speed_mean",
            "body_load_final",
            "window_duration_s",
            "valid_force_ratio",
            "valid_depth_ratio",
            "valid_imu_ratio",
            "valid_radar_ratio",
        ]
    )
    names.extend(f"spike_{spike}_force_depth_slope" for spike in range(4))
    names.extend(
        [
            "force_asymmetry_final",
            "depth_asymmetry_final",
            "total_force_final",
            "total_force_peak",
            "force_depth_energy",
        ]
    )
    return names


def extract_window_features(packets: list[SynchronizedSensorPacket]) -> np.ndarray:
    if not packets:
        raise ValueError("At least one synchronized packet is required")
    times = np.asarray([packet.timestamp_s for packet in packets], dtype=float)
    times = times - times[0]
    force = np.stack([packet.axial_force_n for packet in packets])
    depth = np.stack([packet.penetration_m for packet in packets])
    imu = np.stack(
        [np.concatenate([packet.accelerometer_mps2, packet.gyroscope_rps]) for packet in packets]
    )

    features: list[float] = []
    for values in force.T:
        features.extend([values[-1], values.max(), values.mean(), values.std(), _linear_slope(values, times)])
    for values in depth.T:
        features.extend([values[-1], values.max(), values.mean(), values.std(), _linear_slope(values, times)])
    for values in imu.T:
        features.extend([values.mean(), values.std(), np.abs(values).max()])

    features.extend(packets[-1].radar_frontend.tolist())
    validity = np.stack([packet.valid_mask for packet in packets])
    features.extend(
        [
            packets[-1].proprioception.commanded_probe_load_n,
            float(np.mean([p.proprioception.commanded_foot_speed_mps for p in packets])),
            packets[-1].proprioception.body_weight_on_foot_n,
            times[-1] if len(times) > 1 else 0.0,
            float(validity[:, :4].mean()),
            float(validity[:, 4:8].mean()),
            float(validity[:, 8:14].mean()),
            float(validity[:, 14:19].mean()),
        ]
    )
    for spike in range(4):
        valid = np.ptp(depth[:, spike]) > 1e-5
        features.append(float(np.polyfit(depth[:, spike], force[:, spike], 1)[0]) if valid else 0.0)

    front_force = force[-1, 0] + force[-1, 1]
    rear_force = force[-1, 2] + force[-1, 3]
    front_depth = depth[-1, 0] + depth[-1, 1]
    rear_depth = depth[-1, 2] + depth[-1, 3]
    total_force = force.sum(axis=1)
    mean_depth = depth.mean(axis=1)
    energy = float(np.trapezoid(total_force, mean_depth)) if len(packets) > 1 else 0.0
    features.extend(
        [
            front_force - rear_force,
            front_depth - rear_depth,
            total_force[-1],
            total_force.max(),
            energy,
        ]
    )
    result = np.asarray(features, dtype=float)
    expected = len(feature_names())
    if result.shape != (expected,):
        raise RuntimeError(f"Feature contract drift: expected {expected}, got {result.shape}")
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
