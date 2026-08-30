from __future__ import annotations

from collections.abc import Sequence

import torch

SENSOR_FAULT_MODES = (
    "nominal",
    "one_force_stale",
    "one_probe_saturated",
    "imu_bias_burst",
    "radar_interface_merge",
    "packet_latency_burst",
)


def balanced_fault_codes_by_group(
    group_ids: Sequence[str], *, device: torch.device | str
) -> torch.Tensor:
    """Assign every fault mode round-robin within each named physical group."""

    if not group_ids:
        raise ValueError("group_ids cannot be empty")
    counters: dict[str, int] = {}
    codes = []
    for group_id in group_ids:
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group_ids must contain non-empty strings")
        count = counters.get(group_id, 0)
        codes.append(count % len(SENSOR_FAULT_MODES))
        counters[group_id] = count + 1
    return torch.tensor(codes, dtype=torch.long, device=device)


def apply_sensor_faults(
    force_n: torch.Tensor,
    penetration_m: torch.Tensor,
    acceleration_mps2: torch.Tensor,
    gyro_rps: torch.Tensor,
    radar_frontend: torch.Tensor,
    fault_cycle: torch.Tensor,
    *,
    sample_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the six authored sensor-fault modes without changing the bilateral ABI."""

    num_envs = force_n.shape[0]
    expected = {
        "force_n": (num_envs, 2, 4),
        "penetration_m": (num_envs, 2, 4),
        "acceleration_mps2": (num_envs, 2, 3),
        "gyro_rps": (num_envs, 2, 3),
        "radar_frontend": (num_envs, 2, 5),
        "fault_cycle": (num_envs,),
    }
    for name, value in {
        "force_n": force_n,
        "penetration_m": penetration_m,
        "acceleration_mps2": acceleration_mps2,
        "gyro_rps": gyro_rps,
        "radar_frontend": radar_frontend,
        "fault_cycle": fault_cycle,
    }.items():
        if tuple(value.shape) != expected[name]:
            raise ValueError(f"{name} must have shape {expected[name]}")
    sensor_force = force_n.clone()
    sensor_penetration = penetration_m.clone()
    sensor_acceleration = acceleration_mps2.clone()
    sensor_gyro = gyro_rps.clone()
    sensor_radar = radar_frontend.clone()
    fresh_mask = torch.ones((num_envs, 2, 19), dtype=torch.bool, device=force_n.device)
    environment_ids = torch.arange(num_envs, device=force_n.device)
    fault_code = (environment_ids + fault_cycle) % len(SENSOR_FAULT_MODES)

    stale_force = fault_code == 1
    fresh_mask[stale_force, 0, 0] = False
    saturated = fault_code == 2
    sensor_force[saturated, 0, 1] = 700.0
    sensor_penetration[saturated, 0, 1] = 0.055
    if 20 <= sample_index % 100 < 40:
        imu_bias = fault_code == 3
        sensor_acceleration[imu_bias] += torch.tensor((0.8, -0.5, 0.3), device=force_n.device)
        sensor_gyro[imu_bias] += torch.tensor((0.05, -0.04, 0.03), device=force_n.device)
    merged = fault_code == 4
    sensor_radar[merged, :, 1] = sensor_radar[merged, :, 0]
    sensor_radar[merged, :, 3] = 0.5 * (sensor_radar[merged, :, 2] + sensor_radar[merged, :, 3])
    latency_burst = (fault_code == 5) & (sample_index % 100 < 15)
    fresh_mask[latency_burst] = False
    return (
        sensor_force,
        sensor_penetration,
        sensor_acceleration,
        sensor_gyro,
        sensor_radar,
        fresh_mask,
    )
