from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Proprioception, SynchronizedSensorPacket
from .physics import ProbeEpisodeTruth


@dataclass(frozen=True)
class SensorNoiseConfig:
    force_noise_std_n: float = 1.2
    force_quantization_n: float = 0.25
    force_saturation_n: float = 300.0
    depth_noise_std_m: float = 0.00035
    depth_quantization_m: float = 0.0005
    imu_accel_noise_std_mps2: float = 0.12
    imu_gyro_noise_std_rps: float = 0.012
    radar_resolution_m: float = 0.04
    sample_drop_probability: float = 0.003


class SensorSimulator:
    """Converts simulator truth to values that the planned v1 hardware can measure."""

    def __init__(self, config: SensorNoiseConfig | None = None):
        self.config = config or SensorNoiseConfig()

    @staticmethod
    def _quantize(values: np.ndarray, quantum: float) -> np.ndarray:
        return np.round(values / quantum) * quantum

    def packets(self, truth: ProbeEpisodeTruth, *, seed: int) -> list[SynchronizedSensorPacket]:
        cfg = self.config
        rng = np.random.default_rng(seed)
        force_bias = rng.normal(0.0, 0.8, size=4)
        depth_bias = rng.normal(0.0, 0.00025, size=4)
        accel_bias = rng.normal(0.0, 0.06, size=3)
        gyro_bias = rng.normal(0.0, 0.006, size=3)

        # Five decoded frontend values. Thin crusts are intentionally below range resolution.
        radar = truth.radar_frontend_truth.copy()
        radar[:2] = self._quantize(radar[:2] + rng.normal(0.0, 0.012, size=2), cfg.radar_resolution_m)
        radar[2] = np.clip(radar[2] + rng.normal(0.0, 0.05), 0.0, 1.0)
        radar[3] = np.clip(0.82 * radar[3] + rng.normal(0.08, 0.12), 0.0, 1.0)
        radar[4] = np.clip(0.02 + abs(rng.normal(0.0, 0.025)), 0.0, 0.2)

        packets: list[SynchronizedSensorPacket] = []
        previous_force = np.zeros(4)
        previous_depth = np.zeros(4)
        for idx, timestamp in enumerate(truth.timestamps_s):
            valid_mask = np.ones(19, dtype=bool)
            force = truth.axial_force_n[idx] + force_bias + rng.normal(0.0, cfg.force_noise_std_n, size=4)
            force = 0.72 * force + 0.28 * previous_force
            force = self._quantize(np.clip(force, 0.0, cfg.force_saturation_n), cfg.force_quantization_n)
            depth = truth.penetration_m[idx] + depth_bias + rng.normal(0.0, cfg.depth_noise_std_m, size=4)
            depth = self._quantize(np.clip(depth, 0.0, None), cfg.depth_quantization_m)

            if idx > 0 and rng.random() < cfg.sample_drop_probability:
                force = previous_force.copy()
                depth = previous_depth.copy()
                valid_mask[:8] = False
            previous_force, previous_depth = force, depth

            accel = truth.accelerometer_mps2[idx] + accel_bias + rng.normal(
                0.0, cfg.imu_accel_noise_std_mps2, size=3
            )
            gyro = truth.gyroscope_rps[idx] + gyro_bias + rng.normal(
                0.0, cfg.imu_gyro_noise_std_rps, size=3
            )
            packet = SynchronizedSensorPacket(
                timestamp_s=float(timestamp),
                axial_force_n=force,
                penetration_m=depth,
                accelerometer_mps2=np.clip(accel, -80.0, 80.0),
                gyroscope_rps=np.clip(gyro, -20.0, 20.0),
                radar_frontend=radar.copy(),
                valid_mask=valid_mask,
                proprioception=Proprioception(
                    foot_position_xyz_m=np.array([truth.x_m, truth.y_m, -float(depth.mean())]),
                    foot_velocity_xyz_mps=np.array([0.0, 0.0, -float(truth.commanded_speed_mps[idx])]),
                    pelvis_roll_pitch_yaw_rad=np.zeros(3),
                    commanded_probe_load_n=float(truth.commanded_load_n[idx]),
                    commanded_foot_speed_mps=float(truth.commanded_speed_mps[idx]),
                    body_weight_on_foot_n=float(truth.body_weight_on_foot_n[idx]),
                ),
            )
            packet.vector()  # enforce the contract when producing data
            packets.append(packet)
        return packets
