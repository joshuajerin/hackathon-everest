from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np

SENSOR_CHANNELS = 19


class FootSide(StrEnum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class Proprioception:
    """G1 context paired with each crampon sample; not extra crampon channels."""

    foot_position_xyz_m: np.ndarray
    foot_velocity_xyz_mps: np.ndarray
    pelvis_roll_pitch_yaw_rad: np.ndarray
    commanded_probe_load_n: float
    commanded_foot_speed_mps: float
    body_weight_on_foot_n: float


@dataclass(frozen=True)
class SynchronizedSensorPacket:
    """The hardware-realistic 19-channel crampon contract at one timestamp."""

    timestamp_s: float
    axial_force_n: np.ndarray  # (4,), one scalar along each spike axis
    penetration_m: np.ndarray  # (4,), moving probe/collar displacement
    accelerometer_mps2: np.ndarray  # (3,)
    gyroscope_rps: np.ndarray  # (3,)
    radar_frontend: np.ndarray  # (5,), decoded radar values, not a raw A-scan
    valid_mask: np.ndarray  # (19,), false means held/imputed rather than a fresh sample
    proprioception: Proprioception

    def vector(self) -> np.ndarray:
        values = np.concatenate(
            [
                np.asarray(self.axial_force_n, dtype=float),
                np.asarray(self.penetration_m, dtype=float),
                np.asarray(self.accelerometer_mps2, dtype=float),
                np.asarray(self.gyroscope_rps, dtype=float),
                np.asarray(self.radar_frontend, dtype=float),
            ]
        )
        if values.shape != (SENSOR_CHANNELS,):
            raise ValueError(f"Expected {SENSOR_CHANNELS} crampon channels, got {values.shape}")
        mask = np.asarray(self.valid_mask, dtype=bool)
        if mask.shape != (SENSOR_CHANNELS,):
            raise ValueError(f"Expected a {SENSOR_CHANNELS}-value validity mask, got {mask.shape}")
        return values


@dataclass
class FootTerrainEstimate:
    support_layer_depth_m: float
    void_probability: float
    fracture_probability: float
    slip_probability: float
    void_depth_m: float
    effective_vertical_stiffness_n_per_m: float
    effective_vertical_damping_ns_per_m: float
    bearing_capacity_n: float
    current_sinkage_m: float
    sinkage_rate_mps: float
    shear_capacity_n: float
    effective_friction: float
    slip_margin_n: float
    spike_engagement: np.ndarray
    spike_support_quality: np.ndarray
    center_of_support_xy: np.ndarray
    compaction_state: float
    damage_state: float
    fracture_margin_n: float
    uncertainty: np.ndarray

    def lower_confidence_bearing_n(self, sigma: float = 2.0) -> float:
        return float(self.bearing_capacity_n - sigma * self.uncertainty[3])

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items()}


@dataclass
class BilateralSupportState:
    left_current_load_n: float
    right_current_load_n: float
    left_support_reserve_n: float
    right_support_reserve_n: float
    left_sinkage_m: float
    right_sinkage_m: float
    height_difference_m: float
    total_support_margin_n: float
    support_polygon_margin_m: float
    maximum_safe_transfer_rate_nps: float


@dataclass
class StepDecision:
    action: str
    target_xy_m: np.ndarray
    conservative_score: float
    swing_clearance_m: float
    approach_velocity_mps: float
    probe_load_n: float
    load_transfer_rate_nps: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_xy_m"] = self.target_xy_m.tolist()
        return result
