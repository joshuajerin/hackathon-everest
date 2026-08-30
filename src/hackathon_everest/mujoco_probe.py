from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import SENSOR_CHANNELS, Proprioception, SynchronizedSensorPacket

EXPECTED_SENSOR_LAYOUT = (
    ("spike_0_axial_force", 1),
    ("spike_1_axial_force", 1),
    ("spike_2_axial_force", 1),
    ("spike_3_axial_force", 1),
    ("spike_0_penetration", 1),
    ("spike_1_penetration", 1),
    ("spike_2_penetration", 1),
    ("spike_3_penetration", 1),
    ("crampon_accelerometer", 3),
    ("crampon_gyroscope", 3),
    ("radar_frontend", 5),
)


@dataclass(frozen=True)
class MujocoProbeRun:
    packets: list[SynchronizedSensorPacket]
    target_load_n: np.ndarray
    carriage_position_m: np.ndarray
    lateral_position_m: np.ndarray
    contact_geom_pairs: tuple[tuple[str, str], ...]
    model_path: Path
    timestep_s: float
    slope_deg: float
    lateral_drive_force_n: float

    @property
    def sensor_matrix(self) -> np.ndarray:
        return np.vstack([packet.vector() for packet in self.packets])

    def report(self) -> dict[str, Any]:
        matrix = self.sensor_matrix
        tail = max(1, len(matrix) // 5)
        force_tail = matrix[-tail:, :4]
        total_tail = force_tail.sum(axis=1)
        per_spike = force_tail.mean(axis=0)
        mean_load = float(total_tail.mean())
        return {
            "model": str(self.model_path),
            "physics_timestep_s": self.timestep_s,
            "packet_rate_hz": round(
                1.0 / np.median(np.diff([p.timestamp_s for p in self.packets])), 3
            ),
            "packet_count": len(self.packets),
            "sensor_channels": int(matrix.shape[1]),
            "target_load_n": float(self.target_load_n[-1]),
            "slope_deg": self.slope_deg,
            "lateral_drive_force_n": self.lateral_drive_force_n,
            "final_lateral_position_m": float(self.lateral_position_m[-1]),
            "lateral_travel_m": float(self.lateral_position_m[-1] - self.lateral_position_m[0]),
            "steady_total_force_mean_n": mean_load,
            "steady_total_force_std_n": float(total_tail.std()),
            "steady_load_error_percent": float(
                100.0 * abs(mean_load - self.target_load_n[-1]) / max(self.target_load_n[-1], 1e-9)
            ),
            "steady_per_spike_force_n": per_spike.tolist(),
            "steady_force_balance_cv": float(per_spike.std() / max(per_spike.mean(), 1e-9)),
            "final_probe_travel_m": matrix[-1, 4:8].tolist(),
            "contact_geom_pairs": [list(pair) for pair in self.contact_geom_pairs],
            "contract": "4 axial force + 4 moving-probe travel + 6 IMU + 5 radar frontend = 19",
            "caveat": (
                "The inclined rigid-plane fixture validates geometry, axial-force projection, and sensor wiring. "
                "Its condim=1 plane is not a validated ice-friction or brittle-fracture model."
            ),
        }


def _mujoco_module():
    try:
        import mujoco
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install the MuJoCo extra with: uv sync --extra mujoco") from error
    return mujoco


def _validate_sensor_layout(model: Any, mujoco: Any) -> None:
    actual = []
    for sensor_id in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        actual.append((name, int(model.sensor_dim[sensor_id])))
    if tuple(actual) != EXPECTED_SENSOR_LAYOUT:
        raise ValueError(f"Unexpected MuJoCo sensor layout: {actual}")
    if model.nsensordata != SENSOR_CHANNELS:
        raise ValueError(f"Expected {SENSOR_CHANNELS} sensor values, got {model.nsensordata}")


def _contact_pairs(model: Any, data: Any, mujoco: Any) -> tuple[tuple[str, str], ...]:
    pairs = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        pairs.add((first, second))
    return tuple(sorted(pairs))


def _plane_normal(slope_deg: float) -> np.ndarray:
    """Return the world normal for a plane pitched about the world y axis."""
    angle = float(np.deg2rad(slope_deg))
    return np.array([np.sin(angle), 0.0, np.cos(angle)])


def _configure_plane_slope(model: Any, mujoco: Any, slope_deg: float) -> np.ndarray:
    if not -30.0 <= slope_deg <= 30.0:
        raise ValueError("slope_deg must be between -30 and 30 degrees")
    plane_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ice_plane")
    half_angle = float(np.deg2rad(slope_deg) / 2.0)
    model.geom_quat[plane_id] = [np.cos(half_angle), 0.0, np.sin(half_angle), 0.0]
    return _plane_normal(slope_deg)


def _joint_addresses(model: Any, mujoco: Any, name: str) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _axial_contact_forces(model: Any, data: Any, mujoco: Any) -> np.ndarray:
    """Project solved contact normals onto the fixed spike axes.

    A MuJoCo touch sensor reports contact magnitude. On a slope that is not the
    axial load seen by a vertically mounted spike, so the estimator packet uses
    this explicit projection instead.
    """
    geom_to_spike = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"probe_{index}_geom"): index
        for index in range(4)
    }
    axial = np.zeros(4)
    spike_axis = np.array([0.0, 0.0, 1.0])
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        spike_index = geom_to_spike.get(contact.geom1, geom_to_spike.get(contact.geom2))
        if spike_index is None:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_index, force)
        normal_world = contact.frame[:3]
        axial[spike_index] += abs(float(force[0] * np.dot(normal_world, spike_axis)))
    return axial


def run_mujoco_probe(
    model_path: str | Path,
    *,
    duration_s: float = 1.0,
    ramp_s: float = 0.30,
    target_load_n: float = 150.0,
    packet_rate_hz: float = 100.0,
    radar_frontend: np.ndarray | None = None,
    slope_deg: float = 0.0,
    lateral_drive_force_n: float = 0.0,
) -> MujocoProbeRun:
    """Run a four-probe rigid-plane fixture, optionally pitched and laterally driven.

    The incline is a geometry/sensor gate only. Native MuJoCo plane contact is
    deliberately not presented as a brittle-ice or traction model; use
    :func:`run_hybrid_ice_probe` for the stateful material law.
    """
    if duration_s <= 0 or ramp_s <= 0 or target_load_n <= 0 or packet_rate_hz <= 0:
        raise ValueError("duration, ramp, target load, and packet rate must be positive")
    if not np.isfinite(lateral_drive_force_n):
        raise ValueError("lateral_drive_force_n must be finite")

    mujoco = _mujoco_module()
    model_path = Path(model_path).resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    _configure_plane_slope(model, mujoco, slope_deg)
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    _validate_sensor_layout(model, mujoco)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    sample_stride = max(1, round(1.0 / (model.opt.timestep * packet_rate_hz)))
    radar = np.asarray(
        radar_frontend if radar_frontend is not None else [0.0, 0.50, 0.95, 0.0, 0.02],
        dtype=float,
    )
    if radar.shape != (5,):
        raise ValueError("radar_frontend must have five decoded values")

    total_weight_n = float(mujoco.mj_getTotalmass(model) * abs(model.opt.gravity[2]))
    packets: list[SynchronizedSensorPacket] = []
    target_history: list[float] = []
    carriage_history: list[float] = []
    lateral_history: list[float] = []
    contact_pairs: set[tuple[str, str]] = set()
    step_count = int(np.ceil(duration_s / model.opt.timestep))
    carriage_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe_carriage")
    carriage_x_qpos, carriage_x_dof = _joint_addresses(model, mujoco, "carriage_x")
    carriage_z_qpos, carriage_z_dof = _joint_addresses(model, mujoco, "carriage_z")

    for step in range(step_count):
        desired_load = target_load_n * min(data.time / ramp_s, 1.0)
        # Positive joint force is upward. Gravity compensation minus desired load
        # makes the commanded load equal the eventual vertical reaction on flat ground.
        data.ctrl[0] = total_weight_n - desired_load
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[carriage_x_dof] = lateral_drive_force_n
        mujoco.mj_step(model, data)
        data.sensordata[14:19] = radar
        contact_pairs.update(_contact_pairs(model, data, mujoco))
        if step % sample_stride != 0:
            continue

        vector = data.sensordata.copy()
        vector[:4] = _axial_contact_forces(model, data, mujoco)
        position = data.xpos[carriage_body_id].copy()
        velocity = data.cvel[carriage_body_id, 3:6].copy()
        packet = SynchronizedSensorPacket(
            timestamp_s=float(data.time),
            axial_force_n=vector[:4],
            penetration_m=vector[4:8],
            accelerometer_mps2=vector[8:11],
            gyroscope_rps=vector[11:14],
            radar_frontend=vector[14:19],
            valid_mask=np.ones(SENSOR_CHANNELS, dtype=bool),
            proprioception=Proprioception(
                foot_position_xyz_m=position,
                foot_velocity_xyz_mps=velocity,
                pelvis_roll_pitch_yaw_rad=np.zeros(3),
                commanded_probe_load_n=float(desired_load),
                commanded_foot_speed_mps=float(abs(data.qvel[carriage_z_dof])),
                body_weight_on_foot_n=float(vector[:4].sum()),
            ),
        )
        packet.vector()
        packets.append(packet)
        target_history.append(float(desired_load))
        carriage_history.append(float(data.qpos[carriage_z_qpos]))
        lateral_history.append(float(data.qpos[carriage_x_qpos]))

    return MujocoProbeRun(
        packets=packets,
        target_load_n=np.asarray(target_history),
        carriage_position_m=np.asarray(carriage_history),
        lateral_position_m=np.asarray(lateral_history),
        contact_geom_pairs=tuple(sorted(contact_pairs)),
        model_path=model_path,
        timestep_s=float(model.opt.timestep),
        slope_deg=float(slope_deg),
        lateral_drive_force_n=float(lateral_drive_force_n),
    )


def save_mujoco_probe(run: MujocoProbeRun, out_dir: str | Path) -> dict[str, Any]:
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run.report()
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        out_dir / "packets.npz",
        timestamps_s=np.asarray([packet.timestamp_s for packet in run.packets]),
        sensor_values=run.sensor_matrix,
        valid_masks=np.vstack([packet.valid_mask for packet in run.packets]),
        target_load_n=run.target_load_n,
        carriage_position_m=run.carriage_position_m,
        lateral_position_m=run.lateral_position_m,
        slope_deg=np.asarray(run.slope_deg),
        lateral_drive_force_n=np.asarray(run.lateral_drive_force_n),
    )
    return report
