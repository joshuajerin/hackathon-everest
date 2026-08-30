from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ice import IceContactParameters, StatefulIceSpikeContact
from .models import SENSOR_CHANNELS, Proprioception, SynchronizedSensorPacket
from .mujoco_probe import (
    _configure_plane_slope,
    _joint_addresses,
    _mujoco_module,
    _validate_sensor_layout,
)


@dataclass(frozen=True)
class HybridIceProbeRun:
    packets: list[SynchronizedSensorPacket]
    ice_penetration_m: np.ndarray
    ice_shear_capacity_n: np.ndarray
    ice_contact_force_n: np.ndarray
    fracture_state: np.ndarray
    parameters: tuple[IceContactParameters, ...]
    target_load_n: np.ndarray
    lateral_position_m: np.ndarray
    model_path: Path
    timestep_s: float
    slope_deg: float
    lateral_drive_force_n: float

    @property
    def sensor_matrix(self) -> np.ndarray:
        return np.vstack([packet.vector() for packet in self.packets])

    def report(self) -> dict[str, Any]:
        values = self.sensor_matrix
        tail = max(1, len(values) // 5)
        force = values[-tail:, :4]
        total_force = force.sum(axis=1)
        return {
            "model": str(self.model_path),
            "physics_timestep_s": self.timestep_s,
            "packet_count": len(self.packets),
            "sensor_channels": int(values.shape[1]),
            "target_load_n": float(self.target_load_n[-1]),
            "slope_deg": self.slope_deg,
            "lateral_drive_force_n": self.lateral_drive_force_n,
            "final_lateral_position_m": float(self.lateral_position_m[-1]),
            "lateral_travel_m": float(self.lateral_position_m[-1] - self.lateral_position_m[0]),
            "steady_total_force_mean_n": float(total_force.mean()),
            "steady_total_force_std_n": float(total_force.std()),
            "steady_per_spike_force_n": force.mean(axis=0).tolist(),
            "maximum_ice_penetration_m": self.ice_penetration_m.max(axis=0).tolist(),
            "final_probe_travel_m": values[-1, 4:8].tolist(),
            "final_shear_capacity_n": self.ice_shear_capacity_n[-1].tolist(),
            "final_contact_force_xyz_n": self.ice_contact_force_n[-1].tolist(),
            "fractured_by_spike": self.fracture_state[-1].astype(bool).tolist(),
            "domain_parameters": [asdict(parameters) for parameters in self.parameters],
            "estimator_input": "Only the 19 sensor values. Ice depth, material parameters, and fracture state are diagnostics/labels only.",
            "model_boundary": (
                "MuJoCo supplies rigid-body kinematics and probe mechanics. A stateful reduced-order law supplies "
                "temperature-dependent indentation, brittle force drop, ploughing, residual friction, and crater memory."
            ),
        }


def run_hybrid_ice_probe(
    model_path: str | Path,
    *,
    duration_s: float = 1.5,
    ramp_s: float = 0.40,
    target_load_n: float = 150.0,
    packet_rate_hz: float = 100.0,
    seed: int = 41,
    slope_deg: float = 0.0,
    lateral_drive_force_n: float = 0.0,
) -> HybridIceProbeRun:
    """Run a stateful ice probe on an optional incline with a lateral drive.

    MuJoCo supplies the probe kinematics. The rigid plane is disabled and the
    hybrid law applies normal and bounded tangential forces at each tip. This
    makes slope and lateral-drive results material-law diagnostics, rather than
    treating a low native Coulomb coefficient as brittle ice.
    """
    if duration_s <= 0 or ramp_s <= 0 or target_load_n <= 0 or packet_rate_hz <= 0:
        raise ValueError("duration, ramp, target load, and packet rate must be positive")
    if not np.isfinite(lateral_drive_force_n):
        raise ValueError("lateral_drive_force_n must be finite")

    mujoco = _mujoco_module()
    model_path = Path(model_path).resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    slope_normal = _configure_plane_slope(model, mujoco, slope_deg)
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    _validate_sensor_layout(model, mujoco)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    plane_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ice_plane")
    model.geom_contype[plane_id] = 0
    model.geom_conaffinity[plane_id] = 0
    tip_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"probe_{index}_tip")
        for index in range(4)
    ]
    probe_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"probe_{index}") for index in range(4)
    ]
    probe_geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"probe_{index}_geom")
        for index in range(4)
    ]
    probe_radii_m = np.asarray([model.geom_size[geom_id, 0] for geom_id in probe_geom_ids])
    carriage_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe_carriage")
    carriage_x_qpos, carriage_x_dof = _joint_addresses(model, mujoco, "carriage_x")
    _, carriage_z_dof = _joint_addresses(model, mujoco, "carriage_z")
    material = tuple(
        StatefulIceSpikeContact(IceContactParameters.sample(seed + i)) for i in range(4)
    )
    total_weight_n = float(mujoco.mj_getTotalmass(model) * abs(model.opt.gravity[2]))
    radar = np.array([0.0, 0.50, 0.97, 0.0, 0.02])
    sample_stride = max(1, round(1.0 / (model.opt.timestep * packet_rate_hz)))
    step_count = int(np.ceil(duration_s / model.opt.timestep))
    previous_depth = np.zeros(4)

    packets: list[SynchronizedSensorPacket] = []
    depths: list[np.ndarray] = []
    shear: list[np.ndarray] = []
    contact_forces_history: list[np.ndarray] = []
    fractured: list[np.ndarray] = []
    targets: list[float] = []
    lateral_positions: list[float] = []
    mujoco.mj_forward(model, data)
    previous_tip_positions = np.asarray(
        [data.site_xpos[site_id] for site_id in tip_site_ids]
    ).copy()
    vertical_axis = np.array([0.0, 0.0, 1.0])
    drive_force = np.array([lateral_drive_force_n, 0.0, 0.0])
    drive_tangent = drive_force - slope_normal * float(np.dot(drive_force, slope_normal))

    for step in range(step_count):
        desired_load = target_load_n * min(data.time / ramp_s, 1.0)
        # The vertical load command also has a downhill component on an incline.
        # Shear must oppose that component in addition to the explicit sweep load.
        vertical_load = np.array([0.0, 0.0, -desired_load])
        load_tangent = vertical_load - slope_normal * float(np.dot(vertical_load, slope_normal))
        data.qfrc_applied[:] = 0.0
        tip_positions = np.asarray([data.site_xpos[site_id] for site_id in tip_site_ids]).copy()
        # Signed distance uses each fitted capsule's rounded-end radius along
        # the inclined plane normal, not a hard-coded world-z shortcut.
        signed_tip_clearance = tip_positions @ slope_normal - probe_radii_m
        ice_depth = np.clip(-signed_tip_clearance, 0.0, 0.04)
        penetration_rate = (ice_depth - previous_depth) / model.opt.timestep
        tip_velocity = (tip_positions - previous_tip_positions) / model.opt.timestep
        tangent_velocity = tip_velocity - np.outer(tip_velocity @ slope_normal, slope_normal)
        responses = [
            contact.step(
                float(ice_depth[index]),
                float(penetration_rate[index]),
                lateral_speed_mps=float(np.linalg.norm(tangent_velocity[index])),
                dt_s=float(model.opt.timestep),
            )
            for index, contact in enumerate(material)
        ]
        normal_forces = np.asarray([response.normal_force_n for response in responses])
        shear_capacities = np.asarray([response.shear_capacity_n for response in responses])
        tangent_demand = load_tangent + drive_tangent
        demand_magnitude = float(np.linalg.norm(tangent_demand))
        if demand_magnitude > 1e-12 and shear_capacities.sum() > 0.0:
            force_fraction = min(1.0, demand_magnitude / float(shear_capacities.sum()))
            shear_forces = -np.outer(
                shear_capacities * force_fraction,
                tangent_demand / demand_magnitude,
            )
        else:
            shear_forces = np.zeros((4, 3))
        contact_forces = normal_forces[:, None] * slope_normal + shear_forces
        for index, force_vector in enumerate(contact_forces):
            mujoco.mj_applyFT(
                model,
                data,
                force_vector,
                np.zeros(3),
                tip_positions[index],
                probe_body_ids[index],
                data.qfrc_applied,
            )
        data.qfrc_applied[carriage_x_dof] += lateral_drive_force_n
        data.ctrl[0] = total_weight_n - desired_load
        mujoco.mj_step(model, data)
        # Only hardware-shaped axial loads enter the estimator. Material state,
        # vector contact force, and plane-normal depth are held out as truth.
        axial_forces = np.clip(contact_forces @ vertical_axis, 0.0, None)
        data.sensordata[:4] = axial_forces
        data.sensordata[14:19] = radar
        previous_depth = ice_depth
        previous_tip_positions = tip_positions
        if step % sample_stride != 0:
            continue

        vector = data.sensordata.copy()
        packet = SynchronizedSensorPacket(
            timestamp_s=float(data.time),
            axial_force_n=vector[:4],
            penetration_m=vector[4:8],
            accelerometer_mps2=vector[8:11],
            gyroscope_rps=vector[11:14],
            radar_frontend=vector[14:19],
            valid_mask=np.ones(SENSOR_CHANNELS, dtype=bool),
            proprioception=Proprioception(
                foot_position_xyz_m=data.xpos[carriage_body_id].copy(),
                foot_velocity_xyz_mps=data.cvel[carriage_body_id, 3:6].copy(),
                pelvis_roll_pitch_yaw_rad=np.zeros(3),
                commanded_probe_load_n=float(desired_load),
                commanded_foot_speed_mps=float(abs(data.qvel[carriage_z_dof])),
                body_weight_on_foot_n=float(vector[:4].sum()),
            ),
        )
        packet.vector()
        packets.append(packet)
        depths.append(ice_depth.copy())
        shear.append(shear_capacities)
        contact_forces_history.append(contact_forces.copy())
        fractured.append(np.asarray([response.fractured for response in responses]))
        targets.append(float(desired_load))
        lateral_positions.append(float(data.qpos[carriage_x_qpos]))

    return HybridIceProbeRun(
        packets=packets,
        ice_penetration_m=np.vstack(depths),
        ice_shear_capacity_n=np.vstack(shear),
        ice_contact_force_n=np.stack(contact_forces_history),
        fracture_state=np.vstack(fractured),
        parameters=tuple(contact.parameters for contact in material),
        target_load_n=np.asarray(targets),
        lateral_position_m=np.asarray(lateral_positions),
        model_path=model_path,
        timestep_s=float(model.opt.timestep),
        slope_deg=float(slope_deg),
        lateral_drive_force_n=float(lateral_drive_force_n),
    )


def save_hybrid_ice_probe(run: HybridIceProbeRun, out_dir: str | Path) -> dict[str, Any]:
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
        lateral_position_m=run.lateral_position_m,
        slope_deg=np.asarray(run.slope_deg),
        lateral_drive_force_n=np.asarray(run.lateral_drive_force_n),
    )
    np.savez_compressed(
        out_dir / "simulator_truth_do_not_feed_estimator.npz",
        ice_penetration_m=run.ice_penetration_m,
        ice_shear_capacity_n=run.ice_shear_capacity_n,
        ice_contact_force_n=run.ice_contact_force_n,
        fracture_state=run.fracture_state,
    )
    return report
