from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any

import torch
from isaaclab.envs import ManagerBasedRLEnv

from hackathon_everest.everest_suite import assign_cases, load_suite

from ....contact import (
    BatchedCramponWrenchBridge,
    BatchedStatefulMaterial,
    IsaacArticulationWrenchAdapter,
    build_suite_material_parameters,
    suite_plane_normals,
)
from ....sensors import CramponSensorAdapter, CramponSensorConfig
from ....sensors.faults import (
    SENSOR_FAULT_MODES,
    apply_sensor_faults,
    balanced_fault_codes_by_group,
)


def _quaternion_wxyz_to_rpy(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(
        1e-8
    )
    w, x, y, z = quaternion.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    return torch.stack((roll, pitch, yaw), dim=-1)


class EverestStatefulCramponEnv(ManagerBasedRLEnv):
    """Manager environment with one custom crampon wrench call per 200 Hz physics step."""

    def __init__(self, cfg: Any, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        robot = self.scene["robot"]
        body_ids, body_names = robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )
        if body_names != ["left_ankle_roll_link", "right_ankle_roll_link"]:
            raise RuntimeError(f"Unexpected ankle body order: {body_names}")
        self._everest_ankle_body_ids = torch.as_tensor(
            body_ids, device=self.device, dtype=torch.long
        )
        suite_path = Path(cfg.everest_suite_config_path)
        suite = load_suite(suite_path)
        if cfg.everest_require_complete_coverage:
            self.everest_cases = assign_cases(
                suite, num_envs=self.num_envs, seed=cfg.everest_suite_seed
            )
        else:
            required = (
                len(suite["surfaces"])
                * len(suite["inclines_deg"])
                * len(suite["hazards"])
                * len(suite["contact_modes"])
            )
            candidate_cases = assign_cases(
                suite, num_envs=max(self.num_envs, required), seed=cfg.everest_suite_seed
            )
            if cfg.everest_play_surface_id:
                candidate_cases = [
                    case
                    for case in candidate_cases
                    if case.surface_id == cfg.everest_play_surface_id
                    and case.incline_deg == cfg.everest_play_incline_deg
                    and case.hazard_id == cfg.everest_play_hazard_id
                    and (
                        not cfg.everest_play_contact_mode_id
                        or case.contact_mode_id == cfg.everest_play_contact_mode_id
                    )
                ]
            if not candidate_cases:
                raise RuntimeError("No terrain cases matched the play selection")
            self.everest_cases = [
                candidate_cases[index % len(candidate_cases)] for index in range(self.num_envs)
            ]
        parameters = build_suite_material_parameters(
            suite, self.everest_cases, seed=cfg.everest_suite_seed, device=self.device
        )
        if cfg.everest_nominal_bootstrap_material:
            # Easy end of the project-authored hard-ice prior, used only as a
            # staged curriculum before randomized material adaptation.
            parameters.damping_ns_per_m.fill_(100.0)
            parameters.bearing_capacity_n.fill_(900.0)
            parameters.shear_capacity_n.fill_(450.0)
            parameters.friction.fill_(0.14)
            parameters.ice_indentation_pressure_pa.fill_(65_000_000.0)
            parameters.ice_fracture_energy_j.fill_(0.30)
            parameters.void_present.zero_()
        material = BatchedStatefulMaterial(parameters)
        probe_enabled = torch.ones(parameters.shape, dtype=torch.bool, device=self.device)
        spatial_void_bounds = torch.full(
            (self.num_envs, 2), float("nan"), dtype=torch.float32, device=self.device
        )
        contact_mode_rear_load_targets = {
            "all_points_flat_foot": 0.50,
            "hybrid_contact": 0.35,
            "front_point_contact": 0.20,
        }
        self._everest_target_rear_load_fraction = torch.tensor(
            [contact_mode_rear_load_targets[case.contact_mode_id] for case in self.everest_cases],
            dtype=torch.float32,
            device=self.device,
        )
        for environment, case in enumerate(self.everest_cases):
            if case.hazard_id == "open_crevasse_gap":
                # Collider gap is x=[-0.65, 0.65] around the environment origin.
                # The analytical plane origin is anchored four metres before it.
                relative_gap_center = -float(cfg.everest_terrain_anchor_x_m)
                spatial_void_bounds[environment] = torch.tensor(
                    (relative_gap_center - 0.65, relative_gap_center + 0.65),
                    device=self.device,
                )
        self.everest_wrench_bridge = BatchedCramponWrenchBridge(
            material,
            native_support_collisions_enabled=False,
            probe_enabled_mask=probe_enabled,
            spatial_void_x_bounds_m=spatial_void_bounds,
            virtual_travel_m=cfg.everest_virtual_travel_m,
        )
        self._everest_wrench_adapter = IsaacArticulationWrenchAdapter(
            robot, self._everest_ankle_body_ids
        )
        if cfg.everest_use_case_inclines:
            self._everest_terrain_normal = suite_plane_normals(self.everest_cases, self.device)
        else:
            self._everest_terrain_normal = torch.zeros((self.num_envs, 3), device=self.device)
            self._everest_terrain_normal[:, 2] = 1.0
        self._everest_terrain_origin = self.scene.env_origins.clone()
        manifest_path = os.environ.get("EVEREST_VECTOR_TERRAIN_MANIFEST")
        if manifest_path:
            manifest = json.loads(Path(manifest_path).expanduser().read_text())
            areas = manifest.get("areas")
            if not isinstance(areas, list) or len(areas) != self.num_envs:
                raise RuntimeError(
                    "Terrain manifest area count must exactly match the vector environment count"
                )
            expected_origins = torch.tensor(
                [area["origin_m"] for area in areas], dtype=torch.float32, device=self.device
            )
            mismatch_m = (expected_origins - self.scene.env_origins).abs().amax()
            if float(mismatch_m) > 1.0e-4:
                raise RuntimeError(
                    "Terrain manifest origins do not match Isaac Lab environment origins: "
                    f"maximum error {float(mismatch_m):.6f} m"
                )
        self._everest_terrain_origin[:, 0] += cfg.everest_terrain_anchor_x_m
        self._everest_sensor = CramponSensorAdapter(
            CramponSensorConfig(
                seed=cfg.everest_suite_seed,
                packet_rate_hz=100.0,
                sample_drop_probability=cfg.everest_sample_drop_probability,
            )
        )
        self.everest_sensor_frames: deque[Any] = deque(maxlen=16)
        self.everest_truth_frames: deque[dict[str, torch.Tensor]] = deque(maxlen=16)
        self.everest_latest_wrench = None
        self.everest_latest_sensor_frame = None
        self._everest_previous_ankle_velocity = torch.zeros(
            (self.num_envs, 2, 3), device=self.device
        )
        articulation_masses = robot.root_physx_view.get_masses().numpy()
        self._everest_robot_weight_n = float(articulation_masses[0].sum() * 9.81)
        self._everest_sensor_sample_index = 0
        self._everest_enable_sensor_fault_curriculum = bool(
            cfg.everest_enable_sensor_fault_curriculum
        )
        if tuple(suite["sensor_fault_modes"]) != SENSOR_FAULT_MODES:
            raise RuntimeError("Unexpected sensor fault mode ABI")
        # ManagerBasedEnv resets once after construction. In the complete suite,
        # balance all six faults independently within each contact mode instead
        # of confounding two fault modes with each of the three contact modes.
        environment_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if cfg.everest_require_complete_coverage:
            initial_fault_code = balanced_fault_codes_by_group(
                [case.contact_mode_id for case in self.everest_cases], device=self.device
            )
        else:
            initial_fault_code = environment_ids % len(SENSOR_FAULT_MODES)
        self._everest_initial_fault_code = initial_fault_code
        self._everest_fault_cycle = initial_fault_code - environment_ids - 1
        self._everest_pending_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Native 200 Hz gait-quality accumulators. They are diagnostic only and
        # never enter the deployable sensor packet or policy observation.
        self._everest_gait_previous_stance = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self._everest_gait_swing_active = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self._everest_gait_swing_peak_m = torch.zeros((self.num_envs, 2), device=self.device)
        self._everest_gait_time_s = torch.zeros(self.num_envs, device=self.device)
        self._everest_gait_last_touchdown_time_s = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self._everest_gait_last_touchdown_tangent_m = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self._everest_gait_has_touchdown = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self._everest_gait_last_touchdown_foot = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._everest_gait_swing_count = torch.zeros((self.num_envs, 2), device=self.device)
        self._everest_gait_swing_peak_sum_m = torch.zeros((self.num_envs, 2), device=self.device)
        self._everest_gait_low_swing_count = torch.zeros((self.num_envs, 2), device=self.device)
        self._everest_gait_stride_count = torch.zeros((self.num_envs, 2), device=self.device)
        self._everest_gait_stride_advance_sum_m = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self._everest_gait_stride_duration_sum_s = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self._everest_gait_stride_duration_sq_sum_s2 = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self._everest_gait_touchdown_transition_count = torch.zeros(
            self.num_envs, device=self.device
        )
        self._everest_gait_touchdown_alternating_count = torch.zeros(
            self.num_envs, device=self.device
        )
        self._everest_gait_loaded_force_sum_n = torch.zeros(self.num_envs, device=self.device)
        self._everest_gait_force_weighted_lateral_speed_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._everest_gait_loaded_probe_count = torch.zeros(self.num_envs, device=self.device)
        self._everest_gait_lateral_excess_count = torch.zeros(self.num_envs, device=self.device)
        self._everest_gait_slipping_probe_count = torch.zeros(self.num_envs, device=self.device)
        self._everest_gait_fractured_probe_count = torch.zeros(self.num_envs, device=self.device)
        self._everest_gait_max_lateral_speed_mps = torch.zeros(self.num_envs, device=self.device)
        self._everest_original_write_data_to_sim = self.scene.write_data_to_sim
        self.scene.write_data_to_sim = self._everest_write_data_to_sim

    def _update_everest_gait_metrics(self, wrench) -> None:
        """Accumulate contact-gated gait diagnostics at the physics update rate."""

        active = ~self._everest_pending_reset
        normal_force = wrench.probe_normal_force_n.clamp_min(0.0)
        loaded_probe = normal_force >= 10.0
        stance = loaded_probe.any(dim=-1) & active[:, None]
        normal = self._everest_terrain_normal
        origin = self._everest_terrain_origin
        probe_height_m = (
            (wrench.probe_world_position_m - origin[:, None, None, :]) * normal[:, None, None, :]
        ).sum(dim=-1) - 0.004
        swing_height_m = probe_height_m.amin(dim=-1)
        liftoff = self._everest_gait_previous_stance & ~stance
        self._everest_gait_swing_active |= liftoff
        swinging = self._everest_gait_swing_active & ~stance & active[:, None]
        self._everest_gait_swing_peak_m = torch.where(
            swinging,
            torch.maximum(self._everest_gait_swing_peak_m, swing_height_m),
            self._everest_gait_swing_peak_m,
        )
        touchdown = (~self._everest_gait_previous_stance) & stance & active[:, None]
        completed_swing = touchdown & self._everest_gait_swing_active
        self._everest_gait_swing_count += completed_swing
        self._everest_gait_swing_peak_sum_m += torch.where(
            completed_swing, self._everest_gait_swing_peak_m, torch.zeros_like(swing_height_m)
        )
        self._everest_gait_low_swing_count += (
            completed_swing & (self._everest_gait_swing_peak_m < 0.015)
        ).float()
        self._everest_gait_swing_active &= ~touchdown
        self._everest_gait_swing_peak_m = torch.where(
            touchdown,
            torch.zeros_like(self._everest_gait_swing_peak_m),
            self._everest_gait_swing_peak_m,
        )

        tangent = torch.stack((normal[:, 2], torch.zeros_like(normal[:, 0]), -normal[:, 0]), dim=-1)
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        probe_tangent_m = (wrench.probe_world_position_m * tangent[:, None, None, :]).sum(dim=-1)
        weighted_touchdown_tangent_m = (probe_tangent_m * normal_force).sum(
            dim=-1
        ) / normal_force.sum(dim=-1).clamp_min(1.0)
        for foot in range(2):
            event = touchdown[:, foot]
            prior = event & self._everest_gait_has_touchdown[:, foot]
            advance = (
                weighted_touchdown_tangent_m[:, foot]
                - self._everest_gait_last_touchdown_tangent_m[:, foot]
            )
            duration = self._everest_gait_time_s - self._everest_gait_last_touchdown_time_s[:, foot]
            self._everest_gait_stride_count[:, foot] += prior.float()
            self._everest_gait_stride_advance_sum_m[:, foot] += torch.where(
                prior, advance, torch.zeros_like(advance)
            )
            self._everest_gait_stride_duration_sum_s[:, foot] += torch.where(
                prior, duration, torch.zeros_like(duration)
            )
            self._everest_gait_stride_duration_sq_sum_s2[:, foot] += torch.where(
                prior, duration.square(), torch.zeros_like(duration)
            )
            transition = event & (self._everest_gait_last_touchdown_foot >= 0)
            self._everest_gait_touchdown_transition_count += transition.float()
            self._everest_gait_touchdown_alternating_count += (
                transition & (self._everest_gait_last_touchdown_foot != foot)
            ).float()
            self._everest_gait_last_touchdown_foot = torch.where(
                event,
                torch.full_like(self._everest_gait_last_touchdown_foot, foot),
                self._everest_gait_last_touchdown_foot,
            )
            self._everest_gait_last_touchdown_tangent_m[:, foot] = torch.where(
                event,
                weighted_touchdown_tangent_m[:, foot],
                self._everest_gait_last_touchdown_tangent_m[:, foot],
            )
            self._everest_gait_last_touchdown_time_s[:, foot] = torch.where(
                event,
                self._everest_gait_time_s,
                self._everest_gait_last_touchdown_time_s[:, foot],
            )
            self._everest_gait_has_touchdown[:, foot] |= event

        lateral_speed_mps = wrench.probe_lateral_speed_mps.clamp_min(0.0)
        loaded_force = (normal_force * loaded_probe).sum(dim=(1, 2))
        self._everest_gait_loaded_force_sum_n += loaded_force
        self._everest_gait_force_weighted_lateral_speed_sum += (
            lateral_speed_mps * normal_force * loaded_probe
        ).sum(dim=(1, 2))
        self._everest_gait_loaded_probe_count += loaded_probe.sum(dim=(1, 2))
        self._everest_gait_lateral_excess_count += (loaded_probe & (lateral_speed_mps > 0.12)).sum(
            dim=(1, 2)
        )
        self._everest_gait_slipping_probe_count += (
            loaded_probe & wrench.material_response.slipping
        ).sum(dim=(1, 2))
        self._everest_gait_fractured_probe_count += (
            loaded_probe & wrench.material_response.fractured
        ).sum(dim=(1, 2))
        current_max = torch.where(
            loaded_probe,
            lateral_speed_mps,
            torch.zeros_like(lateral_speed_mps),
        ).amax(dim=(1, 2))
        self._everest_gait_max_lateral_speed_mps = torch.maximum(
            self._everest_gait_max_lateral_speed_mps, current_max
        )
        self._everest_gait_previous_stance.copy_(stance)
        self._everest_gait_time_s += float(self.physics_dt)

    def everest_gait_metrics(self) -> dict[str, torch.Tensor]:
        """Return cumulative native contact gait diagnostics without policy inputs."""

        swing_count = self._everest_gait_swing_count.sum(dim=-1)
        stride_count = self._everest_gait_stride_count.sum(dim=-1)
        stride_duration_mean = self._everest_gait_stride_duration_sum_s.sum(
            dim=-1
        ) / stride_count.clamp_min(1.0)
        stride_duration_second = self._everest_gait_stride_duration_sq_sum_s2.sum(
            dim=-1
        ) / stride_count.clamp_min(1.0)
        stride_duration_std = (
            (stride_duration_second - stride_duration_mean.square()).clamp_min(0.0).sqrt()
        )
        return {
            "completed_swing_count": swing_count.clone(),
            "mean_swing_peak_m": (
                self._everest_gait_swing_peak_sum_m.sum(dim=-1) / swing_count.clamp_min(1.0)
            ).clone(),
            "low_swing_fraction": (
                self._everest_gait_low_swing_count.sum(dim=-1) / swing_count.clamp_min(1.0)
            ).clone(),
            "stride_count": stride_count.clone(),
            "mean_stride_advance_m": (
                self._everest_gait_stride_advance_sum_m.sum(dim=-1) / stride_count.clamp_min(1.0)
            ).clone(),
            "mean_stride_duration_s": stride_duration_mean.clone(),
            "stride_duration_cv": (
                stride_duration_std / stride_duration_mean.clamp_min(1.0e-6)
            ).clone(),
            "touchdown_alternation_fraction": (
                self._everest_gait_touchdown_alternating_count
                / self._everest_gait_touchdown_transition_count.clamp_min(1.0)
            ).clone(),
            "force_weighted_stance_lateral_speed_mps": (
                self._everest_gait_force_weighted_lateral_speed_sum
                / self._everest_gait_loaded_force_sum_n.clamp_min(1.0)
            ).clone(),
            "maximum_stance_lateral_speed_mps": self._everest_gait_max_lateral_speed_mps.clone(),
            "lateral_speed_excess_fraction": (
                self._everest_gait_lateral_excess_count
                / self._everest_gait_loaded_probe_count.clamp_min(1.0)
            ).clone(),
            "material_slipping_fraction": (
                self._everest_gait_slipping_probe_count
                / self._everest_gait_loaded_probe_count.clamp_min(1.0)
            ).clone(),
            "material_fractured_fraction": (
                self._everest_gait_fractured_probe_count
                / self._everest_gait_loaded_probe_count.clamp_min(1.0)
            ).clone(),
        }

    def _everest_write_data_to_sim(self) -> None:
        robot = self.scene["robot"]
        pending_reset = self._everest_pending_reset.clone()
        reset_ids = torch.nonzero(pending_reset, as_tuple=False).squeeze(-1)
        wrote_reset_state = bool(reset_ids.numel())
        if wrote_reset_state:
            robot.permanent_wrench_composer.reset(env_ids=reset_ids)
            self._everest_original_write_data_to_sim()
        ids = self._everest_ankle_body_ids
        ankle_position = robot.data.body_pos_w[:, ids]
        ankle_quaternion = robot.data.body_quat_w[:, ids]
        ankle_linear_velocity = robot.data.body_lin_vel_w[:, ids]
        ankle_angular_velocity = robot.data.body_ang_vel_w[:, ids]
        applied_load = torch.full(
            (self.num_envs, 2), 0.5 * self._everest_robot_weight_n, device=self.device
        )
        normal = self._everest_terrain_normal
        gravity_tangent = torch.tensor(
            (0.0, 0.0, -self._everest_robot_weight_n), device=self.device
        ).expand(self.num_envs, -1)
        gravity_tangent = (
            gravity_tangent - (gravity_tangent * normal).sum(dim=-1, keepdim=True) * normal
        )
        tangential_demand = (
            0.5 * gravity_tangent[:, None, :]
            + float(self.cfg.everest_tangential_velocity_gain_ns_per_m) * ankle_linear_velocity
        )
        wrench = self.everest_wrench_bridge.step(
            ankle_position,
            ankle_quaternion,
            ankle_linear_velocity,
            ankle_angular_velocity,
            self._everest_terrain_origin,
            normal,
            applied_load,
            tangential_demand,
            dt_s=self.physics_dt,
        )
        self.everest_wrench_bridge.lift_probes(wrench.probe_penetration_m <= 0.0)
        if wrote_reset_state:
            self.everest_wrench_bridge.reset_worlds(reset_ids)
            for value in (
                wrench.total_force_n,
                wrench.total_torque_nm,
                wrench.probe_force_n,
                wrench.probe_normal_force_n,
                wrench.probe_axial_force_n,
                wrench.probe_penetration_m,
                wrench.probe_penetration_rate_mps,
                wrench.probe_lateral_speed_mps,
                wrench.material_response.normal_force_n,
                wrench.material_response.shear_capacity_n,
                wrench.material_response.friction_utilization,
                wrench.material_response.damage,
                wrench.material_response.residual_crater_depth_m,
            ):
                value[pending_reset] = 0
            for value in (
                wrench.material_response.slipping,
                wrench.material_response.fractured,
                wrench.material_response.broken_out,
            ):
                value[pending_reset] = False
            self._everest_pending_reset[reset_ids] = False
        self._update_everest_gait_metrics(wrench)
        self._everest_wrench_adapter.apply(wrench)
        self.everest_latest_wrench = wrench
        if self._sim_step_counter % 2 == 0:
            self._everest_sample_sensor(
                wrench, ankle_position, ankle_linear_velocity, ankle_angular_velocity
            )
        self._everest_previous_ankle_velocity.copy_(ankle_linear_velocity)
        if not wrote_reset_state:
            self._everest_original_write_data_to_sim()

    def _everest_sample_sensor(
        self,
        wrench,
        ankle_position: torch.Tensor,
        ankle_linear_velocity: torch.Tensor,
        ankle_angular_velocity: torch.Tensor,
    ) -> None:
        parameters = self.everest_wrench_bridge.material.parameters
        acceleration = (
            ankle_linear_velocity - self._everest_previous_ankle_velocity
        ) / self.physics_dt
        acceleration[..., 2] += 9.81
        support = parameters.support_layer_depth_m.mean(dim=-1)
        has_void = parameters.void_present.any(dim=-1)
        void_depth = torch.where(
            parameters.void_present,
            parameters.void_top_depth_m,
            torch.full_like(parameters.void_top_depth_m, 10.0),
        ).amin(dim=-1)
        return_strength = (
            0.25 + parameters.vertical_stiffness_n_per_m.mean(dim=-1) / 55_000.0
        ).clamp(0.02, 1.0)
        environment_phase = torch.arange(self.num_envs, device=self.device, dtype=torch.float32)[
            :, None
        ]
        foot_phase = torch.arange(2, device=self.device, dtype=torch.float32)[None, :]
        phase = (
            0.37 * float(self._everest_sensor_sample_index)
            + 0.73 * environment_phase
            + 1.17 * foot_phase
        )
        range_noise = 0.006 * torch.sin(phase)
        strength_noise = 0.025 * torch.sin(1.91 * phase + 0.4)
        secondary_range = torch.where(has_void, void_depth, support + 0.18) + range_noise
        secondary_strength = torch.where(
            has_void,
            0.70 * return_strength * torch.exp(-void_depth / 0.45),
            0.04 * return_strength,
        )
        secondary_strength = (secondary_strength + strength_noise).clamp(0.0, 1.0)
        radar = torch.stack(
            (
                support + 0.5 * range_noise,
                secondary_range,
                (return_strength - 0.5 * strength_noise).clamp(0.0, 1.0),
                secondary_strength,
                (0.025 + strength_noise.abs()).clamp(0.0, 0.10),
            ),
            dim=-1,
        )
        radar[..., :2] = torch.round(radar[..., :2] / 0.04) * 0.04
        radar[..., 2:] = torch.round(radar[..., 2:] / 0.05) * 0.05
        command = self.command_manager.get_command("base_velocity")
        # Use contact-relative deployable context, never simulator world coordinates.
        foot_position = torch.zeros_like(ankle_position)
        foot_position[..., 2] = -wrench.probe_penetration_m.mean(dim=-1)
        foot_velocity = torch.zeros_like(ankle_linear_velocity)
        foot_velocity[..., 2] = -wrench.probe_penetration_rate_mps.mean(dim=-1)
        probe_speed = wrench.probe_penetration_rate_mps.abs().mean(dim=-1).clamp(0.0, 0.35)
        root_rpy = _quaternion_wxyz_to_rpy(self.scene["robot"].data.root_quat_w)
        root_rpy[..., 0] = 0.0
        root_rpy[..., 2] = 0.0
        body_load = wrench.probe_normal_force_n.sum(dim=-1).clamp(0.0, 343.0)
        command_load = torch.full_like(body_load, min(0.5 * self._everest_robot_weight_n, 250.0))
        context = {
            "foot_position_xyz_m": foot_position,
            "foot_velocity_xyz_mps": foot_velocity,
            "pelvis_roll_pitch_yaw_rad": root_rpy,
            "commanded_probe_load_n": command_load,
            "commanded_foot_speed_mps": probe_speed,
            "body_weight_on_foot_n": body_load,
        }
        commands = {
            "requested_vx_mps": command[:, 0],
            "requested_vy_mps": command[:, 1],
            "requested_wz_rps": command[:, 2],
            "mode": torch.ones(self.num_envs, device=self.device),
            "probe_load_n": command_load,
            "approach_speed_mps": probe_speed,
        }
        if self._everest_enable_sensor_fault_curriculum and self._everest_sensor_sample_index > 0:
            (
                sensor_force,
                sensor_penetration,
                sensor_acceleration,
                sensor_gyro,
                sensor_radar,
                fresh_mask,
            ) = apply_sensor_faults(
                wrench.sensor_force_n,
                wrench.sensor_penetration_m,
                acceleration,
                ankle_angular_velocity,
                radar,
                self._everest_fault_cycle,
                sample_index=self._everest_sensor_sample_index,
            )
        else:
            sensor_force = wrench.sensor_force_n
            sensor_penetration = wrench.sensor_penetration_m
            sensor_acceleration = acceleration
            sensor_gyro = ankle_angular_velocity
            sensor_radar = radar
            fresh_mask = torch.ones((self.num_envs, 2, 19), dtype=torch.bool, device=self.device)
        timestamp = torch.full(
            (self.num_envs, 2),
            self._everest_sensor_sample_index * 0.01,
            device=self.device,
            dtype=torch.float64,
        )
        frame = self._everest_sensor.observe(
            axial_force_n=sensor_force,
            penetration_m=sensor_penetration,
            accelerometer_mps2=sensor_acceleration,
            gyroscope_rps=sensor_gyro,
            radar_frontend=sensor_radar,
            timestamp_s=timestamp,
            fresh_mask=fresh_mask,
            context=context,
            commands=commands,
        )
        self._everest_sensor_sample_index += 1
        self.everest_latest_sensor_frame = frame
        self.everest_sensor_frames.append(frame)
        response = wrench.material_response
        bearing = (parameters.bearing_capacity_n * (1.0 - 0.5 * response.damage)).mean(dim=-1)
        shear = response.shear_capacity_n.mean(dim=-1)
        fracture_margin = (parameters.fracture_strength_n - wrench.probe_normal_force_n).amin(
            dim=-1
        )
        probe_contact = wrench.probe_penetration_m > 0.0
        probe_slip_margin = response.shear_capacity_n * (1.0 - response.friction_utilization)
        slip_margin = torch.where(probe_contact, probe_slip_margin, 0.0).sum(dim=-1)
        slip_margin = torch.where(
            probe_contact.any(dim=-1), slip_margin, torch.full_like(slip_margin, 500.0)
        )
        void_depth = torch.where(
            parameters.void_present,
            parameters.void_top_depth_m,
            torch.zeros_like(parameters.void_top_depth_m),
        ).amax(dim=-1)
        targets = torch.stack(
            (
                support,
                parameters.vertical_stiffness_n_per_m.mean(dim=-1),
                parameters.damping_ns_per_m.mean(dim=-1),
                bearing,
                shear,
                parameters.friction.mean(dim=-1),
                self.everest_wrench_bridge.material.compaction.mean(dim=-1),
                response.damage.mean(dim=-1),
                fracture_margin,
                slip_margin,
                void_depth,
            ),
            dim=-1,
        )
        events = torch.stack(
            (
                parameters.void_present.any(dim=-1),
                response.fractured.any(dim=-1),
                response.slipping.any(dim=-1),
            ),
            dim=-1,
        )
        self.everest_truth_frames.append(
            {"targets": targets.detach().clone(), "events": events.detach().clone()}
        )

    def pop_everest_sensor_frames(self) -> list[Any]:
        frames = list(self.everest_sensor_frames)
        self.everest_sensor_frames.clear()
        return frames

    def pop_everest_truth_frames(self) -> list[dict[str, torch.Tensor]]:
        frames = list(self.everest_truth_frames)
        self.everest_truth_frames.clear()
        return frames

    def _clear_native_contact_history(self, env_ids: torch.Tensor) -> None:
        """Clear force history that Isaac's timestamp-only sensor reset retains."""

        contact_sensor = self.scene.sensors.get("contact_forces")
        if contact_sensor is None:
            return
        for name in (
            "net_forces_w",
            "net_forces_w_history",
            "force_matrix_w",
            "force_matrix_w_history",
            "friction_forces_w",
        ):
            proxy = getattr(contact_sensor.data, name, None)
            if proxy is not None:
                proxy.torch[env_ids] = 0.0

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        if hasattr(self, "everest_wrench_bridge"):
            self._clear_native_contact_history(env_ids)
            self.everest_wrench_bridge.reset_worlds(env_ids)
            self._everest_pending_reset[env_ids] = True
            self._everest_sensor.mark_environment_reset(env_ids)
            self._everest_fault_cycle[env_ids] += 1
            self._everest_previous_ankle_velocity[env_ids] = 0.0
            self._everest_gait_previous_stance[env_ids] = False
            self._everest_gait_swing_active[env_ids] = False
            self._everest_gait_swing_peak_m[env_ids] = 0.0
            self._everest_gait_last_touchdown_time_s[env_ids] = 0.0
            self._everest_gait_last_touchdown_tangent_m[env_ids] = 0.0
            self._everest_gait_has_touchdown[env_ids] = False
            self._everest_gait_last_touchdown_foot[env_ids] = -1
