from __future__ import annotations

import os
from pathlib import Path

import torch
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1Rewards,
    G1RoughEnvCfg,
)

from ....assets import G1_CRAMPON_CFG, G1_CRAMPON_STATEFUL_CFG


def everest_base_clearance_l2(env, target_clearance_m: float) -> torch.Tensor:
    """Penalize pelvis clearance error relative to the analytical terrain plane."""

    robot = env.scene["robot"]
    clearance = (
        (robot.data.root_pos_w - env._everest_terrain_origin) * env._everest_terrain_normal
    ).sum(dim=-1)
    return (clearance - target_clearance_m).square()


def everest_illegal_contact_after_reset_grace(
    env,
    threshold: float,
    sensor_cfg,
    grace_steps: int,
    minimum_base_clearance_m: float,
) -> torch.Tensor:
    """Reject stale contact history unless the pelvis is also physically low."""

    illegal = base_mdp.illegal_contact(env, threshold=threshold, sensor_cfg=sensor_cfg)
    robot = env.scene["robot"]
    clearance = (
        (robot.data.root_pos_w - env._everest_terrain_origin) * env._everest_terrain_normal
    ).sum(dim=-1)
    physically_low = clearance < minimum_base_clearance_m
    return illegal & physically_low & (env.episode_length_buf >= grace_steps)


def everest_penetration_excess_l2(env, limit_m: float) -> torch.Tensor:
    """Penalize analytical spike travel beyond the curriculum support band."""

    wrench = env.everest_latest_wrench
    if wrench is None:
        return torch.zeros(env.num_envs, device=env.device)
    maximum = wrench.probe_penetration_m.amax(dim=(1, 2))
    return torch.relu(maximum - limit_m).square()


def everest_slip_fraction(env) -> torch.Tensor:
    """Privileged training reward only; never part of the deployable observation."""

    wrench = env.everest_latest_wrench
    if wrench is None:
        return torch.zeros(env.num_envs, device=env.device)
    return wrench.material_response.slipping.float().mean(dim=(1, 2))


def everest_contact_mode_rear_load_l2(env, minimum_contact_load_n: float) -> torch.Tensor:
    """Shape commanded contact mode without deleting physically available support probes."""

    wrench = env.everest_latest_wrench
    if wrench is None:
        return torch.zeros(env.num_envs, device=env.device)
    normal_force = wrench.probe_normal_force_n.clamp_min(0.0)
    total_load = normal_force.sum(dim=(1, 2))
    rear_load = normal_force[:, :, 2:].sum(dim=(1, 2))
    rear_fraction = rear_load / total_load.clamp_min(minimum_contact_load_n)
    error = (rear_fraction - env._everest_target_rear_load_fraction).square()
    return torch.where(total_load >= minimum_contact_load_n, error, torch.zeros_like(error))


def everest_contact_mode_rear_load_exp(
    env, minimum_contact_load_n: float, std: float
) -> torch.Tensor:
    """Bounded positive reward for matching the commanded rear-load fraction."""

    if std <= 0.0:
        raise ValueError("std must be positive")
    wrench = env.everest_latest_wrench
    if wrench is None:
        return torch.zeros(env.num_envs, device=env.device)
    normal_force = wrench.probe_normal_force_n.clamp_min(0.0)
    total_load = normal_force.sum(dim=(1, 2))
    rear_load = normal_force[:, :, 2:].sum(dim=(1, 2))
    rear_fraction = rear_load / total_load.clamp_min(minimum_contact_load_n)
    delta = rear_fraction - env._everest_target_rear_load_fraction
    score = torch.exp(-delta.square() / (std * std))
    return torch.where(total_load >= minimum_contact_load_n, score, torch.zeros_like(score))


def everest_contact_mode_rear_load_linear(env, minimum_contact_load_n: float) -> torch.Tensor:
    """Dense bounded reward for matching the commanded rear-load fraction."""

    wrench = env.everest_latest_wrench
    if wrench is None:
        return torch.zeros(env.num_envs, device=env.device)
    normal_force = wrench.probe_normal_force_n.clamp_min(0.0)
    total_load = normal_force.sum(dim=(1, 2))
    rear_load = normal_force[:, :, 2:].sum(dim=(1, 2))
    rear_fraction = rear_load / total_load.clamp_min(minimum_contact_load_n)
    delta = (rear_fraction - env._everest_target_rear_load_fraction).abs()
    score = (1.0 - delta).clamp(0.0, 1.0)
    return torch.where(total_load >= minimum_contact_load_n, score, torch.zeros_like(score))


def everest_forward_velocity_error_l2(env, command_name: str) -> torch.Tensor:
    """Penalize forward speed error directly to prevent high-speed reward shortcuts."""

    robot = env.scene["robot"]
    requested = env.command_manager.get_command(command_name)[:, 0]
    forward = robot.data.root_lin_vel_b[:, 0]
    return (forward - requested).square()


def everest_sagittal_toe_support_l2(env, target_forward_offset_m: float) -> torch.Tensor:
    """Train the pelvis projection toward the toe-support line using privileged geometry."""

    wrench = env.everest_latest_wrench
    if wrench is None:
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    ankle_center = wrench.ankle_position_m.mean(dim=1)
    normal = env._everest_terrain_normal
    tangent = torch.stack((normal[:, 2], torch.zeros_like(normal[:, 0]), -normal[:, 0]), dim=-1)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    forward_offset = ((robot.data.root_pos_w - ankle_center) * tangent).sum(dim=-1)
    return (forward_offset - target_forward_offset_m).square()


@configclass
class EverestStatefulRewardsCfg(G1Rewards):
    """Rewards excluding native-foot contact terms unavailable to analytical probes."""

    everest_penetration_excess = RewTerm(
        func=everest_penetration_excess_l2,
        weight=-40.0,
        params={"limit_m": 0.020},
    )
    everest_slip = RewTerm(func=everest_slip_fraction, weight=-0.5)
    feet_air_time = None
    feet_slide = None


@configclass
class EverestBootstrapRewardsCfg(EverestStatefulRewardsCfg):
    """Reward curriculum for learning stable analytical-contact support."""

    everest_alive = RewTerm(func=base_mdp.is_alive, weight=5.0)
    everest_base_clearance = RewTerm(
        func=everest_base_clearance_l2,
        weight=-10.0,
        params={"target_clearance_m": 0.72},
    )


@configclass
class EverestFrontPointRewardsCfg(EverestBootstrapRewardsCfg):
    """Smooth gait shaping with physical fallback support left available."""

    everest_contact_mode_rear_load = RewTerm(
        func=everest_contact_mode_rear_load_l2,
        weight=-8.0,
        params={"minimum_contact_load_n": 20.0},
    )
    everest_sagittal_toe_support = RewTerm(
        func=everest_sagittal_toe_support_l2,
        weight=-50.0,
        params={"target_forward_offset_m": 0.065},
    )


@configclass
class EverestFrontPointLoadCorrectedRewardsCfg(EverestFrontPointRewardsCfg):
    """Direct load and speed objectives for bounded residual adaptation."""

    everest_contact_mode_rear_load = RewTerm(
        func=everest_contact_mode_rear_load_l2,
        weight=-40.0,
        params={"minimum_contact_load_n": 20.0},
    )
    everest_sagittal_toe_support = None
    everest_forward_velocity_error = RewTerm(
        func=everest_forward_velocity_error_l2,
        weight=-20.0,
        params={"command_name": "base_velocity"},
    )


@configclass
class EverestFrontPointPositiveTrackingRewardsCfg(EverestFrontPointRewardsCfg):
    """Bounded positive task rewards that cannot favor premature termination."""

    everest_contact_mode_rear_load = RewTerm(
        func=everest_contact_mode_rear_load_exp,
        weight=10.0,
        params={"minimum_contact_load_n": 20.0, "std": 0.15},
    )
    everest_sagittal_toe_support = None


@configclass
class EverestFrontPointLinearTrackingRewardsCfg(EverestFrontPointRewardsCfg):
    """Dense bounded load reward with a useful gradient from stock behavior."""

    everest_contact_mode_rear_load = RewTerm(
        func=everest_contact_mode_rear_load_linear,
        weight=20.0,
        params={"minimum_contact_load_n": 20.0},
    )
    everest_sagittal_toe_support = None


def _set_nominal_reset(events) -> None:
    """Remove inherited root/joint reset impulses for contact curriculum stages."""
    events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


@configclass
class EverestG1CramponRoughEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CRAMPON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Isolation and spacing are increased for crampon/hazard work. Terrain cells remain shared assets.
        self.scene.env_spacing = 16.0
        self.scene.replicate_physics = True
        self.scene.filter_collisions = True


@configclass
class EverestG1CramponRoughEnvCfg_PLAY(EverestG1CramponRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 16.0
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        _set_nominal_reset(self.events)


def _default_suite_path() -> str:
    configured = os.environ.get("EVEREST_REPO_ROOT")
    if configured:
        return str(
            Path(configured).expanduser().resolve() / "configs/isaaclab/everest_terrain_suite.yaml"
        )
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "configs/isaaclab/everest_terrain_suite.yaml"
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("Set EVEREST_REPO_ROOT to the hackathon-everest checkout")


@configclass
class EverestG1CramponStatefulEnvCfg(G1RoughEnvCfg):
    rewards: EverestStatefulRewardsCfg = EverestStatefulRewardsCfg()
    everest_suite_config_path: str = _default_suite_path()
    everest_suite_seed: int = 23
    everest_sample_drop_probability: float = 0.006
    everest_use_case_inclines: bool = False
    everest_terrain_anchor_x_m: float = 0.0
    everest_require_complete_coverage: bool = True
    everest_play_surface_id: str = ""
    everest_play_incline_deg: float = 0.0
    everest_play_hazard_id: str = "none"
    everest_play_contact_mode_id: str = ""
    everest_nominal_bootstrap_material: bool = False
    everest_enable_sensor_fault_curriculum: bool = False
    # Reuse environment zero's sampled material across vector rows for a
    # controlled side-by-side comparison.
    everest_match_material_across_envs: bool = False
    # Per-environment multiplier on tangential crampon grip. A low value is a
    # bare-foot approximation for visual ablations; normal support is retained.
    everest_crampon_grip_scale_by_env: tuple[float, ...] | None = None
    everest_virtual_travel_m: float = 0.055
    everest_tangential_velocity_gain_ns_per_m: float = 400.0

    def __post_init__(self):
        super().__post_init__()
        _set_nominal_reset(self.events)
        self.terminations.base_contact.func = everest_illegal_contact_after_reset_grace
        self.terminations.base_contact.params.update(
            {"grace_steps": 3, "minimum_base_clearance_m": 0.50}
        )
        self.scene.robot = G1_CRAMPON_STATEFUL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.env_spacing = 16.0
        self.scene.replicate_physics = True
        self.scene.filter_collisions = True
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.usd_path = None
        self.scene.terrain.env_spacing = 16.0
        self.curriculum.terrain_levels = None


def _default_vector_terrain_path() -> str:
    configured = os.environ.get("EVEREST_VECTOR_TERRAIN_USD")
    if configured:
        return str(Path(configured).expanduser().resolve())
    repo = Path(_default_suite_path()).parents[2]
    return str(repo / "build/isaaclab/everest_vector_terrain_2160.usdc")


@configclass
class EverestG1CramponStatefulSuiteEnvCfg(EverestG1CramponStatefulEnvCfg):
    """Complete 2,160-case material/incline/hazard route suite."""

    everest_use_case_inclines: bool = True
    everest_terrain_anchor_x_m: float = -4.0
    everest_require_complete_coverage: bool = True
    everest_play_surface_id: str = ""
    everest_enable_sensor_fault_curriculum: bool = True

    def __post_init__(self):
        super().__post_init__()
        _set_nominal_reset(self.events)
        self.scene.num_envs = 2160
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.usd_path = _default_vector_terrain_path()
        self.scene.terrain.env_spacing = 16.0
        self.scene.robot.init_state.pos = (-4.0, 0.0, 0.74)
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.80)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)


@configclass
class EverestG1CramponStatefulBootstrapStandEnvCfg(EverestG1CramponStatefulEnvCfg):
    """Undisturbed hard-ice standing curriculum stage."""

    rewards: EverestBootstrapRewardsCfg = EverestBootstrapRewardsCfg()
    everest_require_complete_coverage: bool = False
    everest_play_surface_id: str = "hard_glacier_ice"
    everest_play_contact_mode_id: str = "all_points_flat_foot"
    everest_nominal_bootstrap_material: bool = True
    everest_virtual_travel_m: float = 0.012

    def __post_init__(self):
        super().__post_init__()
        _set_nominal_reset(self.events)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class EverestG1CramponStatefulBootstrapEnvCfg(EverestG1CramponStatefulBootstrapStandEnvCfg):
    """Undisturbed fixed-speed hard-ice walking curriculum stage."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.everest_alive.weight = 2.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.15)


@configclass
class EverestG1CramponStatefulBootstrapStandRandomizedEnvCfg(
    EverestG1CramponStatefulBootstrapStandEnvCfg
):
    """Hard-ice stand stage with the complete authored material-prior range."""

    everest_nominal_bootstrap_material: bool = False


@configclass
class EverestG1CramponStatefulBootstrapRandomizedEnvCfg(EverestG1CramponStatefulBootstrapEnvCfg):
    """Fixed-speed hard-ice walk stage with randomized authored material priors."""

    everest_nominal_bootstrap_material: bool = False


@configclass
class EverestG1CramponStatefulBootstrapFrontPointStandRandomizedEnvCfg(
    EverestG1CramponStatefulBootstrapStandRandomizedEnvCfg
):
    """Isolated randomized hard-ice toe-loading stance curriculum."""

    rewards: EverestFrontPointRewardsCfg = EverestFrontPointRewardsCfg()
    everest_play_contact_mode_id: str = "front_point_contact"

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.dof_acc_l2.weight = -1.0e-6
        self.rewards.ang_vel_xy_l2.weight = -0.20
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.joint_deviation_hip.weight = -1.0


@configclass
class EverestG1CramponStatefulBootstrapFrontPointRandomizedEnvCfg(
    EverestG1CramponStatefulBootstrapRandomizedEnvCfg
):
    """Smooth stock-anchored hard-ice front-loading adaptation task."""

    rewards: EverestFrontPointRewardsCfg = EverestFrontPointRewardsCfg()
    everest_play_contact_mode_id: str = "front_point_contact"

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.dof_acc_l2.weight = -1.0e-6
        self.rewards.ang_vel_xy_l2.weight = -0.20
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.joint_deviation_hip.weight = -1.0


@configclass
class EverestG1CramponStatefulBootstrapFrontPointLoadCorrectedRandomizedEnvCfg(
    EverestG1CramponStatefulBootstrapFrontPointRandomizedEnvCfg
):
    """Front-load residual task without the obsolete sagittal-position proxy."""

    rewards: EverestFrontPointLoadCorrectedRewardsCfg = EverestFrontPointLoadCorrectedRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.10


@configclass
class EverestG1CramponStatefulBootstrapFrontPointPositiveTrackingRandomizedEnvCfg(
    EverestG1CramponStatefulBootstrapFrontPointRandomizedEnvCfg
):
    """Front-load task with bounded positive load and speed tracking."""

    rewards: EverestFrontPointPositiveTrackingRewardsCfg = (
        EverestFrontPointPositiveTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.10
        self.rewards.track_lin_vel_xy_exp.weight = 6.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.15


@configclass
class EverestG1CramponStatefulBootstrapFrontPointLinearTrackingRandomizedEnvCfg(
    EverestG1CramponStatefulBootstrapFrontPointRandomizedEnvCfg
):
    """Front-load task with dense bounded linear load tracking."""

    rewards: EverestFrontPointLinearTrackingRewardsCfg = EverestFrontPointLinearTrackingRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.10
        self.rewards.track_lin_vel_xy_exp.weight = 6.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.15


@configclass
class EverestG1CramponStatefulEnvCfg_PLAY(EverestG1CramponStatefulEnvCfg):
    everest_require_complete_coverage: bool = False
    everest_play_surface_id: str = "hard_glacier_ice"
    everest_play_contact_mode_id: str = "all_points_flat_foot"

    def __post_init__(self):
        super().__post_init__()
        _set_nominal_reset(self.events)
        self.scene.num_envs = 1
        self.episode_length_s = 40.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.15)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
