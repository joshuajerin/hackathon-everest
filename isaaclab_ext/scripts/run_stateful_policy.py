#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from hackathon_everest_isaaclab.data.schema import VisibleSensorBatch
from hackathon_everest_isaaclab.learning.models import (
    REGRESSION_NAMES,
    CausalBilateralEstimator,
    VisibleOnlySupervisor,
)
from hackathon_everest_isaaclab.learning.safety_priors import (
    FRACTURE_DAMAGE_CAUTION,
    MINIMUM_BEARING_CAPACITY_N,
    SEVERE_SLIP_MARGIN_N,
)
from hackathon_everest_isaaclab.learning.shield import (
    SafetyShield,
    ShieldAction,
    ShieldConfig,
    ShieldSignals,
    conservative_target_safe,
)
from hackathon_everest_isaaclab.runtime import (
    ContactGatedPolicyCorrection,
    ContactGatedPolicyCorrectionConfig,
    EverestController,
    EverestControllerConfig,
    acquire_isaac_process_lock,
    visible_crampon_contact,
)
from hackathon_everest_isaaclab.sensors.faults import SENSOR_FAULT_MODES
from hackathon_everest_isaaclab.tasks import register_cli
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

register_cli()
parser = argparse.ArgumentParser(
    description="Run the complete visible Everest policy on stateful G1"
)
parser.add_argument("--task", default="Everest-Velocity-Flat-G1-Crampon-Stateful-Play-v0")
parser.add_argument("--surface-id", default="hard_glacier_ice")
parser.add_argument("--incline-deg", type=float, default=0.0)
parser.add_argument("--hazard-id", default="none")
parser.add_argument("--contact-mode-id", default="all_points_flat_foot")
parser.add_argument(
    "--suite-config",
    type=Path,
    help="Optional terrain-suite YAML used for this run; assets remain rooted in the checkout.",
)
parser.add_argument(
    "--requested-vx",
    type=float,
    default=0.30,
    help="Requested forward speed sent through the shield and frozen stock G1 policy.",
)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument(
    "--controller-rate-hz",
    type=float,
    default=50.0,
    help="Visible supervisor/stock-command update rate; must divide the 100 Hz sensor rate.",
)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--mode", choices=("shadow", "active"), default="shadow")
parser.add_argument("--stock-policy", type=Path, required=True)
parser.add_argument("--visible-checkpoint", type=Path, required=True)
parser.add_argument("--style-reference-policy", type=Path)
parser.add_argument(
    "--contact-correction-policy",
    type=Path,
    help=(
        "TorchScript trained bounded-residual policy. Its bounded joint correction is "
        "blended in only after a fresh visible crampon contact packet."
    ),
)
parser.add_argument("--contact-correction-max-residual", type=float, default=0.12)
parser.add_argument("--contact-correction-weight-step", type=float, default=0.05)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max-terminations", type=int)
parser.add_argument("--min-base-height-m", type=float)
parser.add_argument("--max-ood-fraction", type=float)
parser.add_argument("--max-stale-fraction", type=float)
parser.add_argument("--max-recovery-fraction", type=float)
parser.add_argument(
    "--min-accepted-safe-vx",
    type=float,
    help="Optional lower gate for the shield-approved forward command in active mode.",
)
parser.add_argument(
    "--max-contact-lateral-speed-mps",
    type=float,
    help="Optional upper gate on normal-load-gated analytical probe lateral speed.",
)
parser.add_argument(
    "--max-contact-slip-fraction",
    type=float,
    help="Optional upper gate on normal-load-gated material slip events.",
)
parser.add_argument(
    "--min-mean-swing-lift-m",
    type=float,
    help="Optional lower gate on completed unloaded-swing peak ankle height.",
)
parser.add_argument(
    "--min-mean-stride-length-m",
    type=float,
    help="Optional lower gate on same-foot touchdown separation along world X.",
)
parser.add_argument("--video-dir", type=Path)
parser.add_argument("--video-length", type=int, default=500)
parser.add_argument(
    "--diagnostic-posture-grid",
    action="store_true",
    help="Apply an evaluation-only square grid of bilateral hip/ankle pitch offsets.",
)
parser.add_argument("--diagnostic-hip-pitch-extent", type=float, default=1.0)
parser.add_argument("--diagnostic-ankle-pitch-extent", type=float, default=1.0)
parser.add_argument(
    "--diagnostic-balance-grid",
    action="store_true",
    help="Scan evaluation-only pitch/pitch-rate ankle feedback gains.",
)
parser.add_argument(
    "--diagnostic-balance-full-grid",
    action="store_true",
    help="Scan 512 bounded hip/ankle bias and pitch-feedback combinations.",
)
parser.add_argument(
    "--diagnostic-lower-body-grid",
    action="store_true",
    help="Scan 256 bounded hip/knee/ankle/torso residual combinations.",
)
parser.add_argument(
    "--diagnostic-candidate-command-grid",
    action="store_true",
    help="Evaluate the selected smooth residual over eight stock-policy forward commands.",
)
parser.add_argument("--diagnostic-stock-vx-maximum", type=float, default=0.80)
parser.add_argument(
    "--diagnostic-zero-stock-command",
    action="store_true",
    help="Zero the stock-policy command for an evaluation-only posture diagnostic.",
)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]
if args.video_dir is not None:
    if args.video_length < 1 or args.video_length > args.steps:
        raise ValueError("video-length must be in [1, steps]")
    args.enable_cameras = True
_ISAAC_PROCESS_LOCK = acquire_isaac_process_lock()


def pack_context(frame: VisibleSensorBatch) -> torch.Tensor:
    context = frame.context
    pelvis = context["pelvis_roll_pitch_yaw_rad"]
    if pelvis.ndim == 2:
        pelvis = pelvis[:, None, :].expand(-1, 2, -1)
    return torch.cat(
        (
            context["foot_position_xyz_m"],
            context["foot_velocity_xyz_mps"],
            pelvis,
            context["commanded_probe_load_n"].unsqueeze(-1),
            context["commanded_foot_speed_mps"].unsqueeze(-1),
            context["body_weight_on_foot_n"].unsqueeze(-1),
        ),
        dim=-1,
    )


def pack_commands(frame: VisibleSensorBatch) -> torch.Tensor:
    command = frame.commands
    return torch.cat(
        (
            command["requested_vx_mps"].unsqueeze(-1),
            command["requested_vy_mps"].unsqueeze(-1),
            command["requested_wz_rps"].unsqueeze(-1),
            command["mode"].unsqueeze(-1),
            command["probe_load_n"],
            command["approach_speed_mps"],
        ),
        dim=-1,
    )


def requested_command(frame: VisibleSensorBatch) -> torch.Tensor:
    command = frame.commands
    return torch.stack(
        (command["requested_vx_mps"], command["requested_vy_mps"], command["requested_wz_rps"]),
        dim=-1,
    )


def create_showcase_slope(surface_id: str, incline_deg: float) -> None:
    """Add a non-colliding visual slab aligned with the analytical support plane."""

    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    colors = {
        "base_camp_patchy_snow": (0.82, 0.86, 0.90),
        "western_cwm_consolidated_snow": (0.90, 0.94, 0.98),
        "lhotse_boot_packed_snow": (0.72, 0.76, 0.80),
        "south_col_wind_pack": (0.86, 0.89, 0.93),
        "summit_ridge_drift": (0.97, 0.98, 1.00),
        "hard_glacier_ice": (0.26, 0.47, 0.66),
        "fractured_blue_ice": (0.17, 0.40, 0.61),
        "polished_wind_ice": (0.34, 0.55, 0.71),
        "thin_snow_over_ice": (0.80, 0.87, 0.93),
    }
    if surface_id not in colors:
        raise ValueError(f"No visual color is registered for surface {surface_id!r}")
    stage = omni.usd.get_context().get_stage()
    ground = stage.GetPrimAtPath("/World/ground")
    if ground.IsValid():
        UsdGeom.Imageable(ground).MakeInvisible()
    angle = math.radians(incline_deg)
    normal = (-math.sin(angle), 0.0, math.cos(angle))
    thickness = 0.12
    cube = UsdGeom.Cube.Define(stage, "/World/EverestShowcaseSlope")
    cube.CreateSizeAttr(1.0)
    transform = UsdGeom.Xformable(cube)
    transform.AddTranslateOp().Set(
        Gf.Vec3d(-0.5 * thickness * normal[0], 0.0, -0.5 * thickness * normal[2])
    )
    transform.AddRotateYOp().Set(-incline_deg)
    transform.AddScaleOp().Set(Gf.Vec3d(20.0, 8.0, thickness))
    material = UsdShade.Material.Define(stage, "/World/EverestShowcaseMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/EverestShowcaseMaterial/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*colors[surface_id])
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.82)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)


def main() -> int:
    if not 0.0 < args.requested_vx <= 0.80:
        raise ValueError("requested-vx must be in (0.0, 0.80]")
    if args.contact_correction_policy is not None and args.mode != "active":
        raise ValueError("contact-correction-policy is only valid in active mode")
    if not 0.0 < args.contact_correction_max_residual <= 0.35:
        raise ValueError("contact-correction-max-residual must be in (0.0, 0.35]")
    if not 0.0 < args.contact_correction_weight_step <= 1.0:
        raise ValueError("contact-correction-weight-step must be in (0.0, 1.0]")
    env_cfg, _ = resolve_task_config(args.task, "")
    suite_config_path = None
    if args.suite_config is not None:
        suite_config_path = args.suite_config.expanduser().resolve()
        if not suite_config_path.is_file():
            raise FileNotFoundError(suite_config_path)
        env_cfg.everest_suite_config_path = str(suite_config_path)
    complete_suite = "Suite-G1-Crampon-Stateful" in args.task
    if not complete_suite:
        env_cfg.everest_require_complete_coverage = False
        env_cfg.everest_play_surface_id = args.surface_id
        env_cfg.everest_play_incline_deg = args.incline_deg
        env_cfg.everest_play_hazard_id = args.hazard_id
        env_cfg.everest_play_contact_mode_id = args.contact_mode_id
        env_cfg.everest_use_case_inclines = True
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args.requested_vx, args.requested_vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    # The velocity arrow is useful for debugging, not for clean gait footage.
    env_cfg.commands.base_velocity.debug_vis = False
    exit_code = 0
    with launch_simulation(env_cfg, args):
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device or "cuda:0"
        if args.video_dir is not None:
            env_cfg.viewer.eye = (2.4, 2.4, 1.4)
            env_cfg.viewer.lookat = (0.0, 0.0, 0.75)
            env_cfg.viewer.origin_type = "asset_root"
            env_cfg.viewer.env_index = 0
            env_cfg.viewer.asset_name = "robot"
        render_mode = "rgb_array" if args.video_dir is not None else None
        env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
        if args.video_dir is not None:
            # Gait footage must not include scanner/contact debug primitives.
            for sensor in env.unwrapped.scene.sensors.values():
                sensor.set_debug_vis(False)
            create_showcase_slope(args.surface_id, args.incline_deg)
            args.video_dir.mkdir(parents=True, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(args.video_dir),
                step_trigger=lambda step: step == 0,
                video_length=args.video_length,
                name_prefix=f"{args.task}-{args.mode}",
                disable_logger=True,
            )
        observation, _ = env.reset()
        device = env.unwrapped.device
        diagnostic_stock_vx: torch.Tensor | None = None
        stock = torch.jit.load(str(args.stock_policy), map_location=device).eval()
        style_reference = (
            torch.jit.load(str(args.style_reference_policy), map_location=device).eval()
            if args.style_reference_policy is not None
            else None
        )
        contact_correction_policy = (
            torch.jit.load(str(args.contact_correction_policy), map_location=device).eval()
            if args.contact_correction_policy is not None
            else None
        )
        contact_correction = (
            ContactGatedPolicyCorrection(
                ContactGatedPolicyCorrectionConfig(
                    maximum_weight_step=args.contact_correction_weight_step,
                    maximum_action_residual=args.contact_correction_max_residual,
                )
            )
            if contact_correction_policy is not None
            else None
        )

        def stock_action(stock_observation: torch.Tensor) -> torch.Tensor:
            if not args.diagnostic_zero_stock_command and diagnostic_stock_vx is None:
                return stock(stock_observation).detach()
            conditioned = stock_observation.clone()
            if args.diagnostic_zero_stock_command:
                conditioned[:, 9:12] = 0.0
            else:
                assert diagnostic_stock_vx is not None
                conditioned[:, 9] = diagnostic_stock_vx
            return stock(conditioned).detach()

        def safe_hold_action(stock_observation: torch.Tensor) -> torch.Tensor:
            """Keep the frozen stock posture path while removing locomotion command."""

            conditioned = stock_observation.clone()
            conditioned[:, 9:12] = 0.0
            return stock(conditioned).detach()

        def contact_corrected_action(
            base_action: torch.Tensor,
            safe_velocity_yaw: torch.Tensor,
            stock_observation: torch.Tensor,
            sensor_frame: VisibleSensorBatch,
            correction_allowed: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Apply the trained residual only on fresh, shield-approved contact."""

            no_contact = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            if contact_correction_policy is None or contact_correction is None:
                return base_action, no_contact
            # Never retain a specialist offset when the safety path becomes stale or unsafe.
            if not bool(correction_allowed.all()) and contact_correction.weight is not None:
                contact_correction.reset(~correction_allowed)
            conditioned = stock_observation.clone()
            conditioned[:, 9:12] = safe_velocity_yaw
            specialist_action = contact_correction_policy(conditioned).detach()
            crampon_in_contact = visible_crampon_contact(
                sensor_frame.packet_values,
                sensor_frame.valid_mask,
                sensor_frame.sample_age_s,
                stale_after_s=controller.config.stale_after_s,
            )
            crampon_in_contact &= correction_allowed
            return (
                contact_correction.step(base_action, specialist_action, crampon_in_contact),
                crampon_in_contact,
            )

        checkpoint = torch.load(args.visible_checkpoint, map_location=device, weights_only=False)
        estimator = CausalBilateralEstimator(**checkpoint["estimator_config"]).to(device)
        estimator.load_state_dict(checkpoint["estimator_state_dict"])
        supervisor = VisibleOnlySupervisor(**checkpoint["supervisor_config"]).to(device)
        supervisor.load_state_dict(checkpoint["supervisor_state_dict"])
        estimator.eval()
        supervisor.eval()

        def locomotion(stock_observation: torch.Tensor, safe_command: torch.Tensor) -> torch.Tensor:
            conditioned = stock_observation.clone()
            conditioned[:, 9:12] = safe_command
            return stock(conditioned).detach()

        controller = EverestController(
            estimator=estimator,
            supervisor=supervisor,
            locomotion_policy=locomotion,
            shield=SafetyShield(ShieldConfig(min_dwell_steps=2, commit_hysteresis_steps=3)),
            config=EverestControllerConfig(
                history_steps=31,
                control_rate_hz=args.controller_rate_hz,
                minimum_history_steps=6,
            ),
        )
        action = (
            safe_hold_action(observation["policy"])
            if args.mode == "active"
            else stock_action(observation["policy"])
        )
        posture_offsets = None
        posture_grid_values = None
        lower_body_grid_values = None
        balance_grid_values = None
        balance_pitch_gain = None
        balance_rate_gain = None
        balance_ankle_bias = None
        ankle_pitch_indices: tuple[int, int] | None = None
        selected_diagnostic_grids = sum(
            bool(value)
            for value in (
                args.diagnostic_posture_grid,
                args.diagnostic_balance_grid,
                args.diagnostic_balance_full_grid,
                args.diagnostic_lower_body_grid,
                args.diagnostic_candidate_command_grid,
            )
        )
        if selected_diagnostic_grids > 1:
            raise ValueError("select only one diagnostic grid")
        if args.diagnostic_zero_stock_command and selected_diagnostic_grids != 1:
            raise ValueError("diagnostic zero command requires one diagnostic grid")
        if selected_diagnostic_grids == 1:
            if args.mode != "shadow":
                raise ValueError("diagnostic action offsets are allowed only in shadow mode")
            grid_side = math.isqrt(args.num_envs)
            if args.diagnostic_balance_full_grid:
                if args.num_envs != 512:
                    raise ValueError("full balance grid requires exactly 512 environments")
            elif args.diagnostic_lower_body_grid:
                if args.num_envs != 256:
                    raise ValueError("lower-body grid requires exactly 256 environments")
            elif args.diagnostic_candidate_command_grid:
                if args.num_envs != 8:
                    raise ValueError("candidate command grid requires exactly eight environments")
            elif grid_side * grid_side != args.num_envs or grid_side < 2:
                raise ValueError("diagnostic grids require a square num-envs >= 4")
            action_term = env.unwrapped.action_manager._terms["joint_pos"]
            joint_index = {name: index for index, name in enumerate(action_term._joint_names)}
            required_names = (
                "left_hip_pitch_joint",
                "right_hip_pitch_joint",
                "left_ankle_pitch_joint",
                "right_ankle_pitch_joint",
                "left_knee_joint",
                "right_knee_joint",
                "torso_joint",
            )
            missing_names = [name for name in required_names if name not in joint_index]
            if missing_names:
                raise RuntimeError(f"diagnostic posture joints missing: {missing_names}")
            ankle_pitch_indices = (
                joint_index["left_ankle_pitch_joint"],
                joint_index["right_ankle_pitch_joint"],
            )
            posture_offsets = torch.zeros_like(action)
            if args.diagnostic_candidate_command_grid:
                if not 0.15 <= args.diagnostic_stock_vx_maximum <= 0.80:
                    raise ValueError("diagnostic stock vx maximum must be in [0.15, 0.80]")
                diagnostic_stock_vx = torch.linspace(
                    0.15, args.diagnostic_stock_vx_maximum, args.num_envs, device=device
                )
                hip_pitch = torch.full((args.num_envs,), -0.20, device=device)
                knee = torch.full((args.num_envs,), -0.50, device=device)
                ankle_pitch = torch.full((args.num_envs,), -1.0 / 6.0, device=device)
                torso = torch.full((args.num_envs,), 0.20, device=device)
                posture_offsets[:, joint_index["left_hip_pitch_joint"]] = hip_pitch
                posture_offsets[:, joint_index["right_hip_pitch_joint"]] = hip_pitch
                posture_offsets[:, joint_index["left_knee_joint"]] = knee
                posture_offsets[:, joint_index["right_knee_joint"]] = knee
                posture_offsets[:, ankle_pitch_indices[0]] = ankle_pitch
                posture_offsets[:, ankle_pitch_indices[1]] = ankle_pitch
                posture_offsets[:, joint_index["torso_joint"]] = torso
                lower_body_grid_values = (
                    torch.stack((hip_pitch, knee, ankle_pitch, torso), dim=-1).cpu().tolist()
                )
            elif args.diagnostic_lower_body_grid:
                environment_index = torch.arange(args.num_envs, device=device)
                torso_index = environment_index % 4
                ankle_index = (environment_index // 4) % 4
                knee_index = (environment_index // 16) % 4
                hip_index = (environment_index // 64) % 4
                hip_pitch = torch.linspace(-0.20, 0.20, 4, device=device)[hip_index]
                knee = torch.linspace(-0.50, 0.50, 4, device=device)[knee_index]
                ankle_pitch = torch.linspace(-0.50, 0.50, 4, device=device)[ankle_index]
                torso = torch.linspace(-0.20, 0.20, 4, device=device)[torso_index]
                posture_offsets[:, joint_index["left_hip_pitch_joint"]] = hip_pitch
                posture_offsets[:, joint_index["right_hip_pitch_joint"]] = hip_pitch
                posture_offsets[:, joint_index["left_knee_joint"]] = knee
                posture_offsets[:, joint_index["right_knee_joint"]] = knee
                posture_offsets[:, ankle_pitch_indices[0]] = ankle_pitch
                posture_offsets[:, ankle_pitch_indices[1]] = ankle_pitch
                posture_offsets[:, joint_index["torso_joint"]] = torso
                lower_body_grid_values = (
                    torch.stack((hip_pitch, knee, ankle_pitch, torso), dim=-1).cpu().tolist()
                )
            elif args.diagnostic_posture_grid:
                if (
                    not math.isfinite(args.diagnostic_hip_pitch_extent)
                    or not math.isfinite(args.diagnostic_ankle_pitch_extent)
                    or args.diagnostic_hip_pitch_extent <= 0.0
                    or args.diagnostic_ankle_pitch_extent <= 0.0
                ):
                    raise ValueError("diagnostic posture extents must be finite and positive")
                hip_pitch = torch.linspace(
                    -args.diagnostic_hip_pitch_extent,
                    args.diagnostic_hip_pitch_extent,
                    grid_side,
                    device=device,
                ).repeat_interleave(grid_side)
                ankle_pitch = torch.linspace(
                    -args.diagnostic_ankle_pitch_extent,
                    args.diagnostic_ankle_pitch_extent,
                    grid_side,
                    device=device,
                ).repeat(grid_side)
                posture_offsets[:, joint_index["left_hip_pitch_joint"]] = hip_pitch
                posture_offsets[:, joint_index["right_hip_pitch_joint"]] = hip_pitch
                posture_offsets[:, ankle_pitch_indices[0]] = ankle_pitch
                posture_offsets[:, ankle_pitch_indices[1]] = ankle_pitch
                posture_grid_values = torch.stack((hip_pitch, ankle_pitch), dim=-1).cpu().tolist()
            elif args.diagnostic_balance_full_grid:
                environment_index = torch.arange(args.num_envs, device=device)
                rate_index = environment_index % 4
                pitch_index = (environment_index // 4) % 8
                ankle_index = (environment_index // 32) % 4
                hip_index = (environment_index // 128) % 4
                hip_bias = torch.linspace(-1.0, 1.0, 4, device=device)[hip_index]
                balance_ankle_bias = torch.linspace(-4.0, 4.0, 4, device=device)[ankle_index]
                balance_pitch_gain = torch.linspace(-8.0, 8.0, 8, device=device)[pitch_index]
                balance_rate_gain = torch.linspace(-3.0, 3.0, 4, device=device)[rate_index]
                posture_offsets[:, joint_index["left_hip_pitch_joint"]] = hip_bias
                posture_offsets[:, joint_index["right_hip_pitch_joint"]] = hip_bias
                balance_grid_values = (
                    torch.stack(
                        (hip_bias, balance_ankle_bias, balance_pitch_gain, balance_rate_gain),
                        dim=-1,
                    )
                    .cpu()
                    .tolist()
                )
            else:
                balance_pitch_gain = torch.linspace(
                    -8.0, 8.0, grid_side, device=device
                ).repeat_interleave(grid_side)
                balance_rate_gain = torch.linspace(-3.0, 3.0, grid_side, device=device).repeat(
                    grid_side
                )
                balance_ankle_bias = torch.full_like(balance_pitch_gain, -1.7)
                hip_bias = torch.full_like(balance_pitch_gain, 0.43)
                posture_offsets[:, joint_index["left_hip_pitch_joint"]] = hip_bias
                posture_offsets[:, joint_index["right_hip_pitch_joint"]] = hip_bias
                balance_grid_values = (
                    torch.stack(
                        (hip_bias, balance_ankle_bias, balance_pitch_gain, balance_rate_gain),
                        dim=-1,
                    )
                    .cpu()
                    .tolist()
                )

        def apply_diagnostic_action_offsets(joint_action: torch.Tensor) -> torch.Tensor:
            if posture_offsets is None:
                return joint_action
            effective_offsets = posture_offsets
            if balance_pitch_gain is not None:
                assert (
                    balance_rate_gain is not None
                    and balance_ankle_bias is not None
                    and ankle_pitch_indices is not None
                )
                robot = env.unwrapped.scene["robot"]
                dynamic_ankle = (
                    balance_ankle_bias
                    + balance_pitch_gain * robot.data.projected_gravity_b[:, 0]
                    + balance_rate_gain * robot.data.root_ang_vel_b[:, 1]
                ).clamp(-4.0, 4.0)
                effective_offsets = posture_offsets.clone()
                effective_offsets[:, ankle_pitch_indices[0]] = dynamic_ankle
                effective_offsets[:, ankle_pitch_indices[1]] = dynamic_ankle
            return joint_action + effective_offsets

        action = apply_diagnostic_action_offsets(action)
        previous = None
        previous_valid = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        conformal = torch.as_tensor(
            checkpoint.get("conformal99", checkpoint["conformal95"]), device=device
        )
        bearing_absolute_radius_n = checkpoint.get("bearing_capacity_absolute_conformal99_n")
        target_index = {name: index for index, name in enumerate(REGRESSION_NAMES)}
        action_counts: Counter[int] = Counter()
        contact_correction_ticks = 0
        contact_correction_action_samples = 0
        contact_correction_rms_sum = 0.0
        contact_correction_rms_max = 0.0
        preference_counts: Counter[int] = Counter()
        shield_reason_counts: Counter[int] = Counter()
        predicate_names = (
            "bearing_unsafe",
            "void_predicted",
            "fracture_predicted",
            "fracture_degraded",
            "bilateral_slip_predicted",
            "severe_bilateral_slip",
            "target_unsafe",
            "stance_unsafe",
            "settling",
        )
        predicate_counts: Counter[str] = Counter()
        predicate_counts_by_environment = {
            name: torch.zeros(args.num_envs, dtype=torch.int64, device=device)
            for name in predicate_names
        }
        target_diagnostic_samples: dict[str, list[float]] = {
            name: []
            for name in (
                "minimum_bearing_mean_n",
                "minimum_bearing_lower_n",
                "maximum_bearing_uncertainty_n",
                "maximum_damage_upper",
                "maximum_fracture_probability",
                "minimum_slip_probability",
                "best_foot_slip_margin_lower_n",
            )
        }
        maximum_abs_z = torch.zeros(estimator.input_feature_size, device=device)
        terminations = 0
        terminations_by_environment = torch.zeros(args.num_envs, dtype=torch.int64, device=device)
        recoveries = 0
        recoveries_by_environment = torch.zeros(args.num_envs, dtype=torch.int64, device=device)
        stale_ticks = 0
        stale_by_environment = torch.zeros(args.num_envs, dtype=torch.int64, device=device)
        ood_ticks = 0
        ood_by_environment = torch.zeros(args.num_envs, dtype=torch.int64, device=device)
        force_sum = 0.0
        force_max = 0.0
        maximum_total_normal_force = 0.0
        final_total_normal_force = 0.0
        penetration_max = 0.0
        rear_load_fraction_sum = torch.zeros(args.num_envs, device=device)
        rear_load_fraction_samples = torch.zeros(args.num_envs, dtype=torch.int64, device=device)
        forward_velocity_error_sum = torch.zeros(args.num_envs, device=device)
        forward_velocity_sum = torch.zeros(args.num_envs, device=device)
        requested_velocity_sum = torch.zeros(args.num_envs, device=device)
        motion_metric_samples = torch.zeros(args.num_envs, dtype=torch.int64, device=device)
        action_delta_rms_sum = torch.zeros(args.num_envs, device=device)
        action_delta_rms_max = torch.zeros(args.num_envs, device=device)
        style_deviation_rms_sum = torch.zeros(args.num_envs, device=device)
        style_deviation_rms_max = torch.zeros(args.num_envs, device=device)
        accepted_safe_vx_sum = torch.zeros(args.num_envs, device=device)
        accepted_safe_vx_samples = torch.zeros(args.num_envs, dtype=torch.int64, device=device)
        previous_joint_action = action.detach().clone()
        sensor_ticks = 0
        reward_sum = 0.0
        base_height_min = float("inf")
        base_height_min_by_environment = torch.full((args.num_envs,), float("inf"), device=device)
        diagnostic_trace: list[dict[str, float | int]] = []
        start = time.perf_counter()
        for _ in range(args.steps):
            observation, reward, terminated, truncated, _ = env.step(action)
            reset_mask = terminated | truncated
            reward_sum += float(reward.mean())
            terminations += int(terminated.sum())
            terminations_by_environment += terminated.to(dtype=torch.int64)
            robot = env.unwrapped.scene["robot"]
            base_height = (
                (robot.data.root_pos_w - env.unwrapped._everest_terrain_origin)
                * env.unwrapped._everest_terrain_normal
            ).sum(dim=-1)
            base_height_min_by_environment = torch.minimum(
                base_height_min_by_environment, base_height
            )
            base_height_min = min(base_height_min, float(base_height.min()))
            valid_motion = ~reset_mask
            requested_velocity = env.unwrapped.command_manager.get_command("base_velocity")[:, 0]
            forward_velocity = robot.data.root_lin_vel_b[:, 0]
            forward_velocity_error_sum += torch.where(
                valid_motion,
                (forward_velocity - requested_velocity).abs(),
                torch.zeros_like(requested_velocity),
            )
            forward_velocity_sum += torch.where(
                valid_motion, forward_velocity, torch.zeros_like(forward_velocity)
            )
            requested_velocity_sum += torch.where(
                valid_motion, requested_velocity, torch.zeros_like(requested_velocity)
            )
            motion_metric_samples += valid_motion.to(dtype=torch.int64)
            latest_wrench = env.unwrapped.everest_latest_wrench
            if latest_wrench is not None:
                normal_force = latest_wrench.probe_normal_force_n.clamp_min(0.0)
                total_load = normal_force.sum(dim=(1, 2))
                rear_fraction = normal_force[:, :, 2:].sum(dim=(1, 2)) / total_load.clamp_min(20.0)
                loaded = total_load >= 20.0
                rear_load_fraction_sum += torch.where(
                    loaded, rear_fraction, torch.zeros_like(rear_fraction)
                )
                rear_load_fraction_samples += loaded.to(dtype=torch.int64)
            frames = env.unwrapped.pop_everest_sensor_frames()
            output = None
            latest_sensor_frame: VisibleSensorBatch | None = None
            for frame in frames:
                latest_sensor_frame = frame
                context = pack_context(frame)
                command_context = pack_commands(frame)
                packet_features = torch.cat(
                    (frame.packet_values, frame.valid_mask.float(), frame.sample_age_s, context),
                    dim=-1,
                )
                z = (packet_features - estimator.input_mean) / estimator.input_std
                maximum_abs_z = torch.maximum(maximum_abs_z, z.abs().amax(dim=(0, 1)))
                ood = z.abs().amax(dim=(1, 2)) > 12.0
                body_load = frame.context["body_weight_on_foot_n"]
                stance_safe = body_load.amax(dim=-1) > 80.0
                settling = frame.packet_values[..., 11:14].abs().amax(dim=(1, 2)) > 6.0
                target_predicates: dict[str, torch.Tensor] = {}
                if previous is None:
                    target_safe = torch.ones(args.num_envs, dtype=torch.bool, device=device)
                else:
                    regression_mean = previous.estimator.regression_mean
                    regression_log_scale = previous.estimator.regression_log_scale
                    event_probability = previous.estimator.event_logits.sigmoid()
                    uncertainty = conformal * regression_log_scale.exp()
                    lower = regression_mean - uncertainty
                    upper = regression_mean + uncertainty
                    bearing_lower = lower[..., target_index["bearing_capacity_n"]]
                    bearing_uncertainty = uncertainty[..., target_index["bearing_capacity_n"]]
                    if bearing_absolute_radius_n is not None:
                        bearing_lower = regression_mean[
                            ..., target_index["bearing_capacity_n"]
                        ] - float(bearing_absolute_radius_n)
                        bearing_uncertainty = torch.full_like(
                            bearing_lower, float(bearing_absolute_radius_n)
                        )
                    bearing_unsafe = bearing_lower.amin(dim=-1) < MINIMUM_BEARING_CAPACITY_N
                    void_predicted = (event_probability[..., 0] > 0.5).any(dim=-1)
                    fracture_predicted = (event_probability[..., 1] > 0.5).any(dim=-1)
                    fracture_degraded = fracture_predicted & (
                        upper[..., target_index["damage_state"]].amax(dim=-1)
                        > FRACTURE_DAMAGE_CAUTION
                    )
                    bilateral_slip_predicted = (event_probability[..., 2] > 0.5).all(dim=-1)
                    severe_bilateral_slip = bilateral_slip_predicted & (
                        lower[..., target_index["slip_margin_n"]].amax(dim=-1)
                        < SEVERE_SLIP_MARGIN_N
                    )
                    target_safe = conservative_target_safe(
                        regression_mean,
                        regression_log_scale,
                        previous.estimator.event_logits,
                        conformal,
                        bearing_capacity_index=target_index["bearing_capacity_n"],
                        bearing_capacity_absolute_radius_n=bearing_absolute_radius_n,
                        damage_index=target_index["damage_state"],
                        slip_margin_index=target_index["slip_margin_n"],
                    )
                    target_safe = torch.where(
                        previous_valid, target_safe, torch.ones_like(target_safe)
                    )
                    target_predicates = {
                        "bearing_unsafe": bearing_unsafe & previous_valid,
                        "void_predicted": void_predicted & previous_valid,
                        "fracture_predicted": fracture_predicted & previous_valid,
                        "fracture_degraded": fracture_degraded & previous_valid,
                        "bilateral_slip_predicted": bilateral_slip_predicted & previous_valid,
                        "severe_bilateral_slip": severe_bilateral_slip & previous_valid,
                        "target_unsafe": ~target_safe,
                    }
                    target_diagnostics = {
                        "minimum_bearing_mean_n": regression_mean[
                            ..., target_index["bearing_capacity_n"]
                        ].amin(dim=-1),
                        "minimum_bearing_lower_n": bearing_lower.amin(dim=-1),
                        "maximum_bearing_uncertainty_n": bearing_uncertainty.amax(dim=-1),
                        "maximum_damage_upper": upper[..., target_index["damage_state"]].amax(
                            dim=-1
                        ),
                        "maximum_fracture_probability": event_probability[..., 1].amax(dim=-1),
                        "minimum_slip_probability": event_probability[..., 2].amin(dim=-1),
                        "best_foot_slip_margin_lower_n": lower[
                            ..., target_index["slip_margin_n"]
                        ].amax(dim=-1),
                    }
                    for name, values in target_diagnostics.items():
                        target_diagnostic_samples[name].extend(values.detach().cpu().tolist())
                target_predicates.update(
                    {
                        "stance_unsafe": ~stance_safe,
                        "settling": settling,
                    }
                )
                for name, predicate in target_predicates.items():
                    predicate_counts[name] += int(predicate.sum())
                    predicate_counts_by_environment[name] += predicate.to(dtype=torch.int64)
                signals = ShieldSignals(
                    stale=torch.zeros(args.num_envs, dtype=torch.bool, device=device),
                    ood=ood,
                    target_safe=target_safe,
                    stance_safe=stance_safe,
                    settling=settling,
                )
                output = controller.step(
                    frame,
                    stock_observation=observation["policy"],
                    requested_velocity_yaw=requested_command(frame),
                    shield_signals=signals,
                    deployable_context=context,
                    deployable_command_gait_context=command_context,
                )
                accepted_safe_vx_sum += output.safe_command.velocity_yaw[:, 0]
                accepted_safe_vx_samples += 1
                previous = output
                previous_valid |= ~reset_mask
                action_counts.update(output.shield.action.cpu().tolist())
                preference_counts.update(output.supervisor_preference.cpu().tolist())
                shield_reason_counts.update(output.shield.reason.cpu().tolist())
                recoveries += int(output.recovery_request.sum())
                recoveries_by_environment += output.recovery_request.to(dtype=torch.int64)
                stale_ticks += int(output.stale.sum())
                stale_by_environment += output.stale.to(dtype=torch.int64)
                ood_ticks += int(ood.sum())
                ood_by_environment += ood.to(dtype=torch.int64)
                wrench = env.unwrapped.everest_latest_wrench
                force_sum += float(wrench.probe_normal_force_n.mean())
                force_max = max(force_max, float(wrench.probe_normal_force_n.max()))
                final_total_normal_force = float(wrench.probe_normal_force_n.sum(dim=(1, 2)).mean())
                maximum_total_normal_force = max(
                    maximum_total_normal_force, final_total_normal_force
                )
                penetration_max = max(penetration_max, float(wrench.probe_penetration_m.max()))
                sensor_ticks += 1
            if len(diagnostic_trace) < 50:
                wrench = env.unwrapped.everest_latest_wrench
                diagnostic_trace.append(
                    {
                        "base_height_m": float(
                            (
                                (robot.data.root_pos_w - env.unwrapped._everest_terrain_origin)
                                * env.unwrapped._everest_terrain_normal
                            )
                            .sum(dim=-1)
                            .mean()
                        ),
                        "mean_probe_force_n": float(wrench.probe_normal_force_n.mean()),
                        "max_penetration_m": float(wrench.probe_penetration_m.max()),
                        "terminated": int(terminated.sum()),
                    }
                )
            correction_contact = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            if args.mode == "active":
                action = (
                    output.joint_action
                    if output is not None
                    else safe_hold_action(observation["policy"])
                )
                base_action = action
                if output is not None and latest_sensor_frame is not None:
                    action, correction_contact = contact_corrected_action(
                        base_action,
                        output.safe_command.velocity_yaw,
                        observation["policy"],
                        latest_sensor_frame,
                        (~output.stale)
                        & (output.shield.action == int(ShieldAction.COMMIT)),
                    )
                    correction_rms = (action - base_action).square().mean(dim=-1).sqrt()
                    contact_correction_ticks += int(correction_contact.sum())
                    if contact_correction_policy is not None:
                        contact_correction_action_samples += args.num_envs
                    contact_correction_rms_sum += float(correction_rms.sum())
                    contact_correction_rms_max = max(
                        contact_correction_rms_max, float(correction_rms.max())
                    )
            else:
                action = stock_action(observation["policy"])
            action = apply_diagnostic_action_offsets(action)
            action_delta_rms = (action - previous_joint_action).square().mean(dim=-1).sqrt()
            action_delta_rms_sum += torch.where(
                valid_motion, action_delta_rms, torch.zeros_like(action_delta_rms)
            )
            action_delta_rms_max = torch.maximum(
                action_delta_rms_max,
                torch.where(valid_motion, action_delta_rms, torch.zeros_like(action_delta_rms)),
            )
            if style_reference is not None:
                reference_observation = observation["policy"]
                if diagnostic_stock_vx is not None:
                    reference_observation = reference_observation.clone()
                    reference_observation[:, 9] = diagnostic_stock_vx
                reference_action = style_reference(reference_observation).detach()
                style_deviation_rms = (action - reference_action).square().mean(dim=-1).sqrt()
                style_deviation_rms_sum += torch.where(
                    valid_motion, style_deviation_rms, torch.zeros_like(style_deviation_rms)
                )
                style_deviation_rms_max = torch.maximum(
                    style_deviation_rms_max,
                    torch.where(
                        valid_motion, style_deviation_rms, torch.zeros_like(style_deviation_rms)
                    ),
                )
            previous_joint_action = action.detach().clone()
            if bool(reset_mask.any()):
                controller.reset_environments(reset_mask)
                if contact_correction is not None:
                    contact_correction.reset(reset_mask)
                previous_valid[reset_mask] = False
                if args.mode == "active":
                    # Stock/JIT outputs are inference tensors. Clone before a
                    # selective reset write so terminated environments return to
                    # the zero-command stock hold without an inference-mode error.
                    action = action.clone()
                    action[reset_mask] = safe_hold_action(observation["policy"])[reset_mask]
                    previous_joint_action[reset_mask] = action[reset_mask]
        duration = time.perf_counter() - start
        feature_names = (
            [f"packet[{index}]" for index in range(19)]
            + [f"valid[{index}]" for index in range(19)]
            + [f"age[{index}]" for index in range(19)]
            + [
                "context.foot_position_x",
                "context.foot_position_y",
                "context.foot_position_z",
                "context.foot_velocity_x",
                "context.foot_velocity_y",
                "context.foot_velocity_z",
                "context.pelvis_roll",
                "context.pelvis_pitch",
                "context.pelvis_yaw",
                "context.commanded_probe_load_n",
                "context.commanded_foot_speed_mps",
                "context.body_weight_on_foot_n",
            ]
        )
        top_z = sorted(
            zip(feature_names, maximum_abs_z.cpu().tolist(), strict=True),
            key=lambda item: item[1],
            reverse=True,
        )[:12]
        sensor_environment_ticks = max(sensor_ticks * args.num_envs, 1)
        target_diagnostic_summary = {}
        for name, samples in target_diagnostic_samples.items():
            values = torch.tensor(samples, dtype=torch.float64)
            target_diagnostic_summary[name] = {
                "minimum": float(values.min()),
                "p01": float(values.quantile(0.01)),
                "median": float(values.median()),
                "p99": float(values.quantile(0.99)),
                "maximum": float(values.max()),
            }
        ood_fraction = ood_ticks / sensor_environment_ticks
        stale_fraction = stale_ticks / sensor_environment_ticks
        recovery_fraction = recoveries / sensor_environment_ticks
        gait_metrics = env.unwrapped.everest_gait_metrics()
        accepted_safe_vx = accepted_safe_vx_sum / accepted_safe_vx_samples.clamp_min(1)
        gate_failures: list[str] = []
        if args.max_terminations is not None and terminations > args.max_terminations:
            gate_failures.append(f"terminations {terminations} > {args.max_terminations}")
        if args.min_base_height_m is not None and base_height_min < args.min_base_height_m:
            gate_failures.append(
                f"base_height_min_m {base_height_min:.6f} < {args.min_base_height_m:.6f}"
            )
        if args.max_ood_fraction is not None and ood_fraction > args.max_ood_fraction:
            gate_failures.append(f"ood_fraction {ood_fraction:.6f} > {args.max_ood_fraction:.6f}")
        if args.max_stale_fraction is not None and stale_fraction > args.max_stale_fraction:
            gate_failures.append(
                f"stale_fraction {stale_fraction:.6f} > {args.max_stale_fraction:.6f}"
            )
        if (
            args.max_recovery_fraction is not None
            and recovery_fraction > args.max_recovery_fraction
        ):
            gate_failures.append(
                f"recovery_fraction {recovery_fraction:.6f} > {args.max_recovery_fraction:.6f}"
            )
        if (
            args.min_accepted_safe_vx is not None
            and float(accepted_safe_vx.min()) < args.min_accepted_safe_vx
        ):
            gate_failures.append(
                f"minimum_accepted_safe_vx {float(accepted_safe_vx.min()):.6f} < {args.min_accepted_safe_vx:.6f}"
            )
        if (
            args.max_contact_lateral_speed_mps is not None
            and float(gait_metrics["force_weighted_stance_lateral_speed_mps"].max())
            > args.max_contact_lateral_speed_mps
        ):
            gate_failures.append(
                "force_weighted_stance_lateral_speed_mps exceeds "
                f"{args.max_contact_lateral_speed_mps:.6f}"
            )
        if (
            args.max_contact_slip_fraction is not None
            and float(gait_metrics["material_slipping_fraction"].max())
            > args.max_contact_slip_fraction
        ):
            gate_failures.append(
                f"material_slipping_fraction exceeds {args.max_contact_slip_fraction:.6f}"
            )
        if (
            args.min_mean_swing_lift_m is not None
            and float(gait_metrics["mean_swing_peak_m"].min()) < args.min_mean_swing_lift_m
        ):
            gate_failures.append(f"mean_swing_peak_m below {args.min_mean_swing_lift_m:.6f}")
        if (
            args.min_mean_stride_length_m is not None
            and float(gait_metrics["mean_stride_advance_m"].min()) < args.min_mean_stride_length_m
        ):
            gate_failures.append(f"mean_stride_advance_m below {args.min_mean_stride_length_m:.6f}")
        fault_cycles = env.unwrapped._everest_fault_cycle.cpu().tolist()
        initial_fault_codes = env.unwrapped._everest_initial_fault_code.cpu().tolist()
        cases = [
            {
                "case_id": case.case_id,
                "surface_id": case.surface_id,
                "hazard_id": case.hazard_id,
                "contact_mode_id": case.contact_mode_id,
                "incline_deg": case.incline_deg,
                "sensor_fault_cycle": int(fault_cycles[index]),
                "sensor_fault_mode_id": SENSOR_FAULT_MODES[int(initial_fault_codes[index])],
                "sensor_fault_final_mode_id": SENSOR_FAULT_MODES[
                    (index + int(fault_cycles[index])) % len(SENSOR_FAULT_MODES)
                ],
                **(
                    {"diagnostic_stock_vx_mps": float(diagnostic_stock_vx[index])}
                    if diagnostic_stock_vx is not None
                    else {}
                ),
                **(
                    {
                        "diagnostic_hip_pitch_action_offset": posture_grid_values[index][0],
                        "diagnostic_ankle_pitch_action_offset": posture_grid_values[index][1],
                    }
                    if posture_grid_values is not None
                    else {}
                ),
                **(
                    {
                        "diagnostic_hip_pitch_action_offset": lower_body_grid_values[index][0],
                        "diagnostic_knee_action_offset": lower_body_grid_values[index][1],
                        "diagnostic_ankle_pitch_action_offset": lower_body_grid_values[index][2],
                        "diagnostic_torso_action_offset": lower_body_grid_values[index][3],
                    }
                    if lower_body_grid_values is not None
                    else {}
                ),
                **(
                    {
                        "diagnostic_hip_pitch_action_offset": balance_grid_values[index][0],
                        "diagnostic_ankle_pitch_action_bias": balance_grid_values[index][1],
                        "diagnostic_pitch_gain": balance_grid_values[index][2],
                        "diagnostic_pitch_rate_gain": balance_grid_values[index][3],
                    }
                    if balance_grid_values is not None
                    else {}
                ),
            }
            for index, case in enumerate(env.unwrapped.everest_cases)
        ]
        result = {
            "status": "passed" if not gate_failures else "failed",
            "gate_failures": gate_failures,
            "mode": args.mode,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "controller_rate_hz": args.controller_rate_hz,
            "contact_correction": {
                "enabled": contact_correction_policy is not None,
                "policy": (
                    str(args.contact_correction_policy)
                    if args.contact_correction_policy is not None
                    else None
                ),
                "maximum_action_residual": args.contact_correction_max_residual,
                "weight_step": args.contact_correction_weight_step,
                "contact_source": "fresh_visible_axial_force_and_penetration",
                "contact_ticks": contact_correction_ticks,
                "mean_action_delta_rms": (
                    contact_correction_rms_sum / max(contact_correction_action_samples, 1)
                ),
                "maximum_action_delta_rms": contact_correction_rms_max,
            },
            "duration_s": duration,
            "sim_steps_per_s": args.steps * args.num_envs / duration,
            "terminations": terminations,
            "terminations_by_environment": terminations_by_environment.cpu().tolist(),
            "cases": cases,
            "mean_reward_per_step": reward_sum / args.steps,
            "base_height_min_m": base_height_min,
            "base_height_min_m_by_environment": base_height_min_by_environment.cpu().tolist(),
            "sensor_ticks": sensor_ticks,
            "requested_vx_mps": args.requested_vx,
            "suite_config_path": str(suite_config_path) if suite_config_path is not None else None,
            "mean_accepted_safe_vx_mps_by_environment": accepted_safe_vx.cpu().tolist(),
            "native_gait_metrics": {
                name: values.cpu().tolist() for name, values in gait_metrics.items()
            },
            "mean_probe_force_n": force_sum / max(sensor_ticks, 1),
            "maximum_probe_force_n": force_max,
            "maximum_total_normal_force_n": maximum_total_normal_force,
            "final_total_normal_force_n": final_total_normal_force,
            "robot_weight_n": env.unwrapped._everest_robot_weight_n,
            "maximum_penetration_m": penetration_max,
            "target_rear_load_fraction_by_environment": (
                env.unwrapped._everest_target_rear_load_fraction.cpu().tolist()
            ),
            "mean_rear_load_fraction_by_environment": (
                rear_load_fraction_sum
                / rear_load_fraction_samples.to(dtype=rear_load_fraction_sum.dtype).clamp_min(1.0)
            )
            .cpu()
            .tolist(),
            "rear_load_fraction_samples_by_environment": (
                rear_load_fraction_samples.cpu().tolist()
            ),
            "mean_instantaneous_forward_velocity_error_mps_by_environment": (
                forward_velocity_error_sum
                / motion_metric_samples.to(dtype=forward_velocity_error_sum.dtype).clamp_min(1.0)
            )
            .cpu()
            .tolist(),
            "mean_forward_velocity_mps_by_environment": (
                forward_velocity_sum
                / motion_metric_samples.to(dtype=forward_velocity_sum.dtype).clamp_min(1.0)
            )
            .cpu()
            .tolist(),
            "mean_requested_velocity_mps_by_environment": (
                requested_velocity_sum
                / motion_metric_samples.to(dtype=requested_velocity_sum.dtype).clamp_min(1.0)
            )
            .cpu()
            .tolist(),
            "time_averaged_forward_velocity_error_mps_by_environment": (
                (forward_velocity_sum - requested_velocity_sum).abs()
                / motion_metric_samples.to(dtype=forward_velocity_sum.dtype).clamp_min(1.0)
            )
            .cpu()
            .tolist(),
            "mean_action_delta_rms_by_environment": (
                action_delta_rms_sum
                / motion_metric_samples.to(dtype=action_delta_rms_sum.dtype).clamp_min(1.0)
            )
            .cpu()
            .tolist(),
            "maximum_action_delta_rms_by_environment": action_delta_rms_max.cpu().tolist(),
            "mean_style_deviation_rms_by_environment": (
                style_deviation_rms_sum
                / motion_metric_samples.to(dtype=style_deviation_rms_sum.dtype).clamp_min(1.0)
            )
            .cpu()
            .tolist(),
            "maximum_style_deviation_rms_by_environment": (style_deviation_rms_max.cpu().tolist()),
            "style_reference_policy": (
                str(args.style_reference_policy)
                if args.style_reference_policy is not None
                else None
            ),
            "final_action_counts": {
                str(key): value for key, value in sorted(action_counts.items())
            },
            "supervisor_preference_counts": {
                str(key): value for key, value in sorted(preference_counts.items())
            },
            "shield_reason_counts": {
                str(key): value for key, value in sorted(shield_reason_counts.items())
            },
            "shield_predicate_counts": {name: predicate_counts[name] for name in predicate_names},
            "shield_predicate_fractions": {
                name: predicate_counts[name] / sensor_environment_ticks for name in predicate_names
            },
            "shield_predicate_counts_by_environment": {
                name: predicate_counts_by_environment[name].cpu().tolist()
                for name in predicate_names
            },
            "shield_target_diagnostic_summary": target_diagnostic_summary,
            "simulator_truth_initial_bearing_capacity_n_by_environment": (
                env.unwrapped.everest_wrench_bridge.material.parameters.bearing_capacity_n.mean(
                    dim=(1, 2)
                )
                .cpu()
                .tolist()
            ),
            "recovery_requests": recoveries,
            "recovery_requests_by_environment": recoveries_by_environment.cpu().tolist(),
            "stale_ticks": stale_ticks,
            "stale_ticks_by_environment": stale_by_environment.cpu().tolist(),
            "ood_ticks": ood_ticks,
            "ood_ticks_by_environment": ood_by_environment.cpu().tolist(),
            "ood_fraction": ood_fraction,
            "stale_fraction": stale_fraction,
            "recovery_fraction": recovery_fraction,
            "sensor_environment_ticks": sensor_environment_ticks,
            "diagnostic_trace": diagnostic_trace,
            "maximum_abs_normalized_features": {name: value for name, value in top_z},
            "claim_boundary": "Full Isaac G1 simulator shadow/active result; not hardware validation.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        exit_code = 0 if not gate_failures else 2
        env.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
