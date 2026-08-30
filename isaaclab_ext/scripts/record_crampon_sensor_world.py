#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from hackathon_everest_isaaclab.runtime import acquire_isaac_process_lock
from hackathon_everest_isaaclab.tasks import register_cli
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

SURFACES = (
    "hard_glacier_ice",
    "fractured_blue_ice",
    "polished_wind_ice",
    "thin_snow_over_ice",
)
CONTACT_MODES = ("all_points_flat_foot", "hybrid_contact", "front_point_contact")
SURFACE_DISPLAY_COLORS = {
    "hard_glacier_ice": (0.52, 0.75, 0.90),
    "fractured_blue_ice": (0.22, 0.50, 0.82),
    "polished_wind_ice": (0.68, 0.88, 0.98),
    "thin_snow_over_ice": (0.88, 0.94, 0.98),
}
CHANNEL_GROUPS = {
    "axial_force_n": [0, 4],
    "penetration_m": [4, 8],
    "accelerometer_mps2": [8, 11],
    "gyroscope_rps": [11, 14],
    "radar_frontend": [14, 19],
}

register_cli()
parser = argparse.ArgumentParser(
    description="Record a native Isaac crampon sensor-lab world with exact bilateral telemetry"
)
parser.add_argument(
    "--task",
    default="Everest-Velocity-Flat-G1-Crampon-Stateful-Bootstrap-FrontPoint-Randomized-v0",
)
parser.add_argument("--policy", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--steps-per-surface", type=int, default=250)
parser.add_argument("--surfaces", nargs="+", choices=SURFACES, default=list(SURFACES))
parser.add_argument("--camera-eye", nargs=3, type=float, default=(2.4, 2.4, 1.4))
parser.add_argument("--camera-lookat", nargs=3, type=float, default=(0.0, 0.0, 0.75))
parser.add_argument("--incline-deg", type=float, default=0.0)
parser.add_argument("--contact-mode", choices=CONTACT_MODES, default="all_points_flat_foot")
parser.add_argument("--scene-seed", type=int, default=0)
parser.add_argument("--requested-vx", type=float, default=0.15)
parser.add_argument("--warmup-steps", type=int, default=100)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]
if args.steps_per_surface < 2:
    raise ValueError("steps-per-surface must be at least two")
if not 0.0 < args.requested_vx <= 0.80:
    raise ValueError("requested-vx must be in (0, 0.80]")
if args.warmup_steps < 0:
    raise ValueError("warmup-steps must be non-negative")
args.enable_cameras = True
_ISAAC_PROCESS_LOCK = acquire_isaac_process_lock()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_visual_ice_slab(surface_id: str, incline_deg: float, scene_seed: int) -> None:
    """Build a visibly sloped, collision-free alpine hill around the robot.

    The authored geometry rises in +X, matching ``suite_plane_normals``.  It is
    presentation geometry only: no collision API is applied, so the stateful
    analytical probe-wrench model remains the sole crampon contact authority.
    """
    import math

    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    native_ground = stage.GetPrimAtPath("/World/ground")
    if native_ground.IsValid():
        # Explicit descendant visibility is needed by the headless renderer;
        # visibility does not remove physics schemas or contact authority.
        for prim in Usd.PrimRange(native_ground):
            if prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeInvisible()
    world = UsdGeom.Xform.Define(stage, "/World/CramponSensorLab")
    world.GetPrim().CreateAttribute("everest:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)
    color = Gf.Vec3f(*SURFACE_DISPLAY_COLORS[surface_id])
    slope_radians = math.radians(incline_deg)

    def sloped_box(
        path: str,
        center: tuple[float, float, float],
        half_extents: tuple[float, float, float],
        display_color: Gf.Vec3f,
    ) -> UsdGeom.Cube:
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(2.0)
        cube.CreateDisplayColorAttr([display_color])
        transform = UsdGeom.Xformable(cube)
        transform.AddTranslateOp().Set(Gf.Vec3d(*center))
        # A negative Y rotation gives an upward normal (-sin(a), 0, cos(a)),
        # which is the exact analytical convention for a plane rising in +X.
        transform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, -incline_deg, 0.0))
        transform.AddScaleOp().Set(Gf.Vec3f(*half_extents))
        cube.GetPrim().CreateAttribute("everest:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)
        return cube

    # The top face passes through z=0 at the robot and extends far enough that
    # close and medium cameras see a continuous hill rather than a flat disk.
    slab_half_thickness = 0.08
    sloped_box(
        "/World/CramponSensorLab/IceHill",
        (0.0, 0.0, -slab_half_thickness * math.cos(slope_radians)),
        (24.0, 14.0, slab_half_thickness),
        color,
    )

    # A broad visual-only approach covers the native support-plane grid on the
    # downhill side.  It ends at x=0, where the commanded +X climb begins.
    approach = UsdGeom.Cube.Define(stage, "/World/CramponSensorLab/SnowApproach")
    approach.CreateSizeAttr(2.0)
    approach.CreateDisplayColorAttr([Gf.Vec3f(0.86, 0.93, 0.98)])
    approach_xform = UsdGeom.Xformable(approach)
    approach_xform.AddTranslateOp().Set(Gf.Vec3d(-24.0, 0.0, -0.02))
    approach_xform.AddScaleOp().Set(Gf.Vec3f(24.0, 30.0, 0.03))
    approach.GetPrim().CreateAttribute("everest:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)

    # Raised side banks make the incline readable from front, side, and rear
    # cameras while keeping the walking corridor visually open.
    bank_color = Gf.Vec3f(0.88, 0.94, 0.99)
    for side in (-1.0, 1.0):
        sloped_box(
            f"/World/CramponSensorLab/SnowBank_{'Left' if side > 0 else 'Right'}",
            (0.0, side * 12.7, 0.42),
            (24.0, 1.15, 0.50),
            bank_color,
        )

    # Thin cross-slope bands expose the grade in every shot.  They follow the
    # visual hill exactly and are spaced away from the robot's start point.
    contour_color = Gf.Vec3f(0.76, 0.89, 0.98)
    for index, x_position in enumerate((-16.0, -12.0, -8.0, -4.0, 4.0, 8.0, 12.0, 16.0)):
        sloped_box(
            f"/World/CramponSensorLab/SlopeBand_{index:02d}",
            (x_position, 0.0, x_position * math.tan(slope_radians) + 0.025),
            (0.07, 13.7, 0.018),
            contour_color,
        )

    # A broad crest at the uphill end makes the scene read as a hill instead
    # of an infinite mathematical plane.
    crest_x = 21.5
    sloped_box(
        "/World/CramponSensorLab/SummitCrest",
        (crest_x, 0.0, crest_x * math.tan(slope_radians) + 0.30),
        (2.2, 14.0, 0.38),
        bank_color,
    )

    sky = UsdGeom.Sphere.Define(stage, "/World/CramponSensorLab/SkyDome")
    sky.CreateRadiusAttr(150.0)
    sky.CreateDisplayColorAttr([Gf.Vec3f(0.32, 0.52, 0.78)])
    sky.GetPrim().CreateAttribute("everest:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)

    # Distant deterministic mountains frame the slope without adding support
    # collision near the robot.
    for index in range(16):
        angle = (2.0 * math.pi * index / 16.0) + 0.21 * (scene_seed % 7)
        radius = 42.0 + 8.0 * ((index * 7 + scene_seed) % 4)
        height = 12.0 + 6.0 * ((index * 5 + scene_seed) % 5)
        width = 7.0 + 1.5 * ((index * 3 + scene_seed) % 4)
        mountain = UsdGeom.Cone.Define(stage, f"/World/CramponSensorLab/Mountain_{index:02d}")
        mountain.CreateRadiusAttr(width)
        mountain.CreateHeightAttr(height)
        shade = 0.56 + 0.07 * ((index + scene_seed) % 4)
        mountain.CreateDisplayColorAttr([Gf.Vec3f(shade, min(0.97, shade + 0.1), 1.0)])
        transform = UsdGeom.Xformable(mountain)
        transform.AddTranslateOp().Set(
            Gf.Vec3d(radius * math.cos(angle), radius * math.sin(angle), height / 2.0 - 0.04)
        )
        mountain.GetPrim().CreateAttribute("everest:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)

    # Low irregular foreground ridges add depth without obscuring the robot.
    for index in range(6):
        angle = (2.0 * math.pi * index / 6.0) + 0.43
        ridge = UsdGeom.Cone.Define(stage, f"/World/CramponSensorLab/ForegroundRidge_{index:02d}")
        ridge.CreateRadiusAttr(4.0 + index % 3)
        ridge.CreateHeightAttr(2.0 + index % 2)
        ridge.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.85, 0.93)])
        transform = UsdGeom.Xformable(ridge)
        transform.AddTranslateOp().Set(
            Gf.Vec3d(28.0 * math.cos(angle), 28.0 * math.sin(angle), 1.0 - 0.04)
        )
        ridge.GetPrim().CreateAttribute("everest:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)


def packet_record(frame, step: int, sensor_tick: int) -> dict:
    return {
        "control_step": step,
        "sensor_tick": sensor_tick,
        "packet_values": frame.packet_values[0].detach().cpu().tolist(),
        "valid_mask": frame.valid_mask[0].detach().cpu().tolist(),
        "sample_age_s": frame.sample_age_s[0].detach().cpu().tolist(),
        "timestamp_s": frame.timestamp_s[0].detach().cpu().tolist(),
    }


def main() -> int:
    output_dir = args.output_dir.expanduser().resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    policy_path = args.policy.expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    report: dict = {
        "schema_version": "1.0.0",
        "artifact_type": "native_isaac_crampon_sensor_world",
        "task": args.task,
        "policy": {"path": str(policy_path), "sha256": sha256(policy_path)},
        "camera": {
            "eye_asset_root_m": list(args.camera_eye),
            "lookat_asset_root_m": list(args.camera_lookat),
        },
        "surfaces": [],
        "channel_groups": CHANNEL_GROUPS,
        "packet_abi": ["B", 2, 19],
        "channel_semantics": {
            "probe_order": ["(+x,+y)", "(+x,-y)", "(-x,+y)", "(-x,-y)"],
            "accelerometer_frame": "world axes; simulator adapter adds +9.81 to world z",
            "gyroscope_frame": "world axes",
            "radar_frontend": "five online adapter outputs; not reconstructed offline",
            "valid_mask": "true means fresh; it does not guarantee fault-free data",
            "surface_identity": "world metadata only; never part of the visible packet",
        },
        "claim_boundary": (
            "Project-authored Isaac simulator sensor demonstration. Material values are simulator "
            "priors, not surveyed Everest conditions or hardware calibration. The colored slab is "
            "a categorical visual aid only and has no contact physics."
        ),
    }
    env_cfg, _ = resolve_task_config(args.task, "")
    with launch_simulation(env_cfg, args):
        policy = torch.jit.load(str(policy_path), map_location=args.device or "cuda:0").eval()
        for surface_id in args.surfaces:
            env_cfg, _ = resolve_task_config(args.task, "")
            env_cfg.scene.num_envs = 1
            env_cfg.sim.device = args.device or "cuda:0"
            env_cfg.everest_require_complete_coverage = False
            env_cfg.everest_nominal_bootstrap_material = False
            env_cfg.everest_play_surface_id = surface_id
            env_cfg.everest_play_contact_mode_id = args.contact_mode
            env_cfg.everest_play_incline_deg = args.incline_deg
            env_cfg.everest_use_case_inclines = True
            env_cfg.viewer.eye = tuple(args.camera_eye)
            env_cfg.viewer.lookat = tuple(args.camera_lookat)
            env_cfg.viewer.origin_type = "asset_root"
            env_cfg.viewer.env_index = 0
            env_cfg.viewer.asset_name = "robot"
            if hasattr(env_cfg.commands, "base_velocity"):
                env_cfg.commands.base_velocity.ranges.lin_vel_x = (
                    args.requested_vx,
                    args.requested_vx,
                )
                env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
                env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
                env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
                env_cfg.commands.base_velocity.debug_vis = False
            surface_dir = raw_dir / surface_id
            surface_dir.mkdir(parents=True, exist_ok=True)
            env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
            add_visual_ice_slab(surface_id, args.incline_deg, args.scene_seed)
            observation, _ = env.reset()
            # Establish a stable stock-policy gait before the recorder receives a frame.
            # This intentionally excludes reset/transient footage from every final video.
            with torch.inference_mode():
                for _ in range(args.warmup_steps):
                    observation, _, _, _, _ = env.step(policy(observation["policy"]))
                    env.unwrapped.pop_everest_sensor_frames()
            start_root_position = (
                env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().clone()
            )
            terrain_normal = env.unwrapped._everest_terrain_normal[0].detach().cpu().clone()
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(surface_dir),
                step_trigger=lambda step: step == 0,
                video_length=args.steps_per_surface,
                name_prefix=surface_id,
                disable_logger=True,
            )
            records: list[dict | None] = []
            all_sensor_packets: list[dict] = []
            terminations = 0
            with torch.inference_mode():
                for step in range(args.steps_per_surface):
                    action = policy(observation["policy"])
                    observation, _, terminated, truncated, _ = env.step(action)
                    terminations += int(torch.count_nonzero(terminated))
                    sensor_frames = env.unwrapped.pop_everest_sensor_frames()
                    tick_start = len(all_sensor_packets)
                    step_packets = [
                        packet_record(frame, step, tick_start + index)
                        for index, frame in enumerate(sensor_frames)
                    ]
                    all_sensor_packets.extend(step_packets)
                    if step_packets:
                        latest = dict(step_packets[-1])
                        latest["video_frame_index"] = step
                        latest["sensor_tick_start"] = tick_start
                        latest["sensor_tick_end_exclusive"] = len(all_sensor_packets)
                        latest["sensor_packets_this_control_step"] = len(step_packets)
                        records.append(latest)
                    else:
                        records.append(None)
                    if bool(truncated.any()):
                        # The environment performs its own selective reset. Recording continues.
                        pass
            end_root_position = (
                env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().clone()
            )
            root_displacement = end_root_position - start_root_position
            expected_climb_m = float(root_displacement[0]) * math.tan(
                math.radians(args.incline_deg)
            )
            env.close()
            videos = sorted(surface_dir.glob("*.mp4"))
            if len(videos) != 1:
                raise RuntimeError(f"Expected one video for {surface_id}, found {len(videos)}")
            video = videos[0]
            surface_report = {
                "surface_id": surface_id,
                "surface_family": "layered" if surface_id == "thin_snow_over_ice" else "ice",
                "physics_plane": "stateful analytical material",
                "display_slab": "visual-only categorical primitive",
                "video": {"path": str(video), "sha256": sha256(video)},
                "steps": args.steps_per_surface,
                "warmup_steps_excluded": args.warmup_steps,
                "requested_vx_mps": args.requested_vx,
                "terminations": terminations,
                "incline_deg": args.incline_deg,
                "contact_mode": args.contact_mode,
                "travel_direction": "uphill_+X",
                "terrain_normal_world": terrain_normal.tolist(),
                "locomotion": {
                    "start_root_position_world_m": start_root_position.tolist(),
                    "end_root_position_world_m": end_root_position.tolist(),
                    "root_displacement_world_m": root_displacement.tolist(),
                    "forward_progress_m": float(root_displacement[0]),
                    "vertical_climb_m": float(root_displacement[2]),
                    "expected_climb_from_progress_m": expected_climb_m,
                    "climb_tracking_error_m": float(root_displacement[2]) - expected_climb_m,
                },
                "video_alignment": (
                    "Each 50 Hz video/control frame displays the latest visible packet from that "
                    "control step. all_sensor_packets preserves every higher-rate sensor tick."
                ),
                "telemetry": records,
                "all_sensor_packets": all_sensor_packets,
            }
            report["surfaces"].append(surface_report)
            (surface_dir / "telemetry.json").write_text(json.dumps(surface_report, indent=2) + "\n")
        report_path = output_dir / "sensor_world.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {**report, "surfaces": [s["surface_id"] for s in report["surfaces"]]}, indent=2
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
