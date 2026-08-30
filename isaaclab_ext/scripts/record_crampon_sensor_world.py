#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
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
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]
if args.steps_per_surface < 2:
    raise ValueError("steps-per-surface must be at least two")
args.enable_cameras = True
_ISAAC_PROCESS_LOCK = acquire_isaac_process_lock()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_visual_ice_slab(surface_id: str) -> None:
    """Add a visual-only primitive; analytical contact remains the authoritative plane."""
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    world = UsdGeom.Xform.Define(stage, "/World/CramponSensorLab")
    world.GetPrim().CreateAttribute("everest:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)
    cube = UsdGeom.Cube.Define(stage, "/World/CramponSensorLab/IceDisplaySlab")
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*SURFACE_DISPLAY_COLORS[surface_id])])
    cube.CreateDisplayOpacityAttr([0.94])
    transform = UsdGeom.Xformable(cube)
    transform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.015))
    transform.AddScaleOp().Set(Gf.Vec3f(9.0, 9.0, 0.04))
    cube.GetPrim().CreateAttribute("everest:surfaceId", Sdf.ValueTypeNames.String).Set(surface_id)
    cube.GetPrim().CreateAttribute("everest:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)


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
            env_cfg.everest_play_contact_mode_id = "all_points_flat_foot"
            env_cfg.viewer.eye = tuple(args.camera_eye)
            env_cfg.viewer.lookat = tuple(args.camera_lookat)
            env_cfg.viewer.origin_type = "asset_root"
            env_cfg.viewer.env_index = 0
            env_cfg.viewer.asset_name = "robot"
            if hasattr(env_cfg.commands, "base_velocity"):
                env_cfg.commands.base_velocity.debug_vis = False
            surface_dir = raw_dir / surface_id
            surface_dir.mkdir(parents=True, exist_ok=True)
            env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
            add_visual_ice_slab(surface_id)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(surface_dir),
                step_trigger=lambda step: step == 0,
                video_length=args.steps_per_surface,
                name_prefix=surface_id,
                disable_logger=True,
            )
            observation, _ = env.reset()
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
                "terminations": terminations,
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
