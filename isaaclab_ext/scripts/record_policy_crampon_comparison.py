#!/usr/bin/env python3
"""Record two policies side by side with crampon support enabled versus ablated."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
COLORS = {
    "hard_glacier_ice": (0.26, 0.47, 0.66),
    "fractured_blue_ice": (0.17, 0.40, 0.61),
    "polished_wind_ice": (0.34, 0.55, 0.71),
    "thin_snow_over_ice": (0.80, 0.87, 0.93),
}

register_cli()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Everest-Velocity-Flat-G1-Crampon-Stateful-Play-v0")
parser.add_argument("--crampon-policy", type=Path, required=True)
parser.add_argument("--baseline-policy", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--surface", choices=SURFACES, required=True)
parser.add_argument("--incline-deg", type=float, required=True)
parser.add_argument("--scene-seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--requested-vx", type=float, default=0.15)
parser.add_argument("--baseline-grip-scale", type=float, default=0.04)
parser.add_argument("--lane-spacing", type=float, default=2.4)
parser.add_argument("--camera-eye", nargs=3, type=float, default=(4.8, 5.8, 2.8))
parser.add_argument("--camera-lookat", nargs=3, type=float, default=(0.0, 0.0, 0.75))
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]
if args.steps < 2:
    raise ValueError("steps must be at least two")
if args.warmup_steps < 0:
    raise ValueError("warmup-steps must be non-negative")
if not 0.0 < args.requested_vx <= 0.80:
    raise ValueError("requested-vx must be in (0, 0.80]")
if not 0.0 <= args.baseline_grip_scale < 1.0:
    raise ValueError("baseline-grip-scale must be in [0, 1)")
if args.lane_spacing <= 1.5:
    raise ValueError("lane-spacing must be greater than 1.5 m")
args.enable_cameras = True
_LOCK = acquire_isaac_process_lock()


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def add_label_overlay(raw_video: Path) -> Path:
    """Add an explicit comparison legend while keeping only one final MP4."""
    from PIL import Image, ImageDraw, ImageFont

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width",
            "-of",
            "json",
            str(raw_video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    width = int(json.loads(probe.stdout)["streams"][0]["width"])

    def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        for candidate in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ):
            if Path(candidate).is_file():
                return ImageFont.truetype(candidate, size)
        return ImageFont.load_default()

    banner = Image.new("RGBA", (width, 94), (0, 0, 0, 184))
    draw = ImageDraw.Draw(banner)
    left = "GREEN: WITH CRAMPONS  |  POLICY A"
    right = "RED: NO CRAMPONS  |  POLICY B"
    main_size = min(30, max(16, width // 42))
    main_font = font(main_size)
    while main_size > 14:
        left_width = draw.textbbox((0, 0), left, font=main_font)[2]
        right_width = draw.textbbox((0, 0), right, font=main_font)[2]
        if left_width + right_width + 114 <= width:
            break
        main_size -= 1
        main_font = font(main_size)
    detail_font = font(min(18, max(14, width // 70)))
    title = f"{args.surface.replace('_', ' ').upper()}  |  {args.incline_deg:g} DEG SLOPE"
    draw.text((38, 14), left, font=main_font, fill=(104, 255, 138, 255))
    right_width = draw.textbbox((0, 0), right, font=main_font)[2]
    draw.text((width - right_width - 38, 14), right, font=main_font, fill=(255, 107, 107, 255))
    title_width = draw.textbbox((0, 0), title, font=detail_font)[2]
    draw.text(((width - title_width) / 2, 63), title, font=detail_font, fill="white")
    banner_path = raw_video.with_name(f".{raw_video.stem}-banner.png")
    output = raw_video.with_name(f"{raw_video.stem}-labeled.mp4")
    banner.save(banner_path)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(raw_video),
                "-i",
                str(banner_path),
                "-filter_complex",
                "overlay=0:0:eof_action=repeat",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            check=True,
        )
        os.replace(output, raw_video)
    finally:
        banner_path.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
    return raw_video


def add_comparison_set(env_origins: torch.Tensor) -> None:
    """Add two visual-only lanes and hide crampon meshes on the baseline robot."""
    import math

    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    ground = stage.GetPrimAtPath("/World/ground")
    if ground.IsValid():
        UsdGeom.Imageable(ground).MakeInvisible()
    root = UsdGeom.Xform.Define(stage, "/World/PolicyComparison")
    root.GetPrim().CreateAttribute("everest:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)
    color = COLORS[args.surface]
    angle = math.radians(args.incline_deg)
    normal = (-math.sin(angle), 0.0, math.cos(angle))
    for index, origin in enumerate(env_origins.detach().cpu().tolist()):
        lane = UsdGeom.Cube.Define(stage, f"/World/PolicyComparison/Lane_{index}")
        lane.CreateSizeAttr(1.0)
        transform = UsdGeom.Xformable(lane)
        transform.AddTranslateOp().Set(
            Gf.Vec3d(
                origin[0] - 0.06 * normal[0],
                origin[1],
                origin[2] - 0.06 * normal[2],
            )
        )
        transform.AddRotateYOp().Set(-args.incline_deg)
        transform.AddScaleOp().Set(Gf.Vec3d(18.0, 1.9, 0.12))
        material = UsdShade.Material.Define(stage, f"/World/PolicyComparison/LaneMaterial_{index}")
        shader = UsdShade.Shader.Define(
            stage, f"/World/PolicyComparison/LaneMaterial_{index}/PreviewSurface"
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        brightness = 1.08 if index == 0 else 0.72
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*(min(1.0, value * brightness) for value in color))
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.78)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(lane.GetPrim()).Bind(material)
        lane.GetPrim().CreateAttribute("everest:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)
        marker_color = Gf.Vec3f(0.18, 0.95, 0.36) if index == 0 else Gf.Vec3f(0.95, 0.20, 0.20)
        for side, offset in (("left", -0.97), ("right", 0.97)):
            rail = UsdGeom.Cube.Define(stage, f"/World/PolicyComparison/Lane_{index}_{side}_marker")
            rail.CreateSizeAttr(1.0)
            rail.CreateDisplayColorAttr([marker_color])
            rail_transform = UsdGeom.Xformable(rail)
            rail_transform.AddTranslateOp().Set(
                Gf.Vec3d(
                    origin[0] - 0.04 * normal[0],
                    origin[1] + offset,
                    origin[2] - 0.04 * normal[2],
                )
            )
            rail_transform.AddRotateYOp().Set(-args.incline_deg)
            rail_transform.AddScaleOp().Set(Gf.Vec3d(18.0, 0.035, 0.04))
            rail.GetPrim().CreateAttribute("everest:visualOnly", Sdf.ValueTypeNames.Bool).Set(True)

    # The baseline keeps the identical G1 articulation and reset state but does
    # not show crampon hardware. Its normal support remains, but grip is reduced by cfg.
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if "/env_1/" in path and prim.GetName() == "EverestCramponVisual":
            UsdGeom.Imageable(prim).MakeInvisible()


def main() -> int:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    crampon_policy_path = args.crampon_policy.expanduser().resolve()
    baseline_policy_path = args.baseline_policy.expanduser().resolve()
    for path in (crampon_policy_path, baseline_policy_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = 2
    env_cfg.scene.env_spacing = args.lane_spacing
    env_cfg.scene.terrain.env_spacing = args.lane_spacing
    env_cfg.sim.device = args.device or "cuda:0"
    env_cfg.everest_require_complete_coverage = False
    env_cfg.everest_nominal_bootstrap_material = False
    env_cfg.everest_play_surface_id = args.surface
    env_cfg.everest_play_contact_mode_id = "all_points_flat_foot"
    env_cfg.everest_play_incline_deg = args.incline_deg
    env_cfg.everest_suite_seed = args.scene_seed
    env_cfg.everest_use_case_inclines = True
    env_cfg.everest_match_material_across_envs = True
    env_cfg.everest_crampon_grip_scale_by_env = (1.0, args.baseline_grip_scale)
    env_cfg.viewer.eye = tuple(args.camera_eye)
    env_cfg.viewer.lookat = tuple(args.camera_lookat)
    env_cfg.viewer.origin_type = "world"
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args.requested_vx, args.requested_vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.debug_vis = False

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
        for sensor in env.unwrapped.scene.sensors.values():
            sensor.set_debug_vis(False)
        add_comparison_set(env.unwrapped.scene.env_origins)
        observation, _ = env.reset()
        device = env.unwrapped.device
        crampon_policy = torch.jit.load(str(crampon_policy_path), map_location=device).eval()
        baseline_policy = torch.jit.load(str(baseline_policy_path), map_location=device).eval()

        def paired_action(policy_observation: torch.Tensor) -> torch.Tensor:
            crampon_action = crampon_policy(policy_observation[0:1]).detach()
            baseline_action = baseline_policy(policy_observation[1:2]).detach()
            return torch.cat((crampon_action, baseline_action), dim=0)

        with torch.inference_mode():
            for _ in range(args.warmup_steps):
                observation, _, _, _, _ = env.step(paired_action(observation["policy"]))

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(output_dir),
            step_trigger=lambda step: step == 0,
            video_length=args.steps,
            name_prefix=f"{args.surface}-{args.incline_deg:g}deg-comparison",
            disable_logger=True,
        )
        start_position = env.unwrapped.scene["robot"].data.root_pos_w.detach().clone()
        terminations = torch.zeros(2, dtype=torch.int64, device=device)
        minimum_height = torch.full((2,), float("inf"), device=device)
        with torch.inference_mode():
            for _ in range(args.steps):
                observation, _, terminated, _, _ = env.step(paired_action(observation["policy"]))
                terminations += terminated.to(torch.int64)
                minimum_height = torch.minimum(
                    minimum_height, env.unwrapped.scene["robot"].data.root_pos_w[:, 2]
                )
        final_position = env.unwrapped.scene["robot"].data.root_pos_w.detach().clone()
        gait = env.unwrapped.everest_gait_metrics()
        env.close()

        videos = sorted(output_dir.glob("*.mp4"))
        if len(videos) != 1:
            raise RuntimeError(f"Expected exactly one comparison video, found {len(videos)}")
        video = add_label_overlay(videos[0])
        labels = ("CRAMPON POLICY", "NO-CRAMPON BASELINE")
        rows = []
        for index, label in enumerate(labels):
            rows.append(
                {
                    "environment_index": index,
                    "label": label,
                    "crampon_visual_visible": index == 0,
                    "tangential_grip_scale": 1.0 if index == 0 else args.baseline_grip_scale,
                    "policy": str(crampon_policy_path if index == 0 else baseline_policy_path),
                    "policy_sha256": sha256(
                        crampon_policy_path if index == 0 else baseline_policy_path
                    ),
                    "terminations": int(terminations[index].cpu()),
                    "minimum_base_height_m": float(minimum_height[index].cpu()),
                    "forward_displacement_m": float(
                        (final_position[index, 0] - start_position[index, 0]).cpu()
                    ),
                    "stance_lateral_speed_mps": float(
                        gait["force_weighted_stance_lateral_speed_mps"][index].cpu()
                    ),
                }
            )
        manifest = {
            "schema_version": "1.0.0",
            "artifact_type": "native_isaac_same_sim_policy_crampon_comparison",
            "video": {"path": str(video), "sha256": sha256(video)},
            "scene": {
                "surface": args.surface,
                "incline_deg": args.incline_deg,
                "scene_seed": args.scene_seed,
                "requested_vx_mps": args.requested_vx,
                "steps": args.steps,
                "warmup_steps_excluded": args.warmup_steps,
                "lane_spacing_m": args.lane_spacing,
                "same_isaac_process": True,
                "matched_material_parameters": True,
            },
            "policies": rows,
            "comparison_semantics": (
                "Both G1 instances run simultaneously with matched terrain, material parameters, "
                "commands, and reset conditions. The baseline uses its own policy, hidden crampon "
                "visuals, and a low-grip bare-foot contact approximation that retains normal support."
            ),
            "claim_boundary": (
                "Controlled simulator ablation for visual policy comparison. The no-crampon lane is "
                "an intentional low-grip proxy, not a validated model of a stock bare G1 foot; "
                "material parameters are project-authored priors, not measured Everest conditions."
            ),
        }
        (output_dir / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
