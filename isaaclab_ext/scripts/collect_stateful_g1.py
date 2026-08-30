#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from hackathon_everest_isaaclab.data.writer import stable_group_hash, write_immutable_shard
from hackathon_everest_isaaclab.runtime import acquire_isaac_process_lock
from hackathon_everest_isaaclab.tasks import register_cli
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

register_cli()
parser = argparse.ArgumentParser(description="Collect full-G1 visible/truth windows")
parser.add_argument("--task", default="Everest-Velocity-Flat-G1-Crampon-Stateful-Bootstrap-v0")
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--surface-id", default="")
parser.add_argument("--incline-deg", type=float, default=0.0)
parser.add_argument("--requested-vx", type=float, default=0.15)
parser.add_argument("--windows-per-env", type=int, default=8)
parser.add_argument("--window-steps", type=int, default=31)
parser.add_argument("--policy", type=Path, required=True)
parser.add_argument("--dataset-root", type=Path, required=True)
parser.add_argument("--dataset-id", required=True)
parser.add_argument("--seed", type=int, default=31)
parser.add_argument(
    "--exclude-reset-windows",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Exclude windows containing an environment termination/reset.",
)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]
_ISAAC_PROCESS_LOCK = acquire_isaac_process_lock()


def stack_visible(frames, group: str | None = None):
    if group is None:
        return {
            "packet_values": torch.stack([frame.packet_values for frame in frames], dim=1),
            "valid_mask": torch.stack([frame.valid_mask for frame in frames], dim=1),
            "timestamp_s": torch.stack([frame.timestamp_s for frame in frames], dim=1),
            "sample_age_s": torch.stack([frame.sample_age_s for frame in frames], dim=1),
        }
    names = frames[0].context if group == "context" else frames[0].commands
    source = lambda frame: frame.context if group == "context" else frame.commands
    return {name: torch.stack([source(frame)[name] for frame in frames], dim=1) for name in names}


def windows(value: torch.Tensor) -> np.ndarray:
    blocks = []
    for window in range(args.windows_per_env):
        start = window * args.window_steps
        blocks.append(value[:, start : start + args.window_steps])
    return torch.cat(blocks, dim=0).cpu().numpy()


def split_for_hash(group_hash: str) -> str:
    bucket = int(group_hash[:8], 16) % 100
    if bucket < 65:
        return "train"
    if bucket < 75:
        return "calibration"
    if bucket < 85:
        return "validation"
    return "sealed_test"


def main() -> int:
    env_cfg, _ = resolve_task_config(args.task, "")
    with launch_simulation(env_cfg, args):
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        if args.surface_id:
            env_cfg.everest_require_complete_coverage = False
            env_cfg.everest_play_surface_id = args.surface_id
            env_cfg.everest_play_incline_deg = args.incline_deg
            env_cfg.everest_play_hazard_id = "none"
            env_cfg.everest_use_case_inclines = True
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args.requested_vx, args.requested_vx)
        env_cfg.sim.device = args.device or "cuda:0"
        env = gym.make(args.task, cfg=env_cfg)
        observation, _ = env.reset()
        device = env.unwrapped.device
        policy = torch.jit.load(str(args.policy), map_location=device).eval()
        action = policy(observation["policy"]).detach()
        visible_frames = []
        truth_frames = []
        terminated_frames = []
        needed = args.windows_per_env * args.window_steps
        control_steps = math.ceil(needed / 2) + 3
        for _ in range(control_steps):
            observation, _, terminated, truncated, _ = env.step(action)
            sensor = env.unwrapped.pop_everest_sensor_frames()
            truth = env.unwrapped.pop_everest_truth_frames()
            if len(sensor) != len(truth):
                raise RuntimeError("Visible/truth frame count mismatch")
            visible_frames.extend(sensor)
            truth_frames.extend(truth)
            terminated_frames.extend([(terminated | truncated).clone()] * len(sensor))
            action = policy(observation["policy"]).detach()
        if len(visible_frames) < needed:
            raise RuntimeError(f"Only {len(visible_frames)} sensor frames for {needed} requested")
        visible_frames = visible_frames[:needed]
        truth_frames = truth_frames[:needed]
        terminated_frames = terminated_frames[:needed]
        visible_values = stack_visible(visible_frames)
        visible = {
            **{name: windows(value) for name, value in visible_values.items()},
            "context": {
                name: windows(value)
                for name, value in stack_visible(visible_frames, "context").items()
            },
            "commands": {
                name: windows(value)
                for name, value in stack_visible(visible_frames, "commands").items()
            },
        }
        truth_targets = torch.stack([frame["targets"] for frame in truth_frames], dim=1)
        truth_events = torch.stack([frame["events"] for frame in truth_frames], dim=1)
        termination = torch.stack(terminated_frames, dim=1)
        truth = {
            "targets": windows(truth_targets),
            "events": windows(truth_events),
            "termination": windows(termination.unsqueeze(-1)).squeeze(-1),
        }
        rows = []
        for window in range(args.windows_per_env):
            for environment, case in enumerate(env.unwrapped.everest_cases):
                group_hash = stable_group_hash(
                    {
                        "source": "full_isaac_g1",
                        "surface": case.surface_id,
                        "incline": case.incline_deg,
                        "hazard": case.hazard_id,
                        "contact": case.contact_mode_id,
                        "environment": environment,
                        "window": window,
                        "seed": args.seed,
                    }
                )
                rows.append(
                    {
                        "episode_id": f"g1-{window:04d}-{environment:05d}",
                        "group_hash": group_hash,
                        "split": split_for_hash(group_hash),
                        "sampling_regime": "full_isaac_policy_rollout",
                        "surface_id": case.surface_id,
                        "incline_deg": case.incline_deg,
                        "hazard_id": case.hazard_id,
                        "contact_mode_id": case.contact_mode_id,
                        "contains_reset": bool(
                            truth["termination"][window * args.num_envs + environment].any()
                        ),
                    }
                )
        requested_windows = len(rows)
        reset_windows = sum(row["contains_reset"] for row in rows)
        if args.exclude_reset_windows and reset_windows:
            keep = np.asarray([not row["contains_reset"] for row in rows], dtype=bool)

            def filter_tree(value):
                if isinstance(value, dict):
                    return {name: filter_tree(child) for name, child in value.items()}
                return value[keep]

            visible = filter_tree(visible)
            truth = filter_tree(truth)
            rows = [row for row, selected in zip(rows, keep, strict=True) if selected]
        if not rows:
            raise RuntimeError("All full-G1 windows contained a reset")
        destination = write_immutable_shard(
            args.dataset_root,
            dataset_id=args.dataset_id,
            shard_id="worker-0000",
            visible=visible,
            truth=truth,
            episode_rows=rows,
            provenance={
                "source": "full Isaac Sim G1 stateful-contact rollout",
                "task": args.task,
                "policy": str(args.policy),
                "seed": args.seed,
                "requested_windows": requested_windows,
                "reset_windows_detected": reset_windows,
                "reset_windows_excluded": reset_windows if args.exclude_reset_windows else 0,
                "claim_boundary": "Synthetic full-articulation data; not hardware measurements.",
            },
        )
        print(f"STATEFUL_G1_DATASET_COMPLETE path={destination} episodes={len(rows)}")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
