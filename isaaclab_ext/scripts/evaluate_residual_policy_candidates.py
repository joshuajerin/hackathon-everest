#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
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

register_cli()
parser = argparse.ArgumentParser(
    description="Evaluate multiple deterministic residual policies in one paired-material Isaac run"
)
parser.add_argument(
    "--task",
    default="Everest-Velocity-Flat-G1-Crampon-Stateful-Bootstrap-FrontPoint-Randomized-v0",
)
parser.add_argument(
    "--policy",
    action="append",
    required=True,
    help="Candidate in NAME=PATH form; may be repeated",
)
parser.add_argument("--style-reference-policy", type=Path, required=True)
parser.add_argument("--replicates", type=int, default=8)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--output", type=Path, required=True)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]
_ISAAC_PROCESS_LOCK = acquire_isaac_process_lock()


def parse_policies(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError("Every --policy must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not name or name in seen or not path.is_file():
            raise ValueError(f"Invalid or duplicate policy candidate: {value}")
        seen.add(name)
        result.append((name, path))
    return result


def pair_material_parameters(env, replicates: int, candidate_count: int) -> list[str]:
    parameters = env.unwrapped.everest_wrench_bridge.material.parameters
    paired = []
    for name, value in vars(parameters).items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and value.shape[0] == env.unwrapped.num_envs
        ):
            baseline = value[:replicates].clone()
            for candidate in range(1, candidate_count):
                value[candidate * replicates : (candidate + 1) * replicates].copy_(baseline)
            paired.append(name)
    return paired


def main() -> int:
    candidates = parse_policies(args.policy)
    if args.replicates < 1 or args.steps < 1:
        raise ValueError("replicates and steps must be positive")
    num_envs = len(candidates) * args.replicates
    env_cfg, _ = resolve_task_config(args.task, "")
    with launch_simulation(env_cfg, args):
        env_cfg.scene.num_envs = num_envs
        env_cfg.sim.device = args.device or "cuda:0"
        env = gym.make(args.task, cfg=env_cfg)
        observation, _ = env.reset()
        device = env.unwrapped.device
        policies = [
            (name, torch.jit.load(str(path), map_location=device).eval())
            for name, path in candidates
        ]
        reference = torch.jit.load(str(args.style_reference_policy), map_location=device).eval()
        paired_fields = pair_material_parameters(env, args.replicates, len(candidates))

        terminations = torch.zeros(num_envs, dtype=torch.long, device=device)
        timeouts = torch.zeros_like(terminations)
        minimum_base_height = torch.full((num_envs,), float("inf"), device=device)
        velocity_sum = torch.zeros(num_envs, device=device)
        requested_velocity_sum = torch.zeros(num_envs, device=device)
        instantaneous_velocity_error_sum = torch.zeros(num_envs, device=device)
        motion_samples = torch.zeros(num_envs, dtype=torch.long, device=device)
        rear_load_sum = torch.zeros(num_envs, device=device)
        rear_load_samples = torch.zeros(num_envs, dtype=torch.long, device=device)
        action_delta_sum = torch.zeros(num_envs, device=device)
        action_delta_max = torch.zeros(num_envs, device=device)
        style_sum = torch.zeros(num_envs, device=device)
        style_max = torch.zeros(num_envs, device=device)
        reward_sum = torch.zeros(num_envs, device=device)

        def actions_for(obs: torch.Tensor) -> torch.Tensor:
            actions = torch.empty((num_envs, 37), device=device)
            for index, (_, policy) in enumerate(policies):
                sl = slice(index * args.replicates, (index + 1) * args.replicates)
                actions[sl] = policy(obs[sl])
            return actions

        with torch.inference_mode():
            previous_action = actions_for(observation["policy"])
            started = time.perf_counter()
            for _ in range(args.steps):
                action = actions_for(observation["policy"])
                reference_action = reference(observation["policy"])
                action_delta = (action - previous_action).square().mean(dim=-1).sqrt()
                style_delta = (action - reference_action).square().mean(dim=-1).sqrt()
                observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                valid = ~done
                terminations += terminated.long()
                timeouts += truncated.long()
                action_delta_sum += torch.where(valid, action_delta, torch.zeros_like(action_delta))
                action_delta_max = torch.maximum(
                    action_delta_max,
                    torch.where(valid, action_delta, torch.zeros_like(action_delta)),
                )
                style_sum += torch.where(valid, style_delta, torch.zeros_like(style_delta))
                style_max = torch.maximum(
                    style_max, torch.where(valid, style_delta, torch.zeros_like(style_delta))
                )
                reward_sum += reward
                robot = env.unwrapped.scene["robot"]
                height = (
                    (robot.data.root_pos_w - env.unwrapped._everest_terrain_origin)
                    * env.unwrapped._everest_terrain_normal
                ).sum(dim=-1)
                minimum_base_height = torch.minimum(minimum_base_height, height)
                requested = env.unwrapped.command_manager.get_command("base_velocity")[:, 0]
                forward = robot.data.root_lin_vel_b[:, 0]
                velocity_sum += torch.where(valid, forward, torch.zeros_like(forward))
                requested_velocity_sum += torch.where(valid, requested, torch.zeros_like(requested))
                instantaneous_velocity_error_sum += torch.where(
                    valid, (forward - requested).abs(), torch.zeros_like(forward)
                )
                motion_samples += valid.long()
                wrench = env.unwrapped.everest_latest_wrench
                normal_force = wrench.probe_normal_force_n.clamp_min(0.0)
                total_load = normal_force.sum(dim=(1, 2))
                rear_fraction = normal_force[:, :, 2:].sum(dim=(1, 2)) / total_load.clamp_min(20.0)
                loaded = total_load >= 20.0
                rear_load_sum += torch.where(loaded, rear_fraction, torch.zeros_like(rear_fraction))
                rear_load_samples += loaded.long()
                previous_action = action
            duration = time.perf_counter() - started

        results = []
        for index, (name, path) in enumerate(candidates):
            sl = slice(index * args.replicates, (index + 1) * args.replicates)
            samples = motion_samples[sl].clamp_min(1)
            load_samples = rear_load_samples[sl].clamp_min(1)
            mean_velocity = velocity_sum[sl] / samples
            mean_requested = requested_velocity_sum[sl] / samples
            time_average_error = (mean_velocity - mean_requested).abs()
            result = {
                "name": name,
                "policy": str(path),
                "terminations": int(terminations[sl].sum()),
                "terminations_by_environment": terminations[sl].cpu().tolist(),
                "timeouts": int(timeouts[sl].sum()),
                "minimum_base_height_m": float(minimum_base_height[sl].min()),
                "mean_forward_velocity_mps": float(mean_velocity.mean()),
                "mean_forward_velocity_mps_by_environment": mean_velocity.cpu().tolist(),
                "mean_time_averaged_velocity_error_mps": float(time_average_error.mean()),
                "mean_instantaneous_velocity_error_mps": float(
                    (instantaneous_velocity_error_sum[sl] / samples).mean()
                ),
                "mean_rear_load_fraction": float((rear_load_sum[sl] / load_samples).mean()),
                "mean_rear_load_fraction_by_environment": (rear_load_sum[sl] / load_samples)
                .cpu()
                .tolist(),
                "mean_action_delta_rms": float((action_delta_sum[sl] / samples).mean()),
                "maximum_action_delta_rms": float(action_delta_max[sl].max()),
                "mean_stock_style_deviation_rms": float((style_sum[sl] / samples).mean()),
                "maximum_stock_style_deviation_rms": float(style_max[sl].max()),
                "mean_reward_per_step": float(reward_sum[sl].mean() / args.steps),
            }
            results.append(result)
        report = {
            "schema_version": "1.0.0",
            "task": args.task,
            "steps": args.steps,
            "replicates_per_policy": args.replicates,
            "paired_material_parameter_fields": paired_fields,
            "duration_s": duration,
            "results": results,
            "claim_boundary": "Deterministic native Isaac simulator selection; not hardware validation.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
