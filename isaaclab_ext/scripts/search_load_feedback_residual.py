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
    description="Search bounded visible-force feedback residuals around the frozen stock G1 policy"
)
parser.add_argument(
    "--task",
    default="Everest-Velocity-Flat-G1-Crampon-Stateful-Bootstrap-FrontPoint-Randomized-v0",
)
parser.add_argument("--stock-policy", type=Path, required=True)
parser.add_argument("--candidates", type=int, default=64)
parser.add_argument("--replicates", type=int, default=8)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=23)
parser.add_argument("--maximum-residual", type=float, default=0.12)
parser.add_argument("--output", type=Path, required=True)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]
_ISAAC_PROCESS_LOCK = acquire_isaac_process_lock()


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


def candidate_parameters(count: int, replicates: int, device: torch.device) -> torch.Tensor:
    if count < 1:
        raise ValueError("candidates must be positive")
    zero = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.20]], device=device)
    if count == 1:
        base = zero
    else:
        sobol = torch.quasirandom.SobolEngine(5, scramble=True, seed=args.seed)
        draws = sobol.draw(count - 1).to(device)
        gains = -0.60 + 1.20 * draws[:, :4]
        alpha = 0.05 + 0.45 * draws[:, 4:5]
        base = torch.cat((zero, torch.cat((gains, alpha), dim=-1)), dim=0)
    return base.repeat_interleave(replicates, dim=0)


def main() -> int:
    if args.replicates < 1 or args.steps < 1:
        raise ValueError("replicates and steps must be positive")
    if not 0.0 < args.maximum_residual <= 1.0:
        raise ValueError("maximum-residual must be in (0, 1]")
    policy_path = args.stock_policy.expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    num_envs = args.candidates * args.replicates
    env_cfg, _ = resolve_task_config(args.task, "")
    with launch_simulation(env_cfg, args):
        env_cfg.scene.num_envs = num_envs
        env_cfg.sim.device = args.device or "cuda:0"
        env = gym.make(args.task, cfg=env_cfg)
        observation, _ = env.reset()
        device = env.unwrapped.device
        stock = torch.jit.load(str(policy_path), map_location=device).eval()
        parameters = candidate_parameters(args.candidates, args.replicates, device)
        paired_fields = pair_material_parameters(env, args.replicates, args.candidates)

        filtered = torch.zeros((num_envs, 37), device=device)
        terminations = torch.zeros(num_envs, dtype=torch.long, device=device)
        minimum_base_height = torch.full((num_envs,), float("inf"), device=device)
        velocity_sum = torch.zeros(num_envs, device=device)
        requested_velocity_sum = torch.zeros(num_envs, device=device)
        motion_samples = torch.zeros(num_envs, dtype=torch.long, device=device)
        rear_load_sum = torch.zeros(num_envs, device=device)
        rear_load_samples = torch.zeros(num_envs, dtype=torch.long, device=device)
        action_delta_sum = torch.zeros(num_envs, device=device)
        style_sum = torch.zeros(num_envs, device=device)

        def feedback_residual() -> torch.Tensor:
            target = torch.zeros_like(filtered)
            frame = env.unwrapped.everest_latest_sensor_frame
            if frame is None:
                return target
            force = frame.packet_values[:, :, :4].clamp_min(0.0)
            foot_total = force.sum(dim=-1)
            foot_rear = force[:, :, 2:].sum(dim=-1)
            loaded = foot_total >= 10.0
            rear_fraction = foot_rear / foot_total.clamp_min(10.0)
            foot_error = torch.where(
                loaded,
                rear_fraction - env.unwrapped._everest_target_rear_load_fraction[:, None],
                torch.zeros_like(rear_fraction),
            )
            total = foot_total.sum(dim=-1)
            global_rear = foot_rear.sum(dim=-1) / total.clamp_min(20.0)
            global_error = torch.where(
                total >= 20.0,
                global_rear - env.unwrapped._everest_target_rear_load_fraction,
                torch.zeros_like(global_rear),
            )
            hip_gain, knee_gain, ankle_gain, torso_gain = parameters[:, :4].unbind(dim=-1)
            target[:, 0] = hip_gain * foot_error[:, 0]
            target[:, 1] = hip_gain * foot_error[:, 1]
            target[:, 11] = knee_gain * foot_error[:, 0]
            target[:, 12] = knee_gain * foot_error[:, 1]
            target[:, 15] = ankle_gain * foot_error[:, 0]
            target[:, 16] = ankle_gain * foot_error[:, 1]
            target[:, 2] = torso_gain * global_error
            return target.clamp(-args.maximum_residual, args.maximum_residual)

        with torch.inference_mode():
            previous_action = stock(observation["policy"])
            started = time.perf_counter()
            for _ in range(args.steps):
                reference = stock(observation["policy"])
                desired = feedback_residual()
                alpha = parameters[:, 4:5]
                filtered.mul_(1.0 - alpha).add_(alpha * desired)
                filtered.clamp_(-args.maximum_residual, args.maximum_residual)
                action = reference + filtered
                action_delta = (action - previous_action).square().mean(dim=-1).sqrt()
                style_delta = filtered.square().mean(dim=-1).sqrt()
                observation, _, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                valid = ~done
                terminations += terminated.long()
                action_delta_sum += torch.where(valid, action_delta, torch.zeros_like(action_delta))
                style_sum += torch.where(valid, style_delta, torch.zeros_like(style_delta))
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
                motion_samples += valid.long()
                frame = env.unwrapped.everest_latest_sensor_frame
                force = frame.packet_values[:, :, :4].clamp_min(0.0)
                total_load = force.sum(dim=(1, 2))
                rear_fraction = force[:, :, 2:].sum(dim=(1, 2)) / total_load.clamp_min(20.0)
                loaded = total_load >= 20.0
                rear_load_sum += torch.where(loaded, rear_fraction, torch.zeros_like(rear_fraction))
                rear_load_samples += loaded.long()
                filtered[done] = 0.0
                previous_action = action
            duration = time.perf_counter() - started

        results = []
        base_parameters = parameters[:: args.replicates]
        for index in range(args.candidates):
            sl = slice(index * args.replicates, (index + 1) * args.replicates)
            samples = motion_samples[sl].clamp_min(1)
            load_samples = rear_load_samples[sl].clamp_min(1)
            velocity = velocity_sum[sl] / samples
            requested = requested_velocity_sum[sl] / samples
            rear = rear_load_sum[sl] / load_samples
            falls = int(terminations[sl].sum().item())
            result = {
                "candidate": index,
                "hip_gain": float(base_parameters[index, 0]),
                "knee_gain": float(base_parameters[index, 1]),
                "ankle_gain": float(base_parameters[index, 2]),
                "torso_gain": float(base_parameters[index, 3]),
                "smoothing_alpha": float(base_parameters[index, 4]),
                "terminations": falls,
                "terminations_by_environment": terminations[sl].cpu().tolist(),
                "minimum_base_height_m": float(minimum_base_height[sl].min()),
                "mean_forward_velocity_mps": float(velocity.mean()),
                "mean_time_averaged_velocity_error_mps": float((velocity - requested).abs().mean()),
                "mean_rear_load_fraction": float(rear.mean()),
                "mean_rear_load_fraction_by_environment": rear.cpu().tolist(),
                "mean_action_delta_rms": float((action_delta_sum[sl] / samples).mean()),
                "mean_stock_style_deviation_rms": float((style_sum[sl] / samples).mean()),
            }
            result["selection_score"] = (
                10000.0 * falls
                + 1000.0 * max(0.0, 0.50 - result["minimum_base_height_m"])
                + 100.0 * abs(result["mean_forward_velocity_mps"] - 0.15)
                + 20.0 * abs(result["mean_rear_load_fraction"] - 0.20)
                + 5.0 * result["mean_stock_style_deviation_rms"]
            )
            results.append(result)
        results.sort(key=lambda item: item["selection_score"])
        report = {
            "status": "completed",
            "claim_boundary": "Native Isaac simulator search using visible force packets; not hardware validation.",
            "stock_policy": str(policy_path),
            "task": args.task,
            "candidates": args.candidates,
            "replicates": args.replicates,
            "steps": args.steps,
            "maximum_residual": args.maximum_residual,
            "paired_material_fields": paired_fields,
            "duration_s": duration,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "best": results[0]}, indent=2))
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
