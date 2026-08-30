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
parser = argparse.ArgumentParser(description="Evaluate a deterministic G1 locomotion policy")
parser.add_argument("--task", required=True)
parser.add_argument("--policy", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max-base-contact-terminations", type=int)
parser.add_argument("--min-completed-episode-steps-mean", type=float)
parser.add_argument("--max-mean-velocity-error-mps", type=float)
parser.add_argument("--min-base-height-m", type=float)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]
_ISAAC_PROCESS_LOCK = acquire_isaac_process_lock()


def main() -> int:
    exit_code = 0
    if args.num_envs < 1 or args.steps < 1:
        raise ValueError("num-envs and steps must be positive")
    env_cfg, _ = resolve_task_config(args.task, "")
    with launch_simulation(env_cfg, args):
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device or "cuda:0"
        env = gym.make(args.task, cfg=env_cfg)
        observation, _ = env.reset()
        device = env.unwrapped.device
        policy = torch.jit.load(str(args.policy), map_location=device).eval()
        episode_steps = torch.zeros(args.num_envs, dtype=torch.long, device=device)
        completed_steps: list[int] = []
        completed_by_environment: list[list[int]] = [[] for _ in range(args.num_envs)]
        base_contact_terminations = 0
        base_contact_by_environment = torch.zeros(args.num_envs, dtype=torch.long, device=device)
        timeouts = 0
        timeouts_by_environment = torch.zeros(args.num_envs, dtype=torch.long, device=device)
        reward_sum = 0.0
        velocity_error_sum = 0.0
        minimum_base_height = float("inf")
        maximum_penetration = 0.0
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(args.steps):
                action = policy(observation["policy"])
                if tuple(action.shape) != (args.num_envs, 37):
                    raise RuntimeError(f"Unexpected policy action shape {tuple(action.shape)}")
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_steps += 1
                done = terminated | truncated
                if bool(done.any()):
                    done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
                    completed = episode_steps[done].cpu().tolist()
                    completed_steps.extend(completed)
                    for environment, length in zip(done_ids.cpu().tolist(), completed, strict=True):
                        completed_by_environment[environment].append(length)
                    episode_steps[done] = 0
                base_contact_terminations += int(torch.count_nonzero(terminated))
                base_contact_by_environment += terminated.long()
                timeouts += int(torch.count_nonzero(truncated))
                timeouts_by_environment += truncated.long()
                reward_sum += float(reward.mean())
                base_velocity = env.unwrapped.scene["robot"].data.root_lin_vel_b[:, :2]
                command = env.unwrapped.command_manager.get_command("base_velocity")[:, :2]
                velocity_error_sum += float(
                    torch.linalg.vector_norm(base_velocity - command, dim=-1).mean()
                )
                minimum_base_height = min(
                    minimum_base_height,
                    float(env.unwrapped.scene["robot"].data.root_pos_w[:, 2].min()),
                )
                wrench = env.unwrapped.everest_latest_wrench
                if wrench is not None:
                    maximum_penetration = max(
                        maximum_penetration, float(wrench.probe_penetration_m.max())
                    )
        duration = time.perf_counter() - started
        censored = episode_steps.cpu().tolist()
        all_observed = completed_steps + censored
        cases = [
            {
                "case_id": case.case_id,
                "surface_id": case.surface_id,
                "hazard_id": case.hazard_id,
                "contact_mode_id": case.contact_mode_id,
                "incline_deg": case.incline_deg,
            }
            for case in env.unwrapped.everest_cases
        ]
        parameters = env.unwrapped.everest_wrench_bridge.material.parameters
        bearing_per_environment = parameters.bearing_capacity_n.mean(dim=(1, 2)).cpu().tolist()
        completed_mean = sum(completed_steps) / len(completed_steps) if completed_steps else None
        result = {
            "status": "diagnostic_complete",
            "task": args.task,
            "policy": str(args.policy),
            "num_envs": args.num_envs,
            "steps": args.steps,
            "control_dt_s": float(env.unwrapped.step_dt),
            "duration_s": duration,
            "simulated_env_steps_per_s": args.num_envs * args.steps / duration,
            "base_contact_terminations": base_contact_terminations,
            "base_contact_terminations_by_environment": base_contact_by_environment.cpu().tolist(),
            "timeouts": timeouts,
            "timeouts_by_environment": timeouts_by_environment.cpu().tolist(),
            "completed_episode_steps_by_environment": completed_by_environment,
            "cases": cases,
            "bearing_capacity_n_by_environment": bearing_per_environment,
            "completed_episodes": len(completed_steps),
            "completed_episode_steps_mean": completed_mean,
            "completed_episode_steps_max": max(completed_steps, default=None),
            "longest_observed_episode_steps": max(all_observed, default=0),
            "unfinished_episode_steps": censored,
            "mean_reward_per_step": reward_sum / args.steps,
            "mean_velocity_error_mps": velocity_error_sum / args.steps,
            "minimum_base_height_m": minimum_base_height,
            "maximum_probe_penetration_m": maximum_penetration,
            "claim_boundary": "Full Isaac G1 simulator result; not hardware validation.",
        }
        gate_failures = []
        if (
            args.max_base_contact_terminations is not None
            and base_contact_terminations > args.max_base_contact_terminations
        ):
            gate_failures.append("base_contact_terminations")
        if args.min_completed_episode_steps_mean is not None and (
            completed_mean is None or completed_mean < args.min_completed_episode_steps_mean
        ):
            gate_failures.append("completed_episode_steps_mean")
        mean_velocity_error = velocity_error_sum / args.steps
        if (
            args.max_mean_velocity_error_mps is not None
            and mean_velocity_error > args.max_mean_velocity_error_mps
        ):
            gate_failures.append("mean_velocity_error_mps")
        if args.min_base_height_m is not None and minimum_base_height < args.min_base_height_m:
            gate_failures.append("minimum_base_height_m")
        gates_requested = any(
            value is not None
            for value in (
                args.max_base_contact_terminations,
                args.min_completed_episode_steps_mean,
                args.max_mean_velocity_error_mps,
                args.min_base_height_m,
            )
        )
        result["gate_failures"] = gate_failures
        if gates_requested:
            result["status"] = "failed" if gate_failures else "passed"
            exit_code = 2 if gate_failures else 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        env.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
