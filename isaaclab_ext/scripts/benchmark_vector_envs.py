#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
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
    description="Bounded headless vector-environment throughput benchmark"
)
parser.add_argument("--task", default="Everest-Velocity-Rough-G1-Crampon-v0")
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--warmup_steps", type=int, default=25)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--output", type=Path, required=True)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra_args]
_ISAAC_PROCESS_LOCK = acquire_isaac_process_lock()


def nvidia_memory_used_mib() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        values = [int(line.strip()) for line in output.splitlines() if line.strip()]
        return sum(values)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def finite(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return True


def save_result(result: dict) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main() -> int:
    env_cfg, _ = resolve_task_config(args.task, "")
    result = {
        "task": args.task,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
    }
    try:
        with launch_simulation(env_cfg, args):
            env_cfg.scene.num_envs = args.num_envs
            env_cfg.sim.device = args.device or "cuda:0"
            start_create = time.perf_counter()
            env = gym.make(args.task, cfg=env_cfg)
            observations, _ = env.reset()
            create_s = time.perf_counter() - start_create
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            print(f"BENCHMARK_WARMUP_START num_envs={args.num_envs}", flush=True)
            for _ in range(args.warmup_steps):
                observations, rewards, _terminated, _truncated, _ = env.step(actions)
            torch.cuda.synchronize()
            print("BENCHMARK_TIMED_START", flush=True)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            finite_state = True
            for _ in range(args.steps):
                observations, rewards, _terminated, _truncated, _ = env.step(actions)
                finite_state &= finite(observations) and finite(rewards)
            torch.cuda.synchronize()
            duration = time.perf_counter() - start
            result.update(
                {
                    "status": "passed" if finite_state else "non_finite",
                    "scene_creation_s": create_s,
                    "duration_s": duration,
                    "vector_steps_per_s": args.steps * args.num_envs / duration,
                    "per_environment_sim_hz": args.steps / duration,
                    "physics_real_time_factor": args.steps
                    * float(env.unwrapped.step_dt)
                    / duration,
                    "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                    "cuda_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                    "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "nvidia_smi_compute_memory_mib": nvidia_memory_used_mib(),
                    "observation_finite": finite_state,
                    "action_shape": list(env.action_space.shape),
                }
            )
            save_result(result)
            env.close()
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"status": "cuda_oom", "error": str(exc)})
    except Exception as exc:  # noqa: BLE001 - benchmark records configuration/runtime failures
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    if not args.output.exists():
        save_result(result)
    return 0 if result.get("status") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
