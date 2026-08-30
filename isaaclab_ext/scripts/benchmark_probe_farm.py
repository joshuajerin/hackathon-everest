#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
from hackathon_everest_isaaclab.contact.stateful_material import (
    SNOW,
    BatchedMaterialParameters,
    BatchedStatefulMaterial,
)
from hackathon_everest_isaaclab.sensors.crampon_sensor import (
    CramponSensorAdapter,
    CramponSensorConfig,
)


def nvidia_memory_used_mib() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return sum(int(line) for line in output.splitlines() if line.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def full(shape, value, device, *, dtype=torch.float32):
    return torch.full(shape, value, dtype=dtype, device=device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    shape = (args.num_envs, 2, 4)
    params = BatchedMaterialParameters(
        material_code=full(shape, SNOW, device, dtype=torch.int64),
        vertical_stiffness_n_per_m=full(shape, 20_000.0, device),
        damping_ns_per_m=full(shape, 60.0, device),
        bearing_capacity_n=full(shape, 720.0, device),
        shear_capacity_n=full(shape, 320.0, device),
        friction=full(shape, 0.35, device),
        support_layer_depth_m=full(shape, 0.08, device),
        crust_thickness_m=full(shape, 0.012, device),
        fracture_strength_n=full(shape, 180.0, device),
        void_present=full(shape, False, device, dtype=torch.bool),
        void_top_depth_m=full(shape, 0.20, device),
        void_height_m=full(shape, 0.0, device),
        ice_indentation_pressure_pa=full(shape, 40e6, device),
        ice_tip_radius_m=full(shape, 0.0004, device),
        ice_cone_half_angle_rad=full(shape, 0.5235988, device),
        ice_shank_radius_m=full(shape, 0.003, device),
        ice_fracture_energy_j=full(shape, 0.12, device),
        ice_post_fracture_strength_ratio=full(shape, 0.45, device),
        breakout_displacement_m=full(shape, 0.004, device),
        maximum_probe_force_n=full(shape, 250.0, device),
    )
    model = BatchedStatefulMaterial(params)
    sensor = CramponSensorAdapter(CramponSensorConfig(sample_drop_probability=0.003))
    depth = full(shape, 0.006, device)
    rate = full(shape, 0.02, device)
    lateral = full(shape, 0.002, device)
    load = full(shape, 80.0, device)
    demand = full(shape, 12.0, device)
    accel = full((args.num_envs, 2, 3), 0.0, device)
    gyro = torch.zeros_like(accel)
    radar = full((args.num_envs, 2, 5), 0.0, device)
    radar[..., 0] = 0.08
    warmup = 20
    response = None
    for step in range(warmup):
        response = model.step(depth, rate, lateral, load, demand, dt_s=0.005)
        if step % 2 == 0:
            sensor.observe(
                axial_force_n=response.normal_force_n,
                penetration_m=depth,
                accelerometer_mps2=accel,
                gyroscope_rps=gyro,
                radar_frontend=radar,
                timestamp_s=torch.full((args.num_envs, 2), step * 0.005, device=device),
            )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for offset in range(args.steps):
        step = warmup + offset
        response = model.step(depth, rate, lateral, load, demand, dt_s=0.005)
        if step % 2 == 0:
            sensor.observe(
                axial_force_n=response.normal_force_n,
                penetration_m=depth,
                accelerometer_mps2=accel,
                gyroscope_rps=gyro,
                radar_frontend=radar,
                timestamp_s=torch.full((args.num_envs, 2), step * 0.005, device=device),
            )
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    result = {
        "status": "passed",
        "num_envs": args.num_envs,
        "physics_steps": args.steps,
        "duration_s": duration,
        "probe_environment_steps_per_s": args.num_envs * args.steps / duration,
        "physics_steps_per_s": args.steps / duration,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "cuda_reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "nvidia_smi_compute_memory_mib": nvidia_memory_used_mib(),
        "finite": bool(torch.isfinite(response.normal_force_n).all().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
