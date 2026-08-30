#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from hackathon_everest_isaaclab.contact.stateful_material import (
    ICE,
    LAYERED_SNOW_ICE,
    SNOW,
    BatchedMaterialParameters,
    BatchedStatefulMaterial,
)
from hackathon_everest_isaaclab.data.writer import stable_group_hash, write_immutable_shard
from hackathon_everest_isaaclab.sensors.crampon_sensor import (
    CramponSensorAdapter,
    CramponSensorConfig,
)
from hackathon_everest_isaaclab.sensors.faults import (
    SENSOR_FAULT_MODES,
    apply_sensor_faults,
)

from hackathon_everest.dataset import EVENT_NAMES, TARGET_NAMES
from hackathon_everest.everest_suite import assign_cases, load_suite


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor(value: np.ndarray, device: torch.device, *, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def uniform(rng: np.random.Generator, bounds: list[float], size: tuple[int, ...]) -> np.ndarray:
    return rng.uniform(float(bounds[0]), float(bounds[1]), size=size)


def build_parameters(config: dict, cases, *, seed: int, device: torch.device):
    rng = np.random.default_rng(seed)
    count = len(cases)
    shape = (count, 2, 4)
    surfaces = {item["id"]: item for item in config["surfaces"]}
    priors = config["physics_priors"]
    arrays = {
        name: np.zeros(shape, dtype=np.float32)
        for name in (
            "vertical_stiffness_n_per_m",
            "damping_ns_per_m",
            "bearing_capacity_n",
            "shear_capacity_n",
            "friction",
            "support_layer_depth_m",
            "crust_thickness_m",
            "fracture_strength_n",
            "void_top_depth_m",
            "void_height_m",
            "ice_indentation_pressure_pa",
            "ice_tip_radius_m",
            "ice_cone_half_angle_rad",
            "ice_shank_radius_m",
            "ice_fracture_energy_j",
            "ice_post_fracture_strength_ratio",
            "breakout_displacement_m",
            "maximum_probe_force_n",
        )
    }
    material_code = np.zeros(shape, dtype=np.int64)
    void_present = np.zeros(shape, dtype=bool)
    for episode, case in enumerate(cases):
        surface = surfaces[case.surface_id]
        prior = priors[surface["physics_prior"]]
        family = surface["family"]
        code = SNOW if family == "snow" else ICE if family == "ice" else LAYERED_SNOW_ICE
        material_code[episode] = code
        local = rng.uniform(0.92, 1.08, size=(2, 4)).astype(np.float32)

        def sampled(
            name: str, fallback: list[float], prior: dict = prior, local: np.ndarray = local
        ) -> np.ndarray:
            base = float(uniform(rng, prior.get(name, fallback), (1,))[0])
            return base * local

        arrays["vertical_stiffness_n_per_m"][episode] = sampled(
            "stiffness_n_per_m", [18_000, 42_000]
        )
        arrays["damping_ns_per_m"][episode] = sampled("damping_ns_per_m", [20, 120])
        arrays["bearing_capacity_n"][episode] = sampled("bearing_n", [300, 900])
        arrays["shear_capacity_n"][episode] = sampled("shear_n", [80, 500])
        arrays["friction"][episode] = np.clip(sampled("friction", [0.03, 0.65]), 0.02, 0.9)
        depth = float(surface["configured_depth_m"])
        arrays["support_layer_depth_m"][episode] = rng.uniform(
            max(0.003, 0.15 * depth), max(0.006, 0.9 * depth), size=(2, 4)
        )
        arrays["crust_thickness_m"][episode] = sampled("crust_m", [0.0, 0.025])
        arrays["fracture_strength_n"][episode] = sampled("fracture_strength_n", [50, 310])
        arrays["ice_indentation_pressure_pa"][episode] = sampled(
            "indentation_pressure_pa", [28e6, 73e6]
        )
        arrays["ice_tip_radius_m"][episode] = rng.uniform(0.00025, 0.00075, size=(2, 4))
        arrays["ice_cone_half_angle_rad"][episode] = np.deg2rad(
            rng.uniform(24.0, 36.0, size=(2, 4))
        )
        arrays["ice_shank_radius_m"][episode] = rng.uniform(0.0025, 0.0040, size=(2, 4))
        arrays["ice_fracture_energy_j"][episode] = sampled("fracture_energy_j", [0.035, 0.30])
        arrays["ice_post_fracture_strength_ratio"][episode] = rng.uniform(0.30, 0.72, size=(2, 4))
        arrays["breakout_displacement_m"][episode] = rng.uniform(0.0015, 0.0060, size=(2, 4))
        arrays["maximum_probe_force_n"][episode] = 250.0
        hazard = case.hazard_id
        if hazard in {
            "buried_shallow_void",
            "buried_deep_void",
            "thin_snow_bridge",
            "open_crevasse_gap",
            "edge_collapse",
        }:
            selected = np.ones((2, 4), dtype=bool)
            if hazard == "edge_collapse":
                selected[:] = False
                selected[:, rng.choice(4, size=2, replace=False)] = True
            void_present[episode] = selected
            if hazard == "buried_shallow_void":
                top, height = rng.uniform(0.03, 0.12), rng.uniform(0.05, 0.25)
            elif hazard == "buried_deep_void":
                top, height = rng.uniform(0.12, 0.35), rng.uniform(0.20, 0.80)
            elif hazard == "thin_snow_bridge":
                top, height = rng.uniform(0.02, 0.12), rng.uniform(0.20, 0.80)
                arrays["crust_thickness_m"][episode] = top
            elif hazard == "open_crevasse_gap":
                top, height = 0.0, 2.0
                arrays["bearing_capacity_n"][episode] *= 0.08
                arrays["shear_capacity_n"][episode] *= 0.08
            else:
                top, height = rng.uniform(0.01, 0.08), rng.uniform(0.10, 0.60)
                arrays["bearing_capacity_n"][episode][selected] *= 0.30
            arrays["void_top_depth_m"][episode] = top
            arrays["void_height_m"][episode] = height
        else:
            arrays["void_top_depth_m"][episode] = 0.5
            arrays["void_height_m"][episode] = 0.0
        if hazard == "exposed_ice_patch" and family == "snow":
            material_code[episode, :, :2] = ICE
    params = BatchedMaterialParameters(
        material_code=tensor(material_code, device, dtype=torch.int64),
        void_present=tensor(void_present, device, dtype=torch.bool),
        **{name: tensor(value, device) for name, value in arrays.items()},
    )
    return params, rng


def split_for_hash(group_hash: str) -> str:
    value = int(group_hash[:8], 16) % 100
    if value < 65:
        return "train"
    if value < 75:
        return "calibration"
    if value < 85:
        return "validation"
    return "sealed_test"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/lambda/nfs/prime/everest/datasets/isaac")
    )
    parser.add_argument("--dataset-id", default="everest_l0_probe_10k_v1")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    config_path = root / "configs/isaaclab/everest_terrain_suite.yaml"
    config = load_suite(config_path)
    cases = assign_cases(config, num_envs=args.episodes, seed=args.seed)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    params, rng = build_parameters(config, cases, seed=args.seed, device=device)
    model = BatchedStatefulMaterial(params)
    damaged = torch.as_tensor(
        [case.hazard_id == "damaged_previous_footprint" for case in cases], device=device
    )
    model.damage[damaged] = torch.rand_like(model.damage[damaged]) * 0.45 + 0.25
    model.compaction[damaged] = torch.rand_like(model.compaction[damaged]) * 0.45 + 0.25

    count = args.episodes
    environment_ids = torch.arange(count, device=device, dtype=torch.long)
    selector = (environment_ids + args.seed) % 10
    fault_code = torch.where(selector < 5, 0, selector - 4)
    fault_cycle = (fault_code - environment_ids) % len(SENSOR_FAULT_MODES)
    fault_codes_cpu = fault_code.cpu().tolist()
    shape = params.shape
    surface_map = {item["id"]: item for item in config["surfaces"]}
    slopes = tensor(np.asarray([case.incline_deg for case in cases], dtype=np.float32), device)
    angle = torch.deg2rad(slopes).view(count, 1, 1)
    modes = [case.contact_mode_id for case in cases]
    weights = torch.full(shape, 0.25, device=device)
    for index, mode in enumerate(modes):
        if mode == "hybrid_contact":
            weights[index] = torch.tensor([[0.35, 0.35, 0.15, 0.15]] * 2, device=device)
        elif mode == "front_point_contact":
            weights[index] = torch.tensor([[0.45, 0.45, 0.05, 0.05]] * 2, device=device)
    load_max = tensor(rng.uniform(100.0, 260.0, size=(count, 2, 1)).astype(np.float32), device)
    approach_speed = tensor(rng.uniform(0.08, 0.35, size=(count, 2, 1)).astype(np.float32), device)
    maximum_depth = tensor(rng.uniform(0.025, 0.065, size=(count, 2, 1)).astype(np.float32), device)
    lateral_ratio = tensor(rng.uniform(0.04, 0.32, size=(count, 2, 1)).astype(np.float32), device)
    mode_factor = torch.as_tensor(
        [
            0.72 if mode == "front_point_contact" else 0.86 if mode == "hybrid_contact" else 1.0
            for mode in modes
        ],
        device=device,
    ).view(count, 1, 1)

    force_gain = tensor(rng.normal(1.0, 0.035, size=shape).astype(np.float32), device)
    force_bias = tensor(rng.normal(0.0, 0.8, size=shape).astype(np.float32), device)
    depth_bias = tensor(rng.normal(0.0, 0.00025, size=shape).astype(np.float32), device)
    imu_accel_bias = tensor(rng.normal(0.0, 0.06, size=(count, 2, 3)).astype(np.float32), device)
    imu_gyro_bias = tensor(rng.normal(0.0, 0.006, size=(count, 2, 3)).astype(np.float32), device)
    canary = tensor(rng.normal(size=(count, 8)).astype(np.float32), device)
    adapter = CramponSensorAdapter(
        CramponSensorConfig(seed=args.seed, packet_rate_hz=100.0, sample_drop_probability=0.006)
    )

    visible_steps = []
    context_steps = {
        name: []
        for name in (
            "foot_position_xyz_m",
            "foot_velocity_xyz_mps",
            "pelvis_roll_pitch_yaw_rad",
            "commanded_probe_load_n",
            "commanded_foot_speed_mps",
            "body_weight_on_foot_n",
        )
    }
    command_steps = {
        name: []
        for name in (
            "requested_vx_mps",
            "requested_vy_mps",
            "requested_wz_rps",
            "mode",
            "probe_load_n",
            "approach_speed_mps",
        )
    }
    truth_steps = {
        name: []
        for name in (
            "targets",
            "events",
            "normal_force_n",
            "shear_capacity_n",
            "penetration_m",
            "friction_utilization",
            "slipping",
            "fractured",
            "damage",
        )
    }
    previous_depth = torch.zeros(shape, device=device)
    previous_total_force = torch.zeros((count, 2), device=device)
    response = None
    for physics_step in range(61):
        time_s = physics_step * 0.005
        ramp = float(np.sin(min(1.0, time_s / 0.30) * np.pi / 2.0) ** 1.35)
        total_load = load_max * ramp
        applied_load = total_load * weights
        desired = torch.minimum(maximum_depth, approach_speed * time_s)
        depth_shape = torch.clamp(weights / 0.25, 0.25, 1.45)
        depth = desired * depth_shape
        rate = torch.clamp((depth - previous_depth) / 0.005, min=0.0)
        slope_demand = torch.sin(angle).abs() * mode_factor
        tangential_demand = applied_load * torch.clamp(lateral_ratio + slope_demand, 0.0, 1.35)
        lateral_speed = torch.where(
            response.slipping
            if response is not None
            else torch.zeros(shape, dtype=torch.bool, device=device),
            0.04 + 0.12 * torch.sin(angle).abs(),
            0.002 * torch.ones_like(depth),
        )
        response = model.step(
            depth, rate, lateral_speed, applied_load, tangential_demand, dt_s=0.005
        )
        previous_depth = depth
        if physics_step % 2:
            continue
        measured_force = (
            torch.round(
                torch.clamp(
                    response.normal_force_n * force_gain
                    + force_bias
                    + torch.randn_like(depth) * 1.2,
                    0.0,
                    300.0,
                )
                / 0.25
            )
            * 0.25
        )
        measured_depth = (
            torch.round(
                torch.clamp(depth + depth_bias + torch.randn_like(depth) * 0.00035, min=0.0)
                / 0.0005
            )
            * 0.0005
        )
        total_force = response.normal_force_n.sum(dim=-1)
        force_rate = (total_force - previous_total_force) / 0.01
        previous_total_force = total_force
        accel = torch.zeros((count, 2, 3), device=device)
        accel[..., 0] = 9.81 * torch.sin(angle.squeeze(-1)) + 0.0025 * force_rate
        accel[..., 2] = 9.81 * torch.cos(angle.squeeze(-1))
        accel += imu_accel_bias + torch.randn_like(accel) * 0.12
        gyro = torch.zeros_like(accel)
        gyro[..., 1] = 0.002 * (
            (response.normal_force_n[..., 0] + response.normal_force_n[..., 1])
            - (response.normal_force_n[..., 2] + response.normal_force_n[..., 3])
        )
        gyro[..., 2] = response.slipping.float().mean(dim=-1) * 1.2
        gyro += imu_gyro_bias + torch.randn_like(gyro) * 0.012
        support = params.support_layer_depth_m.mean(dim=-1)
        void_depth = torch.where(params.void_present, params.void_top_depth_m, 10.0).amin(dim=-1)
        void_prob = params.void_present.float().mean(dim=-1)
        return_strength = torch.clamp(
            0.25 + params.vertical_stiffness_n_per_m.mean(dim=-1) / 55_000.0, 0.02, 1.0
        )
        has_void = void_prob > 0.0
        secondary_range = torch.where(has_void, void_depth, support + 0.18)
        secondary_strength = torch.where(
            has_void,
            0.70 * return_strength * torch.exp(-void_depth / 0.45),
            0.04 * return_strength,
        )
        strength_noise = torch.randn_like(return_strength) * 0.025
        radar = torch.stack(
            [
                support,
                secondary_range,
                (return_strength - 0.5 * strength_noise).clamp(0.0, 1.0),
                (secondary_strength + strength_noise).clamp(0.0, 1.0),
                (0.025 + strength_noise.abs()).clamp(0.0, 0.10),
            ],
            dim=-1,
        )
        radar[..., :2] = (
            torch.round((radar[..., :2] + torch.randn_like(radar[..., :2]) * 0.012) / 0.04) * 0.04
        )
        radar[..., 2:] = torch.round(radar[..., 2:] / 0.05) * 0.05
        if physics_step > 0:
            (
                measured_force,
                measured_depth,
                accel,
                gyro,
                radar,
                fresh_mask,
            ) = apply_sensor_faults(
                measured_force,
                measured_depth,
                accel,
                gyro,
                radar,
                fault_cycle,
                sample_index=physics_step // 2,
            )
        else:
            fresh_mask = torch.ones((count, 2, 19), dtype=torch.bool, device=device)
        timestamp = torch.full((count, 2), physics_step * 0.005, device=device)
        foot_position = torch.zeros((count, 2, 3), device=device)
        foot_position[..., 2] = -measured_depth.mean(dim=-1)
        foot_velocity = torch.zeros_like(foot_position)
        foot_velocity[..., 2] = -approach_speed.squeeze(-1)
        context = {
            "foot_position_xyz_m": foot_position,
            "foot_velocity_xyz_mps": foot_velocity,
            "pelvis_roll_pitch_yaw_rad": torch.stack(
                [torch.zeros_like(slopes), slopes * np.pi / 180.0, torch.zeros_like(slopes)], dim=-1
            ),
            "commanded_probe_load_n": total_load.squeeze(-1),
            "commanded_foot_speed_mps": approach_speed.squeeze(-1),
            "body_weight_on_foot_n": torch.clamp(total_force, 0.0, 343.0),
        }
        commands = {
            "requested_vx_mps": torch.full((count,), 0.20, device=device)
            * torch.cos(angle[:, 0, 0]),
            "requested_vy_mps": torch.zeros(count, device=device),
            "requested_wz_rps": torch.zeros(count, device=device),
            "mode": torch.ones(count, device=device),
            "probe_load_n": total_load.squeeze(-1),
            "approach_speed_mps": approach_speed.squeeze(-1),
        }
        packet = adapter.observe(
            axial_force_n=measured_force,
            penetration_m=measured_depth,
            accelerometer_mps2=accel,
            gyroscope_rps=gyro,
            radar_frontend=radar,
            timestamp_s=timestamp,
            fresh_mask=fresh_mask,
            context=context,
            commands=commands,
        )
        visible_steps.append(packet)
        for name, values in context_steps.items():
            values.append(context[name])
        for name, values in command_steps.items():
            values.append(commands[name])
        bearing = (params.bearing_capacity_n * (1.0 - 0.5 * model.damage)).mean(dim=-1)
        shear = response.shear_capacity_n.mean(dim=-1)
        fracture_margin = torch.clamp(
            params.fracture_strength_n - 4.0 * applied_load, min=0.0
        ).amin(dim=-1)
        probe_contact = depth > 0.0
        probe_slip_margin = response.shear_capacity_n - tangential_demand
        slip_margin = torch.where(probe_contact, probe_slip_margin, 0.0).sum(dim=-1)
        slip_margin = torch.where(
            probe_contact.any(dim=-1), slip_margin, torch.full_like(slip_margin, 500.0)
        )
        labels = torch.stack(
            [
                support,
                params.vertical_stiffness_n_per_m.mean(dim=-1),
                params.damping_ns_per_m.mean(dim=-1),
                bearing,
                shear,
                params.friction.mean(dim=-1),
                model.compaction.mean(dim=-1),
                model.damage.mean(dim=-1),
                fracture_margin,
                slip_margin,
                torch.where(void_prob > 0, void_depth, torch.zeros_like(void_depth)),
            ],
            dim=-1,
        )
        events = torch.stack(
            [void_prob > 0, model.fractured.any(dim=-1), response.slipping.any(dim=-1)], dim=-1
        )
        truth_values = {
            "targets": labels,
            "events": events,
            "normal_force_n": response.normal_force_n,
            "shear_capacity_n": response.shear_capacity_n,
            "penetration_m": depth,
            "friction_utilization": response.friction_utilization,
            "slipping": response.slipping,
            "fractured": model.fractured,
            "damage": model.damage,
        }
        for name, values in truth_steps.items():
            values.append(truth_values[name])

    stack_np = lambda values: torch.stack(values, dim=1).detach().cpu().numpy()
    visible = {
        "packet_values": stack_np([item.packet_values for item in visible_steps]),
        "valid_mask": stack_np([item.valid_mask for item in visible_steps]).astype(bool),
        "timestamp_s": stack_np([item.timestamp_s for item in visible_steps]),
        "sample_age_s": stack_np([item.sample_age_s for item in visible_steps]),
        "context": {name: stack_np(values) for name, values in context_steps.items()},
        "commands": {name: stack_np(values) for name, values in command_steps.items()},
    }
    truth = {name: stack_np(values) for name, values in truth_steps.items()}
    truth["material_code"] = params.material_code.detach().cpu().numpy()
    truth["void_present"] = params.void_present.detach().cpu().numpy()
    truth["truth_canary"] = canary.detach().cpu().numpy()

    final_events = truth["events"][:, -1]
    final_force = truth["normal_force_n"][:, -1].sum(axis=(1, 2))
    rows = []
    for index, case in enumerate(cases):
        surface = surface_map[case.surface_id]
        group_parts = {
            "generator_family": "everest_l0_v2_corrected_contact_radar",
            "parent_seed": args.seed,
            "surface": case.surface_id,
            "incline_deg": case.incline_deg,
            "hazard": case.hazard_id,
            "repetition": case.repetition,
            "geometry_revision": "26-component-usdc-53703057",
            "spatial_block": case.case_id,
        }
        group_hash = stable_group_hash(group_parts)
        any_void, any_fracture, any_slip = final_events[index].any(axis=0)
        if case.hazard_id == "open_crevasse_gap" and final_force[index] < 20.0:
            outcome = "no_contact"
        elif any_slip:
            outcome = "slip"
        elif any_void or any_fracture:
            outcome = "partial_grip"
        else:
            outcome = "stable"
        rows.append(
            {
                "episode_id": index,
                "case_id": case.case_id,
                "group_hash": group_hash,
                "split": split_for_hash(group_hash),
                "surface_id": case.surface_id,
                "surface_family": case.surface_family,
                "physics_prior": surface["physics_prior"],
                "incline_deg": float(case.incline_deg),
                "configured_depth_m": float(surface["configured_depth_m"]),
                "hazard_id": case.hazard_id,
                "contact_mode_id": case.contact_mode_id,
                "sensor_fault_mode": SENSOR_FAULT_MODES[fault_codes_cpu[index]],
                "repetition": case.repetition,
                "outcome": outcome,
                "sampling_regime": "natural_prior"
                if case.hazard_id in {"none", "damaged_previous_footprint", "exposed_ice_patch"}
                else "stress_tail",
            }
        )

    provenance = {
        "seed": args.seed,
        "physics_dt_s": 0.005,
        "packet_rate_hz": 100.0,
        "terrain_suite_sha256": sha256_file(config_path),
        "collector_sha256": sha256_file(Path(__file__)),
        "stateful_material_sha256": sha256_file(
            root / "isaaclab_ext/source/hackathon_everest_isaaclab/"
            "hackathon_everest_isaaclab/contact/stateful_material.py"
        ),
        "crampon_source_sha256": "53703057dff7ea5b2e7e468164289d6c0aba629400952c9a8a9a5f7048f2a660",
        "claim_boundary": config["claim_boundary"],
        "torch_version": torch.__version__,
        "device": str(device),
        "target_names": TARGET_NAMES,
        "event_names": EVENT_NAMES,
        "sensor_fault_modes": SENSOR_FAULT_MODES,
        "sensor_fault_distribution": "50% nominal; 10% each authored fault mode",
    }
    output = write_immutable_shard(
        args.dataset_root,
        dataset_id=args.dataset_id,
        shard_id="worker-0000",
        visible=visible,
        truth=truth,
        episode_rows=rows,
        provenance=provenance,
    )
    summary = {
        "output": str(output),
        "episodes": count,
        "steps": len(visible_steps),
        "outcomes": {
            name: sum(row["outcome"] == name for row in rows)
            for name in sorted({row["outcome"] for row in rows})
        },
        "splits": {
            name: sum(row["split"] == name for row in rows)
            for name in ("train", "calibration", "validation", "sealed_test")
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
