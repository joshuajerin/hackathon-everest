#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import newton
import numpy as np
import warp as wp
import yaml
from newton.solvers import SolverImplicitMPM


def trajectory_depth(time_s: float, maximum_depth_m: float, speed_mps: float) -> float:
    phase_s = maximum_depth_m / speed_mps
    if time_s <= phase_s:
        return speed_mps * time_s
    if time_s <= 2.0 * phase_s:
        return maximum_depth_m - speed_mps * (time_s - phase_s)
    return min(maximum_depth_m, speed_mps * (time_s - 2.0 * phase_s))


def main() -> int:
    parser = argparse.ArgumentParser(description="Newton MPM single-spike snow discrepancy trace")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("selected_cases.yaml")
    )
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-steps", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.voxel_size not in config["numerics"]["voxel_sizes_m"]:
        raise ValueError("voxel size is not one of the authored convergence levels")

    newton.use_coord_layout_targets = True
    wp.set_device(args.device)
    case = config["case"]
    bed = config["bed"]
    material = config["material"]
    geometry = config["geometry"]
    numerics = config["numerics"]
    trajectory = config["trajectory"]
    voxel = float(args.voxel_size)
    dt = float(numerics["dt_s"])
    bed_size = np.asarray(bed["size_xyz_m"], dtype=np.float64)
    particles_per_cell = 2
    spacing = voxel / particles_per_cell
    resolution = np.maximum(2, np.ceil(bed_size / spacing).astype(int))
    cell = bed_size / resolution
    radius = float(cell.max() * 0.5)
    mass = float(np.prod(cell) * material["density_kg_per_m3"])

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_particle_grid(
        pos=wp.vec3(*(-0.5 * bed_size[:2]), 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=int(resolution[0]) + 1,
        dim_y=int(resolution[1]) + 1,
        dim_z=int(resolution[2]) + 1,
        cell_x=float(cell[0]),
        cell_y=float(cell[1]),
        cell_z=float(cell[2]),
        mass=mass,
        jitter=0.0,
        radius_mean=radius,
    )
    half_height = float(geometry["newton_cone_half_height_m"])
    bed_top = float(bed_size[2])
    probe_center_z = bed_top + half_height
    probe = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, probe_center_z), wp.quat_identity()),
        is_kinematic=True,
        label="single_crampon_spike",
    )
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=float(material["friction"]))
    shape_cfg.margin = 0.25 * voxel
    builder.add_shape_cone(
        probe,
        radius=float(geometry["newton_cone_radius_m"]),
        half_height=half_height,
        cfg=shape_cfg,
        label="sharp_cone_geometry_limit",
    )
    model = builder.finalize(device=args.device)
    model.set_gravity((0.0, 0.0, -9.81))
    state_0 = model.state()
    state_1 = model.state()

    positions = state_0.particle_q.numpy()
    boundary_width = 1.5 * spacing
    boundary = (
        (positions[:, 2] <= boundary_width)
        | (np.abs(positions[:, 0]) >= 0.5 * bed_size[0] - boundary_width)
        | (np.abs(positions[:, 1]) >= 0.5 * bed_size[1] - boundary_width)
    )
    boundary_indices = wp.array(np.flatnonzero(boundary), dtype=wp.int32, device=model.device)
    model.particle_mass[boundary_indices].fill_(0.0)
    state_0.mpm.particle_Jp.fill_(0.975)
    model.mpm.young_modulus.fill_(float(material["young_modulus_pa"]))
    model.mpm.poisson_ratio.fill_(float(material["poisson_ratio"]))
    model.mpm.friction.fill_(float(material["friction"]))
    model.mpm.damping.fill_(float(material["damping_s"]))
    model.mpm.yield_pressure.fill_(float(material["yield_pressure_pa"]))
    model.mpm.tensile_yield_ratio.fill_(float(material["tensile_yield_ratio"]))
    model.mpm.yield_stress.fill_(float(material["yield_stress_pa"]))
    model.mpm.hardening.fill_(float(material["hardening"]))
    model.mpm.dilatancy.fill_(float(material["dilatancy"]))

    options = SolverImplicitMPM.Config()
    options.voxel_size = voxel
    options.grid_type = str(numerics["grid_type"])
    options.collider_velocity_mode = str(numerics["collider_velocity_mode"])
    options.max_iterations = int(numerics["max_iterations"])
    options.tolerance = float(numerics["tolerance"])
    solver = SolverImplicitMPM(model, config=options)

    phase_s = float(trajectory["maximum_depth_m"] / trajectory["indentation_speed_mps"])
    steps = int(np.ceil(3.0 * phase_s / dt))
    if args.maximum_steps is not None:
        steps = min(steps, args.maximum_steps)
    trace_time = np.empty(steps, dtype=np.float64)
    trace_depth = np.empty(steps, dtype=np.float64)
    trace_reaction = np.zeros((steps, 3), dtype=np.float64)
    trace_jp = np.empty((steps, 3), dtype=np.float64)
    collider_bodies = solver.collider_body_index.numpy()

    for step in range(steps):
        time_s = (step + 1) * dt
        depth = trajectory_depth(
            time_s,
            float(trajectory["maximum_depth_m"]),
            float(trajectory["indentation_speed_mps"]),
        )
        body_q = state_0.body_q.numpy()
        body_q[probe] = (0.0, 0.0, probe_center_z - depth, 0.0, 0.0, 0.0, 1.0)
        state_0.body_q.assign(body_q)
        body_qd = state_0.body_qd.numpy()
        body_qd[probe] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        state_0.body_qd.assign(body_qd)
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0
        impulses, _, collider_ids = solver.collect_collider_impulses(state_0)
        impulse_np = impulses.numpy()
        ids_np = collider_ids.numpy()
        if len(ids_np):
            selected = collider_bodies[ids_np] == probe
            if selected.any():
                trace_reaction[step] = impulse_np[selected, :3].sum(axis=0) / dt
        jp = state_0.mpm.particle_Jp.numpy()
        trace_time[step] = time_s
        trace_depth[step] = depth
        trace_jp[step] = (float(jp.min()), float(jp.mean()), float(jp.max()))

    final_positions = state_0.particle_q.numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        time_s=trace_time,
        commanded_depth_m=trace_depth,
        collider_reaction_xyz_n=trace_reaction,
        particle_jp_min_mean_max=trace_jp,
        final_particle_positions_m=final_positions,
    )
    manifest = {
        "status": "trace_complete",
        "case_id": case["id"],
        "newton_version": newton.__version__,
        "device": args.device,
        "voxel_size_m": voxel,
        "dt_s": dt,
        "steps": steps,
        "particle_count": int(model.particle_count),
        "output": str(args.output),
        "claim_boundary": config["claim_boundary"],
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
