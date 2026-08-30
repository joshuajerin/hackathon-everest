"""Export the saved Blender fit into G1-local MuJoCo assets and metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SPIKE_OFFSETS_BASE_M = ((0.075, 0.045), (0.075, -0.045), (-0.075, 0.045), (-0.075, -0.045))
PROBE_BODY_Z_BASE_M = -0.005
PROBE_AXIS_LENGTH_BASE_M = 0.027608
PROBE_RADIUS_BASE_M = 0.003
IMU_BASE_M = (0.010, 0.0, 0.018)
RADAR_BASE_M = (0.055, 0.0, -0.006)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def triangle_vertices_in_anchor(
    obj: bpy.types.Object, anchor: bpy.types.Object
) -> list[tuple[Vector, Vector, Vector]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    mesh.calc_loop_triangles()
    transform = anchor.matrix_world.inverted() @ evaluated.matrix_world
    triangles = [
        tuple(transform @ mesh.vertices[index].co for index in triangle.vertices)
        for triangle in mesh.loop_triangles
    ]
    evaluated.to_mesh_clear()
    return triangles


def write_binary_stl(path: Path, triangles: list[tuple[Vector, Vector, Vector]], label: str) -> None:
    header = f"Hackathon Everest Blender fit {label}".encode()[:80].ljust(80, b"\0")
    records = bytearray()
    for first, second, third in triangles:
        normal = (second - first).cross(third - first)
        if normal.length_squared > 1e-20:
            normal.normalize()
        else:
            normal = Vector((0.0, 0.0, 0.0))
        records.extend(struct.pack("<12fH", *normal, *first, *second, *third, 0))
    path.write_bytes(header + struct.pack("<I", len(triangles)) + records)


def bounds(triangles: list[tuple[Vector, Vector, Vector]]) -> dict[str, list[float]]:
    points = [point for triangle in triangles for point in triangle]
    minimum = [min(point[index] for point in points) for index in range(3)]
    maximum = [max(point[index] for point in points) for index in range(3)]
    return {
        "minimum_m": minimum,
        "maximum_m": maximum,
        "extents_m": [maximum[index] - minimum[index] for index in range(3)],
    }


def vector_list(value: Vector) -> list[float]:
    return [float(component) for component in value]


def object_transform(obj: bpy.types.Object) -> dict[str, list[float]]:
    return {
        "location_m": vector_list(obj.location),
        "rotation_euler_rad": vector_list(obj.rotation_euler),
        "scale": vector_list(obj.scale),
    }


def main() -> None:
    args = arguments()
    blend = args.blend.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bpy.context.view_layer.update()

    anchor = bpy.data.objects["LEFT_ANKLE_ROLL_FRAME"]
    fine = bpy.data.objects["LEFT_FINE_TUNE"]
    fit_matrix = anchor.matrix_world.inverted() @ fine.matrix_world
    scale_xyz = [fit_matrix.to_3x3().col[index].length for index in range(3)]
    if max(scale_xyz) - min(scale_xyz) > 1e-6:
        raise ValueError(f"Nonuniform Blender fit scale is not supported: {scale_xyz}")
    uniform_scale = sum(scale_xyz) / 3.0
    rotation = fit_matrix.to_3x3().normalized().to_quaternion()

    components = {}
    all_triangles = []
    for object_name, filename in (
        ("EDITABLE__LEFT_CRAMPON_FRAME", "crampon_frame_fitted.stl"),
        ("EDITABLE__LEFT_MOUNT_PLATE", "mount_plate_fitted.stl"),
    ):
        triangles = triangle_vertices_in_anchor(bpy.data.objects[object_name], anchor)
        write_binary_stl(out / filename, triangles, object_name)
        component_bounds = bounds(triangles)
        components[object_name] = {
            "file": filename,
            "triangle_count": len(triangles),
            **component_bounds,
        }
        all_triangles.extend(triangles)

    probe_bodies = []
    for x_m, y_m in SPIKE_OFFSETS_BASE_M:
        point = fit_matrix @ Vector((x_m, y_m, PROBE_BODY_Z_BASE_M))
        probe_bodies.append(vector_list(point))
    imu = fit_matrix @ Vector(IMU_BASE_M)
    radar = fit_matrix @ Vector(RADAR_BASE_M)
    combined = bounds(all_triangles)
    lowest_probe_tip = min(
        body[2] - uniform_scale * (PROBE_AXIS_LENGTH_BASE_M + PROBE_RADIUS_BASE_M)
        for body in probe_bodies
    )
    metadata = {
        "schema_version": 1,
        "source_blend": blend.name,
        "source_blend_sha256": hashlib.sha256(blend.read_bytes()).hexdigest(),
        "controller": object_transform(bpy.data.objects["CRAMPON_FIT_CONTROL"]),
        "left_fine_tune": object_transform(fine),
        "right_fine_tune": object_transform(bpy.data.objects["RIGHT_FINE_TUNE"]),
        "fit_matrix_g1_local": [[float(value) for value in row] for row in fit_matrix],
        "uniform_scale_from_base": uniform_scale,
        "rotation_quaternion_wxyz": [rotation.w, rotation.x, rotation.y, rotation.z],
        "components": components,
        "combined_bounds_m": combined,
        "probe": {
            "body_positions_m": probe_bodies,
            "axis_length_m": uniform_scale * PROBE_AXIS_LENGTH_BASE_M,
            "radius_m": uniform_scale * PROBE_RADIUS_BASE_M,
            "lowest_tip_z_m": lowest_probe_tip,
        },
        "imu_position_m": vector_list(imu),
        "radar_position_m": vector_list(radar),
        "note": "Exported from the saved Blender controls in G1 ankle-roll coordinates.",
    }
    (out / "blender_fit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
