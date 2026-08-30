"""Create an editable G1 Blender fit from a positioned USD crampon assembly.

The USD assembly is aligned to the saved crampon geometry in g1_crampon_fit.blend.
Every USD component keeps its authored relative transform. A shared Blender control
moves both feet, with optional per-foot fine-tune controls.

Run with Blender, not regular Python:
  blender --background --python blender/setup_usd_component_fit.py -- \
    --base-blend blender/g1_crampon_fit.blend \
    --source-usd /path/Shoe_with_crampons_and_components.usdc \
    --output blender/g1_crampon_components_fit.blend
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector, kdtree

RAW_STL_TO_METERS_SCALE = 100.0
MAIN_FRAME_VERTEX_COUNT = 34_265


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-blend", type=Path, required=True)
    parser.add_argument("--source-usd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--metadata", type=Path)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def move_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for existing in list(obj.users_collection):
        existing.objects.unlink(obj)
    collection.objects.link(obj)


def select_only(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def mesh_world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    lows: list[Vector] = []
    highs: list[Vector] = []
    for obj in objects:
        points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        lows.append(Vector(tuple(min(point[axis] for point in points) for axis in range(3))))
        highs.append(Vector(tuple(max(point[axis] for point in points) for axis in range(3))))
    return (
        Vector(tuple(min(point[axis] for point in lows) for axis in range(3))),
        Vector(tuple(max(point[axis] for point in highs) for axis in range(3))),
    )


def centroid_world(obj: bpy.types.Object) -> Vector:
    total = Vector((0.0, 0.0, 0.0))
    for vertex in obj.data.vertices:
        total += obj.matrix_world @ vertex.co
    return total / len(obj.data.vertices)


def normalized_rotation(matrix: Matrix) -> Matrix:
    rotation = matrix.to_3x3().normalized()
    value = Matrix.Identity(4)
    for row in range(3):
        for col in range(3):
            value[row][col] = rotation[row][col]
    return value


def uniform_scale(matrix: Matrix) -> float:
    axes = matrix.to_3x3()
    return sum(axes.col[index].length for index in range(3)) / 3.0


def nearest_rms(source: bpy.types.Object, target: bpy.types.Object, transform: Matrix) -> tuple[float, float]:
    tree = kdtree.KDTree(len(target.data.vertices))
    for index, vertex in enumerate(target.data.vertices):
        tree.insert(target.matrix_world @ vertex.co, index)
    tree.balance()
    squared = 0.0
    maximum = 0.0
    for vertex in source.data.vertices:
        mapped = transform @ (source.matrix_world @ vertex.co)
        _, _, distance = tree.find(mapped)
        squared += distance * distance
        maximum = max(maximum, distance)
    return (squared / len(source.data.vertices)) ** 0.5, maximum


def friendly_name(source_name: str) -> str:
    if source_name == "mesh_003":
        return "CRAMPON_FRAME"
    if source_name == "_MF_Mesh_005":
        return "MOUNT_PLATE"
    return source_name.upper().replace(" ", "_")


def create_empty(name: str, collection: bpy.types.Collection, display: str, size: float) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    collection.objects.link(obj)
    obj.empty_display_type = display
    obj.empty_display_size = size
    return obj


def parent_local(child: bpy.types.Object, parent: bpy.types.Object, local_matrix: Matrix) -> None:
    child.parent = parent
    child.matrix_parent_inverse.identity()
    child.matrix_local = local_matrix


def main() -> None:
    args = arguments()
    base_blend = args.base_blend.resolve()
    source_usd = args.source_usd.resolve()
    output = args.output.resolve()
    if not base_blend.exists():
        raise FileNotFoundError(base_blend)
    if not source_usd.exists():
        raise FileNotFoundError(source_usd)

    bpy.ops.wm.open_mainfile(filepath=str(base_blend))
    target = bpy.data.objects.get("EDITABLE__LEFT_CRAMPON_FRAME")
    left_ankle = bpy.data.objects.get("LEFT_ANKLE_ROLL_FRAME")
    right_ankle = bpy.data.objects.get("RIGHT_ANKLE_ROLL_FRAME")
    if target is None or left_ankle is None or right_ankle is None:
        raise RuntimeError("Base Blender file does not contain the expected G1 fitting objects")

    # Keep the previous authoritative fit in the file, but hide it so there is no overlapping geometry.
    for name in (
        "EDITABLE__LEFT_CRAMPON_FRAME",
        "EDITABLE__LEFT_MOUNT_PLATE",
        "EDITABLE__RIGHT_CRAMPON_FRAME",
        "EDITABLE__RIGHT_MOUNT_PLATE",
        "CRAMPON_FIT_CONTROL",
        "LEFT_FINE_TUNE",
        "RIGHT_FINE_TUNE",
    ):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_viewport = True
            obj.hide_render = True
            obj.hide_select = True

    existing = set(bpy.data.objects)
    bpy.ops.wm.usd_import(
        filepath=str(source_usd),
        import_cameras=False,
        import_lights=False,
        import_materials=True,
        import_meshes=True,
    )
    imported = list(set(bpy.data.objects) - existing)
    imported_meshes = [obj for obj in imported if obj.type == "MESH" and len(obj.data.vertices)]
    if not imported_meshes:
        raise RuntimeError(f"USD import created no mesh objects: {source_usd}")
    source_frame = next(
        (obj for obj in imported_meshes if len(obj.data.vertices) == MAIN_FRAME_VERTEX_COUNT),
        None,
    )
    if source_frame is None:
        raise RuntimeError(f"Could not identify {MAIN_FRAME_VERTEX_COUNT}-vertex main frame in USD")

    # The current STL mesh uses a fixed 100x source-unit conversion. The USD is already in metres.
    # Its authored mesh orientation matches the normalized destination mesh frame. Align centroids
    # after applying that rotation and the saved uniform fit scale.
    target_rotation = normalized_rotation(target.matrix_world)
    fit_scale = uniform_scale(target.matrix_world) / RAW_STL_TO_METERS_SCALE
    source_center = centroid_world(source_frame)
    target_center = centroid_world(target)
    source_to_target = target_rotation.copy()
    for row in range(3):
        for col in range(3):
            source_to_target[row][col] *= fit_scale
    mapped_center = source_to_target @ source_center
    source_to_target.translation = target_center - mapped_center

    rms_error_m, maximum_error_m = nearest_rms(source_frame, target, source_to_target)
    if rms_error_m > 1e-5 or maximum_error_m > 5e-5:
        raise RuntimeError(
            f"USD-to-saved-fit alignment is not exact enough: rms={rms_error_m:.9g} m, "
            f"max={maximum_error_m:.9g} m"
        )

    low, high = mesh_world_bounds(imported_meshes)
    source_pivot = (low + high) * 0.5
    pivot_matrix = Matrix.Translation(source_pivot)
    source_centered = Matrix.Translation(-source_pivot)

    # Express the shared control in ankle-local coordinates and put its origin at the asset center.
    source_to_left_ankle = left_ankle.matrix_world.inverted() @ source_to_target
    control_matrix = source_to_left_ankle @ pivot_matrix

    work = bpy.data.collections.new("04_EDITABLE_USD_COMPONENT_ASSEMBLY")
    bpy.context.scene.collection.children.link(work)
    templates = bpy.data.collections.new("98_INTERNAL_USD_SOURCE_TEMPLATE")
    bpy.context.scene.collection.children.link(templates)
    templates.hide_render = True
    templates.hide_viewport = True

    control = create_empty("USD_ASSET_POSITION_CONTROL", work, "CIRCLE", 0.13)
    control.matrix_world = control_matrix
    control.show_name = True
    control["instructions"] = "Use G/R/S or Item > Transform to move both complete USD assemblies."
    control["source_usdc"] = str(source_usd)
    control["source_sha256"] = hashlib.sha256(source_usd.read_bytes()).hexdigest()
    control["source_world_bounds_m"] = json.dumps({"min": list(low), "max": list(high)})
    control["source_pivot_world_m"] = list(source_pivot)
    control["alignment_rms_m"] = rms_error_m
    control["alignment_max_m"] = maximum_error_m
    control["alignment_method"] = "Exact main-mesh geometry match to saved G1 Blender fit"

    # Preserve the untouched imported objects as hidden source templates.
    source_matrices: dict[str, Matrix] = {}
    for obj in imported:
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        source_matrix = obj.matrix_world.copy()
        source_name = obj.name
        obj.name = f"SOURCE__USD__{friendly_name(source_name)}"
        obj["usd_source_name"] = source_name
        source_matrices[obj.name] = source_matrix
        move_to_collection(obj, templates)
        obj.hide_render = True
        obj.hide_viewport = True
        obj.hide_select = True

    for side, ankle in (("LEFT", left_ankle), ("RIGHT", right_ankle)):
        follow = create_empty(f"FOLLOW__{side}_USD_ASSET", work, "CUBE", 0.055)
        parent_local(follow, ankle, Matrix.Identity(4))
        follow.hide_select = True
        # Raw transform-property drivers preserve the control's normal Blender T/R/S composition.
        # COPY_TRANSFORMS rotates the translated control origin when the control has a non-zero
        # source-axis rotation, which would move the assembly away from the saved fit.
        for data_path in ("location", "rotation_euler", "scale"):
            for axis in range(3):
                fcurve = follow.driver_add(data_path, axis)
                driver = fcurve.driver
                driver.type = "SCRIPTED"
                variable = driver.variables.new()
                variable.name = "value"
                variable.type = "SINGLE_PROP"
                variable.targets[0].id = control
                variable.targets[0].data_path = f"{data_path}[{axis}]"
                driver.expression = "value"

        fine = create_empty(f"{side}_USD_FINE_TUNE", work, "ARROWS", 0.03)
        parent_local(fine, follow, Matrix.Identity(4))
        fine["instructions"] = f"Optional {side.lower()}-only G/R/S correction after the shared USD control."

        for template in [obj for obj in templates.objects if obj.type == "MESH"]:
            original_name = template.name.removeprefix("SOURCE__USD__")
            part = template.copy()
            part.data = template.data.copy()
            work.objects.link(part)
            part.name = f"EDITABLE__USD_{side}__{original_name}"
            part.hide_render = False
            part.hide_viewport = False
            part.hide_select = False
            local_matrix = source_centered @ source_matrices[template.name]
            parent_local(part, fine, local_matrix)
            part["usd_original_object"] = template["usd_source_name"]
            part["source_relative_matrix"] = json.dumps([list(row) for row in local_matrix])

    # Restore useful workbench state.
    bpy.context.scene["README_USD_COMPONENT_FIT"] = (
        "Select USD_ASSET_POSITION_CONTROL and use G/R/S. "
        "LEFT_USD_FINE_TUNE and RIGHT_USD_FINE_TUNE make per-foot corrections. "
        "The USD contained no humanoid; it was aligned by exact geometry to the saved G1 fit."
    )
    bpy.context.scene["source_usdc"] = str(source_usd)
    bpy.context.scene["source_usdc_sha256"] = control["source_sha256"]
    bpy.context.scene["usd_alignment_rms_m"] = rms_error_m
    bpy.context.scene["usd_alignment_max_m"] = maximum_error_m

    metadata = {
        "schema_version": "1.0.0",
        "source_usdc": source_usd.name,
        "source_sha256": control["source_sha256"],
        "source_stage": {"up_axis": "Z", "meters_per_unit": 1.0, "default_prim": "/root"},
        "source_contains_humanoid": False,
        "source_mesh_objects": len(imported_meshes),
        "source_world_bounds_m": {"min": list(low), "max": list(high)},
        "source_pivot_world_m": list(source_pivot),
        "alignment": {
            "target": "saved EDITABLE__LEFT_CRAMPON_FRAME in g1_crampon_fit.blend",
            "method": "exact main-mesh geometry match",
            "rms_error_m": rms_error_m,
            "maximum_error_m": maximum_error_m,
        },
        "control": {
            "name": control.name,
            "location_m": list(control.location),
            "rotation_euler_rad": list(control.rotation_euler),
            "scale": list(control.scale),
        },
        "per_foot_controls": ["LEFT_USD_FINE_TUNE", "RIGHT_USD_FINE_TUNE"],
        "output_blend": output.name,
    }
    if args.metadata is not None:
        metadata_path = args.metadata.resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    select_only(control)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    if args.render is not None:
        render = args.render.resolve()
        render.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(render)
        bpy.ops.render.render(write_still=True)
    print(f"Saved editable USD component fit: {output}")
    print(f"USD alignment RMS: {rms_error_m:.9g} m; max: {maximum_error_m:.9g} m")
    print(f"Control transform: location={tuple(control.location)}, rotation={tuple(control.rotation_euler)}, scale={tuple(control.scale)}")


if __name__ == "__main__":
    main()
