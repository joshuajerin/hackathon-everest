"""Build an editable Blender workbench for fitting the crampon to the G1 ankle.

Run with Blender, not regular Python:
  blender --background --python blender/setup_crampon_fit.py -- \
    --source-stl /path/Shoe_with_crampons_separated.stl \
    --menagerie vendor/mujoco_menagerie/unitree_g1 \
    --output blender/g1_crampon_fit.blend
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

BASE_STL_SCALE = 100.0
DEFAULT_FIT_SCALE = 1.08
ANKLE_SEPARATION_M = 0.237


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-stl", type=Path, required=True)
    parser.add_argument("--menagerie", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", type=Path)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


def make_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def move_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for existing in list(obj.users_collection):
        existing.objects.unlink(obj)
    collection.objects.link(obj)


def select_only(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def import_stl(path: Path, collection: bpy.types.Collection, name: str) -> bpy.types.Object:
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=str(path.resolve()), global_scale=1.0)
    else:
        bpy.ops.import_mesh.stl(filepath=str(path.resolve()), global_scale=1.0)
    created = list(set(bpy.data.objects) - before)
    if len(created) != 1:
        raise RuntimeError(f"Expected one imported object for {path}, got {len(created)}")
    obj = created[0]
    obj.name = name
    move_to_collection(obj, collection)
    return obj


def separate_loose_parts(obj: bpy.types.Object) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    parts = [obj, *list(set(bpy.data.objects) - before)]
    if len(parts) != 2:
        raise RuntimeError(f"Expected two STL components, got {len(parts)}")
    return parts


def material(name: str, color: tuple[float, float, float, float], metallic: float, roughness: float):
    value = bpy.data.materials.new(name)
    value.diffuse_color = color
    value.metallic = metallic
    value.roughness = roughness
    value.use_nodes = True
    principled = value.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Metallic"].default_value = metallic
        principled.inputs["Roughness"].default_value = roughness
    return value


def empty(name: str, collection: bpy.types.Collection, display: str, size: float) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    collection.objects.link(obj)
    obj.empty_display_type = display
    obj.empty_display_size = size
    return obj


def parent_local(child: bpy.types.Object, parent: bpy.types.Object) -> None:
    child.parent = parent
    child.matrix_parent_inverse.identity()


def look_at(camera: bpy.types.Object, point: tuple[float, float, float]) -> None:
    camera.rotation_euler = (Vector(point) - camera.location).to_track_quat("-Z", "Y").to_euler()


def add_camera(name: str, location: tuple[float, float, float], target: tuple[float, float, float], collection):
    camera_data = bpy.data.cameras.new(name)
    camera = bpy.data.objects.new(name, camera_data)
    collection.objects.link(camera)
    camera.location = location
    camera.data.lens = 52
    camera.data.clip_start = 0.005
    camera.data.clip_end = 20.0
    look_at(camera, target)
    return camera


def create_foot(
    side: str,
    anchor_y: float,
    source_parts: tuple[bpy.types.Object, bpy.types.Object],
    controller: bpy.types.Object,
    work: bpy.types.Collection,
    refs: bpy.types.Collection,
    menagerie: Path,
    gray: bpy.types.Material,
    mount_material: bpy.types.Material,
    frame_material: bpy.types.Material,
) -> None:
    anchor = empty(f"{side}_ANKLE_ROLL_FRAME", work, "ARROWS", 0.06)
    anchor.location.y = anchor_y

    # Official lower-leg reference geometry at zero joint angles.
    roll = import_stl(menagerie / "assets" / f"{side.lower()}_ankle_roll_link.STL", refs, f"REFERENCE__{side}_ANKLE_ROLL")
    parent_local(roll, anchor)
    roll.data.materials.clear()
    roll.data.materials.append(gray)
    roll.hide_select = True
    pitch = import_stl(menagerie / "assets" / f"{side.lower()}_ankle_pitch_link.STL", refs, f"REFERENCE__{side}_ANKLE_PITCH")
    parent_local(pitch, anchor)
    pitch.location = (0.0, 0.0, 0.017558)
    pitch.data.materials.clear()
    pitch.data.materials.append(gray)
    pitch.hide_select = True
    knee = import_stl(menagerie / "assets" / f"{side.lower()}_knee_link.STL", refs, f"REFERENCE__{side}_KNEE")
    parent_local(knee, anchor)
    lateral = 0.000094445 if side == "LEFT" else -0.000094445
    knee.location = (0.0, lateral, 0.317568)
    knee.data.materials.clear()
    knee.data.materials.append(gray)
    knee.hide_select = True

    driven = empty(f"FOLLOW__{side}_CRAMPON", work, "CUBE", 0.055)
    parent_local(driven, anchor)
    driven.hide_select = True
    copy_transform = driven.constraints.new(type="COPY_TRANSFORMS")
    copy_transform.name = "Follow CRAMPON_FIT_CONTROL"
    copy_transform.target = controller
    copy_transform.target_space = "LOCAL"
    copy_transform.owner_space = "LOCAL"
    fine = empty(f"{side}_FINE_TUNE", work, "ARROWS", 0.025)
    parent_local(fine, driven)
    fine["instructions"] = "Optional per-foot correction after the shared CRAMPON_FIT_CONTROL transform."
    source_axis = empty(f"FIXED__{side}_SOURCE_AXIS_AND_UNIT_SCALE", work, "ARROWS", 0.035)
    parent_local(source_axis, fine)
    source_axis.rotation_euler.z = math.radians(-90.0)
    source_axis.scale = (BASE_STL_SCALE,) * 3
    source_axis.hide_select = True

    for original in source_parts:
        part = original.copy()
        part.data = original.data
        work.objects.link(part)
        is_mount = "MOUNT" in original.name
        part.name = f"EDITABLE__{side}_{'MOUNT_PLATE' if is_mount else 'CRAMPON_FRAME'}"
        part.hide_render = False
        part.hide_viewport = False
        part.hide_set(False)
        parent_local(part, source_axis)
        part.data.materials.clear()
        part.data.materials.append(mount_material if is_mount else frame_material)


def configure_scene(output: Path, render_path: Path | None) -> None:
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.unit_settings.scale_length = 1.0
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 700
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.world.color = (0.008, 0.012, 0.020)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    scene["README"] = "Select CRAMPON_FIT_CONTROL. Use G/R/S or the standard Transform panel. LEFT_FINE_TUNE and RIGHT_FINE_TUNE are optional."
    scene["source_stl_uniform_scale"] = BASE_STL_SCALE
    scene["current_fit_factor"] = DEFAULT_FIT_SCALE
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output.resolve()))
    if render_path is not None:
        render_path.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(render_path.resolve())
        bpy.ops.render.render(write_still=True)


def main() -> None:
    args = arguments()
    source = args.source_stl.resolve()
    menagerie = args.menagerie.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if not (menagerie / "g1.xml").exists():
        raise FileNotFoundError(menagerie / "g1.xml")

    clear_scene()
    work = make_collection("01_EDITABLE_FIT")
    refs = make_collection("02_G1_REFERENCE")
    guides = make_collection("03_GUIDES_CAMERAS")
    templates = make_collection("99_INTERNAL_SOURCE_TEMPLATE")
    templates.hide_render = True

    gray = material("G1_REFERENCE_GRAY", (0.22, 0.25, 0.29, 1.0), 0.5, 0.32)
    mount_mat = material("MOUNT_PLATE_BLACK", (0.025, 0.03, 0.035, 1.0), 0.45, 0.25)
    frame_mat = material("CRAMPON_FRAME_BLUE_STEEL", (0.035, 0.12, 0.20, 1.0), 0.9, 0.22)
    floor_mat = material("WORKBENCH_FLOOR", (0.055, 0.075, 0.10, 1.0), 0.1, 0.55)

    imported = import_stl(source, templates, "SOURCE__USER_STL")
    parts = separate_loose_parts(imported)
    parts.sort(key=lambda part: sum(vertex.co.z for vertex in part.data.vertices) / len(part.data.vertices))
    parts[0].name = "TEMPLATE__CRAMPON_FRAME"
    parts[1].name = "TEMPLATE__MOUNT_PLATE"
    for part in parts:
        part.hide_render = True
        part.hide_viewport = True

    controller = empty("CRAMPON_FIT_CONTROL", work, "CIRCLE", 0.13)
    controller.scale = (DEFAULT_FIT_SCALE,) * 3
    controller.show_name = True
    controller["instructions"] = "Use G to move, R to rotate, and S to scale both crampons. Use the Item/Transform panel for exact values."
    controller["fixed_source_unit_scale"] = BASE_STL_SCALE

    source_parts = (parts[0], parts[1])
    create_foot("LEFT", ANKLE_SEPARATION_M / 2, source_parts, controller, work, refs, menagerie, gray, mount_mat, frame_mat)
    create_foot("RIGHT", -ANKLE_SEPARATION_M / 2, source_parts, controller, work, refs, menagerie, gray, mount_mat, frame_mat)

    # Workbench floor at the enlarged visual tip plane.
    bpy.ops.mesh.primitive_plane_add(size=1.3, location=(0, 0, -0.0384567))
    floor = bpy.context.object
    floor.name = "GUIDE__TIP_PLANE"
    move_to_collection(floor, guides)
    floor.data.materials.append(floor_mat)
    floor.hide_select = True


    # Studio lights.
    for name, location, energy, size in (
        ("KEY_LIGHT", (0.4, -0.45, 0.7), 120.0, 0.35),
        ("FILL_LIGHT", (-0.4, 0.45, 0.45), 55.0, 0.30),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        guides.objects.link(light)
        light.location = location
        look_at(light, (0.0, 0.0, 0.05))

    perspective = add_camera("CAMERA__FIT_PERSPECTIVE", (0.48, -0.62, 0.30), (0.01, 0.0, 0.07), guides)
    add_camera("CAMERA__TOP", (0.02, 0.0, 0.72), (0.02, 0.0, 0.02), guides)
    add_camera("CAMERA__SIDE", (-0.55, 0.0, 0.10), (0.02, 0.0, 0.06), guides)
    bpy.context.scene.camera = perspective

    select_only(controller)
    configure_scene(args.output, args.render)
    print(f"Saved editable Blender fit scene: {args.output.resolve()}")


if __name__ == "__main__":
    main()
