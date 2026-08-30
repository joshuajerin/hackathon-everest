#!/usr/bin/env python3
"""Build a derived official G1 MJCF with the sensorized crampon attached."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

UPSTREAM_REVISION = "da76818e269b82289eba39808e2fb91d679d6994"
OFFICIAL_STAND_ANKLE_WORLD_Z_M = 0.0331362476


def find_body(root: ET.Element, name: str) -> ET.Element:
    result = root.find(f".//body[@name='{name}']")
    if result is None:
        raise ValueError(f"Body {name!r} was not found in official G1 model")
    return result


def vector(values: list[float]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def add_crampon(root: ET.Element, side: str, fit: dict[str, object]) -> None:
    ankle_pitch = find_body(root, f"{side}_ankle_pitch_link")
    for geom in list(ankle_pitch.findall("geom")):
        if geom.get("class") == "collision":
            ankle_pitch.remove(geom)

    ankle = find_body(root, f"{side}_ankle_roll_link")
    for geom in list(ankle.findall("geom")):
        if geom.get("class") == "foot":
            ankle.remove(geom)

    ET.SubElement(
        ankle,
        "geom",
        {
            "name": f"{side}_crampon_frame_visual",
            "type": "mesh",
            "mesh": "everest_crampon_frame",
            "material": "everest_crampon_metal",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "2",
        },
    )
    ET.SubElement(
        ankle,
        "geom",
        {
            "name": f"{side}_crampon_mount_visual",
            "type": "mesh",
            "mesh": "everest_mount_plate",
            "material": "black",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "2",
        },
    )
    ET.SubElement(
        ankle,
        "site",
        {
            "name": f"{side}_crampon_imu",
            "pos": vector(fit["imu_position_m"]),
            "size": "0.006",
            "rgba": "0.1 1 0.1 1",
        },
    )
    ET.SubElement(
        ankle,
        "site",
        {
            "name": f"{side}_crampon_radar",
            "pos": vector(fit["radar_position_m"]),
            "size": "0.007",
            "rgba": "0.8 0.1 1 1",
        },
    )
    probe_fit = fit["probe"]
    axis_length = float(probe_fit["axis_length_m"])
    radius = float(probe_fit["radius_m"])
    quaternion = vector(fit["rotation_quaternion_wxyz"])
    for index, position in enumerate(probe_fit["body_positions_m"]):
        probe = ET.SubElement(
            ankle,
            "body",
            {
                "name": f"{side}_crampon_probe_{index}",
                "pos": vector(position),
                "quat": quaternion,
            },
        )
        ET.SubElement(
            probe,
            "joint",
            {
                "name": f"{side}_crampon_probe_{index}_slide",
                "type": "slide",
                "axis": "0 0 1",
                "range": "0 0.020",
                "limited": "true",
                "springref": "0",
                "stiffness": "8000",
                "damping": "120",
                "armature": "0.0001",
            },
        )
        ET.SubElement(
            probe,
            "geom",
            {
                "name": f"{side}_crampon_spike_{index}",
                "type": "capsule",
                "fromto": f"0 0 0 0 0 {-axis_length:.10g}",
                "size": f"{radius:.10g}",
                "density": "7800",
                "rgba": "0.12 0.14 0.16 1",
                "condim": "3",
                "friction": "0.08 0.004 0.0001",
            },
        )
        ET.SubElement(
            probe,
            "site",
            {
                "name": f"{side}_crampon_probe_{index}_tip",
                "type": "sphere",
                "pos": f"0 0 {-axis_length:.10g}",
                "size": f"{radius * 1.25:.10g}",
                "rgba": "0 0 0 0",
            },
        )


def add_sensors(root: ET.Element, side: str) -> None:
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    for index in range(4):
        ET.SubElement(
            sensor,
            "touch",
            {"name": f"{side}_crampon_spike_{index}_axial_force", "site": f"{side}_crampon_probe_{index}_tip"},
        )
    for index in range(4):
        ET.SubElement(
            sensor,
            "jointpos",
            {"name": f"{side}_crampon_spike_{index}_penetration", "joint": f"{side}_crampon_probe_{index}_slide"},
        )
    ET.SubElement(sensor, "accelerometer", {"name": f"{side}_crampon_accelerometer", "site": f"{side}_crampon_imu"})
    ET.SubElement(sensor, "gyro", {"name": f"{side}_crampon_gyroscope", "site": f"{side}_crampon_imu"})
    ET.SubElement(sensor, "user", {"name": f"{side}_crampon_radar_frontend", "dim": "5", "needstage": "vel"})


def joint_qpos_width(joint_type: int, mujoco: object) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def transfer_stand_key(
    original: object, derived: object, mujoco: object, root_raise_m: float
) -> tuple[np.ndarray, np.ndarray]:
    original_key = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if original_key < 0:
        raise ValueError("Official model does not have the expected stand keyframe")
    qpos = np.zeros(derived.nq)
    for joint_id in range(original.njnt):
        name = mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        derived_id = mujoco.mj_name2id(derived, mujoco.mjtObj.mjOBJ_JOINT, name)
        width = joint_qpos_width(int(original.jnt_type[joint_id]), mujoco)
        old_address = int(original.jnt_qposadr[joint_id])
        new_address = int(derived.jnt_qposadr[derived_id])
        qpos[new_address : new_address + width] = original.key_qpos[original_key, old_address : old_address + width]
    root_id = mujoco.mj_name2id(derived, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    qpos[int(derived.jnt_qposadr[root_id]) + 2] += root_raise_m

    ctrl = np.zeros(derived.nu)
    for actuator_id in range(original.nu):
        name = mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        derived_id = mujoco.mj_name2id(derived, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        ctrl[derived_id] = original.key_ctrl[original_key, actuator_id]
    return qpos, ctrl


def numbers(values: np.ndarray) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--menagerie", type=Path, default=Path("vendor/mujoco_menagerie/unitree_g1"))
    parser.add_argument("--crampon-assets", type=Path, default=Path("assets/crampon"))
    parser.add_argument(
        "--fit-metadata", type=Path, default=Path("assets/crampon/blender_fit_metadata.json")
    )
    parser.add_argument("--out", type=Path, default=Path("build/mujoco_g1/unitree_g1"))
    args = parser.parse_args()

    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError("Install MuJoCo with: uv sync --extra mujoco") from error

    source = args.menagerie.resolve()
    crampon_assets = args.crampon_assets.resolve()
    fit_path = args.fit_metadata.resolve()
    fit = json.loads(fit_path.read_text())
    lowest_z = min(
        float(fit["combined_bounds_m"]["minimum_m"][2]),
        float(fit["probe"]["lowest_tip_z_m"]),
    )
    root_raise_m = max(0.002, 0.0005 - (OFFICIAL_STAND_ANKLE_WORLD_Z_M + lowest_z))
    out = args.out.resolve()
    if not (source / "g1.xml").exists():
        raise FileNotFoundError(f"Fetch the pinned Menagerie model first; missing {source / 'g1.xml'}")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(source, out)
    fitted_files = ("crampon_frame_fitted.stl", "mount_plate_fitted.stl")
    for filename in fitted_files:
        shutil.copy2(crampon_assets / filename, out / "assets" / filename)

    original_model = mujoco.MjModel.from_xml_path(str(source / "g1.xml"))
    tree = ET.parse(out / "g1.xml")
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise ValueError("Official model is missing its asset element")
    ET.SubElement(asset, "material", {"name": "everest_crampon_metal", "rgba": "0.16 0.24 0.32 1", "metallic": "0.8", "roughness": "0.28"})
    ET.SubElement(asset, "mesh", {"name": "everest_crampon_frame", "file": "crampon_frame_fitted.stl"})
    ET.SubElement(asset, "mesh", {"name": "everest_mount_plate", "file": "mount_plate_fitted.stl"})
    for side in ("left", "right"):
        add_crampon(root, side, fit)
        add_sensors(root, side)

    old_keyframe = root.find("keyframe")
    if old_keyframe is not None:
        root.remove(old_keyframe)
    ET.indent(tree, space="  ")
    derived_path = out / "g1_crampon.xml"
    tree.write(derived_path, encoding="unicode")
    derived_model = mujoco.MjModel.from_xml_path(str(derived_path))
    qpos, ctrl = transfer_stand_key(original_model, derived_model, mujoco, root_raise_m)
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", {"name": "stand", "qpos": numbers(qpos), "ctrl": numbers(ctrl)})
    ET.indent(tree, space="  ")
    tree.write(derived_path, encoding="unicode")

    scene_tree = ET.parse(out / "scene.xml")
    include = scene_tree.getroot().find("include")
    if include is None:
        raise ValueError("Official scene.xml has no include element")
    include.set("file", "g1_crampon.xml")
    ET.indent(scene_tree, space="  ")
    scene_path = out / "scene_crampon.xml"
    scene_tree.write(scene_path, encoding="unicode")

    checked = mujoco.MjModel.from_xml_path(str(scene_path))
    provenance = {
        "upstream_repository": "https://github.com/google-deepmind/mujoco_menagerie",
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_model": "unitree_g1/g1.xml",
        "derived_model": derived_path.name,
        "derived_scene": scene_path.name,
        "nq": checked.nq,
        "nv": checked.nv,
        "nu": checked.nu,
        "nsensordata": checked.nsensordata,
        "blender_fit_metadata": str(fit_path),
        "blender_fit_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
        "blender_controller": fit["controller"],
        "root_raise_m": root_raise_m,
        "note": "Generated from the saved Blender fit; official g1.xml remains unmodified.",
    }
    (out / "everest_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
