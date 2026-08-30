#!/usr/bin/env python3
"""Synchronize the standalone MuJoCo fixture with exported Blender fit metadata."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def vector(values: list[float]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def require(root: ET.Element, path: str) -> ET.Element:
    value = root.find(path)
    if value is None:
        raise ValueError(f"Missing expected MJCF element: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("assets/crampon/blender_fit_metadata.json"))
    parser.add_argument("--model", type=Path, default=Path("mujoco/crampon_probe.xml"))
    args = parser.parse_args()

    fit = json.loads(args.metadata.read_text())
    tree = ET.parse(args.model)
    root = tree.getroot()
    require(root, ".//mesh[@name='crampon_frame_mesh']").set("file", "crampon_frame_fitted.stl")
    require(root, ".//mesh[@name='mount_plate_mesh']").set("file", "mount_plate_fitted.stl")
    require(root, ".//site[@name='crampon_imu']").set("pos", vector(fit["imu_position_m"]))
    require(root, ".//site[@name='radar_origin']").set("pos", vector(fit["radar_position_m"]))

    probe_fit = fit["probe"]
    axis_length = float(probe_fit["axis_length_m"])
    radius = float(probe_fit["radius_m"])
    quaternion = vector(fit["rotation_quaternion_wxyz"])
    for index, position in enumerate(probe_fit["body_positions_m"]):
        body = require(root, f".//body[@name='probe_{index}']")
        body.set("pos", vector(position))
        body.set("quat", quaternion)
        geom = require(root, f".//geom[@name='probe_{index}_geom']")
        geom.set("fromto", f"0 0 0 0 0 {-axis_length:.10g}")
        geom.set("size", f"{radius:.10g}")
        site = require(root, f".//site[@name='probe_{index}_tip']")
        site.set("pos", f"0 0 {-axis_length:.10g}")
        site.set("size", f"{radius * 1.25:.10g}")

    key = require(root, ".//key[@name='touchdown']")
    key.set("qpos", "0 0 0 0 0 0")
    carriage_z = -float(probe_fit["lowest_tip_z_m"])
    require(root, ".//body[@name='probe_carriage']").set("pos", f"0 0 {carriage_z:.10g}")
    ET.indent(tree, space="  ")
    tree.write(args.model, encoding="unicode")
    print(
        json.dumps(
            {
                "model": str(args.model),
                "metadata": str(args.metadata),
                "carriage_z_m": carriage_z,
                "probe_radius_m": radius,
                "probe_axis_length_m": axis_length,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
