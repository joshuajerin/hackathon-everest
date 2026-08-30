#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation_matrix_from_quaternion(value) -> np.ndarray:
    w = float(value.GetReal())
    x, y, z = (float(v) for v in value.GetImaginary())
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quaternion_from_rotation_matrix(matrix: np.ndarray):
    from pxr import Gf

    trace = float(np.trace(matrix))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w, x, y, z = (
            0.25 * s,
            (matrix[2, 1] - matrix[1, 2]) / s,
            (matrix[0, 2] - matrix[2, 0]) / s,
            (matrix[1, 0] - matrix[0, 1]) / s,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w, x, y, z = (
                (matrix[2, 1] - matrix[1, 2]) / s,
                0.25 * s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
            )
        elif index == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w, x, y, z = (
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                0.25 * s,
                (matrix[1, 2] + matrix[2, 1]) / s,
            )
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w, x, y, z = (
                (matrix[1, 0] - matrix[0, 1]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
                (matrix[1, 2] + matrix[2, 1]) / s,
                0.25 * s,
            )
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def add_mass_prior(stage, link_path: str, config: dict) -> dict:
    from pxr import Gf, UsdPhysics

    prim = stage.GetPrimAtPath(link_path)
    api = UsdPhysics.MassAPI(prim)
    original_mass = float(api.GetMassAttr().Get())
    original_com = np.asarray(api.GetCenterOfMassAttr().Get(), dtype=float)
    original_diag = np.asarray(api.GetDiagonalInertiaAttr().Get(), dtype=float)
    original_axes = api.GetPrincipalAxesAttr().Get()
    original_rotation = rotation_matrix_from_quaternion(original_axes)
    original_inertia = original_rotation @ np.diag(original_diag) @ original_rotation.T

    added_mass = float(config["nominal_added_mass_per_foot_kg"])
    added_com = np.asarray(config["nominal_com_ankle_local_m"], dtype=float)
    dx, dy, dz = (float(v) for v in config["box_envelope_m"])
    added_inertia = np.diag(
        [
            added_mass * (dy * dy + dz * dz) / 12.0,
            added_mass * (dx * dx + dz * dz) / 12.0,
            added_mass * (dx * dx + dy * dy) / 12.0,
        ]
    )
    combined_mass = original_mass + added_mass
    combined_com = (original_mass * original_com + added_mass * added_com) / combined_mass

    def shift(inertia: np.ndarray, mass: float, center: np.ndarray) -> np.ndarray:
        delta = center - combined_com
        return inertia + mass * (float(delta @ delta) * np.eye(3) - np.outer(delta, delta))

    combined_inertia = shift(original_inertia, original_mass, original_com) + shift(
        added_inertia, added_mass, added_com
    )
    eigenvalues, eigenvectors = np.linalg.eigh(combined_inertia)
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1.0
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
        raise RuntimeError(f"Invalid combined inertia for {link_path}")
    api.GetMassAttr().Set(combined_mass)
    api.GetCenterOfMassAttr().Set(Gf.Vec3f(*(float(v) for v in combined_com)))
    api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*(float(v) for v in eigenvalues)))
    api.GetPrincipalAxesAttr().Set(quaternion_from_rotation_matrix(eigenvectors))
    return {
        "link": link_path,
        "original_mass_kg": original_mass,
        "added_mass_kg": added_mass,
        "combined_mass_kg": combined_mass,
        "original_com_m": original_com.tolist(),
        "added_com_m": added_com.tolist(),
        "combined_com_m": combined_com.tolist(),
        "combined_diagonal_inertia_kg_m2": eigenvalues.tolist(),
        "status": config["status"],
    }


def build(
    repo: Path,
    official: Path,
    clean: Path,
    config_path: Path,
    output: Path,
    manifest_path: Path,
    *,
    contact_model: str = "rigid_baseline",
) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    config = yaml.safe_load(config_path.read_text())
    g1 = config["official_g1"]
    if sha256_file(official) != g1["sha256"]:
        raise RuntimeError("Official G1 SHA-256 mismatch")
    clean_manifest = json.loads(clean.with_suffix(".manifest.json").read_text())
    if sha256_file(clean) != clean_manifest["output"]["sha256"]:
        raise RuntimeError("Clean crampon payload SHA-256 mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.usdc")
    layer = Sdf.Layer.CreateNew(str(temporary))
    source_layer = Sdf.Layer.FindOrOpen(str(official))
    if source_layer is None:
        raise RuntimeError("Could not open official G1 layer")
    copied_root_prims = []
    for root_prim in source_layer.rootPrims:
        source_path = Sdf.Path(f"/{root_prim.name}")
        if not Sdf.CopySpec(source_layer, source_path, layer, source_path):
            raise RuntimeError(f"Could not copy official G1 root prim: {source_path}")
        copied_root_prims.append(str(source_path))
    if g1["default_prim"] not in copied_root_prims:
        raise RuntimeError("Official G1 default prim was not copied")
    layer.defaultPrim = g1["default_prim"].lstrip("/")
    layer.Save()
    stage = Usd.Stage.Open(str(temporary))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.GetPrimAtPath(g1["default_prim"]))

    authority = yaml.safe_load((repo / "configs/isaaclab/g1_crampon_asset.yaml").read_text())
    fit = json.loads((repo / authority["editable_fit"]["metadata"]).read_text())
    control = fit["control"]
    pivot = fit["source_pivot_world_m"]
    clean_reference = os.path.relpath(clean, output.parent)
    probes = config["analytical_probes"]
    if contact_model not in {"rigid_baseline", "stateful_material"}:
        raise ValueError("contact_model must be rigid_baseline or stateful_material")
    native_probe_collision = bool(probes[f"{contact_model}_native_probe_collision"])
    mass_reports = []
    probe_paths = []
    for side in ("left", "right"):
        ankle_path = g1["ankle_links"][side]
        if not stage.GetPrimAtPath(ankle_path).IsValid():
            raise RuntimeError(f"Missing ankle link: {ankle_path}")
        visual_path = f"{ankle_path}/EverestCramponVisual"
        visual = UsdGeom.Xform.Define(stage, visual_path)
        visual.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in control["location_m"])))
        visual.AddRotateXYZOp().Set(
            Gf.Vec3f(*(float(np.rad2deg(v)) for v in control["rotation_euler_rad"]))
        )
        visual.AddScaleOp().Set(Gf.Vec3f(*(float(v) for v in control["scale"])))
        centered = UsdGeom.Xform.Define(stage, f"{visual_path}/SourceCentered")
        centered.AddTranslateOp().Set(Gf.Vec3d(*(-float(v) for v in pivot)))
        centered.GetPrim().GetReferences().AddReference(clean_reference)

        group_path = f"{ankle_path}/EverestAnalyticalProbes"
        UsdGeom.Xform.Define(stage, group_path)
        for index, xy in enumerate(probes["xy_m"]):
            probe_path = f"{group_path}/probe_{index}"
            sphere = UsdGeom.Sphere.Define(stage, probe_path)
            sphere.GetRadiusAttr().Set(float(probes["radius_m"]))
            sphere.AddTranslateOp().Set(
                Gf.Vec3d(float(xy[0]), float(xy[1]), float(probes["ankle_local_z_m"]))
            )
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim()).GetCollisionEnabledAttr().Set(
                native_probe_collision
            )
            sphere.GetPrim().CreateAttribute(
                "everest:axisAnkleLocal", Sdf.ValueTypeNames.Float3
            ).Set(Gf.Vec3f(*(float(v) for v in probes["axis_ankle_local"])))
            sphere.GetPrim().CreateAttribute(
                "everest:virtualTravelM", Sdf.ValueTypeNames.Float
            ).Set(float(probes["virtual_travel_m"]))
            sphere.GetPrim().CreateAttribute(
                "everest:customMaterialDisablesNativeContact", Sdf.ValueTypeNames.Bool
            ).Set(True)
            probe_paths.append(probe_path)
        mass_reports.append(add_mass_prior(stage, ankle_path, config["mass_properties"]))

    for collision_path in g1["stock_support_collisions_to_disable"]:
        prim = stage.GetPrimAtPath(collision_path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"Pinned stock collision path did not resolve: {collision_path}")
        UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)

    stage.GetRootLayer().Save()
    # Re-open before publication and prove composition did not add articulations.
    check = Usd.Stage.Open(str(temporary))
    mesh_counts = {}
    for side in ("left", "right"):
        ankle = g1["ankle_links"][side]
        mesh_counts[side] = sum(
            1
            for prim in Usd.PrimRange(check.GetPrimAtPath(f"{ankle}/EverestCramponVisual"))
            if prim.IsA(UsdGeom.Mesh)
        )
    if mesh_counts != {"left": 26, "right": 26}:
        raise RuntimeError(f"Expected 26 visual meshes per foot, got {mesh_counts}")
    if any(check.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI) for path in probe_paths):
        raise RuntimeError("Analytical probes must not create rigid bodies")
    output_sha = sha256_file(temporary)
    os.replace(temporary, output)
    report = {
        "schema_version": "1.0.0",
        "claim_boundary": config["claim_boundary"],
        "official_g1": {
            "path": str(official),
            "source_uri": g1["source_uri"],
            "sha256": g1["sha256"],
            "default_prim": g1["default_prim"],
        },
        "authoritative_crampon": {
            "clean_payload": str(clean),
            "sha256": clean_manifest["output"]["sha256"],
            "source_sha256": authority["source"]["sha256"],
            "component_count_per_foot": 26,
        },
        "fit": {
            "control": control,
            "source_pivot_world_m": pivot,
            "left_fine_tune": "identity",
            "right_fine_tune": "identity",
        },
        "analytical_probes": {
            "paths": probe_paths,
            "count": len(probe_paths),
            "config": probes,
            "contact_model": contact_model,
            "native_probe_collision_enabled": native_probe_collision,
        },
        "stock_collisions_disabled": g1["stock_support_collisions_to_disable"],
        "mass_properties": mass_reports,
        "validation": {
            "visual_mesh_counts": mesh_counts,
            "added_rigid_bodies": 0,
            "added_joints": 0,
            "legacy_stl_used": False,
        },
        "output": {"path": str(output), "sha256": output_sha, "size_bytes": output.stat().st_size},
    }
    temp_manifest = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
    temp_manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temp_manifest, manifest_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--official-g1", type=Path, required=True)
    parser.add_argument(
        "--clean-payload",
        type=Path,
        default=Path("build/isaaclab/g1_crampon_components_clean.usdc"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/isaaclab/g1_crampon_simulation.yaml")
    )
    parser.add_argument(
        "--contact-model",
        choices=("rigid_baseline", "stateful_material"),
        default="rigid_baseline",
    )
    parser.add_argument("--output", type=Path, default=Path("build/isaaclab/g1_crampon.usdc"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("build/isaaclab/g1_crampon.manifest.json")
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    resolve = lambda value: value.resolve() if value.is_absolute() else (root / value).resolve()
    report = build(
        root,
        resolve(args.official_g1),
        resolve(args.clean_payload),
        resolve(args.config),
        resolve(args.output),
        resolve(args.manifest),
        contact_model=args.contact_model,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
