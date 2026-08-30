#!/usr/bin/env python3
"""Build the sanitized complete-component crampon USD payload.

Run this script with the Python environment shipped with the pinned Isaac Sim/
Isaac Lab installation. The module deliberately imports ``pxr`` only inside the
USD build path, so authority/config tests and ``--help`` work without Isaac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import yaml

SOURCE_SHA256 = "53703057dff7ea5b2e7e468164289d6c0aba629400952c9a8a9a5f7048f2a660"
SOURCE_DEFAULT_PRIM = "/root"
SOURCE_UP_AXIS = "Z"
SOURCE_METERS_PER_UNIT = 1.0
FIT_METADATA_SCHEMA_VERSION = "1.0.0"
MANIFEST_SCHEMA_VERSION = "1.0.0"
FIT_SOURCE_BOUNDS_M = {
    "min": [1.2769708633422852, -0.11180946975946426, -0.0439583994448185],
    "max": [1.3697125911712646, 0.1503179967403412, 0.028608979657292366],
}
FIT_SOURCE_PIVOT_M = [
    1.323341727256775,
    0.01925426349043846,
    -0.007674709893763065,
]
FIT_CONTROL = {
    "name": "USD_ASSET_POSITION_CONTROL",
    "location_m": [
        0.04122297838330269,
        0.0008915653452277184,
        -0.04183308780193329,
    ],
    "rotation_euler_rad": [0.0, 0.0, -1.570796251296997],
    "scale": [1.0800000429153442, 1.0800000429153442, 1.0800000429153442],
}
FIT_ALIGNMENT_RMS_M = 4.828731508112456e-06
FIT_ALIGNMENT_MAX_M = 4.95659787702607e-06

# This is an allowlist, not a discovery result. A source with a different prim
# set must fail preflight even if it still happens to contain 26 meshes.
COMPONENT_PRIM_PAIRS: tuple[tuple[str, str], ...] = (
    ("/root/Cylinder_015", "/root/Cylinder_015/Cylinder_015"),
    ("/root/Plane_022", "/root/Plane_022/Plane_022"),
    ("/root/Cylinder_013", "/root/Cylinder_013/Cylinder_013"),
    ("/root/Plane_017", "/root/Plane_017/Plane_017"),
    ("/root/Cylinder_018", "/root/Cylinder_018/Cylinder_018"),
    ("/root/Plane_020", "/root/Plane_020/Plane_020"),
    ("/root/Plane_027", "/root/Plane_027/Plane_027"),
    ("/root/Cylinder_014", "/root/Cylinder_014/Cylinder_014"),
    ("/root/Plane_019", "/root/Plane_019/Plane_019"),
    ("/root/Plane_021", "/root/Plane_021/Plane_021"),
    ("/root/Plane_024", "/root/Plane_024/Plane_024"),
    ("/root/Cylinder_012", "/root/Cylinder_012/Cylinder_012"),
    ("/root/Cylinder_011", "/root/Cylinder_011/Cylinder_011"),
    ("/root/Plane_018", "/root/Plane_018/Plane_018"),
    ("/root/Plane_023", "/root/Plane_023/Plane_023"),
    ("/root/Plane_016", "/root/Plane_016/Plane_016"),
    ("/root/Plane_026", "/root/Plane_026/Plane_026"),
    ("/root/mesh_node_003", "/root/mesh_node_003/mesh_003"),
    ("/root/Cube_003", "/root/Cube_003/Cube_003"),
    ("/root/Cylinder_016", "/root/Cylinder_016/Cylinder_016"),
    ("/root/Plane_025", "/root/Plane_025/Plane_025"),
    ("/root/Cube_002", "/root/Cube_002/Cube_002"),
    ("/root/Cylinder_010", "/root/Cylinder_010/Cylinder_010"),
    ("/root/Cylinder_017", "/root/Cylinder_017/Cylinder_017"),
    ("/root/Plane_015", "/root/Plane_015/Plane_015"),
    ("/root/_MF_Object_003", "/root/_MF_Object_003/_MF_Mesh_005"),
)

EXPECTED_XFORM_OP_NAMES = (
    "xformOp:translate",
    "xformOp:rotateXYZ",
    "xformOp:scale",
)


class AuthorityError(RuntimeError):
    """Raised when an authority or USD preflight check fails."""


@dataclass(frozen=True)
class Authority:
    repo_root: Path
    config_path: Path
    fit_metadata_path: Path
    source_path: Path
    blend_path: Path
    generator_path: Path
    config: dict[str, Any]
    fit_metadata: dict[str, Any]
    source_sha256: str
    config_sha256: str
    fit_metadata_sha256: str
    generator_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def _resolved(root: Path, value: object, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _is_stl_dependency(value: object) -> bool:
    if isinstance(value, Path):
        text = value.as_posix()
    elif isinstance(value, str):
        text = value
    else:
        return False
    # Handle common USD reference strings and URI query/fragment suffixes.
    text = text.strip().split("<", 1)[0].strip("@")
    # USD package-relative paths can look like archive.usdz[meshes/part.stl].
    candidates = re.split(r"[\[\]]", text.replace("\\", "/"))
    normalized = (
        candidate.split(":SDF_FORMAT_ARGS:", 1)[0].strip("@")
        for candidate in candidates
        if candidate
    )
    return any(
        PurePosixPath(urlsplit(candidate).path).suffix.casefold() == ".stl"
        for candidate in normalized
    )


def assert_no_stl_dependencies(dependencies: Iterable[object], context: str) -> None:
    rejected = sorted({str(item) for item in dependencies if _is_stl_dependency(item)})
    if rejected:
        raise AuthorityError(f"{context} contains disallowed STL dependencies: {rejected}")


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} does not exist: {path}")
    try:
        if path.suffix.casefold() in {".yaml", ".yml"}:
            value = yaml.safe_load(path.read_text())
        else:
            value = json.loads(path.read_text())
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise AuthorityError(f"Could not parse {label} {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must contain a mapping: {path}")
    return value


def load_authority(
    repo_root: Path,
    config_path: Path | None = None,
    fit_metadata_path: Path | None = None,
) -> Authority:
    """Load and cross-check the existing YAML and fit metadata without pxr."""
    root = repo_root.resolve()
    config_file = (
        config_path.resolve()
        if config_path is not None
        else root / "configs/isaaclab/g1_crampon_asset.yaml"
    )
    config = _load_mapping(config_file, "asset authority YAML")

    _require(config.get("schema_version") == "1.0.0", "Unexpected authority schema_version")
    _require(
        config.get("status") == "authoritative_for_isaac_lab",
        "Asset YAML is not marked authoritative_for_isaac_lab",
    )
    source_cfg = config.get("source")
    editable_cfg = config.get("editable_fit")
    isaac_cfg = config.get("isaac_lab")
    legacy_cfg = config.get("legacy")
    _require(isinstance(source_cfg, dict), "Asset YAML is missing source mapping")
    _require(isinstance(editable_cfg, dict), "Asset YAML is missing editable_fit mapping")
    _require(isinstance(isaac_cfg, dict), "Asset YAML is missing isaac_lab mapping")
    _require(isinstance(legacy_cfg, dict), "Asset YAML is missing legacy mapping")

    source_path = _resolved(root, source_cfg.get("usdc"), "source.usdc")
    blend_path = _resolved(root, editable_cfg.get("blend"), "editable_fit.blend")
    _require(
        editable_cfg.get("generator") == "blender/setup_usd_component_fit.py",
        "Unexpected editable-fit generator",
    )
    generator_path = _resolved(root, editable_cfg.get("generator"), "editable_fit.generator")
    metadata_file = (
        fit_metadata_path.resolve()
        if fit_metadata_path is not None
        else _resolved(root, editable_cfg.get("metadata"), "editable_fit.metadata")
    )
    fit = _load_mapping(metadata_file, "USD component fit metadata")

    selected_inputs = (source_path, blend_path, generator_path, metadata_file, config_file)
    assert_no_stl_dependencies(selected_inputs, "Selected Isaac authority inputs")
    _require(source_path.suffix.casefold() == ".usdc", "Authoritative source must be USDC")
    _require(source_path.is_file(), f"Authoritative source does not exist: {source_path}")
    _require(blend_path.is_file(), f"Authoritative fit Blend does not exist: {blend_path}")
    _require(generator_path.is_file(), f"Fit generator does not exist: {generator_path}")
    _require(legacy_cfg.get("use_old_two_mesh_fit_for_isaac_lab") is False, "Legacy fit enabled")
    _require(isaac_cfg.get("use_this_asset") is True, "Isaac asset selection is disabled")
    _require(
        isaac_cfg.get("compose_with_official_g1_usd") is True,
        "Official G1 USD composition is disabled",
    )
    _require(
        isaac_cfg.get("estimator_packet_contract") == "19_values_per_foot",
        "Estimator packet contract drifted from 19_values_per_foot",
    )
    _require(
        isaac_cfg.get("preserve_component_relative_transforms") is True,
        "Component relative transform preservation is disabled",
    )
    _require(isaac_cfg.get("visual_geometry") == "all_usd_meshes", "Wrong visual geometry")
    _require(
        isaac_cfg.get("collision_geometry") == "analytical_crampon_proxies_only",
        "High-detail USD meshes must not be selected for collision",
    )

    actual_source_sha = sha256_file(source_path)
    _require(source_cfg.get("sha256") == SOURCE_SHA256, "YAML source SHA is not the golden SHA")
    _require(actual_source_sha == SOURCE_SHA256, "USDC bytes do not match the golden SHA")
    _require(source_cfg.get("default_prim") == SOURCE_DEFAULT_PRIM, "Wrong YAML default prim")
    _require(source_cfg.get("stage_up_axis") == SOURCE_UP_AXIS, "Wrong YAML up axis")
    _require(
        float(source_cfg.get("meters_per_unit", -1.0)) == SOURCE_METERS_PER_UNIT,
        "Wrong YAML metres-per-unit",
    )
    _require(source_cfg.get("mesh_objects") == len(COMPONENT_PRIM_PAIRS), "Wrong mesh count")

    _require(fit.get("schema_version") == FIT_METADATA_SCHEMA_VERSION, "Wrong fit schema")
    _require(fit.get("source_usdc") == source_path.name, "Fit metadata source name mismatch")
    _require(fit.get("source_sha256") == actual_source_sha, "Fit metadata source SHA mismatch")
    _require(fit.get("source_mesh_objects") == len(COMPONENT_PRIM_PAIRS), "Fit mesh mismatch")
    fit_stage = fit.get("source_stage")
    _require(isinstance(fit_stage, dict), "Fit metadata is missing source_stage")
    _require(fit_stage.get("default_prim") == SOURCE_DEFAULT_PRIM, "Fit default prim mismatch")
    _require(fit_stage.get("up_axis") == SOURCE_UP_AXIS, "Fit up axis mismatch")
    _require(
        float(fit_stage.get("meters_per_unit", -1.0)) == SOURCE_METERS_PER_UNIT,
        "Fit metres-per-unit mismatch",
    )
    _require(fit.get("source_contains_humanoid") is False, "Source unexpectedly has humanoid")
    _require(fit.get("source_world_bounds_m") == FIT_SOURCE_BOUNDS_M, "Fit source bounds drifted")
    _require(fit.get("source_pivot_world_m") == FIT_SOURCE_PIVOT_M, "Fit source pivot drifted")
    alignment = fit.get("alignment")
    _require(isinstance(alignment, dict), "Fit metadata is missing alignment")
    _require(
        alignment.get("method") == "exact main-mesh geometry match",
        "Fit alignment method drifted",
    )
    _require(
        alignment.get("rms_error_m") == FIT_ALIGNMENT_RMS_M,
        "Fit alignment RMS drifted",
    )
    _require(
        alignment.get("maximum_error_m") == FIT_ALIGNMENT_MAX_M,
        "Fit alignment maximum error drifted",
    )
    control = fit.get("control")
    _require(control == FIT_CONTROL, "Saved complete-component control transform drifted")
    _require(
        control.get("name") == editable_cfg.get("shared_control"),
        "Shared fit control name mismatch",
    )
    _require(fit.get("output_blend") == blend_path.name, "Fit output Blend name mismatch")
    expected_fine = editable_cfg.get("per_foot_controls")
    _require(isinstance(expected_fine, dict), "YAML per-foot controls are missing")
    _require(
        fit.get("per_foot_controls") == [expected_fine.get("left"), expected_fine.get("right")],
        "Per-foot fit control names mismatch",
    )

    return Authority(
        repo_root=root,
        config_path=config_file,
        fit_metadata_path=metadata_file,
        source_path=source_path,
        blend_path=blend_path,
        generator_path=generator_path,
        config=config,
        fit_metadata=fit,
        source_sha256=actual_source_sha,
        config_sha256=sha256_file(config_file),
        fit_metadata_sha256=sha256_file(metadata_file),
        generator_sha256=sha256_file(generator_path),
    )


def _import_pxr() -> tuple[Any, Any, Any, Any]:
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdUtils
    except ImportError as exc:
        raise AuthorityError(
            "pxr is required to build the clean payload; run this script with the pinned "
            "Isaac Sim/Isaac Lab Python environment"
        ) from exc
    return Sdf, Usd, UsdGeom, UsdUtils


def _asset_path_text(value: object) -> str | None:
    if value is None:
        return None
    path = getattr(value, "path", None)
    if isinstance(path, str):
        return path
    if isinstance(value, str):
        return value
    return None


def _stage_asset_attributes(stage: Any, sdf: Any) -> set[str]:
    result: set[str] = set()
    for prim in stage.TraverseAll():
        for attribute in prim.GetAttributes():
            type_name = attribute.GetTypeName()
            if type_name == sdf.ValueTypeNames.Asset:
                text = _asset_path_text(attribute.Get())
                if text:
                    result.add(text)
            elif type_name == sdf.ValueTypeNames.AssetArray:
                for value in attribute.Get() or ():
                    text = _asset_path_text(value)
                    if text:
                        result.add(text)
    return result


def _computed_dependencies(path: Path, usd_utils: Any) -> tuple[set[str], set[str]]:
    """Return (dependencies, unresolved) across supported USD Python bindings."""
    try:
        result = usd_utils.ComputeAllDependencies(str(path))
    except Exception as exc:
        raise AuthorityError(f"Could not compute USD dependencies for {path}: {exc}") from exc
    _require(
        isinstance(result, tuple) and len(result) == 3,
        "UsdUtils.ComputeAllDependencies returned an unsupported result",
    )
    layers, assets, unresolved = result
    dependencies = {str(item) for item in assets}
    for layer in layers:
        identifier = getattr(layer, "identifier", None)
        if identifier:
            dependencies.add(str(identifier))
    return dependencies, {str(item) for item in unresolved}


def _layer_external_dependencies(layer: Any) -> set[str]:
    method = getattr(layer, "GetExternalAssetDependencies", None)
    if method is None:
        return set()
    return {str(item) for item in method()}


def _validate_stage_authority(stage: Any, authority: Authority, usd_geom: Any) -> None:
    default_prim = stage.GetDefaultPrim()
    _require(default_prim.IsValid(), "Source stage has no valid default prim")
    _require(str(default_prim.GetPath()) == SOURCE_DEFAULT_PRIM, "Source default prim mismatch")
    _require(
        str(usd_geom.GetStageUpAxis(stage)) == SOURCE_UP_AXIS,
        "Source stage up axis mismatch",
    )
    _require(
        float(usd_geom.GetStageMetersPerUnit(stage)) == SOURCE_METERS_PER_UNIT,
        "Source stage metres-per-unit mismatch",
    )

    actual_meshes = {str(prim.GetPath()) for prim in stage.TraverseAll() if prim.IsA(usd_geom.Mesh)}
    expected_meshes = {mesh for _, mesh in COMPONENT_PRIM_PAIRS}
    _require(
        actual_meshes == expected_meshes,
        f"Source Mesh prim set mismatch; missing={sorted(expected_meshes - actual_meshes)}, "
        f"unexpected={sorted(actual_meshes - expected_meshes)}",
    )

    for xform_path, mesh_path in COMPONENT_PRIM_PAIRS:
        xform_prim = stage.GetPrimAtPath(xform_path)
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        _require(xform_prim.IsA(usd_geom.Xform), f"Expected Xform prim: {xform_path}")
        _require(mesh_prim.IsA(usd_geom.Mesh), f"Expected Mesh prim: {mesh_path}")
        ordered = usd_geom.Xformable(xform_prim).GetOrderedXformOps()
        names = tuple(str(op.GetOpName()) for op in ordered)
        _require(names == EXPECTED_XFORM_OP_NAMES, f"Unexpected xform op order at {xform_path}")

    config_source = authority.config["source"]
    _require(len(actual_meshes) == config_source["mesh_objects"], "Configured mesh count drift")


def preflight_source(authority: Authority) -> dict[str, Any]:
    """Validate the raw source under pxr while tolerating its omitted light dependency."""
    sdf, usd, usd_geom, usd_utils = _import_pxr()
    source_layer = sdf.Layer.FindOrOpen(str(authority.source_path))
    _require(source_layer is not None, f"Could not open source Sdf layer: {authority.source_path}")
    try:
        stage = usd.Stage.Open(source_layer, load=usd.Stage.LoadNone)
    except Exception as exc:
        raise AuthorityError(f"Could not open source USD stage: {exc}") from exc
    _require(stage is not None, "Could not open source USD stage")
    _validate_stage_authority(stage, authority, usd_geom)

    computed, unresolved = _computed_dependencies(authority.source_path, usd_utils)
    dependencies = computed | _layer_external_dependencies(source_layer)
    dependencies |= _stage_asset_attributes(stage, sdf)
    assert_no_stl_dependencies(dependencies | unresolved, "Raw source USD")
    return {
        "source_layer": source_layer,
        "dependencies": sorted(dependencies),
        "unresolved_dependencies": sorted(unresolved),
    }


def _temp_usdc_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.tmp.usdc")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _snapshot_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    snapshot = path.with_name(f".{path.name}.{uuid.uuid4().hex}.rollback")
    try:
        os.link(path, snapshot)
    except OSError:
        shutil.copyfile(path, snapshot)
    return snapshot


def _restore_snapshot(path: Path, snapshot: Path | None) -> None:
    if snapshot is None:
        path.unlink(missing_ok=True)
    elif path.exists() and os.path.samefile(path, snapshot):
        snapshot.unlink()
    else:
        os.replace(snapshot, path)


def _publish_payload_and_manifest(
    temporary_payload: Path,
    output: Path,
    manifest_file: Path,
    manifest: dict[str, Any],
) -> None:
    """Publish two individually atomic files and roll both back on failure.

    POSIX has no portable two-path rename transaction. The JSON manifest is
    replaced last as the completion marker, and readers must verify its payload
    hash. Snapshots preserve the previous complete pair if either replace fails.
    """
    output_snapshot: Path | None = None
    manifest_snapshot: Path | None = None
    try:
        output_snapshot = _snapshot_file(output)
        manifest_snapshot = _snapshot_file(manifest_file)
    except OSError:
        if output_snapshot is not None:
            output_snapshot.unlink(missing_ok=True)
        if manifest_snapshot is not None:
            manifest_snapshot.unlink(missing_ok=True)
        raise

    try:
        os.replace(temporary_payload, output)
        _atomic_write_json(manifest_file, manifest)
    except Exception as exc:
        try:
            _restore_snapshot(output, output_snapshot)
            output_snapshot = None
            _restore_snapshot(manifest_file, manifest_snapshot)
            manifest_snapshot = None
        except OSError as rollback_exc:
            raise AuthorityError(
                f"Publication failed and rollback could not restore the previous pair: {rollback_exc}"
            ) from exc
        raise
    finally:
        if output_snapshot is not None:
            output_snapshot.unlink(missing_ok=True)
        if manifest_snapshot is not None:
            manifest_snapshot.unlink(missing_ok=True)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _validate_clean_payload(
    path: Path,
    authority: Authority,
    sdf: Any,
    usd: Any,
    usd_geom: Any,
    usd_utils: Any,
) -> dict[str, Any]:
    layer = sdf.Layer.FindOrOpen(str(path))
    _require(layer is not None, f"Could not reopen clean payload layer: {path}")
    stage = usd.Stage.Open(layer, load=usd.Stage.LoadNone)
    _require(stage is not None, f"Could not reopen clean payload stage: {path}")
    _validate_stage_authority(stage, authority, usd_geom)
    _require(not stage.GetPrimAtPath("/root/env_light").IsValid(), "env_light leaked into payload")

    dependencies, unresolved = _computed_dependencies(path, usd_utils)
    dependencies |= _layer_external_dependencies(layer)
    dependencies |= _stage_asset_attributes(stage, sdf)
    # Do not treat the payload's own layer identifier as an external dependency.
    own_identifiers = {str(path), str(path.resolve()), str(layer.identifier)}
    dependencies -= own_identifiers
    assert_no_stl_dependencies(dependencies | unresolved, "Clean payload USD")
    _require(not unresolved, f"Clean payload has unresolved dependencies: {sorted(unresolved)}")
    return {"dependencies": sorted(dependencies), "unresolved_dependencies": []}


def build_clean_payload(
    authority: Authority,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Copy the allowlisted source specs and atomically publish payload + manifest."""
    sdf, usd, usd_geom, usd_utils = _import_pxr()
    preflight = preflight_source(authority)
    source_layer = preflight["source_layer"]

    output = output_path.resolve()
    manifest_file = manifest_path.resolve()
    assert_no_stl_dependencies((output, manifest_file), "Build outputs")
    _require(output.suffix.casefold() == ".usdc", "Clean payload output must end in .usdc")
    _require(output != authority.source_path, "Refusing to overwrite the authoritative source")
    _require(manifest_file != output, "Manifest path must differ from payload path")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    temporary = _temp_usdc_path(output)
    try:
        clean_layer = sdf.Layer.CreateNew(str(temporary))
        _require(clean_layer is not None, f"Could not create temporary layer: {temporary}")
        root_spec = sdf.CreatePrimInLayer(clean_layer, SOURCE_DEFAULT_PRIM)
        root_spec.specifier = sdf.SpecifierDef
        root_spec.typeName = "Xform"

        material_path = sdf.Path("/root/_materials")
        _require(
            source_layer.GetPrimAtPath(material_path) is not None,
            "Source is missing /root/_materials",
        )
        _require(
            sdf.CopySpec(source_layer, material_path, clean_layer, material_path),
            "Could not copy /root/_materials",
        )
        for xform_path, _mesh_path in COMPONENT_PRIM_PAIRS:
            path = sdf.Path(xform_path)
            _require(
                sdf.CopySpec(source_layer, path, clean_layer, path),
                f"Could not copy component spec {xform_path}",
            )

        clean_stage = usd.Stage.Open(clean_layer)
        _require(clean_stage is not None, "Could not open clean stage while authoring metadata")
        usd_geom.SetStageUpAxis(clean_stage, usd_geom.Tokens.z)
        usd_geom.SetStageMetersPerUnit(clean_stage, SOURCE_METERS_PER_UNIT)
        clean_stage.SetDefaultPrim(clean_stage.GetPrimAtPath(SOURCE_DEFAULT_PRIM))
        clean_layer.Save()
        _fsync_file(temporary)

        clean_validation = _validate_clean_payload(
            temporary, authority, sdf, usd, usd_geom, usd_utils
        )
        output_sha = sha256_file(temporary)
        output_size = temporary.stat().st_size

        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "authority": {
                "config": _relative_or_absolute(authority.config_path, authority.repo_root),
                "config_sha256": authority.config_sha256,
                "fit_metadata": _relative_or_absolute(
                    authority.fit_metadata_path, authority.repo_root
                ),
                "fit_metadata_sha256": authority.fit_metadata_sha256,
                "fit_generator": _relative_or_absolute(
                    authority.generator_path, authority.repo_root
                ),
                "fit_generator_sha256": authority.generator_sha256,
                "source": _relative_or_absolute(authority.source_path, authority.repo_root),
                "source_sha256": authority.source_sha256,
            },
            "source_stage": {
                "default_prim": SOURCE_DEFAULT_PRIM,
                "up_axis": SOURCE_UP_AXIS,
                "meters_per_unit": SOURCE_METERS_PER_UNIT,
                "mesh_count": len(COMPONENT_PRIM_PAIRS),
            },
            "copied": {
                "materials_prim": "/root/_materials",
                "component_prim_pairs": [
                    {"xform": xform, "mesh": mesh} for xform, mesh in COMPONENT_PRIM_PAIRS
                ],
            },
            "omitted": ["/root/env_light"],
            "raw_source_dependencies": preflight["dependencies"],
            "raw_source_unresolved_dependencies": preflight["unresolved_dependencies"],
            "output": {
                "path": _relative_or_absolute(output, authority.repo_root),
                "sha256": output_sha,
                "size_bytes": output_size,
                "dependencies": clean_validation["dependencies"],
                "unresolved_dependencies": clean_validation["unresolved_dependencies"],
            },
        }

        _publish_payload_and_manifest(temporary, output, manifest_file, manifest)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def _parser(default_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--fit-metadata", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/isaaclab/g1_crampon_components_clean.usdc"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("build/isaaclab/g1_crampon_components_clean.manifest.json"),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate authority and source USD without writing output",
    )
    return parser


def _from_root(root: Path, value: Path | None) -> Path | None:
    if value is None:
        return None
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    inferred_root = Path(__file__).resolve().parents[2]
    args = _parser(inferred_root).parse_args(argv)
    root = args.repo_root.resolve()
    authority = load_authority(
        root,
        config_path=_from_root(root, args.config),
        fit_metadata_path=_from_root(root, args.fit_metadata),
    )
    if args.preflight_only:
        result = preflight_source(authority)
        summary = {
            "source": _relative_or_absolute(authority.source_path, root),
            "source_sha256": authority.source_sha256,
            "mesh_count": len(COMPONENT_PRIM_PAIRS),
            "dependencies": result["dependencies"],
            "unresolved_dependencies": result["unresolved_dependencies"],
        }
    else:
        output = _from_root(root, args.output)
        manifest_file = _from_root(root, args.manifest)
        assert output is not None and manifest_file is not None
        summary = build_clean_payload(authority, output, manifest_file)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuthorityError as exc:
        print(f"asset authority error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
