from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "isaaclab_ext/scripts/build_clean_crampon_payload.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_clean_crampon_payload", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


asset = _load_script()

EXPECTED_XFORMS = {
    "/root/Cylinder_015",
    "/root/Plane_022",
    "/root/Cylinder_013",
    "/root/Plane_017",
    "/root/Cylinder_018",
    "/root/Plane_020",
    "/root/Plane_027",
    "/root/Cylinder_014",
    "/root/Plane_019",
    "/root/Plane_021",
    "/root/Plane_024",
    "/root/Cylinder_012",
    "/root/Cylinder_011",
    "/root/Plane_018",
    "/root/Plane_023",
    "/root/Plane_016",
    "/root/Plane_026",
    "/root/mesh_node_003",
    "/root/Cube_003",
    "/root/Cylinder_016",
    "/root/Plane_025",
    "/root/Cube_002",
    "/root/Cylinder_010",
    "/root/Cylinder_017",
    "/root/Plane_015",
    "/root/_MF_Object_003",
}


def test_pure_python_authority_files_agree() -> None:
    authority = asset.load_authority(REPO_ROOT)

    assert authority.source_sha256 == asset.SOURCE_SHA256
    assert authority.source_path.name == "g1_crampon_components_source.usdc"
    assert authority.fit_metadata["source_sha256"] == authority.source_sha256
    assert authority.config["source"]["mesh_objects"] == 26
    assert authority.fit_metadata["source_mesh_objects"] == 26
    assert authority.config["legacy"]["use_old_two_mesh_fit_for_isaac_lab"] is False
    assert authority.config["isaac_lab"]["visual_geometry"] == "all_usd_meshes"
    assert authority.config["isaac_lab"]["collision_geometry"] == "analytical_crampon_proxies_only"


def test_component_allowlist_is_exact_and_unique() -> None:
    pairs = asset.COMPONENT_PRIM_PAIRS
    xforms = [xform for xform, _mesh in pairs]
    meshes = [mesh for _xform, mesh in pairs]

    assert len(pairs) == 26
    assert set(xforms) == EXPECTED_XFORMS
    assert len(set(xforms)) == len(xforms)
    assert len(set(meshes)) == len(meshes)
    assert all(mesh.startswith(f"{xform}/") for xform, mesh in pairs)
    assert dict(pairs)["/root/mesh_node_003"] == "/root/mesh_node_003/mesh_003"
    assert dict(pairs)["/root/_MF_Object_003"] == "/root/_MF_Object_003/_MF_Mesh_005"
    assert "/root/env_light" not in xforms


def test_fit_controls_are_the_saved_complete_component_controls() -> None:
    authority = asset.load_authority(REPO_ROOT)
    fit = authority.fit_metadata

    assert fit["source_pivot_world_m"] == [
        1.323341727256775,
        0.01925426349043846,
        -0.007674709893763065,
    ]
    assert fit["control"] == {
        "name": "USD_ASSET_POSITION_CONTROL",
        "location_m": [
            0.04122297838330269,
            0.0008915653452277184,
            -0.04183308780193329,
        ],
        "rotation_euler_rad": [0.0, 0.0, -1.570796251296997],
        "scale": [
            1.0800000429153442,
            1.0800000429153442,
            1.0800000429153442,
        ],
    }
    assert fit["per_foot_controls"] == [
        "LEFT_USD_FINE_TUNE",
        "RIGHT_USD_FINE_TUNE",
    ]


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("isaac_lab", "compose_with_official_g1_usd", False, "Official G1 USD composition"),
        ("isaac_lab", "estimator_packet_contract", "38_values_combined", "packet contract"),
        ("editable_fit", "generator", "other.py", "editable-fit generator"),
    ],
)
def test_production_authority_rejects_isaac_contract_drift(
    tmp_path: Path, section: str, key: str, value: object, message: str
) -> None:
    config = asset.yaml.safe_load(
        (REPO_ROOT / "configs/isaaclab/g1_crampon_asset.yaml").read_text()
    )
    config[section][key] = value
    path = tmp_path / "authority.yaml"
    path.write_text(asset.yaml.safe_dump(config))

    with pytest.raises(asset.AuthorityError, match=message):
        asset.load_authority(REPO_ROOT, config_path=path)


def test_production_authority_rejects_saved_fit_drift(tmp_path: Path) -> None:
    fit = json.loads((REPO_ROOT / "assets/crampon/usd_component_fit_metadata.json").read_text())
    fit["control"]["location_m"] = [999.0, 999.0, 999.0]
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(fit))

    with pytest.raises(asset.AuthorityError, match="control transform drifted"):
        asset.load_authority(REPO_ROOT, fit_metadata_path=path)


@pytest.mark.parametrize(
    "dependency",
    [
        "part.stl",
        "PART.STL",
        "@./meshes/part.stl@</Mesh>",
        "omniverse://server/assets/part.StL?version=2#mesh",
        "@archive.usdz[meshes/part.stl]@",
        "part.stl:SDF_FORMAT_ARGS:target=usd",
        Path("relative/part.stl"),
    ],
)
def test_stl_dependency_rejection_is_case_insensitive(dependency: object) -> None:
    with pytest.raises(asset.AuthorityError, match="disallowed STL"):
        asset.assert_no_stl_dependencies([dependency], "test")


def test_non_stl_usd_dependencies_are_allowed_by_extension_gate() -> None:
    asset.assert_no_stl_dependencies(
        ["asset.usdc", "texture.hdr", "material.mdl", "asset.usd#prim"], "test"
    )


def test_help_works_without_importing_pxr() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--preflight-only" in result.stdout


def test_atomic_json_writer_publishes_complete_sorted_json(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    asset._atomic_write_json(output, {"z": 2, "a": {"value": 1}})

    assert json.loads(output.read_text()) == {"a": {"value": 1}, "z": 2}
    assert output.read_text().startswith('{\n  "a"')
    assert not list(tmp_path.glob("*.tmp"))


def test_pair_publication_restores_previous_files_if_manifest_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary = tmp_path / ".new.tmp.usdc"
    output = tmp_path / "clean.usdc"
    manifest = tmp_path / "clean.manifest.json"
    temporary.write_bytes(b"new payload")
    output.write_bytes(b"old payload")
    manifest.write_text('{"old": true}\n')

    def fail_manifest(_path: Path, _value: dict[str, object]) -> None:
        raise OSError("injected manifest failure")

    monkeypatch.setattr(asset, "_atomic_write_json", fail_manifest)
    with pytest.raises(OSError, match="injected manifest failure"):
        asset._publish_payload_and_manifest(temporary, output, manifest, {"new": True})

    assert output.read_bytes() == b"old payload"
    assert json.loads(manifest.read_text()) == {"old": True}
    assert not list(tmp_path.glob("*.rollback"))


@pytest.mark.skipif(importlib.util.find_spec("pxr") is None, reason="pxr is not installed")
def test_pxr_build_copies_only_authoritative_components(tmp_path: Path) -> None:
    authority = asset.load_authority(REPO_ROOT)
    output = tmp_path / "clean.usdc"
    manifest_path = tmp_path / "clean.manifest.json"

    manifest = asset.build_clean_payload(authority, output, manifest_path)

    assert output.is_file()
    assert manifest_path.is_file()
    assert manifest["output"]["sha256"] == asset.sha256_file(output)
    assert manifest["output"]["unresolved_dependencies"] == []
    assert manifest["omitted"] == ["/root/env_light"]
    assert len(manifest["copied"]["component_prim_pairs"]) == 26
    assert not any(asset._is_stl_dependency(item) for item in manifest["output"]["dependencies"])

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(output))
    assert stage
    assert not stage.GetPrimAtPath("/root/env_light").IsValid()
    meshes = {str(prim.GetPath()) for prim in stage.TraverseAll() if prim.IsA(UsdGeom.Mesh)}
    assert meshes == {mesh for _xform, mesh in asset.COMPONENT_PRIM_PAIRS}
