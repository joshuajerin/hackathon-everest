#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble an audited Everest policy bundle")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--stock-policy", type=Path, required=True)
    parser.add_argument("--stack-lock", type=Path, required=True)
    parser.add_argument("--asset-config", type=Path, required=True)
    parser.add_argument("--simulation-config", type=Path, required=True)
    parser.add_argument("--terrain-config", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--selected-video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    audit = require_json(args.model_dir / "audit.json")
    if audit.get("status") != "passed":
        raise RuntimeError("visible policy audit has not passed")
    deployment = require_json(args.export_dir / "deployment_manifest.json")
    if deployment.get("deployment_conformal_quantile") != 0.99:
        raise RuntimeError("deployable policy must use 99% conformal calibration")
    if deployment.get("bearing_capacity_conformal_score") != "absolute_residual":
        raise RuntimeError("bearing capacity must use absolute-residual conformal calibration")
    bearing_radius = deployment.get("bearing_capacity_absolute_conformal99_n")
    if not isinstance(bearing_radius, (int, float)) or bearing_radius < 0.0:
        raise RuntimeError("bearing capacity absolute 99% conformal radius is missing")
    abi = deployment.get("sensor_abi", "")
    if "[B,T,2,19]" not in abi or "not flattened" not in abi:
        raise RuntimeError("deployment manifest does not preserve the bilateral sensor ABI")

    evaluations = []
    for path in args.evaluation:
        result = require_json(path)
        if result.get("status") != "passed":
            raise RuntimeError(f"rollout evaluation did not pass: {path}")
        evaluations.append(
            {
                "source": str(path),
                "sha256": sha256(path),
                "task": result.get("task"),
                "mode": result.get("mode"),
                "num_envs": result.get("num_envs"),
                "steps": result.get("steps"),
                "terminations": result.get("terminations", result.get("base_contact_terminations")),
                "claim_boundary": result.get("claim_boundary"),
            }
        )

    required = {
        "policy/visible_policy.pt": args.model_dir / "visible_policy.pt",
        "policy/estimator_dynamic_jit.pt": args.export_dir / "estimator_dynamic_jit.pt",
        "policy/visible_policy_dynamic_jit.pt": args.export_dir / "visible_policy_dynamic_jit.pt",
        "policy/stock_g1_locomotion.pt": args.stock_policy,
        "manifests/policy_audit.json": args.model_dir / "audit.json",
        "manifests/training_metrics.json": args.model_dir / "metrics.json",
        "manifests/deployment_manifest.json": args.export_dir / "deployment_manifest.json",
        "manifests/stack.lock.json": args.stack_lock,
        "config/g1_crampon_asset.yaml": args.asset_config,
        "config/g1_crampon_simulation.yaml": args.simulation_config,
        "config/everest_terrain_suite.yaml": args.terrain_config,
    }
    for index, source in enumerate(args.evaluation):
        required[f"evaluations/{index:02d}_{source.name}"] = source
    if args.selected_video is not None:
        if not args.selected_video.is_file() or args.selected_video.stat().st_size == 0:
            raise RuntimeError("selected video is missing or empty")
        required[f"media/{args.selected_video.name}"] = args.selected_video
    for source in required.values():
        if not source.is_file():
            raise FileNotFoundError(source)

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for relative, source in required.items():
        destination = args.output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    files = {
        str(path.relative_to(args.output_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": "1.0.0",
        "status": "simulator_evaluation_bundle",
        "sensor_abi": "values/mask/age remain separate [B,T,2,19] tensors",
        "feet_flattened_at_boundary": False,
        "deployment_conformal_quantile": 0.99,
        "evaluations": evaluations,
        "files": files,
        "hardware_gates_remaining": [
            "measure exact G1 revision, mass, center of mass, and inertia",
            "calibrate each force, penetration, IMU, gyro, and radar channel",
            "identify physical snow/ice constitutive parameters with instrumented rig tests",
            "integrate and validate hardware watchdog and independent emergency stop",
            "validate timing, packet loss, thermal, power, and actuator limits on hardware",
            "perform staged tethered hardware tests and independent expert safety review",
        ],
        "claim_boundary": (
            "Synthetic Isaac/Newton simulator evidence only. This bundle does not establish "
            "real-hardware safety or Everest readiness."
        ),
    }
    manifest_path = args.output_dir / "BUNDLE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
