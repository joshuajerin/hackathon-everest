from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def write(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def inputs(tmp_path: Path, *, audit_status: str = "passed") -> list[str]:
    model = tmp_path / "model"
    export = tmp_path / "export"
    write(model / "visible_policy.pt")
    write(model / "metrics.json", "{}")
    write(model / "audit.json", json.dumps({"status": audit_status}))
    write(export / "estimator_dynamic_jit.pt")
    write(export / "visible_policy_dynamic_jit.pt")
    write(
        export / "deployment_manifest.json",
        json.dumps(
            {
                "deployment_conformal_quantile": 0.99,
                "bearing_capacity_conformal_score": "absolute_residual",
                "bearing_capacity_absolute_conformal99_n": 200.0,
                "sensor_abi": "separate [B,T,2,19] values/mask/age; feet are not flattened",
            }
        ),
    )
    stock = write(tmp_path / "stock.pt")
    stack = write(tmp_path / "stack.json")
    asset = write(tmp_path / "asset.yaml")
    simulation = write(tmp_path / "simulation.yaml")
    terrain = write(tmp_path / "terrain.yaml")
    evaluation = write(
        tmp_path / "evaluation.json",
        json.dumps({"status": "passed", "task": "test", "claim_boundary": "sim only"}),
    )
    script = Path(__file__).parents[2] / "scripts/assemble_deployment_bundle.py"
    return [
        sys.executable,
        str(script),
        "--model-dir",
        str(model),
        "--export-dir",
        str(export),
        "--stock-policy",
        str(stock),
        "--stack-lock",
        str(stack),
        "--asset-config",
        str(asset),
        "--simulation-config",
        str(simulation),
        "--terrain-config",
        str(terrain),
        "--evaluation",
        str(evaluation),
        "--output-dir",
        str(tmp_path / "bundle"),
    ]


def test_bundle_requires_passed_audits_and_preserves_abi(tmp_path: Path) -> None:
    subprocess.run(inputs(tmp_path), check=True, capture_output=True, text=True)
    manifest = json.loads((tmp_path / "bundle/BUNDLE_MANIFEST.json").read_text())
    assert manifest["feet_flattened_at_boundary"] is False
    assert manifest["deployment_conformal_quantile"] == 0.99
    assert manifest["hardware_gates_remaining"]
    assert (tmp_path / "bundle/policy/stock_g1_locomotion.pt").is_file()


def test_bundle_rejects_failed_visible_policy_audit(tmp_path: Path) -> None:
    result = subprocess.run(
        inputs(tmp_path, audit_status="failed"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "visible policy audit has not passed" in result.stderr
