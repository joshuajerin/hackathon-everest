from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/analyze_policy_crampon_comparison.py"
SPEC = importlib.util.spec_from_file_location("analyze_policy_crampon_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def write_manifest(
    root: Path,
    *,
    seed: int,
    crampon: tuple[int, float, float, float],
    baseline: tuple[int, float, float, float],
) -> None:
    directory = root / f"trial_{seed}"
    directory.mkdir()

    def row(label: str, visible: bool, grip: float, values: tuple[int, float, float, float]) -> dict:
        terminations, height, displacement, lateral_speed = values
        return {
            "label": label,
            "crampon_visual_visible": visible,
            "tangential_grip_scale": grip,
            "policy_sha256": f"policy-{label}",
            "terminations": terminations,
            "minimum_base_height_m": height,
            "forward_displacement_m": displacement,
            "stance_lateral_speed_mps": lateral_speed,
        }

    document = {
        "artifact_type": "native_isaac_same_sim_policy_crampon_comparison",
        "scene": {
            "surface": "hard_glacier_ice",
            "incline_deg": 20,
            "scene_seed": seed,
            "requested_vx_mps": 0.15,
            "steps": 500,
            "same_isaac_process": True,
            "matched_material_parameters": True,
        },
        "policies": [
            row("CRAMPON", True, 1.0, crampon),
            row("BASELINE", False, 0.04, baseline),
        ],
    }
    (directory / "comparison_manifest.json").write_text(json.dumps(document))


def test_analysis_uses_paired_favorable_directions(tmp_path: Path) -> None:
    write_manifest(
        tmp_path,
        seed=1,
        crampon=(0, 0.58, 1.2, 0.03),
        baseline=(1, 0.45, 0.8, 0.10),
    )
    write_manifest(
        tmp_path,
        seed=2,
        crampon=(0, 0.56, 1.1, 0.04),
        baseline=(2, 0.44, 0.7, 0.12),
    )

    result = analysis.analyze(tmp_path, bootstrap_samples=200, seed=9)

    metrics = result["overall_micro_average"]
    assert metrics["terminations_avoided"]["favorable_delta_mean"] == 1.5
    assert metrics["minimum_base_height_gain_m"]["favorable_delta_mean"] == 0.125
    assert metrics["forward_displacement_gain_m"]["favorable_delta_mean"] == 0.4
    assert metrics["stance_lateral_speed_reduction_mps"]["favorable_delta_mean"] == 0.075
    assert metrics["terminations_avoided"]["crampon_wins"] == 2
    assert result["adequacy"]["status"] == "insufficient_per_condition_repeats"


def test_analysis_rejects_duplicate_paired_identity(tmp_path: Path) -> None:
    write_manifest(
        tmp_path,
        seed=1,
        crampon=(0, 0.5, 1.0, 0.03),
        baseline=(1, 0.4, 0.5, 0.10),
    )
    original = tmp_path / "trial_1" / "comparison_manifest.json"
    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    (duplicate_dir / "comparison_manifest.json").write_text(original.read_text())

    try:
        analysis.analyze(tmp_path, bootstrap_samples=200)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate identity should fail")
