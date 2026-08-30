#!/usr/bin/env python3
"""Aggregate paired Isaac crampon comparison manifests.

The input can be one manifest or a directory containing comparison_manifest.json files.
Positive paired deltas always favor the crampon arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

METRICS = {
    "terminations_avoided": ("terminations", -1.0, "count"),
    "minimum_base_height_gain_m": ("minimum_base_height_m", 1.0, "m"),
    "forward_displacement_gain_m": ("forward_displacement_m", 1.0, "m"),
    "stance_lateral_speed_reduction_mps": ("stance_lateral_speed_mps", -1.0, "m/s"),
}


def manifest_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    return sorted(input_path.rglob("comparison_manifest.json"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_interval(
    values: list[float], *, samples: int, seed: int
) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    generator = random.Random(seed)
    means = [
        statistics.fmean(generator.choices(values, k=len(values))) for _ in range(samples)
    ]
    return percentile(means, 0.025), percentile(means, 0.975)


def exact_two_sided_sign_pvalue(positive: int, negative: int) -> float | None:
    non_ties = positive + negative
    if non_ties == 0:
        return None
    tail = min(positive, negative)
    probability = sum(math.comb(non_ties, k) for k in range(tail + 1)) / (2**non_ties)
    return min(1.0, 2.0 * probability)


def validate_and_pair(path: Path, document: dict[str, Any]) -> dict[str, Any]:
    if document.get("artifact_type") != "native_isaac_same_sim_policy_crampon_comparison":
        raise ValueError(f"{path}: unexpected artifact_type")
    scene = document.get("scene")
    if not isinstance(scene, dict):
        raise TypeError(f"{path}: scene must be an object")
    if scene.get("same_isaac_process") is not True:
        raise ValueError(f"{path}: comparison is not marked same-process")
    if scene.get("matched_material_parameters") is not True:
        raise ValueError(f"{path}: material parameters are not marked matched")
    policies = document.get("policies")
    if not isinstance(policies, list) or len(policies) != 2:
        raise ValueError(f"{path}: exactly two policy rows are required")
    crampon_rows = [row for row in policies if row.get("crampon_visual_visible") is True]
    baseline_rows = [row for row in policies if row.get("crampon_visual_visible") is False]
    if len(crampon_rows) != 1 or len(baseline_rows) != 1:
        raise ValueError(f"{path}: cannot identify one crampon and one baseline row")
    crampon, baseline = crampon_rows[0], baseline_rows[0]
    missing = [
        source
        for source, _, _ in METRICS.values()
        if source not in crampon or source not in baseline
    ]
    if missing:
        raise ValueError(f"{path}: missing metric values {sorted(set(missing))}")
    return {
        "manifest": str(path),
        "manifest_sha256": sha256(path),
        "surface": scene.get("surface"),
        "incline_deg": float(scene.get("incline_deg")),
        "scene_seed": int(scene.get("scene_seed")),
        "requested_vx_mps": float(scene.get("requested_vx_mps")),
        "steps": int(scene.get("steps")),
        "crampon_policy_sha256": crampon.get("policy_sha256"),
        "baseline_policy_sha256": baseline.get("policy_sha256"),
        "crampon_grip_scale": float(crampon.get("tangential_grip_scale")),
        "baseline_grip_scale": float(baseline.get("tangential_grip_scale")),
        "raw": {"crampon": crampon, "baseline": baseline},
        "favorable_deltas": {
            output: direction * (float(crampon[source]) - float(baseline[source]))
            for output, (source, direction, _) in METRICS.items()
        },
    }


def summarize_metric(
    trials: list[dict[str, Any]], metric: str, *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    source, _, unit = METRICS[metric]
    crampon = [float(trial["raw"]["crampon"][source]) for trial in trials]
    baseline = [float(trial["raw"]["baseline"][source]) for trial in trials]
    deltas = [trial["favorable_deltas"][metric] for trial in trials]
    positive = sum(delta > 0 for delta in deltas)
    negative = sum(delta < 0 for delta in deltas)
    ties = len(deltas) - positive - negative
    interval = bootstrap_mean_interval(deltas, samples=bootstrap_samples, seed=seed)
    return {
        "unit": unit,
        "positive_delta_favors": "crampon",
        "n_paired_trials": len(trials),
        "crampon_mean": statistics.fmean(crampon),
        "baseline_mean": statistics.fmean(baseline),
        "favorable_delta_mean": statistics.fmean(deltas),
        "favorable_delta_median": statistics.median(deltas),
        "favorable_delta_bootstrap_95pct_ci": list(interval) if interval else None,
        "crampon_wins": positive,
        "baseline_wins": negative,
        "ties": ties,
        "crampon_paired_win_fraction_excluding_ties": (
            positive / (positive + negative) if positive + negative else None
        ),
        "exact_two_sided_sign_test_pvalue": exact_two_sided_sign_pvalue(positive, negative),
    }


def condition_key(trial: dict[str, Any]) -> str:
    return (
        f"{trial['surface']}|{trial['incline_deg']:g}deg|"
        f"{trial['requested_vx_mps']:g}mps"
    )


def analyze(input_path: Path, *, bootstrap_samples: int = 10_000, seed: int = 20260829) -> dict[str, Any]:
    paths = manifest_paths(input_path)
    if not paths:
        raise ValueError(f"No comparison_manifest.json files found below {input_path}")
    trials = [validate_and_pair(path, json.loads(path.read_text())) for path in paths]
    identities = [
        (trial["surface"], trial["incline_deg"], trial["requested_vx_mps"], trial["scene_seed"])
        for trial in trials
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate surface/incline/speed/seed trial identity")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        grouped.setdefault(condition_key(trial), []).append(trial)

    overall = {
        metric: summarize_metric(
            trials, metric, bootstrap_samples=bootstrap_samples, seed=seed + index
        )
        for index, metric in enumerate(METRICS)
    }
    condition_means = []
    for group in grouped.values():
        condition_means.append(
            {
                "raw": {
                    arm: {
                        source: statistics.fmean(
                            float(trial["raw"][arm][source]) for trial in group
                        )
                        for source, _, _ in METRICS.values()
                    }
                    for arm in ("crampon", "baseline")
                },
                "favorable_deltas": {
                    metric: statistics.fmean(
                        trial["favorable_deltas"][metric] for trial in group
                    )
                    for metric in METRICS
                },
            }
        )
    macro = {
        metric: summarize_metric(
            condition_means,
            metric,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 500 + index,
        )
        for index, metric in enumerate(METRICS)
    }
    by_condition = {
        key: {
            metric: summarize_metric(
                group,
                metric,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1000 + group_index * len(METRICS) + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
        for group_index, (key, group) in enumerate(sorted(grouped.items()))
    }
    smallest_condition_n = min(len(group) for group in grouped.values())
    return {
        "schema_version": "1.0.0",
        "artifact_type": "native_isaac_policy_crampon_comparison_summary",
        "input": str(input_path),
        "n_paired_trials": len(trials),
        "n_conditions": len(grouped),
        "smallest_condition_n": smallest_condition_n,
        "adequacy": {
            "status": "initial_sample_target_met" if smallest_condition_n >= 30 else "insufficient_per_condition_repeats",
            "initial_target_paired_trials_per_condition": 30,
            "note": (
                "This is a simulator precision check, not hardware validation. Statistical power "
                "depends on the event rate and declared primary effect, not only this rule of thumb."
            ),
        },
        "overall_micro_average": overall,
        "overall_macro_average_equal_condition_weight": macro,
        "by_condition": by_condition,
        "trials": trials,
        "interpretation_limits": [
            "The baseline is a low-grip project-authored proxy, not a validated stock G1 foot model.",
            "If the two policy hashes differ, hardware and policy effects are confounded.",
            "Micro-averages weight trials; use the macro summary for equal condition weighting.",
            "Forward displacement can be misleading after automatic environment reset.",
            "Simulation results are not real-world fall probabilities or safety certification.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Manifest path or directory searched recursively")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    result = analyze(
        args.input.expanduser().resolve(),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
