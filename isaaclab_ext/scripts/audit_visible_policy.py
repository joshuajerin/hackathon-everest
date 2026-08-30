#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit sealed visible-policy safety gates")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-false-safe-rate", type=float, default=0.002)
    parser.add_argument("--min-void-recall", type=float, default=0.99)
    parser.add_argument("--min-fracture-recall", type=float, default=0.85)
    parser.add_argument("--min-slip-recall", type=float, default=0.95)
    parser.add_argument("--max-unsafe-commit-rate", type=float, default=0.002)
    parser.add_argument("--min-supervisor-accuracy", type=float, default=0.70)
    parser.add_argument("--min-supervisor-commit-recall", type=float, default=0.50)
    parser.add_argument("--min-articulated-safe-coverage", type=float, default=0.75)
    args = parser.parse_args()

    metrics_path = args.model_dir / "metrics.json"
    checkpoint_path = args.model_dir / "visible_policy.pt"
    metrics = json.loads(metrics_path.read_text())
    estimator = metrics["estimator_sealed_test"]
    articulated_estimator = metrics.get("articulated_estimator_sealed_test")
    supervisor = metrics["supervisor_sealed_test"]
    failures: list[str] = []
    if articulated_estimator is None:
        failures.append("articulated_estimator_sealed_test")

    gates = (
        (estimator["false_safe_rate"] <= args.max_false_safe_rate, "false_safe_rate"),
        (estimator["void_present_recall"] >= args.min_void_recall, "void_present_recall"),
        (estimator["fractured_recall"] >= args.min_fracture_recall, "fractured_recall"),
        (estimator["slipping_recall"] >= args.min_slip_recall, "slipping_recall"),
        (
            supervisor["unsafe_commit_rate"] <= args.max_unsafe_commit_rate,
            "unsafe_commit_rate",
        ),
        (
            supervisor["teacher_action_accuracy"] >= args.min_supervisor_accuracy,
            "teacher_action_accuracy",
        ),
        (
            supervisor["commit_recall"] >= args.min_supervisor_commit_recall,
            "commit_recall",
        ),
        (metrics["deployment_conformal_quantile"] == 0.99, "deployment_conformal_quantile"),
    )
    failures.extend(name for passed, name in gates if not passed)
    if articulated_estimator is not None:
        if articulated_estimator["false_safe_rate"] > args.max_false_safe_rate:
            failures.append("articulated_false_safe_rate")
        if articulated_estimator["predicted_safe_coverage"] < args.min_articulated_safe_coverage:
            failures.append("articulated_predicted_safe_coverage")
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != metrics["checkpoint_sha256"]:
        failures.append("checkpoint_sha256")

    source_audit = []
    for source in metrics.get("dataset_sources", []):
        shard = Path(source["shard"])
        manifest_hash = sha256(shard / "manifest.json")
        complete_hash = sha256(shard / "_COMPLETE")
        valid = (
            manifest_hash == source["manifest_sha256"]
            and complete_hash == source["complete_marker_sha256"]
        )
        if not valid:
            failures.append(f"dataset_source:{shard}")
        source_audit.append({"shard": str(shard), "hashes_match": valid})
    if not source_audit:
        failures.append("dataset_sources")

    result = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "model_dir": str(args.model_dir),
        "checkpoint_sha256": checkpoint_hash,
        "selected_estimator_epoch": metrics.get("selected_estimator_epoch"),
        "estimator_selection_score": metrics.get("estimator_selection_score"),
        "selected_supervisor_epoch": metrics.get("selected_supervisor_epoch"),
        "supervisor_selection_score": metrics.get("supervisor_selection_score"),
        "supervisor_commit_logit_subtraction": metrics.get("supervisor_commit_logit_subtraction"),
        "supervisor_validation_maximum_unsafe_commit_rate": metrics.get(
            "supervisor_validation_maximum_unsafe_commit_rate"
        ),
        "estimator_sealed_test": estimator,
        "articulated_estimator_sealed_test": articulated_estimator,
        "supervisor_sealed_test": supervisor,
        "training_window_counts": metrics.get("training_window_counts"),
        "dataset_sources": source_audit,
        "claim_boundary": "Synthetic simulator-policy audit; not hardware safety validation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
