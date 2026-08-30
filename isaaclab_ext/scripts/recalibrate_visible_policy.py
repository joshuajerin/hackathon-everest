#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import train_visible_policy as training
import zarr
from hackathon_everest_isaaclab.learning.models import (
    REGRESSION_NAMES,
    CausalBilateralEstimator,
)


def load_shard(path: Path):
    visible = zarr.open_group(str(path / "visible.zarr"), mode="r")
    truth = zarr.open_group(str(path / "truth.zarr"), mode="r")
    episodes = pq.read_table(path / "episodes.parquet")
    return (
        visible["packet_values"][:],
        visible["valid_mask"][:],
        visible["sample_age_s"][:],
        training.pack_visible_context(visible),
        truth["targets"][:],
        truth["events"][:],
        np.asarray(episodes["split"].to_pylist()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply target-specific absolute-residual bearing conformal calibration"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.model_dir.resolve() == args.output_dir.resolve():
        raise ValueError("output-dir must differ from model-dir")
    checkpoint_path = args.model_dir / "visible_policy.pt"
    metrics_path = args.model_dir / "metrics.json"
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("deployment_conformal_quantile") != 0.99:
        raise RuntimeError("source model must retain 99% deployment calibration")
    sources = [Path(item["shard"]) for item in metrics.get("dataset_sources", [])]
    if len(sources) < 2:
        raise RuntimeError("broad primary and articulated auxiliary shards are required")
    loaded = [load_shard(path) for path in sources]
    primary_count = len(loaded[0][0])
    combined = [np.concatenate(parts, axis=0) for parts in zip(*loaded, strict=True)]
    values = torch.from_numpy(combined[0]).to(args.device)
    masks = torch.from_numpy(combined[1]).to(args.device)
    ages = torch.from_numpy(combined[2]).to(args.device)
    context = torch.from_numpy(combined[3]).to(args.device)
    targets = torch.from_numpy(combined[4]).to(args.device)
    events = torch.from_numpy(combined[5]).to(args.device)
    split = combined[6]
    calibration = torch.from_numpy(np.flatnonzero(split == "calibration")).to(args.device)
    sealed = torch.from_numpy(np.flatnonzero(split == "sealed_test")).to(args.device)
    articulated_sealed = sealed[sealed >= primary_count]
    if len(calibration) == 0 or len(articulated_sealed) == 0:
        raise RuntimeError("calibration and articulated sealed-test splits must be non-empty")

    estimator = CausalBilateralEstimator(**checkpoint["estimator_config"]).to(args.device)
    estimator.load_state_dict(checkpoint["estimator_state_dict"])
    estimator.eval()
    cal_pred, _, cal_truth, _, _ = training.evaluate_estimator(
        estimator,
        values,
        masks,
        ages,
        context,
        targets,
        events,
        calibration,
        args.batch_size,
    )
    absolute_residual = np.abs(cal_pred - cal_truth)
    absolute95 = np.quantile(absolute_residual, 0.95, axis=(0, 1)).astype(np.float32)
    absolute99 = np.quantile(absolute_residual, 0.99, axis=(0, 1)).astype(np.float32)
    bearing_index = REGRESSION_NAMES.index("bearing_capacity_n")
    bearing95 = float(absolute95[bearing_index])
    bearing99 = float(absolute99[bearing_index])
    conformal99 = np.asarray(checkpoint["conformal99"], dtype=np.float32)
    sealed_metrics = training.estimator_metrics(
        estimator,
        values,
        masks,
        ages,
        context,
        targets,
        events,
        sealed,
        args.batch_size,
        conformal99,
        bearing99,
    )
    articulated_metrics = training.estimator_metrics(
        estimator,
        values,
        masks,
        ages,
        context,
        targets,
        events,
        articulated_sealed,
        args.batch_size,
        conformal99,
        bearing99,
    )

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    shutil.copytree(args.model_dir, args.output_dir)
    source_checkpoint_hash = training.sha256(checkpoint_path)
    checkpoint.update(
        {
            "bearing_capacity_absolute_conformal95_n": bearing95,
            "bearing_capacity_absolute_conformal99_n": bearing99,
            "bearing_capacity_conformal_score": "absolute_residual",
            "bearing_capacity_calibration_source_checkpoint_sha256": source_checkpoint_hash,
        }
    )
    output_checkpoint = args.output_dir / "visible_policy.pt"
    torch.save(checkpoint, output_checkpoint)
    metrics.update(
        {
            "estimator_sealed_test": sealed_metrics,
            "articulated_estimator_sealed_test": articulated_metrics,
            "bearing_capacity_absolute_conformal95_n": bearing95,
            "bearing_capacity_absolute_conformal99_n": bearing99,
            "bearing_capacity_conformal_score": "absolute_residual",
            "bearing_capacity_calibration_source_checkpoint_sha256": source_checkpoint_hash,
            "checkpoint_sha256": training.sha256(output_checkpoint),
        }
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    result = {
        "status": "passed",
        "source_model_dir": str(args.model_dir),
        "output_model_dir": str(args.output_dir),
        "source_checkpoint_sha256": source_checkpoint_hash,
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "bearing_capacity_absolute_conformal95_n": bearing95,
        "bearing_capacity_absolute_conformal99_n": bearing99,
        "estimator_sealed_test": sealed_metrics,
        "articulated_estimator_sealed_test": articulated_metrics,
        "claim_boundary": "Synthetic split-conformal calibration; not hardware validation.",
    }
    (args.output_dir / "bearing_recalibration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
