#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
import zarr
from sklearn.metrics import mean_absolute_error, recall_score

from hackathon_everest.dataset import EVENT_NAMES, TARGET_NAMES
from hackathon_everest.estimation import TerrainStateEstimator
from hackathon_everest.features import feature_names

PREFIX_INDICES = (5, 10, 15, 22, 30)


def slope(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    centered_t = times - times.mean()
    denominator = float(np.sum(centered_t**2))
    centered_v = values - values.mean(axis=1, keepdims=True)
    return np.sum(centered_v * centered_t[None, :, None, None], axis=1) / max(denominator, 1e-12)


def extract_features(
    values: np.ndarray,
    masks: np.ndarray,
    timestamps: np.ndarray,
    context: dict[str, np.ndarray],
    end: int,
) -> np.ndarray:
    window = values[:, : end + 1]
    valid = masks[:, : end + 1]
    times = timestamps[0, : end + 1, 0] - timestamps[0, 0, 0]
    force = window[..., :4]
    depth = window[..., 4:8]
    imu = window[..., 8:14]
    parts = []
    for group in (force, depth):
        stats = np.stack(
            [
                group[:, -1],
                group.max(axis=1),
                group.mean(axis=1),
                group.std(axis=1),
                slope(group, times),
            ],
            axis=-1,
        )
        parts.append(stats.reshape(len(values), 2, -1))
    imu_stats = np.stack([imu.mean(axis=1), imu.std(axis=1), np.abs(imu).max(axis=1)], axis=-1)
    parts.append(imu_stats.reshape(len(values), 2, -1))
    parts.append(window[:, -1, :, 14:19])
    command_load = context["commanded_probe_load_n"][:, end]
    command_speed = context["commanded_foot_speed_mps"][:, : end + 1].mean(axis=1)
    body_load = context["body_weight_on_foot_n"][:, end]
    duration = np.full_like(command_load, times[-1] if len(times) > 1 else 0.0)
    ratios = np.stack(
        [
            valid[..., :4].mean(axis=(1, 3)),
            valid[..., 4:8].mean(axis=(1, 3)),
            valid[..., 8:14].mean(axis=(1, 3)),
            valid[..., 14:19].mean(axis=(1, 3)),
        ],
        axis=-1,
    )
    parts.append(np.stack([command_load, command_speed, body_load, duration], axis=-1))
    parts.append(ratios)
    depth_centered = depth - depth.mean(axis=1, keepdims=True)
    force_centered = force - force.mean(axis=1, keepdims=True)
    fd_slope = np.sum(depth_centered * force_centered, axis=1) / np.maximum(
        np.sum(depth_centered**2, axis=1), 1e-12
    )
    parts.append(fd_slope)
    front_force = force[:, -1, :, 0] + force[:, -1, :, 1]
    rear_force = force[:, -1, :, 2] + force[:, -1, :, 3]
    front_depth = depth[:, -1, :, 0] + depth[:, -1, :, 1]
    rear_depth = depth[:, -1, :, 2] + depth[:, -1, :, 3]
    total_force = force.sum(axis=-1)
    mean_depth = depth.mean(axis=-1)
    energy = np.trapezoid(total_force, mean_depth, axis=1)
    parts.append(
        np.stack(
            [
                front_force - rear_force,
                front_depth - rear_depth,
                total_force[:, -1],
                total_force.max(axis=1),
                energy,
            ],
            axis=-1,
        )
    )
    result = np.concatenate(parts, axis=-1)
    if result.shape[-1] != len(feature_names()):
        raise RuntimeError(f"Feature contract drift: {result.shape[-1]} != {len(feature_names())}")
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    visible = zarr.open_group(str(args.shard / "visible.zarr"), mode="r")
    truth = zarr.open_group(str(args.shard / "truth.zarr"), mode="r")
    values = visible["packet_values"][:]
    masks = visible["valid_mask"][:]
    timestamps = visible["timestamp_s"][:]
    context = {
        name: visible[f"context/{name}"][:]
        for name in ("commanded_probe_load_n", "commanded_foot_speed_mps", "body_weight_on_foot_n")
    }
    target_values = truth["targets"][:]
    event_values = truth["events"][:].astype(np.int8)
    table = pq.read_table(args.shard / "episodes.parquet")
    episode_split = np.asarray(table["split"].to_pylist())
    episode_regime = np.asarray(table["sampling_regime"].to_pylist())
    features = []
    targets = []
    events = []
    splits = []
    regimes = []
    for end in PREFIX_INDICES:
        block = extract_features(values, masks, timestamps, context, end)
        features.append(block.reshape(-1, block.shape[-1]))
        targets.append(target_values[:, end].reshape(-1, len(TARGET_NAMES)))
        events.append(event_values[:, end].reshape(-1, len(EVENT_NAMES)))
        splits.append(np.repeat(episode_split, 2))
        regimes.append(np.repeat(episode_regime, 2))
    x = np.concatenate(features)
    y = np.concatenate(targets)
    e = np.concatenate(events)
    split = np.concatenate(splits)
    regime = np.concatenate(regimes)
    train = split == "train"
    sealed = split == "sealed_test"
    natural = sealed & (regime == "natural_prior")
    estimator = TerrainStateEstimator(n_estimators=args.n_estimators, random_state=args.seed).fit(
        x[train], y[train], e[train]
    )
    pred, std = estimator.predict_continuous(x[sealed])
    probs = estimator.predict_events(x[sealed])
    ti = {name: i for i, name in enumerate(TARGET_NAMES)}
    ei = {name: i for i, name in enumerate(EVENT_NAMES)}
    safe_true = y[sealed, ti["bearing_capacity_n"]] >= 343.0
    safe_pred = pred[:, ti["bearing_capacity_n"]] - 2 * std[:, ti["bearing_capacity_n"]] >= 343.0
    metrics = {
        "rows": {
            "train": int(train.sum()),
            "sealed_test": int(sealed.sum()),
            "sealed_natural_prior": int(natural.sum()),
        },
        "support_depth_mae_m": float(
            mean_absolute_error(
                y[sealed, ti["support_layer_depth_m"]], pred[:, ti["support_layer_depth_m"]]
            )
        ),
        "bearing_relative_error_median": float(
            np.median(
                np.abs(pred[:, ti["bearing_capacity_n"]] - y[sealed, ti["bearing_capacity_n"]])
                / np.maximum(y[sealed, ti["bearing_capacity_n"]], 1.0)
            )
        ),
        "shear_relative_error_median": float(
            np.median(
                np.abs(pred[:, ti["shear_capacity_n"]] - y[sealed, ti["shear_capacity_n"]])
                / np.maximum(y[sealed, ti["shear_capacity_n"]], 1.0)
            )
        ),
        "false_safe_rate": float(np.mean(safe_pred[~safe_true])) if np.any(~safe_true) else 0.0,
        "event_recall": {
            name: float(recall_score(e[sealed, ei[name]], probs[name] >= 0.5, zero_division=0))
            for name in EVENT_NAMES
        },
        "claim_boundary": "Synthetic L0 held-out metrics; not real snow/ice or hardware validation.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, args.output_dir / "terrain_estimator.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": "ExtraTrees baseline",
                "feature_names": feature_names(),
                "target_names": TARGET_NAMES,
                "event_names": EVENT_NAMES,
                "prefix_indices": PREFIX_INDICES,
                "dataset_shard": str(args.shard),
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
