#!/usr/bin/env python3
"""Fail-closed post-training calibration of a visible-policy supervisor commit margin."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from hackathon_everest_isaaclab.learning.models import (
    CausalBilateralEstimator,
    VisibleEverestPolicy,
    VisibleOnlySupervisor,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--primary-shard", type=Path, required=True)
parser.add_argument("--aux-shard", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--validation-max-unsafe-commit-rate", type=float, default=0.00075)
parser.add_argument("--sealed-max-unsafe-commit-rate", type=float, default=0.0015)
parser.add_argument("--batch-size", type=int, default=512)
args = parser.parse_args()
if not 0.0 <= args.validation_max_unsafe_commit_rate <= args.sealed_max_unsafe_commit_rate <= 1.0:
    raise ValueError("require 0 <= validation maximum <= sealed maximum <= 1")


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def load_training_module():
    source = Path(__file__).with_name("train_visible_policy.py")
    spec = importlib.util.spec_from_file_location("everest_train_visible_policy", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_arrays(shard: Path, trainer):
    visible = zarr.open_group(str(shard / "visible.zarr"), mode="r")
    truth = zarr.open_group(str(shard / "truth.zarr"), mode="r")
    episodes = pq.read_table(shard / "episodes.parquet")
    return (
        visible["packet_values"][:],
        visible["valid_mask"][:],
        visible["sample_age_s"][:],
        trainer.pack_visible_context(visible),
        trainer.pack_command_context(visible),
        truth["targets"][:],
        truth["events"][:],
        np.asarray(episodes["split"].to_pylist()),
    )


def main() -> int:
    trainer = load_training_module()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    primary, auxiliary = (
        load_arrays(path.expanduser().resolve(), trainer)
        for path in (args.primary_shard, args.aux_shard)
    )
    combined = [np.concatenate(parts, axis=0) for parts in zip(primary, auxiliary, strict=True)]
    values, masks, ages, context, commands, targets, events, split = (
        torch.from_numpy(value).to(device) if index < 7 else value
        for index, value in enumerate(combined)
    )
    indices = {
        name: torch.from_numpy(np.flatnonzero(split == name)).to(device)
        for name in ("validation", "sealed_test")
    }
    estimator = CausalBilateralEstimator(**checkpoint["estimator_config"]).to(device)
    estimator.load_state_dict(checkpoint["estimator_state_dict"])
    estimator.eval()
    supervisor = VisibleOnlySupervisor(**checkpoint["supervisor_config"]).to(device)
    supervisor.load_state_dict(checkpoint["supervisor_state_dict"])
    supervisor.eval()
    latent_blocks, command_blocks, target_blocks, event_blocks = [], [], [], []
    all_indices = torch.arange(len(values), device=device)
    with torch.no_grad():
        for end in trainer.PREFIX_INDICES:
            latent_parts = []
            for offset in range(0, len(values), args.batch_size):
                batch = all_indices[offset : offset + args.batch_size]
                latent_parts.append(
                    estimator(
                        values[batch, : end + 1],
                        masks[batch, : end + 1],
                        ages[batch, : end + 1],
                        context[batch, : end + 1],
                    ).bilateral_latent
                )
            latent_blocks.append(torch.cat(latent_parts))
            command_blocks.append(commands[:, end])
            target_blocks.append(targets[:, end])
            event_blocks.append(events[:, end])
    latent, command = torch.cat(latent_blocks), torch.cat(command_blocks)
    teacher, _ = trainer.make_teacher(torch.cat(target_blocks), torch.cat(event_blocks), command)
    episodes = len(values)
    sample = {
        name: torch.cat(
            [indices[name] + prefix * episodes for prefix in range(len(trainer.PREFIX_INDICES))]
        )
        for name in indices
    }
    before = trainer.evaluate_supervisor(supervisor, latent, command, teacher, sample["validation"])
    extra = trainer.calibrate_commit_logit_subtraction(
        supervisor,
        latent,
        command,
        teacher,
        sample["validation"],
        maximum_unsafe_commit_rate=args.validation_max_unsafe_commit_rate,
    )
    with torch.no_grad():
        supervisor.action_head.bias[0].sub_(extra)
    validation = trainer.evaluate_supervisor(
        supervisor, latent, command, teacher, sample["validation"]
    )
    sealed = trainer.evaluate_supervisor(
        supervisor, latent, command, teacher, sample["sealed_test"]
    )
    if validation["unsafe_commit_rate"] > args.validation_max_unsafe_commit_rate:
        raise RuntimeError("validation margin calibration did not meet its maximum")
    if sealed["unsafe_commit_rate"] > args.sealed_max_unsafe_commit_rate:
        raise RuntimeError(
            f"sealed unsafe commit rate {sealed['unsafe_commit_rate']:.6f} exceeds {args.sealed_max_unsafe_commit_rate:.6f}"
        )
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    checkpoint["supervisor_state_dict"] = supervisor.state_dict()
    checkpoint["supervisor_commit_logit_subtraction"] = float(
        checkpoint["supervisor_commit_logit_subtraction"]
    ) + float(extra)
    checkpoint["supervisor_validation_maximum_unsafe_commit_rate"] = (
        args.validation_max_unsafe_commit_rate
    )
    checkpoint["post_training_commit_margin_calibration"] = {
        "validation_before": before,
        "validation_after": validation,
        "sealed": sealed,
        "additional_commit_logit_subtraction": float(extra),
        "sealed_maximum_unsafe_commit_rate": args.sealed_max_unsafe_commit_rate,
    }
    policy = VisibleEverestPolicy(estimator, supervisor).to(device).eval()
    torch.save(checkpoint, out / "visible_policy.pt")
    example = (values[:1], masks[:1], ages[:1], context[:1], commands[:1, -1])
    torch.jit.trace(trainer.EstimatorExport(estimator), example[:4], check_trace=False).save(
        str(out / "estimator_jit.pt")
    )
    torch.jit.trace(trainer.PolicyExport(policy), example, check_trace=False).save(
        str(out / "visible_policy_jit.pt")
    )
    metrics = {
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_sha256": sha256(out / "visible_policy.pt"),
        "post_training_commit_margin_calibration": checkpoint[
            "post_training_commit_margin_calibration"
        ],
        "claim_boundary": "Synthetic simulator calibration; full articulated G1 evaluation remains required.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
