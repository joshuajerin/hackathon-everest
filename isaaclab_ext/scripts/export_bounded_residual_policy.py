#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from hackathon_everest_isaaclab.learning.residual_policy import BoundedResidualMLPModel
from tensordict import TensorDict


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a hash-audited bounded residual G1 actor")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stock-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    stock_checkpoint = args.stock_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file() or not stock_checkpoint.is_file():
        raise FileNotFoundError("checkpoint and stock-checkpoint must exist")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(20260830)
    sample = TensorDict({"policy": torch.randn(32, 310)}, batch_size=[32])
    model = BoundedResidualMLPModel(
        obs=sample,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=37,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.02,
            "std_type": "scalar",
        },
        stock_checkpoint_path=str(stock_checkpoint),
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor_state = saved.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        raise TypeError("checkpoint is missing actor_state_dict")
    model.load_state_dict(actor_state, strict=True)
    model.eval()

    with torch.inference_mode():
        python_action = model(sample)
        stock_action = model.stock_mlp(sample["policy"])
    maximum_residual = float((python_action - stock_action).abs().max())
    if maximum_residual > 0.120001:
        raise RuntimeError(f"residual bound violated: {maximum_residual}")
    scripted = torch.jit.script(model.as_jit())
    with torch.inference_mode():
        scripted_action = scripted(sample["policy"])
    if not torch.equal(scripted_action, python_action):
        maximum_difference = float((scripted_action - python_action).abs().max())
        if maximum_difference > 1.0e-6:
            raise RuntimeError(f"TorchScript mismatch: {maximum_difference}")

    policy_path = output_dir / "policy.pt"
    scripted.save(str(policy_path))
    manifest = {
        "schema_version": "1.0.0",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "stock_checkpoint": str(stock_checkpoint),
        "stock_checkpoint_sha256": sha256(stock_checkpoint),
        "policy_sha256": sha256(policy_path),
        "observation_shape": ["B", 310],
        "action_shape": ["B", 37],
        "maximum_configured_residual": 0.12,
        "maximum_sampled_residual": maximum_residual,
        "dynamic_batch_checks": [1, 7, 32],
    }
    loaded = torch.jit.load(str(policy_path)).eval()
    for batch in manifest["dynamic_batch_checks"]:
        output = loaded(torch.randn(batch, 310))
        if tuple(output.shape) != (batch, 37) or not bool(torch.isfinite(output).all()):
            raise RuntimeError(f"dynamic export check failed for batch {batch}")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
