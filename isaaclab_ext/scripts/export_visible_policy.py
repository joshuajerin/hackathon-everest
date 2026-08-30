#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from hackathon_everest_isaaclab.learning.models import (
    CausalBilateralEstimator,
    VisibleOnlySupervisor,
)
from torch import nn


class DeployableEstimatorCore(nn.Module):
    """Dynamic-shape tensor-only export core; boundary validation stays in the runtime."""

    def __init__(self, source: CausalBilateralEstimator) -> None:
        super().__init__()
        self.context_dim = source.context_dim
        self.input_feature_size = source.input_feature_size
        self.hidden_size = source.hidden_size
        self.maximum_normalized_input = source.maximum_normalized_input
        self.foot_gru = source.foot_gru
        self.regression_mean_head = source.regression_mean_head
        self.regression_log_scale_head = source.regression_log_scale_head
        self.event_head = source.event_head
        self.bilateral_fusion = source.bilateral_fusion
        self.register_buffer("input_mean", source.input_mean.detach().clone())
        self.register_buffer("input_std", source.input_std.detach().clone())
        self.register_buffer("target_mean", source.target_mean.detach().clone())
        self.register_buffer("target_std", source.target_std.detach().clone())

    def forward(
        self,
        packet_history: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_age_s: torch.Tensor,
        deployable_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.cat(
            (
                packet_history,
                valid_mask.to(dtype=packet_history.dtype),
                sample_age_s.to(dtype=packet_history.dtype),
                deployable_context.to(dtype=packet_history.dtype),
            ),
            dim=-1,
        )
        features = (features - self.input_mean) / self.input_std
        features = features.clamp(
            min=-self.maximum_normalized_input,
            max=self.maximum_normalized_input,
        )
        batch = features.size(0)
        time = features.size(1)
        sequence = features.permute(0, 2, 1, 3).reshape(batch * 2, time, self.input_feature_size)
        _, hidden = self.foot_gru(sequence)
        per_foot_latent = hidden[-1].reshape(batch, 2, self.hidden_size)
        normalized_mean = self.regression_mean_head(per_foot_latent)
        normalized_log_scale = self.regression_log_scale_head(per_foot_latent).clamp(-7.0, 3.0)
        regression_mean = normalized_mean * self.target_std + self.target_mean
        regression_log_scale = normalized_log_scale + torch.log(self.target_std)
        event_logits = self.event_head(per_foot_latent)
        bilateral_latent = self.bilateral_fusion(per_foot_latent.reshape(batch, -1))
        return (
            regression_mean,
            regression_log_scale,
            event_logits,
            per_foot_latent,
            bilateral_latent,
        )


class DeployablePolicyCore(nn.Module):
    def __init__(
        self, estimator: CausalBilateralEstimator, supervisor: VisibleOnlySupervisor
    ) -> None:
        super().__init__()
        self.estimator = DeployableEstimatorCore(estimator)
        self.supervisor_trunk = supervisor.trunk
        self.action_head = supervisor.action_head
        self.proposal_head = supervisor.proposal_head
        self.register_buffer("proposal_lower", supervisor.proposal_lower.detach().clone())
        self.register_buffer("proposal_upper", supervisor.proposal_upper.detach().clone())

    def forward(
        self,
        packet_history: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_age_s: torch.Tensor,
        deployable_context: torch.Tensor,
        deployable_command_gait_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_scale, events, _, bilateral = self.estimator(
            packet_history, valid_mask, sample_age_s, deployable_context
        )
        visible = torch.cat((bilateral, deployable_command_gait_context), dim=-1)
        encoded = self.supervisor_trunk(visible)
        action_logits = self.action_head(encoded)
        proposals = self.proposal_lower + (
            self.proposal_upper - self.proposal_lower
        ) * torch.sigmoid(self.proposal_head(encoded))
        return mean, log_scale, events, action_logits, proposals


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    estimator = CausalBilateralEstimator(**checkpoint["estimator_config"])
    estimator.load_state_dict(checkpoint["estimator_state_dict"])
    supervisor = VisibleOnlySupervisor(**checkpoint["supervisor_config"])
    supervisor.load_state_dict(checkpoint["supervisor_state_dict"])
    estimator.eval()
    supervisor.eval()
    estimator_core = DeployableEstimatorCore(estimator).eval()
    policy_core = DeployablePolicyCore(estimator, supervisor).eval()
    scripted_estimator = torch.jit.script(estimator_core)
    scripted_policy = torch.jit.script(policy_core)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    estimator_path = args.output_dir / "estimator_dynamic_jit.pt"
    policy_path = args.output_dir / "visible_policy_dynamic_jit.pt"
    scripted_estimator.save(str(estimator_path))
    scripted_policy.save(str(policy_path))
    # Prove the export is not frozen to one trace batch or history length.
    dynamic_checks = []
    context_dim = int(checkpoint["estimator_config"]["context_dim"])
    command_dim = int(checkpoint["supervisor_config"]["command_gait_context_dim"])
    with torch.no_grad():
        for batch, time in ((1, 31), (7, 13), (32, 6)):
            values = torch.randn(batch, time, 2, 19)
            mask = torch.rand(batch, time, 2, 19) > 0.05
            age = torch.rand(batch, time, 2, 19) * 0.03
            context = torch.randn(batch, time, 2, context_dim)
            command = torch.randn(batch, command_dim)
            outputs = scripted_policy(values, mask, age, context, command)
            expected = ((batch, 2, 11), (batch, 2, 11), (batch, 2, 3), (batch, 3), (batch, 6))
            actual = tuple(tuple(value.shape) for value in outputs)
            if actual != expected or not all(
                bool(torch.isfinite(value).all()) for value in outputs
            ):
                raise RuntimeError(f"Dynamic export check failed: {actual} != {expected}")
            dynamic_checks.append({"batch": batch, "history": time, "shapes": actual})
    manifest = {
        "schema_version": "1.0.0",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "selected_estimator_epoch": checkpoint.get("selected_estimator_epoch"),
        "estimator_selection_score": checkpoint.get("estimator_selection_score"),
        "estimator_selection_policy": checkpoint.get("estimator_selection_policy"),
        "selected_supervisor_epoch": checkpoint.get("selected_supervisor_epoch"),
        "supervisor_selection_score": checkpoint.get("supervisor_selection_score"),
        "supervisor_selection_policy": checkpoint.get("supervisor_selection_policy"),
        "supervisor_commit_logit_subtraction": checkpoint.get(
            "supervisor_commit_logit_subtraction", 0.0
        ),
        "supervisor_validation_maximum_unsafe_commit_rate": checkpoint.get(
            "supervisor_validation_maximum_unsafe_commit_rate"
        ),
        "dataset_sources": checkpoint.get("dataset_sources", []),
        "training_window_counts": checkpoint.get("training_window_counts", {}),
        "estimator_jit_sha256": sha256(estimator_path),
        "policy_jit_sha256": sha256(policy_path),
        "dynamic_shape_checks": dynamic_checks,
        "sensor_abi": "separate [B,T,2,19] values/mask/age tensors; feet are not flattened at the boundary",
        "context_order": checkpoint["context_order"],
        "command_context_order": checkpoint["command_context_order"],
        "regression_names": checkpoint["regression_names"],
        "event_names": checkpoint["event_names"],
        "action_names": checkpoint["action_names"],
        "proposal_names": list(supervisor.proposal_names),
        "deployment_conformal_quantile": checkpoint.get("deployment_conformal_quantile", 0.95),
        "deployment_conformal_multipliers": checkpoint.get(
            "conformal99", checkpoint["conformal95"]
        ),
        "bearing_capacity_conformal_score": checkpoint.get(
            "bearing_capacity_conformal_score", "normalized_residual"
        ),
        "bearing_capacity_absolute_conformal99_n": checkpoint.get(
            "bearing_capacity_absolute_conformal99_n"
        ),
        "claim_boundary": checkpoint["claim_boundary"],
    }
    (args.output_dir / "deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
