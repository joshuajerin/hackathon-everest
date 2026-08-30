#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from hackathon_everest_isaaclab.learning.models import (
    ACTION_NAMES,
    DEFAULT_PROPOSAL_BOUNDS,
    EVENT_NAMES,
    REGRESSION_NAMES,
    CausalBilateralEstimator,
    VisibleEverestPolicy,
    VisibleOnlySupervisor,
    calibrated_commit_logit_subtraction,
    supervisor_selection_score,
)
from hackathon_everest_isaaclab.learning.safety_priors import (
    DEGRADED_SLIP_MARGIN_N,
    FRACTURE_DAMAGE_CAUTION,
    MINIMUM_BEARING_CAPACITY_N,
    SEVERE_SLIP_MARGIN_N,
    SUPERVISOR_VALIDATION_MAXIMUM_UNSAFE_COMMIT_RATE,
)
from torch import nn

PREFIX_INDICES = (5, 10, 15, 22, 30)
CONTEXT_ORDER = (
    "foot_position_xyz_m[3]",
    "foot_velocity_xyz_mps[3]",
    "pelvis_roll_pitch_yaw_rad[3]",
    "commanded_probe_load_n",
    "commanded_foot_speed_mps",
    "body_weight_on_foot_n",
)
COMMAND_CONTEXT_ORDER = (
    "requested_vx_mps",
    "requested_vy_mps",
    "requested_wz_rps",
    "mode",
    "probe_load_n[left]",
    "probe_load_n[right]",
    "approach_speed_mps[left]",
    "approach_speed_mps[right]",
)


def estimator_selection_score(
    validation: dict[str, float], articulated_validation: dict[str, float] | None = None
) -> float:
    """Rank checkpoints for safety, hazard recall, and articulated liveness."""
    score = (
        validation["bearing_capacity_n_mae"]
        + validation["shear_capacity_n_mae"]
        + 500.0 * validation["support_layer_depth_m_mae"]
        + 100_000.0 * validation["false_safe_rate"]
        + 500.0 * validation["false_unsafe_rate"]
        + 500.0 * (1.0 - validation["void_present_recall"])
        + 500.0 * (1.0 - validation["fractured_recall"])
        + 500.0 * (1.0 - validation["slipping_recall"])
    )
    if articulated_validation is not None:
        score += 100_000.0 * articulated_validation["false_safe_rate"]
        score += 1_000.0 * articulated_validation["false_unsafe_rate"]
    return score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pack_visible_context(root) -> np.ndarray:
    context = root["context"]
    foot_position = context["foot_position_xyz_m"][:]
    foot_velocity = context["foot_velocity_xyz_mps"][:]
    pelvis = context["pelvis_roll_pitch_yaw_rad"][:]
    if pelvis.ndim == 3:
        pelvis = np.repeat(pelvis[:, :, None, :], foot_position.shape[2], axis=2)
    vectors = [foot_position, foot_velocity, pelvis]
    scalars = [
        context["commanded_probe_load_n"][:],
        context["commanded_foot_speed_mps"][:],
        context["body_weight_on_foot_n"][:],
    ]
    return np.concatenate([*vectors, *(value[..., None] for value in scalars)], axis=-1).astype(
        np.float32
    )


def pack_command_context(root) -> np.ndarray:
    commands = root["commands"]
    count, steps, feet = commands["probe_load_n"].shape
    bilateral_scalars = []
    for name in ("requested_vx_mps", "requested_vy_mps", "requested_wz_rps", "mode"):
        value = commands[name][:]
        if value.shape != (count, steps):
            raise RuntimeError(f"Unexpected {name} shape: {value.shape}")
        bilateral_scalars.append(value[..., None])
    return np.concatenate(
        [
            *bilateral_scalars,
            commands["probe_load_n"][:].reshape(count, steps, feet),
            commands["approach_speed_mps"][:].reshape(count, steps, feet),
        ],
        axis=-1,
    ).astype(np.float32)


def stats(value: torch.Tensor, train_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    selected = value.index_select(0, train_indices).float()
    return selected.mean(dim=(0, 1, 2)), selected.std(dim=(0, 1, 2)).clamp_min(1e-5)


def target_stats(
    targets: torch.Tensor, train_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = [targets.index_select(0, train_indices)[:, end] for end in PREFIX_INDICES]
    selected = torch.cat(blocks, dim=0).float()
    return selected.mean(dim=(0, 1)), selected.std(dim=(0, 1)).clamp_min(1e-4)


def estimator_loss(output, target, event, target_std, event_pos_weight):
    normalized_error = (output.regression_mean - target) / target_std
    normalized_log_scale = output.regression_log_scale - torch.log(target_std)
    inverse_variance = torch.exp(-2.0 * normalized_log_scale.clamp(-7.0, 7.0))
    regression = (0.5 * normalized_error.square() * inverse_variance + normalized_log_scale).mean()
    classification = nn.functional.binary_cross_entropy_with_logits(
        output.event_logits, event.float(), pos_weight=event_pos_weight
    )
    return regression + 0.5 * classification, regression, classification


@torch.no_grad()
def evaluate_estimator(model, values, masks, ages, context, targets, events, indices, batch_size):
    model.eval()
    prediction = []
    scale = []
    truth = []
    event_prob = []
    event_truth = []
    for end in PREFIX_INDICES:
        for offset in range(0, len(indices), batch_size):
            batch = indices[offset : offset + batch_size]
            output = model(
                values[batch, : end + 1],
                masks[batch, : end + 1],
                ages[batch, : end + 1],
                context[batch, : end + 1],
            )
            prediction.append(output.regression_mean.cpu())
            scale.append(output.regression_log_scale.exp().cpu())
            truth.append(targets[batch, end].cpu())
            event_prob.append(output.event_logits.sigmoid().cpu())
            event_truth.append(events[batch, end].cpu())
    return tuple(
        torch.cat(value).numpy() for value in (prediction, scale, truth, event_prob, event_truth)
    )


def estimator_metrics(
    model,
    values,
    masks,
    ages,
    context,
    targets,
    events,
    indices,
    batch_size,
    conformal=None,
    bearing_absolute_radius_n=None,
):
    pred, scale, truth, prob, event_truth = evaluate_estimator(
        model, values, masks, ages, context, targets, events, indices, batch_size
    )
    ti = {name: i for i, name in enumerate(REGRESSION_NAMES)}
    ei = {name: i for i, name in enumerate(EVENT_NAMES)}
    result = {}
    for name in (
        "support_layer_depth_m",
        "bearing_capacity_n",
        "shear_capacity_n",
        "slip_margin_n",
    ):
        i = ti[name]
        result[f"{name}_mae"] = float(np.mean(np.abs(pred[..., i] - truth[..., i])))
        if name.endswith("_n"):
            result[f"{name}_median_relative_error"] = float(
                np.median(
                    np.abs(pred[..., i] - truth[..., i]) / np.maximum(np.abs(truth[..., i]), 1.0)
                )
            )
    for name, i in ei.items():
        positive = event_truth[..., i].astype(bool)
        predicted = prob[..., i] >= 0.5
        result[f"{name}_recall"] = float(np.mean(predicted[positive])) if positive.any() else 1.0
    q = 1.96 if conformal is None else np.asarray(conformal)[ti["bearing_capacity_n"]]
    bearing_i = ti["bearing_capacity_n"]
    actual_safe = truth[..., bearing_i] >= 343.0
    bearing_uncertainty = (
        q * scale[..., bearing_i]
        if bearing_absolute_radius_n is None
        else float(bearing_absolute_radius_n)
    )
    predicted_safe = pred[..., bearing_i] - bearing_uncertainty >= 343.0
    result["false_safe_rate"] = (
        float(np.mean(predicted_safe[~actual_safe])) if (~actual_safe).any() else 0.0
    )
    result["false_unsafe_rate"] = (
        float(np.mean(~predicted_safe[actual_safe])) if actual_safe.any() else 0.0
    )
    result["actual_safe_coverage"] = float(np.mean(actual_safe))
    result["predicted_safe_coverage"] = float(np.mean(predicted_safe))
    return result


def make_teacher(targets: torch.Tensor, events: torch.Tensor, command_context: torch.Tensor):
    ti = {name: i for i, name in enumerate(REGRESSION_NAMES)}
    ei = {name: i for i, name in enumerate(EVENT_NAMES)}
    bearing = targets[..., ti["bearing_capacity_n"]]
    damage = targets[..., ti["damage_state"]]
    slip_margin = targets[..., ti["slip_margin_n"]]
    void = events[..., ei["void_present"]].bool().any(dim=-1)
    fracture = events[..., ei["fractured"]].bool().any(dim=-1)
    slip_by_foot = events[..., ei["slipping"]].bool()
    any_slip = slip_by_foot.any(dim=-1)
    bilateral_slip = slip_by_foot.all(dim=-1)
    minimum_bearing = bearing.amin(dim=-1)
    maximum_damage = damage.amax(dim=-1)
    best_stance_slip_margin = slip_margin.amax(dim=-1)
    severe_slip = bilateral_slip & (best_stance_slip_margin < SEVERE_SLIP_MARGIN_N)
    degraded_slip = any_slip & (best_stance_slip_margin < DEGRADED_SLIP_MARGIN_N)
    fracture_caution = fracture & (maximum_damage > FRACTURE_DAMAGE_CAUTION)
    recovery = (
        severe_slip
        | (void & (minimum_bearing < MINIMUM_BEARING_CAPACITY_N))
        | (minimum_bearing < 120.0)
        | (maximum_damage > 0.75)
    )
    hold = (~recovery) & (
        fracture_caution
        | degraded_slip
        | (minimum_bearing < MINIMUM_BEARING_CAPACITY_N)
        | (best_stance_slip_margin < DEGRADED_SLIP_MARGIN_N)
    )
    labels = torch.zeros_like(minimum_bearing, dtype=torch.long)
    labels[hold] = 1
    labels[recovery] = 2
    commit = labels == 0
    proposals = torch.zeros((*labels.shape, len(DEFAULT_PROPOSAL_BOUNDS)), device=targets.device)
    proposals[..., 0] = torch.where(
        commit, torch.ones_like(minimum_bearing), torch.zeros_like(minimum_bearing)
    )
    proposals[..., 1] = proposals[..., 0]
    proposals[..., 2] = command_context[..., 4:6].mean(dim=-1).clamp(0.0, 250.0)
    proposals[..., 3] = command_context[..., 6:8].mean(dim=-1).clamp(0.0, 0.20) * proposals[..., 0]
    proposals[..., 4] = (
        0.04
        + targets[..., ti["support_layer_depth_m"]].amax(dim=-1)
        + targets[..., ti["void_depth_m"]].amax(dim=-1)
    ).clamp(0.0, 0.20)
    proposals[..., 5] = (minimum_bearing / 343.0).clamp(0.0, 1.0) * proposals[..., 0]
    return labels, proposals


@torch.inference_mode()
def supervisor_metrics_from_logits(
    logits: torch.Tensor,
    expected: torch.Tensor,
    *,
    commit_logit_subtraction: float = 0.0,
) -> dict[str, float | dict[str, int]]:
    adjusted = logits.clone()
    adjusted[:, 0] -= commit_logit_subtraction
    predicted = adjusted.argmax(dim=-1)
    unsafe = expected != 0
    teacher_commit = expected == 0
    return {
        "teacher_action_accuracy": float((predicted == expected).float().mean()),
        "unsafe_commit_rate": (
            float((predicted[unsafe] == 0).float().mean()) if bool(unsafe.any()) else 0.0
        ),
        "commit_recall": (
            float((predicted[teacher_commit] == 0).float().mean())
            if bool(teacher_commit.any())
            else 1.0
        ),
        "predicted_commit_coverage": float((predicted == 0).float().mean()),
        "teacher_class_counts": {
            ACTION_NAMES[i]: int((expected == i).sum()) for i in range(len(ACTION_NAMES))
        },
    }


@torch.inference_mode()
def evaluate_supervisor(
    supervisor: VisibleOnlySupervisor,
    latent: torch.Tensor,
    command: torch.Tensor,
    teacher: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    commit_logit_subtraction: float = 0.0,
) -> dict[str, float | dict[str, int]]:
    supervisor.eval()
    logits = supervisor(
        latent.index_select(0, sample_indices),
        command.index_select(0, sample_indices),
    ).action_logits
    expected = teacher.index_select(0, sample_indices)
    return supervisor_metrics_from_logits(
        logits,
        expected,
        commit_logit_subtraction=commit_logit_subtraction,
    )


@torch.inference_mode()
def calibrate_commit_logit_subtraction(
    supervisor: VisibleOnlySupervisor,
    latent: torch.Tensor,
    command: torch.Tensor,
    teacher: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    maximum_unsafe_commit_rate: float = SUPERVISOR_VALIDATION_MAXIMUM_UNSAFE_COMMIT_RATE,
) -> float:
    """Find the smallest conservative commit-logit offset on held-out data."""
    supervisor.eval()
    logits = supervisor(
        latent.index_select(0, sample_indices),
        command.index_select(0, sample_indices),
    ).action_logits
    expected = teacher.index_select(0, sample_indices)
    return calibrated_commit_logit_subtraction(
        logits,
        expected,
        maximum_unsafe_commit_rate=maximum_unsafe_commit_rate,
    )


class EstimatorExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, values, mask, age, context):
        output = self.model(values, mask, age, context)
        return (
            output.regression_mean,
            output.regression_log_scale,
            output.event_logits,
            output.per_foot_latent,
            output.bilateral_latent,
        )


class PolicyExport(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, values, mask, age, context, command_context):
        output = self.policy(values, mask, age, context, command_context)
        return (
            output.estimator.regression_mean,
            output.estimator.regression_log_scale,
            output.estimator.event_logits,
            output.supervisor.action_logits,
            output.supervisor.continuous_proposals,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--aux-shard", type=Path)
    parser.add_argument("--aux-repeat", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--estimator-epochs", type=int, default=40)
    parser.add_argument("--supervisor-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.aux_repeat < 1:
        raise ValueError("aux-repeat must be at least one")

    def load_arrays(shard: Path):
        visible_root = zarr.open_group(str(shard / "visible.zarr"), mode="r")
        truth = zarr.open_group(str(shard / "truth.zarr"), mode="r")
        episodes = pq.read_table(shard / "episodes.parquet")
        contains_reset = (
            np.asarray(episodes["contains_reset"].to_pylist(), dtype=bool)
            if "contains_reset" in episodes.column_names
            else np.zeros(len(episodes), dtype=bool)
        )
        return (
            visible_root["packet_values"][:],
            visible_root["valid_mask"][:],
            visible_root["sample_age_s"][:],
            pack_visible_context(visible_root),
            pack_command_context(visible_root),
            truth["targets"][:],
            truth["events"][:],
            np.asarray(episodes["split"].to_pylist()),
            contains_reset,
        )

    loaded = [load_arrays(args.shard)]
    primary_count = len(loaded[0][0])
    if args.aux_shard is not None:
        loaded.append(load_arrays(args.aux_shard))
    combined = [np.concatenate(parts, axis=0) for parts in zip(*loaded, strict=True)]
    values = torch.from_numpy(combined[0]).to(device)
    masks = torch.from_numpy(combined[1]).to(device)
    ages = torch.from_numpy(combined[2]).to(device)
    context = torch.from_numpy(combined[3]).to(device)
    commands = torch.from_numpy(combined[4]).to(device)
    targets = torch.from_numpy(combined[5]).to(device)
    events = torch.from_numpy(combined[6]).to(device)
    split = combined[7]
    contains_reset = combined[8].astype(bool, copy=False)
    indices = {
        name: torch.from_numpy(np.flatnonzero(split == name)).to(device)
        for name in ("train", "calibration", "validation", "sealed_test")
    }
    auxiliary_total = len(values) - primary_count
    auxiliary_reset_windows = int(contains_reset[primary_count:].sum())
    auxiliary_train = indices["train"][indices["train"] >= primary_count]
    auxiliary_validation = indices["validation"][indices["validation"] >= primary_count]
    # Articulated calibration is disjoint from its sealed test split.
    auxiliary_calibration = indices["calibration"][indices["calibration"] >= primary_count]
    auxiliary_sealed_test = indices["sealed_test"][indices["sealed_test"] >= primary_count]
    if args.aux_shard is not None and (
        len(auxiliary_validation) == 0 or len(auxiliary_sealed_test) == 0
    ):
        raise RuntimeError("Auxiliary shard must contain validation and sealed-test windows")
    if len(auxiliary_train):
        keep = torch.from_numpy(~contains_reset[auxiliary_train.cpu().numpy()]).to(device)
        auxiliary_train = auxiliary_train[keep]
    primary_train = indices["train"][indices["train"] < primary_count]
    non_training = indices["train"][indices["train"] >= primary_count]
    if len(non_training) != len(auxiliary_train):
        # Reset-contaminated auxiliary windows are excluded, never reclassified.
        indices["train"] = torch.cat((primary_train, auxiliary_train))
    if args.aux_shard is not None and args.aux_repeat > 1 and len(auxiliary_train):
        indices["train"] = torch.cat(
            (indices["train"], *([auxiliary_train] * (args.aux_repeat - 1)))
        )
    training_window_counts = {
        "primary_total": primary_count,
        "auxiliary_total": auxiliary_total,
        "auxiliary_reset_windows_excluded": auxiliary_reset_windows,
        "auxiliary_train_unique": len(auxiliary_train),
        "auxiliary_validation": len(auxiliary_validation),
        "auxiliary_sealed_test": len(auxiliary_sealed_test),
        "auxiliary_repeat": args.aux_repeat if args.aux_shard is not None else 0,
        "effective_train_indices": len(indices["train"]),
    }
    source_shards = [args.shard] + ([args.aux_shard] if args.aux_shard is not None else [])
    dataset_sources = [
        {
            "shard": str(shard),
            "manifest_sha256": sha256(shard / "manifest.json"),
            "complete_marker_sha256": sha256(shard / "_COMPLETE"),
        }
        for shard in source_shards
    ]

    input_parts = []
    for tensor in (values, masks.float(), ages, context):
        mean, std = stats(tensor, indices["train"])
        input_parts.append((mean, std))
    input_mean = torch.cat([item[0] for item in input_parts])
    input_std = torch.cat([item[1] for item in input_parts])
    target_mean, target_std = target_stats(targets, indices["train"])
    train_event = torch.cat(
        [events.index_select(0, indices["train"])[:, end] for end in PREFIX_INDICES], dim=0
    ).float()
    positive = train_event.sum(dim=(0, 1))
    negative = train_event.shape[0] * 2 - positive
    event_pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 30.0)
    estimator = CausalBilateralEstimator(
        hidden_size=128,
        bilateral_latent_size=64,
        num_layers=2,
        context_dim=context.shape[-1],
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
    ).to(device)
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=6e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.estimator_epochs, 1)
    )
    history = []
    best = math.inf
    best_state = None
    best_epoch = 0
    started = time.time()
    for epoch in range(args.estimator_epochs):
        estimator.train()
        loss_sum = 0.0
        steps = 0
        prefix_order = torch.randperm(len(PREFIX_INDICES), device=device)
        for prefix_slot in prefix_order.tolist():
            end = PREFIX_INDICES[prefix_slot]
            order = indices["train"][torch.randperm(len(indices["train"]), device=device)]
            for offset in range(0, len(order), args.batch_size):
                batch = order[offset : offset + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                output = estimator(
                    values[batch, : end + 1],
                    masks[batch, : end + 1],
                    ages[batch, : end + 1],
                    context[batch, : end + 1],
                )
                loss, _, _ = estimator_loss(
                    output, targets[batch, end], events[batch, end], target_std, event_pos_weight
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(estimator.parameters(), 5.0)
                optimizer.step()
                loss_sum += float(loss.detach())
                steps += 1
        scheduler.step()
        validation = estimator_metrics(
            estimator,
            values,
            masks,
            ages,
            context,
            targets,
            events,
            indices["validation"],
            args.batch_size,
        )
        articulated_validation = (
            estimator_metrics(
                estimator,
                values,
                masks,
                ages,
                context,
                targets,
                events,
                auxiliary_validation,
                args.batch_size,
            )
            if len(auxiliary_validation)
            else None
        )
        # Safety-critical checkpoint selection must not prefer a lower-MAE model
        # that increases false-safe decisions, misses hazards, or remains inert
        # on held-out articulated gait windows.
        score = estimator_selection_score(validation, articulated_validation)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": loss_sum / max(steps, 1),
                "selection_score": score,
                "validation": validation,
                "articulated_validation": articulated_validation,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if score < best:
            best = score
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone() for name, value in estimator.state_dict().items()
            }
    if best_state is not None:
        estimator.load_state_dict(best_state)
    estimator.eval()
    cal_pred, cal_scale, cal_truth, _, _ = evaluate_estimator(
        estimator,
        values,
        masks,
        ages,
        context,
        targets,
        events,
        auxiliary_calibration if len(auxiliary_calibration) else indices["calibration"],
        args.batch_size,
    )
    absolute_residual = np.abs(cal_pred - cal_truth)
    normalized_residual = absolute_residual / np.maximum(cal_scale, 1e-6)
    conformal95 = np.quantile(normalized_residual, 0.95, axis=(0, 1)).astype(np.float32)
    conformal99 = np.quantile(normalized_residual, 0.99, axis=(0, 1)).astype(np.float32)
    absolute_conformal95 = np.quantile(absolute_residual, 0.95, axis=(0, 1)).astype(np.float32)
    absolute_conformal99 = np.quantile(absolute_residual, 0.99, axis=(0, 1)).astype(np.float32)
    bearing_index = REGRESSION_NAMES.index("bearing_capacity_n")
    bearing_absolute_radius95_n = float(absolute_conformal95[bearing_index])
    bearing_absolute_radius99_n = float(absolute_conformal99[bearing_index])
    sealed_estimator = estimator_metrics(
        estimator,
        values,
        masks,
        ages,
        context,
        targets,
        events,
        indices["sealed_test"],
        args.batch_size,
        conformal99,
        bearing_absolute_radius99_n,
    )
    articulated_sealed_estimator = (
        estimator_metrics(
            estimator,
            values,
            masks,
            ages,
            context,
            targets,
            events,
            auxiliary_sealed_test,
            args.batch_size,
            conformal99,
            bearing_absolute_radius99_n,
        )
        if len(auxiliary_sealed_test)
        else None
    )
    # Precompute causal latents and teacher controls for supervisor training.
    latent_blocks = []
    command_blocks = []
    target_blocks = []
    event_blocks = []
    with torch.no_grad():
        all_indices = torch.arange(len(values), device=device)
        for end in PREFIX_INDICES:
            prefix_latents = []
            for offset in range(0, len(values), args.batch_size):
                batch = all_indices[offset : offset + args.batch_size]
                output = estimator(
                    values[batch, : end + 1],
                    masks[batch, : end + 1],
                    ages[batch, : end + 1],
                    context[batch, : end + 1],
                )
                prefix_latents.append(output.bilateral_latent)
            latent_blocks.append(torch.cat(prefix_latents))
            command_blocks.append(commands[:, end])
            target_blocks.append(targets[:, end])
            event_blocks.append(events[:, end])
    latent_all = torch.cat(latent_blocks)
    command_all = torch.cat(command_blocks)
    target_all = torch.cat(target_blocks)
    event_all = torch.cat(event_blocks)
    teacher_label, teacher_proposal = make_teacher(target_all, event_all, command_all)
    episode_count = len(values)
    sample_train = torch.cat(
        [indices["train"] + prefix * episode_count for prefix in range(len(PREFIX_INDICES))]
    )
    sample_validation = torch.cat(
        [indices["validation"] + prefix * episode_count for prefix in range(len(PREFIX_INDICES))]
    )
    sample_sealed = torch.cat(
        [indices["sealed_test"] + prefix * episode_count for prefix in range(len(PREFIX_INDICES))]
    )
    # Unsafe commits are asymmetric failures. Do not inverse-weight the minority COMMIT class.
    class_weight = torch.tensor((1.0, 1.0, 1.5), device=device)
    supervisor = VisibleOnlySupervisor(
        64, hidden_size=128, command_gait_context_dim=command_all.shape[-1]
    ).to(device)
    supervisor_optimizer = torch.optim.AdamW(supervisor.parameters(), lr=8e-4, weight_decay=1e-5)
    lower = supervisor.proposal_lower
    upper = supervisor.proposal_upper
    supervisor_history = []
    best_supervisor_score = math.inf
    best_supervisor_epoch = 0
    best_supervisor_state = None
    best_commit_logit_subtraction = 0.0
    for epoch in range(args.supervisor_epochs):
        supervisor.train()
        order = sample_train[torch.randperm(len(sample_train), device=device)]
        running = 0.0
        steps = 0
        for offset in range(0, len(order), args.batch_size * 2):
            batch = order[offset : offset + args.batch_size * 2]
            supervisor_optimizer.zero_grad(set_to_none=True)
            output = supervisor(latent_all[batch], command_all[batch])
            classification = nn.functional.cross_entropy(
                output.action_logits, teacher_label[batch], weight=class_weight
            )
            action_probability = output.action_logits.softmax(dim=-1)
            unsafe_mask = teacher_label[batch] != 0
            recovery_mask = teacher_label[batch] == 2
            unsafe_commit_penalty = action_probability[unsafe_mask, 0].mean()
            recovery_commit_penalty = action_probability[recovery_mask, 0].mean()
            proposal_target = (teacher_proposal[batch] - lower) / (upper - lower)
            proposal_pred = (output.continuous_proposals - lower) / (upper - lower)
            proposal_loss = nn.functional.smooth_l1_loss(proposal_pred, proposal_target)
            loss = (
                classification
                + 6.0 * unsafe_commit_penalty
                + 6.0 * recovery_commit_penalty
                + 0.5 * proposal_loss
            )
            loss.backward()
            supervisor_optimizer.step()
            running += float(loss.detach())
            steps += 1
        commit_logit_subtraction = calibrate_commit_logit_subtraction(
            supervisor,
            latent_all,
            command_all,
            teacher_label,
            sample_validation,
        )
        validation = evaluate_supervisor(
            supervisor,
            latent_all,
            command_all,
            teacher_label,
            sample_validation,
            commit_logit_subtraction=commit_logit_subtraction,
        )
        selection_score = supervisor_selection_score(validation)
        supervisor_history.append(
            {
                "epoch": epoch + 1,
                "loss": running / max(steps, 1),
                "selection_score": selection_score,
                "commit_logit_subtraction": commit_logit_subtraction,
                "validation": validation,
            }
        )
        print(json.dumps({"supervisor": supervisor_history[-1]}), flush=True)
        if selection_score < best_supervisor_score:
            best_supervisor_score = selection_score
            best_supervisor_epoch = epoch + 1
            best_commit_logit_subtraction = commit_logit_subtraction
            best_supervisor_state = {
                name: value.detach().cpu().clone()
                for name, value in supervisor.state_dict().items()
            }
    if best_supervisor_state is not None:
        supervisor.load_state_dict(best_supervisor_state)
    with torch.no_grad():
        supervisor.action_head.bias[0].sub_(best_commit_logit_subtraction)
    supervisor_metrics = evaluate_supervisor(
        supervisor, latent_all, command_all, teacher_label, sample_sealed
    )
    policy = VisibleEverestPolicy(estimator, supervisor).to(device).eval()
    checkpoint = {
        "schema_version": "1.0.0",
        "selected_estimator_epoch": best_epoch,
        "estimator_selection_score": best,
        "estimator_selection_policy": "safety_hazard_recall_and_articulated_liveness_penalties",
        "selected_supervisor_epoch": best_supervisor_epoch,
        "supervisor_selection_score": best_supervisor_score,
        "supervisor_selection_policy": "calibrated_unsafe_commit_plus_accuracy_and_commit_recall",
        "supervisor_commit_logit_subtraction": best_commit_logit_subtraction,
        "supervisor_validation_maximum_unsafe_commit_rate": SUPERVISOR_VALIDATION_MAXIMUM_UNSAFE_COMMIT_RATE,
        "training_window_counts": training_window_counts,
        "dataset_sources": dataset_sources,
        "estimator_state_dict": estimator.state_dict(),
        "supervisor_state_dict": supervisor.state_dict(),
        "estimator_config": {
            "hidden_size": 128,
            "bilateral_latent_size": 64,
            "num_layers": 2,
            "context_dim": int(context.shape[-1]),
        },
        "supervisor_config": {
            "bilateral_latent_size": 64,
            "hidden_size": 128,
            "command_gait_context_dim": int(command_all.shape[-1]),
        },
        "regression_names": REGRESSION_NAMES,
        "event_names": EVENT_NAMES,
        "action_names": ACTION_NAMES,
        "context_order": CONTEXT_ORDER,
        "command_context_order": COMMAND_CONTEXT_ORDER,
        "conformal95": conformal95.tolist(),
        "conformal99": conformal99.tolist(),
        "bearing_capacity_absolute_conformal95_n": bearing_absolute_radius95_n,
        "bearing_capacity_absolute_conformal99_n": bearing_absolute_radius99_n,
        "bearing_capacity_conformal_score": "absolute_residual",
        "deployment_conformal_quantile": 0.99,
        "articulated_estimator_sealed_test": articulated_sealed_estimator,
        "claim_boundary": "Synthetic L0 estimator and teacher-derived supervisor. Not hardware validated.",
    }
    checkpoint_path = args.output_dir / "visible_policy.pt"
    torch.save(checkpoint, checkpoint_path)
    example = (values[:1], masks[:1], ages[:1], context[:1], commands[:1, -1])
    torch.jit.trace(
        EstimatorExport(estimator),
        (example[0], example[1], example[2], example[3]),
        check_trace=False,
    ).save(str(args.output_dir / "estimator_jit.pt"))
    torch.jit.trace(PolicyExport(policy), example, check_trace=False).save(
        str(args.output_dir / "visible_policy_jit.pt")
    )
    metrics = {
        "device": str(device),
        "selected_estimator_epoch": best_epoch,
        "estimator_selection_score": best,
        "estimator_selection_policy": "safety_hazard_recall_and_articulated_liveness_penalties",
        "selected_supervisor_epoch": best_supervisor_epoch,
        "supervisor_selection_score": best_supervisor_score,
        "supervisor_selection_policy": "calibrated_unsafe_commit_plus_accuracy_and_commit_recall",
        "supervisor_commit_logit_subtraction": best_commit_logit_subtraction,
        "supervisor_validation_maximum_unsafe_commit_rate": SUPERVISOR_VALIDATION_MAXIMUM_UNSAFE_COMMIT_RATE,
        "training_window_counts": checkpoint["training_window_counts"],
        "dataset_sources": checkpoint["dataset_sources"],
        "duration_s": time.time() - started,
        "estimator_sealed_test": sealed_estimator,
        "articulated_estimator_sealed_test": articulated_sealed_estimator,
        "supervisor_sealed_test": supervisor_metrics,
        "conformal95": dict(zip(REGRESSION_NAMES, conformal95.tolist(), strict=True)),
        "conformal99": dict(zip(REGRESSION_NAMES, conformal99.tolist(), strict=True)),
        "bearing_capacity_absolute_conformal95_n": bearing_absolute_radius95_n,
        "bearing_capacity_absolute_conformal99_n": bearing_absolute_radius99_n,
        "bearing_capacity_conformal_score": "absolute_residual",
        "deployment_conformal_quantile": 0.99,
        "checkpoint_sha256": sha256(checkpoint_path),
        "claim_boundary": "Synthetic L0 results. Full articulated G1 shadow-mode evaluation remains required.",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps({"estimator": history, "supervisor": supervisor_history}, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
