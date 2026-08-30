from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

FOOT_COUNT = 2
SENSOR_CHANNELS = 19
REGRESSION_DIM = 11
EVENT_DIM = 3

REGRESSION_NAMES = (
    "support_layer_depth_m",
    "effective_vertical_stiffness_n_per_m",
    "effective_vertical_damping_ns_per_m",
    "bearing_capacity_n",
    "shear_capacity_n",
    "effective_friction",
    "compaction_state",
    "damage_state",
    "fracture_margin_n",
    "slip_margin_n",
    "void_depth_m",
)
EVENT_NAMES = ("void_present", "fractured", "slipping")
ACTION_NAMES = ("COMMIT", "HOLD_DOUBLE_SUPPORT", "REQUEST_RECOVERY")
DEFAULT_PROPOSAL_BOUNDS = (
    ("velocity_scale", 0.0, 1.0),
    ("yaw_scale", 0.0, 1.0),
    ("probe_load_n", 0.0, 250.0),
    ("approach_speed_mps", 0.0, 0.20),
    ("clearance_m", 0.0, 0.20),
    ("transfer_rate_scale", 0.0, 1.0),
)


def calibrated_commit_logit_subtraction(
    logits: torch.Tensor,
    expected_actions: torch.Tensor,
    *,
    maximum_unsafe_commit_rate: float = 0.0015,
) -> float:
    """Return the smallest COMMIT-logit offset meeting a held-out unsafe-rate bound."""
    if logits.ndim != 2 or logits.shape[-1] != len(ACTION_NAMES):
        raise ValueError("logits must have shape [N,3]")
    if expected_actions.shape != logits.shape[:1] or expected_actions.dtype != torch.int64:
        raise ValueError("expected_actions must be int64 with shape [N]")
    if not 0.0 <= maximum_unsafe_commit_rate < 1.0:
        raise ValueError("maximum_unsafe_commit_rate must be in [0,1)")
    unsafe_logits = logits[expected_actions != 0]
    if len(unsafe_logits) == 0:
        return 0.0
    unsafe_margin = unsafe_logits[:, 0] - unsafe_logits[:, 1:].amax(dim=-1)
    allowed = int(maximum_unsafe_commit_rate * len(unsafe_margin))
    positive = unsafe_margin[unsafe_margin > 0.0]
    if len(positive) <= allowed:
        return 0.0
    descending = positive.sort(descending=True).values
    threshold = descending[allowed]
    return float(torch.nextafter(threshold, torch.full_like(threshold, float("inf"))))


def supervisor_selection_score(metrics: Mapping[str, float]) -> float:
    """Prefer low unsafe-commit rate without collapsing to a never-commit policy."""
    return (
        100_000.0 * metrics["unsafe_commit_rate"]
        + 100.0 * (1.0 - metrics["teacher_action_accuracy"])
        + 100.0 * (1.0 - metrics["commit_recall"])
    )


@dataclass(frozen=True)
class EstimatorOutput:
    """Auditable last-prefix estimates from a bilateral history."""

    regression_mean: torch.Tensor
    regression_log_scale: torch.Tensor
    event_logits: torch.Tensor
    per_foot_latent: torch.Tensor
    bilateral_latent: torch.Tensor


@dataclass(frozen=True)
class SupervisorOutput:
    """Visible-only action preferences and bounded command proposals."""

    action_logits: torch.Tensor
    continuous_proposals: torch.Tensor

    @property
    def proposals(self) -> torch.Tensor:
        """Short alias retained for callers that do not need proposal names."""

        return self.continuous_proposals


@dataclass(frozen=True)
class EverestPolicyOutput:
    estimator: EstimatorOutput
    supervisor: SupervisorOutput


def _normalization_vector(
    value: torch.Tensor | Sequence[float] | None,
    *,
    width: int,
    default: float,
    name: str,
) -> torch.Tensor:
    result = (
        torch.full((width,), default, dtype=torch.float32)
        if value is None
        else torch.as_tensor(value, dtype=torch.float32)
    )
    if tuple(result.shape) != (width,):
        raise ValueError(f"{name} must have shape ({width},)")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return result


class CausalBilateralEstimator(nn.Module):
    """Shared causal per-foot GRU followed by bilateral fusion.

    The inference ABI contains only deployable tensors: packet values, validity,
    per-channel sample age, and an explicitly packed deployable context. The
    first three keep exact ``[B, T, 2, 19]`` layouts. Context is
    ``[B, T, 2, context_dim]`` in a caller-owned, documented allowlist order.
    Features are concatenated per foot; feet are folded into the batch only while
    the one shared GRU is evaluated.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        bilateral_latent_size: int = 32,
        num_layers: int = 1,
        context_dim: int = 0,
        input_mean: torch.Tensor | Sequence[float] | None = None,
        input_std: torch.Tensor | Sequence[float] | None = None,
        target_mean: torch.Tensor | Sequence[float] | None = None,
        target_std: torch.Tensor | Sequence[float] | None = None,
        normalize_inputs: bool = True,
        minimum_log_scale: float = -7.0,
        maximum_log_scale: float = 3.0,
        maximum_normalized_input: float = 12.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or bilateral_latent_size <= 0 or num_layers <= 0:
            raise ValueError("hidden sizes and num_layers must be positive")
        if context_dim < 0:
            raise ValueError("context_dim must be non-negative")
        if minimum_log_scale >= maximum_log_scale:
            raise ValueError("minimum_log_scale must be smaller than maximum_log_scale")
        if maximum_normalized_input <= 0.0:
            raise ValueError("maximum_normalized_input must be positive")

        self.context_dim = int(context_dim)
        self.input_feature_size = 3 * SENSOR_CHANNELS + self.context_dim
        mean = _normalization_vector(
            input_mean, width=self.input_feature_size, default=0.0, name="input_mean"
        )
        std = _normalization_vector(
            input_std, width=self.input_feature_size, default=1.0, name="input_std"
        )
        if bool((std <= 0.0).any().item()):
            raise ValueError("input_std entries must be positive")
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std)
        output_mean = _normalization_vector(
            target_mean, width=REGRESSION_DIM, default=0.0, name="target_mean"
        )
        output_std = _normalization_vector(
            target_std, width=REGRESSION_DIM, default=1.0, name="target_std"
        )
        if bool((output_std <= 0.0).any().item()):
            raise ValueError("target_std entries must be positive")
        self.register_buffer("target_mean", output_mean)
        self.register_buffer("target_std", output_std)
        self.normalize_inputs = bool(normalize_inputs)
        self.hidden_size = int(hidden_size)
        self.bilateral_latent_size = int(bilateral_latent_size)
        self.minimum_log_scale = float(minimum_log_scale)
        self.maximum_log_scale = float(maximum_log_scale)
        self.maximum_normalized_input = float(maximum_normalized_input)

        # One module is deliberately shared across LEFT and RIGHT.
        self.foot_gru = nn.GRU(
            self.input_feature_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.regression_mean_head = nn.Linear(hidden_size, REGRESSION_DIM)
        self.regression_log_scale_head = nn.Linear(hidden_size, REGRESSION_DIM)
        self.event_head = nn.Linear(hidden_size, EVENT_DIM)
        self.bilateral_fusion = nn.Sequential(
            nn.Linear(FOOT_COUNT * hidden_size, bilateral_latent_size),
            nn.Tanh(),
        )

    def set_input_normalization(
        self,
        mean: torch.Tensor | Sequence[float],
        std: torch.Tensor | Sequence[float],
    ) -> None:
        """Update deployable normalization buffers without changing the ABI."""

        new_mean = _normalization_vector(
            mean, width=self.input_feature_size, default=0.0, name="mean"
        )
        new_std = _normalization_vector(std, width=self.input_feature_size, default=1.0, name="std")
        if bool((new_std <= 0.0).any().item()):
            raise ValueError("std entries must be positive")
        self.input_mean.copy_(new_mean.to(device=self.input_mean.device))
        self.input_std.copy_(new_std.to(device=self.input_std.device))

    def set_target_normalization(
        self,
        mean: torch.Tensor | Sequence[float],
        std: torch.Tensor | Sequence[float],
    ) -> None:
        new_mean = _normalization_vector(mean, width=REGRESSION_DIM, default=0.0, name="mean")
        new_std = _normalization_vector(std, width=REGRESSION_DIM, default=1.0, name="std")
        if bool((new_std <= 0.0).any().item()):
            raise ValueError("std entries must be positive")
        self.target_mean.copy_(new_mean.to(device=self.target_mean.device))
        self.target_std.copy_(new_std.to(device=self.target_std.device))

    @staticmethod
    def _validate_history(packet_history: torch.Tensor) -> tuple[int, int]:
        if not isinstance(packet_history, torch.Tensor):
            raise TypeError("packet_history must be a Torch tensor")
        if packet_history.ndim != 4 or tuple(packet_history.shape[-2:]) != (
            FOOT_COUNT,
            SENSOR_CHANNELS,
        ):
            raise ValueError("packet_history must have exact shape [B, T, 2, 19]")
        batch, time = int(packet_history.shape[0]), int(packet_history.shape[1])
        if batch <= 0 or time <= 0:
            raise ValueError("packet_history batch and time dimensions must be non-empty")
        if not packet_history.is_floating_point():
            raise TypeError("packet_history must use a floating-point dtype")
        if not bool(torch.isfinite(packet_history).all().item()):
            raise ValueError("packet_history contains non-finite values")
        return batch, time

    def forward(
        self,
        packet_history: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_age_s: torch.Tensor,
        deployable_context: torch.Tensor | None = None,
    ) -> EstimatorOutput:
        batch, time = self._validate_history(packet_history)
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be a bool Torch tensor")
        if tuple(valid_mask.shape) != tuple(packet_history.shape):
            raise ValueError("valid_mask must have exact shape [B, T, 2, 19]")
        if valid_mask.device != packet_history.device:
            raise ValueError("valid_mask and packet_history must share a device")
        if not isinstance(sample_age_s, torch.Tensor):
            raise TypeError("sample_age_s must be a Torch tensor")
        if tuple(sample_age_s.shape) != tuple(packet_history.shape):
            raise ValueError("sample_age_s must have exact shape [B, T, 2, 19]")
        if not sample_age_s.is_floating_point():
            raise TypeError("sample_age_s must use a floating-point dtype")
        if sample_age_s.device != packet_history.device:
            raise ValueError("sample_age_s and packet_history must share a device")
        if not bool(torch.isfinite(sample_age_s).all().item()) or bool(
            (sample_age_s < 0.0).any().item()
        ):
            raise ValueError("sample_age_s must contain finite non-negative values")

        context_shape = (batch, time, FOOT_COUNT, self.context_dim)
        if deployable_context is None:
            if self.context_dim != 0:
                raise ValueError("deployable_context is required when context_dim is non-zero")
            deployable_context = packet_history.new_empty(context_shape)
        else:
            if not isinstance(deployable_context, torch.Tensor):
                raise TypeError("deployable_context must be a Torch tensor")
            if tuple(deployable_context.shape) != context_shape:
                raise ValueError(f"deployable_context must have shape {context_shape}")
            if not deployable_context.is_floating_point():
                raise TypeError("deployable_context must use a floating-point dtype")
            if deployable_context.device != packet_history.device:
                raise ValueError("deployable_context and packet_history must share a device")
            if not bool(torch.isfinite(deployable_context).all().item()):
                raise ValueError("deployable_context contains non-finite values")

        features = torch.cat(
            (
                packet_history,
                valid_mask.to(dtype=packet_history.dtype),
                sample_age_s.to(dtype=packet_history.dtype),
                deployable_context.to(dtype=packet_history.dtype),
            ),
            dim=-1,
        )
        if not bool(torch.isfinite(features).all().item()):
            raise ValueError("visible features overflowed the packet dtype")
        if self.normalize_inputs:
            features = (features - self.input_mean) / self.input_std
            if not bool(torch.isfinite(features).all().item()):
                raise ValueError("normalized visible features are non-finite")
            features = features.clamp(
                min=-self.maximum_normalized_input,
                max=self.maximum_normalized_input,
            )

        per_foot_sequences = features.permute(0, 2, 1, 3).reshape(
            batch * FOOT_COUNT, time, self.input_feature_size
        )
        _, hidden = self.foot_gru(per_foot_sequences)
        per_foot_latent = hidden[-1].reshape(batch, FOOT_COUNT, self.hidden_size)

        normalized_mean = self.regression_mean_head(per_foot_latent)
        normalized_log_scale = self.regression_log_scale_head(per_foot_latent).clamp(
            min=self.minimum_log_scale,
            max=self.maximum_log_scale,
        )
        regression_mean = normalized_mean * self.target_std + self.target_mean
        regression_log_scale = normalized_log_scale + torch.log(self.target_std)
        event_logits = self.event_head(per_foot_latent)
        bilateral_latent = self.bilateral_fusion(per_foot_latent.reshape(batch, -1))
        return EstimatorOutput(
            regression_mean=regression_mean,
            regression_log_scale=regression_log_scale,
            event_logits=event_logits,
            per_foot_latent=per_foot_latent,
            bilateral_latent=bilateral_latent,
        )


class VisibleOnlySupervisor(nn.Module):
    """Student supervisor whose forward API contains no privileged data plane.

    The optional command/gait context is a packed deployable tensor with shape
    ``[B, command_gait_context_dim]``. Proposal bounds are registered buffers, so
    the exact train/export bounds travel with the state dict. ``REQUEST_RECOVERY`` asks a validated upstream
    controller for help; it does not claim an exact foot replant.
    """

    def __init__(
        self,
        bilateral_latent_size: int = 32,
        *,
        hidden_size: int = 64,
        command_gait_context_dim: int = 0,
        proposal_bounds: Sequence[tuple[str, float, float]] = DEFAULT_PROPOSAL_BOUNDS,
    ) -> None:
        super().__init__()
        if bilateral_latent_size <= 0 or hidden_size <= 0:
            raise ValueError("bilateral_latent_size and hidden_size must be positive")
        if command_gait_context_dim < 0:
            raise ValueError("command_gait_context_dim must be non-negative")
        if not proposal_bounds:
            raise ValueError("proposal_bounds cannot be empty")
        names: list[str] = []
        lower: list[float] = []
        upper: list[float] = []
        for name, low, high in proposal_bounds:
            if not name or name in names:
                raise ValueError("proposal names must be non-empty and unique")
            bound = torch.tensor((float(low), float(high)), dtype=torch.float32)
            if not bool(torch.isfinite(bound).all().item()):
                raise ValueError(f"proposal bounds for {name} must be finite float32 values")
            if not bool((bound[0] < bound[1]).item()):
                raise ValueError(f"proposal bound for {name} must satisfy low < high")
            names.append(str(name))
            lower.append(float(bound[0].item()))
            upper.append(float(bound[1].item()))
        self.bilateral_latent_size = int(bilateral_latent_size)
        self.command_gait_context_dim = int(command_gait_context_dim)
        self.proposal_names = tuple(names)
        self.register_buffer("proposal_lower", torch.tensor(lower, dtype=torch.float32))
        self.register_buffer("proposal_upper", torch.tensor(upper, dtype=torch.float32))
        self.trunk = nn.Sequential(
            nn.Linear(bilateral_latent_size + self.command_gait_context_dim, hidden_size),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden_size, len(ACTION_NAMES))
        self.proposal_head = nn.Linear(hidden_size, len(names))

    def forward(
        self,
        bilateral_latent: torch.Tensor,
        deployable_command_gait_context: torch.Tensor | None = None,
    ) -> SupervisorOutput:
        if not isinstance(bilateral_latent, torch.Tensor):
            raise TypeError("bilateral_latent must be a Torch tensor")
        if bilateral_latent.ndim != 2 or bilateral_latent.shape[-1] != self.bilateral_latent_size:
            raise ValueError(f"bilateral_latent must have shape [B, {self.bilateral_latent_size}]")
        if not bilateral_latent.is_floating_point():
            raise TypeError("bilateral_latent must use a floating-point dtype")
        if not bool(torch.isfinite(bilateral_latent).all().item()):
            raise ValueError("bilateral_latent contains non-finite values")

        context_shape = (int(bilateral_latent.shape[0]), self.command_gait_context_dim)
        if deployable_command_gait_context is None:
            if self.command_gait_context_dim != 0:
                raise ValueError(
                    "deployable_command_gait_context is required when "
                    "command_gait_context_dim is non-zero"
                )
            deployable_command_gait_context = bilateral_latent.new_empty(context_shape)
        else:
            if not isinstance(deployable_command_gait_context, torch.Tensor):
                raise TypeError("deployable_command_gait_context must be a Torch tensor")
            if tuple(deployable_command_gait_context.shape) != context_shape:
                raise ValueError(f"deployable_command_gait_context must have shape {context_shape}")
            if not deployable_command_gait_context.is_floating_point():
                raise TypeError("deployable_command_gait_context must use a floating-point dtype")
            if deployable_command_gait_context.device != bilateral_latent.device:
                raise ValueError(
                    "deployable_command_gait_context and bilateral_latent must share a device"
                )
            if not bool(torch.isfinite(deployable_command_gait_context).all().item()):
                raise ValueError("deployable_command_gait_context contains non-finite values")
        visible_features = torch.cat(
            (
                bilateral_latent,
                deployable_command_gait_context.to(dtype=bilateral_latent.dtype),
            ),
            dim=-1,
        )
        if not bool(torch.isfinite(visible_features).all().item()):
            raise ValueError("supervisor visible features overflowed the latent dtype")
        encoded = self.trunk(visible_features)
        raw_proposals = self.proposal_head(encoded)
        proposals = self.proposal_lower + (
            self.proposal_upper - self.proposal_lower
        ) * torch.sigmoid(raw_proposals)
        return SupervisorOutput(
            action_logits=self.action_head(encoded),
            continuous_proposals=proposals,
        )


class VisibleEverestPolicy(nn.Module):
    """Composition whose forward ABI exposes deployable tensors only."""

    def __init__(
        self,
        estimator: CausalBilateralEstimator | None = None,
        supervisor: VisibleOnlySupervisor | None = None,
    ) -> None:
        super().__init__()
        self.estimator = estimator or CausalBilateralEstimator()
        self.supervisor = supervisor or VisibleOnlySupervisor(self.estimator.bilateral_latent_size)
        if self.supervisor.bilateral_latent_size != self.estimator.bilateral_latent_size:
            raise ValueError("estimator and supervisor bilateral latent sizes must match")

    def forward(
        self,
        packet_history: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_age_s: torch.Tensor,
        deployable_context: torch.Tensor | None = None,
        deployable_command_gait_context: torch.Tensor | None = None,
    ) -> EverestPolicyOutput:
        estimates = self.estimator(packet_history, valid_mask, sample_age_s, deployable_context)
        proposals = self.supervisor(estimates.bilateral_latent, deployable_command_gait_context)
        return EverestPolicyOutput(estimator=estimates, supervisor=proposals)


# Descriptive aliases for training scripts while keeping one implementation.
CausalFootGRUEstimator = CausalBilateralEstimator
VisibleSupervisor = VisibleOnlySupervisor
