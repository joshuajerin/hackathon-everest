from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

# These are the only stock-policy joints a terrain policy may alter by default.
# The order is the learned policy output ABI and must be retained in an export manifest.
DEFAULT_TERRAIN_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
)
DEFAULT_MAXIMUM_RESIDUAL_RAD = (0.05, 0.06, 0.08, 0.05, 0.05, 0.06, 0.08, 0.05, 0.04)


@dataclass(frozen=True)
class SensorJointResidualConfig:
    """Physical bounds for sensor-conditioned lower-body position corrections.

    ``maximum_residual_rad`` is expressed after the stock action is transformed
    into a joint-position target.  It is deliberately not a bound in the stock
    policy's normalized action space.
    """

    joint_names: tuple[str, ...] = DEFAULT_TERRAIN_JOINT_NAMES
    maximum_residual_rad: tuple[float, ...] = DEFAULT_MAXIMUM_RESIDUAL_RAD
    maximum_target_step_rad: float = 0.015

    def __post_init__(self) -> None:
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        if len(self.maximum_residual_rad) != len(self.joint_names):
            raise ValueError("maximum_residual_rad must have one entry per joint")
        if any(value <= 0.0 for value in self.maximum_residual_rad):
            raise ValueError("maximum_residual_rad entries must be positive")
        if self.maximum_target_step_rad <= 0.0:
            raise ValueError("maximum_target_step_rad must be positive")


@dataclass(frozen=True)
class SensorJointResidualOutput:
    """Safe raw action and the corresponding physical joint targets."""

    action: torch.Tensor
    joint_targets_rad: torch.Tensor
    applied_residual_rad: torch.Tensor


class SensorGatedJointResidual:
    """Apply a bounded sensor-policy residual after stock action scaling.

    The sensor policy provides one unconstrained value per configured joint. It
    is converted with ``limit * tanh(raw)`` into radians, rate limited, clamped
    to the actual joint limits, and converted back to the stock action ABI.
    Any false ``enabled`` row fails closed: its correction is removed before
    the final action is returned.

    The caller supplies action scale, offset, and physical limits from the exact
    deployed ``joint_pos`` action term.  This prevents guessing a 37-action G1
    joint ordering or treating normalized stock actions as radians.
    """

    def __init__(self, config: SensorJointResidualConfig | None = None) -> None:
        self.config = config or SensorJointResidualConfig()
        self._applied_residual_rad: torch.Tensor | None = None

    @property
    def applied_residual_rad(self) -> torch.Tensor | None:
        return self._applied_residual_rad

    def reset(self, environment_mask: torch.Tensor | None = None) -> None:
        if environment_mask is None:
            self._applied_residual_rad = None
            return
        if self._applied_residual_rad is None:
            raise RuntimeError("cannot selectively reset before the first residual step")
        if not isinstance(environment_mask, torch.Tensor) or environment_mask.dtype != torch.bool:
            raise TypeError("environment_mask must be a bool Torch tensor")
        if tuple(environment_mask.shape) != tuple(self._applied_residual_rad.shape[:1]):
            raise ValueError("environment_mask must have shape [B]")
        if environment_mask.device != self._applied_residual_rad.device:
            raise ValueError("environment_mask and residual state must share a device")
        self._applied_residual_rad = torch.where(
            environment_mask.unsqueeze(-1),
            torch.zeros_like(self._applied_residual_rad),
            self._applied_residual_rad,
        )

    @staticmethod
    def _vector(
        value: torch.Tensor,
        *,
        name: str,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point Torch tensor")
        if tuple(value.shape) != (action_dim,):
            raise ValueError(f"{name} must have shape [{action_dim}]")
        if value.device != device:
            raise ValueError(f"{name} and stock_action must share a device")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must be finite")
        return value.to(dtype=dtype)

    @torch.inference_mode()
    def step(
        self,
        stock_action: torch.Tensor,
        raw_residual: torch.Tensor,
        *,
        action_joint_names: Sequence[str],
        action_scale: torch.Tensor,
        action_offset: torch.Tensor,
        joint_position_lower_rad: torch.Tensor,
        joint_position_upper_rad: torch.Tensor,
        enabled: torch.Tensor,
    ) -> SensorJointResidualOutput:
        """Return an action vector with safe, named terrain joint corrections.

        ``action_joint_names`` is the exact deployed action order. ``action_scale``
        and ``action_offset`` implement the stock action term's
        ``q_target = action_scale * action + action_offset`` transform.
        """

        if not isinstance(stock_action, torch.Tensor) or not stock_action.is_floating_point():
            raise TypeError("stock_action must be a floating-point Torch tensor")
        if stock_action.ndim != 2 or int(stock_action.shape[0]) <= 0:
            raise ValueError("stock_action must have shape [B, A]")
        if not bool(torch.isfinite(stock_action).all().item()):
            raise ValueError("stock_action must be finite")
        batch, action_dim = stock_action.shape
        if not isinstance(raw_residual, torch.Tensor) or not raw_residual.is_floating_point():
            raise TypeError("raw_residual must be a floating-point Torch tensor")
        count = len(self.config.joint_names)
        if tuple(raw_residual.shape) != (batch, count):
            raise ValueError(f"raw_residual must have shape [{batch}, {count}]")
        if raw_residual.device != stock_action.device or not bool(
            torch.isfinite(raw_residual).all().item()
        ):
            raise ValueError("raw_residual must be finite and share the stock action device")
        if (
            isinstance(action_joint_names, (str, bytes))
            or not isinstance(action_joint_names, Sequence)
            or not all(isinstance(name, str) and name for name in action_joint_names)
        ):
            raise TypeError("action_joint_names must be a sequence of non-empty strings")
        if len(action_joint_names) != action_dim:
            raise ValueError(f"action_joint_names must have {action_dim} entries")
        if len(set(action_joint_names)) != action_dim:
            raise ValueError("action_joint_names must be unique")
        index_by_name = {name: index for index, name in enumerate(action_joint_names)}
        missing = [name for name in self.config.joint_names if name not in index_by_name]
        if missing:
            raise ValueError(
                f"configured residual joints are missing from the action ABI: {missing}"
            )
        joint_indices = torch.tensor(
            [index_by_name[name] for name in self.config.joint_names],
            dtype=torch.long,
            device=stock_action.device,
        )
        if not isinstance(enabled, torch.Tensor) or enabled.dtype != torch.bool:
            raise TypeError("enabled must be a bool Torch tensor")
        if tuple(enabled.shape) != (batch,) or enabled.device != stock_action.device:
            raise ValueError("enabled must have shape [B] on the action device")

        scale = self._vector(
            action_scale,
            name="action_scale",
            action_dim=action_dim,
            device=stock_action.device,
            dtype=stock_action.dtype,
        )
        if bool((scale == 0.0).any().item()):
            raise ValueError("action_scale must not contain zero")
        offset = self._vector(
            action_offset,
            name="action_offset",
            action_dim=action_dim,
            device=stock_action.device,
            dtype=stock_action.dtype,
        )
        lower = self._vector(
            joint_position_lower_rad,
            name="joint_position_lower_rad",
            action_dim=action_dim,
            device=stock_action.device,
            dtype=stock_action.dtype,
        )
        upper = self._vector(
            joint_position_upper_rad,
            name="joint_position_upper_rad",
            action_dim=action_dim,
            device=stock_action.device,
            dtype=stock_action.dtype,
        )
        if bool((lower > upper).any().item()):
            raise ValueError("joint position lower limits must not exceed upper limits")

        limits = stock_action.new_tensor(self.config.maximum_residual_rad)
        requested = limits * torch.tanh(raw_residual)
        if self._applied_residual_rad is None:
            self._applied_residual_rad = torch.zeros_like(requested)
        elif tuple(self._applied_residual_rad.shape) != tuple(requested.shape):
            raise ValueError("residual batch changed; call reset before reuse")
        elif self._applied_residual_rad.device != stock_action.device:
            raise ValueError("residual device changed; call reset before reuse")

        next_residual = self._applied_residual_rad + (requested - self._applied_residual_rad).clamp(
            -self.config.maximum_target_step_rad, self.config.maximum_target_step_rad
        )
        # A stale/unsafe row must not retain its previous joint correction.
        next_residual = torch.where(
            enabled.unsqueeze(-1), next_residual, torch.zeros_like(next_residual)
        )

        stock_targets = stock_action * scale + offset
        selected_stock_targets = stock_targets.index_select(dim=1, index=joint_indices)
        selected_lower = lower.index_select(dim=0, index=joint_indices)
        selected_upper = upper.index_select(dim=0, index=joint_indices)
        selected_targets = (selected_stock_targets + next_residual).clamp(
            min=selected_lower, max=selected_upper
        )
        applied = selected_targets - selected_stock_targets
        self._applied_residual_rad = applied.detach().clone()

        corrected_action = stock_action.clone()
        corrected_values = (
            selected_targets - offset.index_select(dim=0, index=joint_indices)
        ) / scale.index_select(dim=0, index=joint_indices)
        corrected_action[:, joint_indices] = corrected_values
        joint_targets = stock_targets.clone()
        joint_targets[:, joint_indices] = selected_targets
        return SensorJointResidualOutput(
            action=corrected_action,
            joint_targets_rad=joint_targets,
            applied_residual_rad=applied.clone(),
        )
