from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SmoothPolicyBlendConfig:
    """Bounds for visually smooth transitions from the proven stock policy."""

    maximum_weight_step: float = 0.05
    maximum_action_residual: float = 0.35

    def __post_init__(self) -> None:
        if not 0.0 < self.maximum_weight_step <= 1.0:
            raise ValueError("maximum_weight_step must be in (0, 1]")
        if self.maximum_action_residual <= 0.0:
            raise ValueError("maximum_action_residual must be positive")


class SmoothPolicyBlend:
    """Stateful, bounded and eased mixture of stock and specialist joint actions."""

    def __init__(self, config: SmoothPolicyBlendConfig | None = None) -> None:
        self.config = SmoothPolicyBlendConfig() if config is None else config
        self._weight: torch.Tensor | None = None

    @property
    def weight(self) -> torch.Tensor | None:
        return self._weight

    def reset(self, environment_mask: torch.Tensor | None = None) -> None:
        if environment_mask is None:
            self._weight = None
            return
        if self._weight is None:
            raise RuntimeError("cannot selectively reset before the first blend step")
        if not isinstance(environment_mask, torch.Tensor) or environment_mask.dtype != torch.bool:
            raise TypeError("environment_mask must be a bool Torch tensor")
        if tuple(environment_mask.shape) != tuple(self._weight.shape):
            raise ValueError("environment_mask must match the blend batch")
        if environment_mask.device != self._weight.device:
            raise ValueError("environment_mask and blend state must share a device")
        self._weight = torch.where(environment_mask, torch.zeros_like(self._weight), self._weight)

    def step(
        self,
        stock_action: torch.Tensor,
        specialist_action: torch.Tensor,
        target_weight: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(stock_action, torch.Tensor) or not stock_action.is_floating_point():
            raise TypeError("stock_action must be a floating-point Torch tensor")
        if (
            not isinstance(specialist_action, torch.Tensor)
            or not specialist_action.is_floating_point()
        ):
            raise TypeError("specialist_action must be a floating-point Torch tensor")
        if stock_action.ndim != 2 or tuple(specialist_action.shape) != tuple(stock_action.shape):
            raise ValueError("stock and specialist actions must share shape [B, A]")
        if specialist_action.device != stock_action.device:
            raise ValueError("stock and specialist actions must share a device")
        batch = int(stock_action.shape[0])
        if not isinstance(target_weight, torch.Tensor) or not target_weight.is_floating_point():
            raise TypeError("target_weight must be a floating-point Torch tensor")
        if tuple(target_weight.shape) != (batch,) or target_weight.device != stock_action.device:
            raise ValueError("target_weight must have shape [B] on the action device")
        if not bool(
            torch.isfinite(stock_action).all()
            and torch.isfinite(specialist_action).all()
            and torch.isfinite(target_weight).all()
        ):
            raise ValueError("blend inputs must be finite")
        if bool(((target_weight < 0.0) | (target_weight > 1.0)).any()):
            raise ValueError("target_weight must be in [0, 1]")
        if self._weight is None:
            self._weight = torch.zeros(batch, dtype=stock_action.dtype, device=stock_action.device)
        elif tuple(self._weight.shape) != (batch,) or self._weight.device != stock_action.device:
            raise ValueError("blend batch or device changed; call reset before reuse")

        maximum_step = self.config.maximum_weight_step
        self._weight = self._weight + (target_weight - self._weight).clamp(
            -maximum_step, maximum_step
        )
        eased_weight = self._weight.square() * (3.0 - 2.0 * self._weight)
        residual = (specialist_action - stock_action).clamp(
            -self.config.maximum_action_residual,
            self.config.maximum_action_residual,
        )
        return stock_action + eased_weight.unsqueeze(-1) * residual
