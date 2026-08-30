from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.schema import AXIAL_FORCE_SLICE, PENETRATION_SLICE, SENSOR_CHANNELS
from .policy_blend import SmoothPolicyBlend, SmoothPolicyBlendConfig


@dataclass(frozen=True)
class ContactGatedPolicyCorrectionConfig:
    """Limits for applying a specialist policy only while crampons touch ice."""

    maximum_weight_step: float = 0.05
    maximum_action_residual: float = 0.12

    def blend_config(self) -> SmoothPolicyBlendConfig:
        return SmoothPolicyBlendConfig(
            maximum_weight_step=self.maximum_weight_step,
            maximum_action_residual=self.maximum_action_residual,
        )


def visible_crampon_contact(
    packet_values: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_age_s: torch.Tensor,
    *,
    stale_after_s: float,
) -> torch.Tensor:
    """Return per-environment contact from fresh deployable force/penetration channels.

    This intentionally never reads the simulator wrench.  A probe needs fresh,
    valid axial-force and penetration channels, with both values positive.
    """

    if not isinstance(packet_values, torch.Tensor) or not packet_values.is_floating_point():
        raise TypeError("packet_values must be a floating-point Torch tensor")
    if packet_values.ndim != 3 or packet_values.shape[-2:] != (2, SENSOR_CHANNELS):
        raise ValueError("packet_values must have shape [B, 2, 19]")
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be a bool Torch tensor")
    if tuple(valid_mask.shape) != tuple(packet_values.shape):
        raise ValueError("valid_mask must match packet_values")
    if not isinstance(sample_age_s, torch.Tensor) or not sample_age_s.is_floating_point():
        raise TypeError("sample_age_s must be a floating-point Torch tensor")
    if tuple(sample_age_s.shape) != tuple(packet_values.shape):
        raise ValueError("sample_age_s must match packet_values")
    if (
        valid_mask.device != packet_values.device
        or sample_age_s.device != packet_values.device
    ):
        raise ValueError("packet values, validity, and age must share a device")
    if stale_after_s < 0.0:
        raise ValueError("stale_after_s must be non-negative")

    axial_force = packet_values[..., AXIAL_FORCE_SLICE]
    penetration = packet_values[..., PENETRATION_SLICE]
    fresh = (
        valid_mask[..., AXIAL_FORCE_SLICE]
        & valid_mask[..., PENETRATION_SLICE]
        & (sample_age_s[..., AXIAL_FORCE_SLICE] <= stale_after_s)
        & (sample_age_s[..., PENETRATION_SLICE] <= stale_after_s)
    )
    return (fresh & (axial_force > 0.0) & (penetration > 0.0)).any(dim=(1, 2))


class ContactGatedPolicyCorrection:
    """Blend a trained specialist action into stock control after visible contact.

    The correction is zero before contact. It is slewed in and out after a
    contact edge and clamps the specialist-to-stock joint residual.
    """

    def __init__(self, config: ContactGatedPolicyCorrectionConfig | None = None) -> None:
        self.config = config or ContactGatedPolicyCorrectionConfig()
        self._blend = SmoothPolicyBlend(self.config.blend_config())

    @property
    def weight(self) -> torch.Tensor | None:
        return self._blend.weight

    def reset(self, environment_mask: torch.Tensor | None = None) -> None:
        self._blend.reset(environment_mask)

    def step(
        self,
        stock_action: torch.Tensor,
        specialist_action: torch.Tensor,
        crampon_in_contact: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(crampon_in_contact, torch.Tensor) or crampon_in_contact.dtype != torch.bool:
            raise TypeError("crampon_in_contact must be a bool Torch tensor")
        if stock_action.ndim != 2:
            raise ValueError("stock_action must have shape [B, A]")
        batch = int(stock_action.shape[0])
        if tuple(crampon_in_contact.shape) != (batch,):
            raise ValueError("crampon_in_contact must have shape [B]")
        if crampon_in_contact.device != stock_action.device:
            raise ValueError("crampon_in_contact and stock_action must share a device")
        return self._blend.step(
            stock_action,
            specialist_action,
            crampon_in_contact.to(dtype=stock_action.dtype),
        )
