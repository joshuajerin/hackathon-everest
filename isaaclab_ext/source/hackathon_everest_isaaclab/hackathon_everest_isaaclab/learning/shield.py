from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from hackathon_everest_isaaclab.learning.safety_priors import (
    FRACTURE_DAMAGE_CAUTION,
    MINIMUM_BEARING_CAPACITY_N,
    SEVERE_SLIP_MARGIN_N,
)


class ShieldAction(IntEnum):
    """Final high-level action. Recovery is a request, never an exact-replant claim."""

    COMMIT = 0
    HOLD_DOUBLE_SUPPORT = 1
    REQUEST_RECOVERY = 2


class ShieldReason(IntEnum):
    SAFE_TO_COMMIT = 0
    STALE_OR_OOD = 1
    UNSAFE_TARGET = 2
    UNSAFE_STANCE_OR_SETTLING = 3
    RELEASE_HYSTERESIS = 4


@dataclass(frozen=True)
class ShieldConfig:
    """Temporal release guards.

    Active hazards always take effect immediately and in fixed precedence order.
    These settings only delay a relaxation back to ``COMMIT``.
    """

    min_dwell_steps: int = 2
    commit_hysteresis_steps: int = 2

    def __post_init__(self) -> None:
        if self.min_dwell_steps < 1:
            raise ValueError("min_dwell_steps must be at least one")
        if self.commit_hysteresis_steps < 1:
            raise ValueError("commit_hysteresis_steps must be at least one")


@dataclass(frozen=True)
class ShieldSignals:
    """Deployable safety predicates, each with shape ``[B]`` and bool dtype."""

    stale: torch.Tensor
    ood: torch.Tensor
    target_safe: torch.Tensor
    stance_safe: torch.Tensor
    settling: torch.Tensor


@dataclass(frozen=True)
class ShieldOutput:
    action: torch.Tensor
    reason: torch.Tensor
    changed: torch.Tensor
    safe_streak: torch.Tensor
    dwell_steps: torch.Tensor


def conservative_target_safe(
    regression_mean: torch.Tensor,
    regression_log_scale: torch.Tensor,
    event_logits: torch.Tensor,
    conformal_multipliers: torch.Tensor,
    *,
    bearing_capacity_index: int,
    damage_index: int,
    slip_margin_index: int,
    minimum_bearing_capacity_n: float = MINIMUM_BEARING_CAPACITY_N,
    bearing_capacity_absolute_radius_n: float | None = None,
    fracture_damage_threshold: float = FRACTURE_DAMAGE_CAUTION,
    severe_slip_margin_n: float = SEVERE_SLIP_MARGIN_N,
    event_probability_threshold: float = 0.5,
) -> torch.Tensor:
    """Conservative support gate with stance-aware fracture and slip semantics."""

    if regression_mean.ndim != 3 or regression_mean.shape[1] != 2:
        raise ValueError("regression_mean must have shape [B,2,R]")
    if regression_log_scale.shape != regression_mean.shape:
        raise ValueError("regression_log_scale must match regression_mean")
    if event_logits.ndim != 3 or event_logits.shape[:2] != regression_mean.shape[:2]:
        raise ValueError("event_logits must have shape [B,2,E]")
    if event_logits.shape[-1] < 3:
        raise ValueError("event_logits must contain void, fracture, and slip events")
    if (
        conformal_multipliers.ndim != 1
        or conformal_multipliers.shape[0] != regression_mean.shape[-1]
    ):
        raise ValueError("conformal_multipliers must have shape [R]")
    for name, index in {
        "bearing_capacity_index": bearing_capacity_index,
        "damage_index": damage_index,
        "slip_margin_index": slip_margin_index,
    }.items():
        if not 0 <= index < regression_mean.shape[-1]:
            raise ValueError(f"{name} is out of range")
    if bearing_capacity_absolute_radius_n is not None and bearing_capacity_absolute_radius_n < 0.0:
        raise ValueError("bearing_capacity_absolute_radius_n must be non-negative")
    uncertainty = conformal_multipliers * regression_log_scale.exp()
    lower = regression_mean - uncertainty
    upper = regression_mean + uncertainty
    bearing_lower = lower[..., bearing_capacity_index]
    if bearing_capacity_absolute_radius_n is not None:
        bearing_lower = (
            regression_mean[..., bearing_capacity_index] - bearing_capacity_absolute_radius_n
        )
    bearing_safe = bearing_lower.amin(dim=-1) >= minimum_bearing_capacity_n
    event_probability = event_logits.sigmoid()
    any_void = (event_probability[..., 0] > event_probability_threshold).any(dim=-1)
    any_fracture = (event_probability[..., 1] > event_probability_threshold).any(dim=-1)
    bilateral_slip = (event_probability[..., 2] > event_probability_threshold).all(dim=-1)
    fracture_degraded = any_fracture & (
        upper[..., damage_index].amax(dim=-1) > fracture_damage_threshold
    )
    severe_bilateral_slip = bilateral_slip & (
        lower[..., slip_margin_index].amax(dim=-1) < severe_slip_margin_n
    )
    return bearing_safe & ~any_void & ~fracture_degraded & ~severe_bilateral_slip


class SafetyShield:
    """Deterministic, batched, stateful final-authority shield.

    Precedence is unconditional: stale/OOD -> hold, unsafe target -> request
    recovery, unsafe stance/settling -> hold, otherwise commit. Minimum dwell and
    temporal hysteresis apply only when all hazards have cleared. This component
    requests recovery from a separately validated controller. It does not promise
    or command an exact replant.
    """

    def __init__(self, config: ShieldConfig | None = None) -> None:
        self.config = config or ShieldConfig()
        self._last_action: torch.Tensor | None = None
        self._dwell_steps: torch.Tensor | None = None
        self._safe_streak: torch.Tensor | None = None

    @property
    def initialized(self) -> bool:
        return self._last_action is not None

    def reset(self, batch_size: int | None = None, *, device: torch.device | str = "cpu") -> None:
        """Clear state, or initialize a batch in the neutral COMMIT state."""

        if batch_size is None:
            self._last_action = None
            self._dwell_steps = None
            self._safe_streak = None
            return
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._last_action = torch.full(
            (batch_size,), int(ShieldAction.COMMIT), dtype=torch.int64, device=device
        )
        self._dwell_steps = torch.full(
            (batch_size,), self.config.min_dwell_steps, dtype=torch.int64, device=device
        )
        self._safe_streak = torch.full(
            (batch_size,), self.config.commit_hysteresis_steps, dtype=torch.int64, device=device
        )

    @torch.inference_mode()
    def reset_environments(self, environment_ids: torch.Tensor) -> None:
        """Reset selected rows without disturbing other vector environments."""

        if self._last_action is None:
            return
        assert self._dwell_steps is not None and self._safe_streak is not None
        ids = environment_ids.to(device=self._last_action.device)
        if ids.dtype == torch.bool:
            if tuple(ids.shape) != tuple(self._last_action.shape):
                raise ValueError("boolean environment_ids must have shape [B]")
        else:
            if ids.ndim != 1:
                raise ValueError("environment_ids must be a one-dimensional index tensor")
            ids = ids.to(dtype=torch.long)
        self._last_action[ids] = int(ShieldAction.COMMIT)
        self._dwell_steps[ids] = self.config.min_dwell_steps
        self._safe_streak[ids] = self.config.commit_hysteresis_steps

    @staticmethod
    def _validate_signals(signals: ShieldSignals) -> tuple[int, torch.device]:
        values = {
            "stale": signals.stale,
            "ood": signals.ood,
            "target_safe": signals.target_safe,
            "stance_safe": signals.stance_safe,
            "settling": signals.settling,
        }
        shapes: set[tuple[int, ...]] = set()
        devices: set[torch.device] = set()
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a Torch tensor")
            if value.dtype != torch.bool:
                raise TypeError(f"{name} must have bool dtype")
            if value.ndim != 1:
                raise ValueError(f"{name} must have shape [B]")
            shapes.add(tuple(value.shape))
            devices.add(value.device)
        if len(shapes) != 1:
            raise ValueError("all safety signals must share one [B] shape")
        if len(devices) != 1:
            raise ValueError("all safety signals must be on one device")
        batch = next(iter(shapes))[0]
        if batch <= 0:
            raise ValueError("safety signal batch cannot be empty")
        return batch, next(iter(devices))

    def _ensure_state(self, batch: int, device: torch.device) -> None:
        if self._last_action is None:
            self.reset(batch, device=device)
            return
        assert self._dwell_steps is not None and self._safe_streak is not None
        if tuple(self._last_action.shape) != (batch,):
            raise ValueError("shield batch size changed; call reset before reusing it")
        if self._last_action.device != device:
            raise ValueError("shield device changed; call reset before reusing it")

    @staticmethod
    def precedence(signals: ShieldSignals) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the memoryless decision before release hysteresis is applied."""

        batch, device = SafetyShield._validate_signals(signals)
        action = torch.full((batch,), int(ShieldAction.COMMIT), dtype=torch.int64, device=device)
        reason = torch.full(
            (batch,), int(ShieldReason.SAFE_TO_COMMIT), dtype=torch.int64, device=device
        )

        unsafe_stance = ~signals.stance_safe | signals.settling
        action = torch.where(
            unsafe_stance,
            torch.full_like(action, int(ShieldAction.HOLD_DOUBLE_SUPPORT)),
            action,
        )
        reason = torch.where(
            unsafe_stance,
            torch.full_like(reason, int(ShieldReason.UNSAFE_STANCE_OR_SETTLING)),
            reason,
        )

        # Apply higher precedence conditions last so they cannot be overwritten.
        unsafe_target = ~signals.target_safe
        action = torch.where(
            unsafe_target,
            torch.full_like(action, int(ShieldAction.REQUEST_RECOVERY)),
            action,
        )
        reason = torch.where(
            unsafe_target,
            torch.full_like(reason, int(ShieldReason.UNSAFE_TARGET)),
            reason,
        )
        stale_or_ood = signals.stale | signals.ood
        action = torch.where(
            stale_or_ood,
            torch.full_like(action, int(ShieldAction.HOLD_DOUBLE_SUPPORT)),
            action,
        )
        reason = torch.where(
            stale_or_ood,
            torch.full_like(reason, int(ShieldReason.STALE_OR_OOD)),
            reason,
        )
        return action, reason

    def step(self, signals: ShieldSignals) -> ShieldOutput:
        batch, device = self._validate_signals(signals)
        self._ensure_state(batch, device)
        assert self._last_action is not None
        assert self._dwell_steps is not None
        assert self._safe_streak is not None

        candidate, reason = self.precedence(signals)
        candidate_commit = candidate == int(ShieldAction.COMMIT)
        safe_streak = torch.where(
            candidate_commit,
            self._safe_streak + 1,
            torch.zeros_like(self._safe_streak),
        )
        relaxing = candidate_commit & (self._last_action != int(ShieldAction.COMMIT))
        release_ready = (safe_streak >= self.config.commit_hysteresis_steps) & (
            self._dwell_steps >= self.config.min_dwell_steps
        )
        retain = relaxing & ~release_ready
        action = torch.where(retain, self._last_action, candidate)
        reason = torch.where(
            retain,
            torch.full_like(reason, int(ShieldReason.RELEASE_HYSTERESIS)),
            reason,
        )
        changed = action != self._last_action
        dwell_steps = torch.where(
            changed,
            torch.ones_like(self._dwell_steps),
            self._dwell_steps + 1,
        )

        self._last_action = action.clone()
        self._dwell_steps = dwell_steps.clone()
        self._safe_streak = safe_streak.clone()
        return ShieldOutput(
            action=action.clone(),
            reason=reason.clone(),
            changed=changed.clone(),
            safe_streak=safe_streak.clone(),
            dwell_steps=dwell_steps.clone(),
        )

    def decide(
        self,
        *,
        stale: bool,
        ood: bool,
        target_safe: bool,
        stance_safe: bool,
        settling: bool,
    ) -> ShieldAction:
        """Scalar convenience API with the same state and precedence as ``step``."""

        signals = ShieldSignals(
            stale=torch.tensor([stale], dtype=torch.bool),
            ood=torch.tensor([ood], dtype=torch.bool),
            target_safe=torch.tensor([target_safe], dtype=torch.bool),
            stance_safe=torch.tensor([stance_safe], dtype=torch.bool),
            settling=torch.tensor([settling], dtype=torch.bool),
        )
        value = int(self.step(signals).action.item())
        return ShieldAction(value)


SafetyAction = ShieldAction
