from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from ..data.schema import FOOT_COUNT, SENSOR_CHANNELS, VisibleSensorBatch
from ..learning.models import EstimatorOutput, SupervisorOutput
from ..learning.shield import SafetyShield, ShieldAction, ShieldOutput, ShieldSignals

LOCOMOTION_ACTION_DIM = 37
COMMAND_DIM = 3
ESTIMATOR_RATE_HZ = 100.0


class Estimator(Protocol):
    """Visible estimator interface used by the runtime bridge."""

    context_dim: int

    def __call__(
        self,
        packet_history: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_age_s: torch.Tensor,
        deployable_context: torch.Tensor | None = None,
    ) -> EstimatorOutput: ...


class Supervisor(Protocol):
    """Visible supervisor interface used by the runtime bridge."""

    proposal_names: tuple[str, ...]
    command_gait_context_dim: int

    def __call__(
        self,
        bilateral_latent: torch.Tensor,
        deployable_command_gait_context: torch.Tensor | None = None,
    ) -> SupervisorOutput: ...


LocomotionPolicy = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class EverestControllerConfig:
    """Runtime rates and stale-data limits.

    Sensor and estimator steps are fixed at 100 Hz. ``control_rate_hz`` must be
    an integer divisor of 100 Hz so supervisor and locomotion calls have an
    exact sample-and-hold schedule.
    """

    history_steps: int = 50
    control_rate_hz: float = 50.0
    stale_after_s: float = 0.05
    cadence_tolerance_s: float = 1.0e-6
    minimum_history_steps: int = 6

    def __post_init__(self) -> None:
        if self.history_steps < 1:
            raise ValueError("history_steps must be at least one")
        if not 0.0 < self.control_rate_hz <= ESTIMATOR_RATE_HZ:
            raise ValueError("control_rate_hz must be in (0, 100]")
        ratio = ESTIMATOR_RATE_HZ / self.control_rate_hz
        if abs(ratio - round(ratio)) > 1.0e-9:
            raise ValueError("control_rate_hz must divide 100 Hz exactly")
        if self.stale_after_s < 0.0:
            raise ValueError("stale_after_s must be non-negative")
        if self.cadence_tolerance_s < 0.0:
            raise ValueError("cadence_tolerance_s must be non-negative")
        if not 1 <= self.minimum_history_steps <= self.history_steps:
            raise ValueError("minimum_history_steps must be in [1, history_steps]")

    @property
    def control_decimation(self) -> int:
        return round(ESTIMATOR_RATE_HZ / self.control_rate_hz)

    @property
    def estimator_period_s(self) -> float:
        return 1.0 / ESTIMATOR_RATE_HZ


@dataclass(frozen=True)
class PacketHistory:
    """A device-resident rolling bilateral packet history."""

    values: torch.Tensor
    mask: torch.Tensor
    age_s: torch.Tensor
    context: torch.Tensor | None

    @property
    def packet_values(self) -> torch.Tensor:
        return self.values

    @property
    def valid_mask(self) -> torch.Tensor:
        return self.mask

    @property
    def sample_age_s(self) -> torch.Tensor:
        return self.age_s


class RollingPacketHistory:
    """Append-only rolling window that never flattens the bilateral foot axis."""

    def __init__(self, max_steps: int) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least one")
        self.max_steps = int(max_steps)
        self._values: torch.Tensor | None = None
        self._mask: torch.Tensor | None = None
        self._age_s: torch.Tensor | None = None
        self._context: torch.Tensor | None = None
        self._context_dim: int | None = None

    @property
    def initialized(self) -> bool:
        return self._values is not None

    def reset(self) -> None:
        self._values = None
        self._mask = None
        self._age_s = None
        self._context = None
        self._context_dim = None

    @torch.inference_mode()
    def reset_environments(self, environment_ids: torch.Tensor) -> None:
        """Clear selected vector rows while retaining the shared history length."""

        if self._values is None:
            return
        assert self._mask is not None and self._age_s is not None
        ids = environment_ids.to(device=self._values.device)
        if ids.dtype == torch.bool:
            if tuple(ids.shape) != (int(self._values.shape[0]),):
                raise ValueError("boolean environment_ids must have shape [B]")
        else:
            if ids.ndim != 1:
                raise ValueError("environment_ids must be a one-dimensional index tensor")
            ids = ids.to(dtype=torch.long)
        self._values[ids] = 0.0
        self._mask[ids] = False
        self._age_s[ids] = 0.0
        if self._context is not None:
            self._context[ids] = 0.0

    def append(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        age_s: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> PacketHistory:
        _validate_current_packet(values, mask, age_s)
        batch = int(values.shape[0])
        if context is not None:
            if not isinstance(context, torch.Tensor) or not context.is_floating_point():
                raise TypeError("deployable_context must be a floating-point Torch tensor")
            if context.ndim != 3 or tuple(context.shape[:2]) != (batch, FOOT_COUNT):
                raise ValueError("deployable_context must have shape [B, 2, C]")
            if context.device != values.device:
                raise ValueError("deployable_context and packet values must share a device")
            if not bool(torch.isfinite(context).all().item()):
                raise ValueError("deployable_context contains non-finite values")
        context_dim = 0 if context is None else int(context.shape[-1])

        if self._values is not None:
            assert self._mask is not None and self._age_s is not None
            if int(self._values.shape[0]) != batch:
                raise ValueError("packet batch size changed; call reset before reusing the history")
            if self._values.device != values.device:
                raise ValueError("packet device changed; call reset before reusing the history")
            if self._values.dtype != values.dtype or self._age_s.dtype != age_s.dtype:
                raise ValueError("packet dtype changed; call reset before reusing the history")
            if self._context_dim != context_dim:
                raise ValueError("context width changed; call reset before reusing the history")

        next_values = values.detach().clone().unsqueeze(1)
        next_mask = mask.detach().clone().unsqueeze(1)
        next_age = age_s.detach().clone().unsqueeze(1)
        next_context = None if context is None else context.detach().clone().unsqueeze(1)
        if self._values is not None:
            next_values = torch.cat((self._values, next_values), dim=1)
            next_mask = torch.cat((self._mask, next_mask), dim=1)
            next_age = torch.cat((self._age_s, next_age), dim=1)
            if next_context is not None:
                assert self._context is not None
                next_context = torch.cat((self._context, next_context), dim=1)

        self._values = next_values[:, -self.max_steps :]
        self._mask = next_mask[:, -self.max_steps :]
        self._age_s = next_age[:, -self.max_steps :]
        self._context = None if next_context is None else next_context[:, -self.max_steps :]
        self._context_dim = context_dim
        return self.snapshot()

    def snapshot(self) -> PacketHistory:
        if self._values is None or self._mask is None or self._age_s is None:
            raise RuntimeError("packet history is empty")
        return PacketHistory(
            values=self._values.clone(),
            mask=self._mask.clone(),
            age_s=self._age_s.clone(),
            context=None if self._context is None else self._context.clone(),
        )


def _validate_current_packet(values: torch.Tensor, mask: torch.Tensor, age_s: torch.Tensor) -> None:
    if not isinstance(values, torch.Tensor) or not values.is_floating_point():
        raise TypeError("packet_values must be a floating-point Torch tensor")
    if values.ndim != 3 or tuple(values.shape[-2:]) != (FOOT_COUNT, SENSOR_CHANNELS):
        raise ValueError("packet_values must have exact shape [B, 2, 19]")
    if int(values.shape[0]) <= 0:
        raise ValueError("packet batch cannot be empty")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("packet_values contains non-finite values")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("valid_mask must be a bool Torch tensor")
    if tuple(mask.shape) != tuple(values.shape):
        raise ValueError("valid_mask must have exact shape [B, 2, 19]")
    if mask.device != values.device:
        raise ValueError("valid_mask and packet_values must share a device")
    if not isinstance(age_s, torch.Tensor) or not age_s.is_floating_point():
        raise TypeError("sample_age_s must be a floating-point Torch tensor")
    if tuple(age_s.shape) != tuple(values.shape):
        raise ValueError("sample_age_s must have exact shape [B, 2, 19]")
    if age_s.device != values.device:
        raise ValueError("sample_age_s and packet_values must share a device")
    if not bool(torch.isfinite(age_s).all().item()) or bool((age_s < 0.0).any().item()):
        raise ValueError("sample_age_s must contain finite non-negative values")


@dataclass(frozen=True)
class SafeCommand:
    """Shield-approved G1 command. Recovery is only a request to another controller."""

    velocity_yaw: torch.Tensor
    recovery_request: torch.Tensor

    @property
    def command(self) -> torch.Tensor:
        return self.velocity_yaw


class CommandAdapter:
    """Map the shield's final high-level action to the stock G1 command ABI."""

    def __init__(self, proposal_names: tuple[str, ...]) -> None:
        try:
            self._velocity_scale_index = proposal_names.index("velocity_scale")
            self._yaw_scale_index = proposal_names.index("yaw_scale")
        except ValueError as exc:
            raise ValueError("supervisor proposals require velocity_scale and yaw_scale") from exc

    def __call__(
        self,
        requested_velocity_yaw: torch.Tensor,
        continuous_proposals: torch.Tensor,
        final_action: torch.Tensor,
    ) -> SafeCommand:
        if (
            not isinstance(requested_velocity_yaw, torch.Tensor)
            or not requested_velocity_yaw.is_floating_point()
        ):
            raise TypeError("requested_velocity_yaw must be a floating-point Torch tensor")
        if requested_velocity_yaw.ndim != 2 or requested_velocity_yaw.shape[-1] != COMMAND_DIM:
            raise ValueError("requested_velocity_yaw must have shape [B, 3]")
        if not bool(torch.isfinite(requested_velocity_yaw).all().item()):
            raise ValueError("requested_velocity_yaw contains non-finite values")
        batch = int(requested_velocity_yaw.shape[0])
        if not isinstance(continuous_proposals, torch.Tensor):
            raise TypeError("continuous_proposals must be a Torch tensor")
        needed = max(self._velocity_scale_index, self._yaw_scale_index) + 1
        if continuous_proposals.ndim != 2 or tuple(continuous_proposals.shape[:1]) != (batch,):
            raise ValueError("continuous_proposals must have shape [B, P]")
        if int(continuous_proposals.shape[1]) < needed:
            raise ValueError("continuous_proposals is missing command scales")
        if continuous_proposals.device != requested_velocity_yaw.device:
            raise ValueError("proposals and requested command must share a device")
        if not bool(torch.isfinite(continuous_proposals).all().item()):
            raise ValueError("continuous_proposals contains non-finite values")
        if not isinstance(final_action, torch.Tensor) or final_action.dtype != torch.int64:
            raise TypeError("final_action must be an int64 Torch tensor")
        if (
            tuple(final_action.shape) != (batch,)
            or final_action.device != requested_velocity_yaw.device
        ):
            raise ValueError("final_action must have shape [B] on the command device")
        if bool(
            (
                (final_action < int(ShieldAction.COMMIT))
                | (final_action > int(ShieldAction.REQUEST_RECOVERY))
            )
            .any()
            .item()
        ):
            raise ValueError("final_action contains an unknown shield action")

        velocity_scale = continuous_proposals[:, self._velocity_scale_index].clamp(0.0, 1.0)
        yaw_scale = continuous_proposals[:, self._yaw_scale_index].clamp(0.0, 1.0)
        scaled = requested_velocity_yaw * torch.stack(
            (velocity_scale, velocity_scale, yaw_scale), dim=-1
        ).to(dtype=requested_velocity_yaw.dtype)
        commit = final_action == int(ShieldAction.COMMIT)
        safe = torch.where(commit.unsqueeze(-1), scaled, torch.zeros_like(scaled))
        recovery = final_action == int(ShieldAction.REQUEST_RECOVERY)
        return SafeCommand(velocity_yaw=safe, recovery_request=recovery)


@dataclass(frozen=True)
class EverestControllerOutput:
    """Typed output for one 100 Hz bridge step."""

    joint_action: torch.Tensor
    safe_command: SafeCommand
    estimator: EstimatorOutput
    supervisor: SupervisorOutput
    shield: ShieldOutput
    supervisor_preference: torch.Tensor
    history: PacketHistory
    stale: torch.Tensor
    control_updated: bool

    @property
    def recovery_request(self) -> torch.Tensor:
        return self.safe_command.recovery_request

    @property
    def joint_targets(self) -> torch.Tensor:
        return self.joint_action

    @property
    def locomotion_action(self) -> torch.Tensor:
        return self.joint_action


class EverestController:
    """Simulator-neutral in-process bridge to an existing G1 locomotion policy.

    ``step`` consumes one exact bilateral 100 Hz sensor frame. The estimator runs
    on every call. Supervisor, shield, command adaptation, and locomotion run at
    ``control_rate_hz``; their outputs are held between control ticks. Safety
    hazards are sampled at 100 Hz and latched until the next control tick. The
    locomotion policy alone creates the 37 joint actions. This bridge never adds
    residuals, mixes raw joints, or claims an exact recovery replant.
    """

    def __init__(
        self,
        *,
        estimator: Estimator,
        supervisor: Supervisor,
        locomotion_policy: LocomotionPolicy,
        shield: SafetyShield | None = None,
        config: EverestControllerConfig | None = None,
    ) -> None:
        self.estimator = estimator
        self.supervisor = supervisor
        self.locomotion_policy = locomotion_policy
        self.shield = shield or SafetyShield()
        self.config = config or EverestControllerConfig()
        self.history = RollingPacketHistory(self.config.history_steps)
        self.command_adapter = CommandAdapter(tuple(supervisor.proposal_names))
        self._tick = 0
        self._last_timestamp_s: torch.Tensor | None = None
        self._held_supervisor: SupervisorOutput | None = None
        self._held_shield: ShieldOutput | None = None
        self._held_command: SafeCommand | None = None
        self._held_joint_action: torch.Tensor | None = None
        self._held_preference: torch.Tensor | None = None
        self._pending_signals: ShieldSignals | None = None
        self._history_counts: torch.Tensor | None = None
        self._timestamp_initialized: torch.Tensor | None = None
        self._force_control_update = False

    def reset(self) -> None:
        """Clear history, timebase, sample-and-hold state, and shield state."""

        self.history.reset()
        self.shield.reset()
        self._tick = 0
        self._last_timestamp_s = None
        self._held_supervisor = None
        self._held_shield = None
        self._held_command = None
        self._held_joint_action = None
        self._held_preference = None
        self._pending_signals = None
        self._history_counts = None
        self._timestamp_initialized = None
        self._force_control_update = False

    @torch.inference_mode()
    def reset_environments(self, environment_ids: torch.Tensor) -> None:
        """Reset selected vector environments without clearing unaffected rows."""

        if self._history_counts is None:
            return
        ids = environment_ids.to(device=self._history_counts.device)
        if ids.dtype == torch.bool:
            if tuple(ids.shape) != tuple(self._history_counts.shape):
                raise ValueError("boolean environment_ids must have shape [B]")
        else:
            if ids.ndim != 1:
                raise ValueError("environment_ids must be a one-dimensional index tensor")
            ids = ids.to(dtype=torch.long)
        self.history.reset_environments(ids)
        self.shield.reset_environments(ids)
        self._history_counts[ids] = 0
        assert self._timestamp_initialized is not None
        self._timestamp_initialized[ids] = False
        if self._pending_signals is not None:
            self._pending_signals.stale[ids] = False
            self._pending_signals.ood[ids] = False
            self._pending_signals.target_safe[ids] = True
            self._pending_signals.stance_safe[ids] = True
            self._pending_signals.settling[ids] = False
        self._force_control_update = True

    def _timestamp_stale(
        self, timestamp_s: torch.Tensor, packet_values: torch.Tensor
    ) -> torch.Tensor:
        batch = int(packet_values.shape[0])
        if not isinstance(timestamp_s, torch.Tensor) or not timestamp_s.is_floating_point():
            raise TypeError("timestamp_s must be a floating-point Torch tensor")
        if tuple(timestamp_s.shape) != (batch, FOOT_COUNT):
            raise ValueError("timestamp_s must have exact shape [B, 2]")
        if timestamp_s.device != packet_values.device:
            raise ValueError("timestamp_s and packet_values must share a device")
        if not bool(torch.isfinite(timestamp_s).all().item()):
            raise ValueError("timestamp_s contains non-finite values")
        if self._last_timestamp_s is None:
            self._last_timestamp_s = torch.zeros_like(timestamp_s)
            self._timestamp_initialized = torch.zeros(
                batch, dtype=torch.bool, device=packet_values.device
            )
        elif self._last_timestamp_s.device != timestamp_s.device:
            raise ValueError("timestamp device changed; call reset before reusing the controller")
        assert self._timestamp_initialized is not None
        if tuple(self._timestamp_initialized.shape) != (batch,):
            raise ValueError(
                "timestamp batch size changed; call reset before reusing the controller"
            )
        delta = timestamp_s - self._last_timestamp_s
        initialized_feet = self._timestamp_initialized.unsqueeze(-1)
        if bool(((delta <= 0.0) & initialized_feet).any().item()):
            raise ValueError("timestamp_s must increase for each foot on every step")
        cadence_error = torch.abs(delta - self.config.estimator_period_s)
        stale = self._timestamp_initialized & (cadence_error > self.config.cadence_tolerance_s).any(
            dim=-1
        )
        self._last_timestamp_s.copy_(timestamp_s)
        self._timestamp_initialized.fill_(True)
        return stale

    @staticmethod
    def _validate_frame(frame: VisibleSensorBatch) -> tuple[torch.Tensor, ...]:
        if not isinstance(frame, VisibleSensorBatch):
            raise TypeError("sensor_frame must be a VisibleSensorBatch")
        values = frame.packet_values
        mask = frame.valid_mask
        age = frame.sample_age_s
        timestamp = frame.timestamp_s
        _validate_current_packet(values, mask, age)
        return values, mask, age, timestamp

    @staticmethod
    def _validate_stock_inputs(
        stock_observation: torch.Tensor,
        requested_velocity_yaw: torch.Tensor,
        *,
        batch: int,
        device: torch.device,
    ) -> None:
        if (
            not isinstance(stock_observation, torch.Tensor)
            or not stock_observation.is_floating_point()
        ):
            raise TypeError("stock_observation must be a floating-point Torch tensor")
        if stock_observation.ndim != 2 or int(stock_observation.shape[0]) != batch:
            raise ValueError("stock_observation must have shape [B, O]")
        if stock_observation.device != device:
            raise ValueError("stock_observation and sensor frame must share a device")
        if not bool(torch.isfinite(stock_observation).all().item()):
            raise ValueError("stock_observation contains non-finite values")
        if (
            not isinstance(requested_velocity_yaw, torch.Tensor)
            or not requested_velocity_yaw.is_floating_point()
        ):
            raise TypeError("requested_velocity_yaw must be a floating-point Torch tensor")
        if tuple(requested_velocity_yaw.shape) != (batch, COMMAND_DIM):
            raise ValueError("requested_velocity_yaw must have shape [B, 3]")
        if requested_velocity_yaw.device != device:
            raise ValueError("requested_velocity_yaw and sensor frame must share a device")
        if not bool(torch.isfinite(requested_velocity_yaw).all().item()):
            raise ValueError("requested_velocity_yaw contains non-finite values")

    @staticmethod
    def _validate_shield_signals(
        signals: ShieldSignals, *, batch: int, device: torch.device
    ) -> None:
        for name in ("stale", "ood", "target_safe", "stance_safe", "settling"):
            value = getattr(signals, name)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
                raise TypeError(f"shield signal {name} must be a bool Torch tensor")
            if tuple(value.shape) != (batch,):
                raise ValueError(f"shield signal {name} must have shape [B]")
            if value.device != device:
                raise ValueError(f"shield signal {name} must be on the sensor frame device")

    def _latch_signals(self, signals: ShieldSignals, sensor_stale: torch.Tensor) -> ShieldSignals:
        current = ShieldSignals(
            stale=signals.stale | sensor_stale,
            ood=signals.ood.clone(),
            target_safe=signals.target_safe.clone(),
            stance_safe=signals.stance_safe.clone(),
            settling=signals.settling.clone(),
        )
        if self._pending_signals is None:
            self._pending_signals = current
        else:
            pending = self._pending_signals
            self._pending_signals = ShieldSignals(
                stale=pending.stale | current.stale,
                ood=pending.ood | current.ood,
                target_safe=pending.target_safe & current.target_safe,
                stance_safe=pending.stance_safe & current.stance_safe,
                settling=pending.settling | current.settling,
            )
        return self._pending_signals

    @torch.inference_mode()
    def step(
        self,
        sensor_frame: VisibleSensorBatch,
        *,
        stock_observation: torch.Tensor,
        requested_velocity_yaw: torch.Tensor,
        shield_signals: ShieldSignals,
        deployable_context: torch.Tensor | None = None,
        deployable_command_gait_context: torch.Tensor | None = None,
    ) -> EverestControllerOutput:
        """Advance one 100 Hz sensor tick and return the held or updated G1 action."""

        values, mask, age, timestamp = self._validate_frame(sensor_frame)
        batch = int(values.shape[0])
        self._validate_stock_inputs(
            stock_observation, requested_velocity_yaw, batch=batch, device=values.device
        )
        expected_context_dim = int(self.estimator.context_dim)
        if deployable_context is not None and not isinstance(deployable_context, torch.Tensor):
            raise TypeError("deployable_context must be a Torch tensor")
        if expected_context_dim == 0:
            if deployable_context is not None and tuple(deployable_context.shape) != (
                batch,
                FOOT_COUNT,
                0,
            ):
                raise ValueError("estimator is configured without deployable context")
            deployable_context = None
        elif deployable_context is None:
            raise ValueError("deployable_context is required by the estimator")
        elif deployable_context.ndim != 3 or tuple(deployable_context.shape) != (
            batch,
            FOOT_COUNT,
            expected_context_dim,
        ):
            raise ValueError(
                f"deployable_context must have shape [{batch}, 2, {expected_context_dim}]"
            )

        cadence_stale = self._timestamp_stale(timestamp, values)
        age_stale = (age > self.config.stale_after_s).reshape(batch, -1).any(dim=-1)
        if self._history_counts is None:
            self._history_counts = torch.zeros(batch, dtype=torch.long, device=values.device)
        elif tuple(self._history_counts.shape) != (batch,):
            raise ValueError("packet batch size changed; call reset before reusing the controller")
        self._history_counts += 1
        warmup_stale = self._history_counts < self.config.minimum_history_steps
        sensor_stale = cadence_stale | age_stale | warmup_stale
        self._validate_shield_signals(shield_signals, batch=batch, device=values.device)
        effective_signals = self._latch_signals(shield_signals, sensor_stale)
        history = self.history.append(values, mask, age, deployable_context)
        estimate = self.estimator(
            history.values,
            history.mask,
            history.age_s,
            history.context,
        )

        control_updated = (
            self._force_control_update or self._tick % self.config.control_decimation == 0
        )
        if control_updated:
            self._force_control_update = False
            supervisor = self.supervisor(estimate.bilateral_latent, deployable_command_gait_context)
            preference = torch.argmax(supervisor.action_logits, dim=-1)
            shield_override = self.shield.step(effective_signals)
            self._pending_signals = None
            # A safe COMMIT from the deterministic shield means "no override". In that
            # case the visible supervisor's COMMIT/HOLD/RECOVERY preference is honored.
            # Any active shield hazard keeps final authority and cannot be overwritten.
            final_action = torch.where(
                shield_override.action == int(ShieldAction.COMMIT),
                preference.to(dtype=torch.int64),
                shield_override.action,
            )
            previous_action = (
                torch.full_like(final_action, int(ShieldAction.COMMIT))
                if self._held_shield is None
                else self._held_shield.action
            )
            shield_output = ShieldOutput(
                action=final_action,
                reason=shield_override.reason,
                changed=final_action != previous_action,
                safe_streak=shield_override.safe_streak,
                dwell_steps=shield_override.dwell_steps,
            )
            safe_command = self.command_adapter(
                requested_velocity_yaw,
                supervisor.continuous_proposals,
                final_action,
            )
            joint_action = self.locomotion_policy(
                stock_observation,
                safe_command.velocity_yaw,
            )
            if not isinstance(joint_action, torch.Tensor) or not joint_action.is_floating_point():
                raise TypeError("locomotion policy must return a floating-point Torch tensor")
            if tuple(joint_action.shape) != (batch, LOCOMOTION_ACTION_DIM):
                raise ValueError("locomotion policy must return exact shape [B, 37]")
            if joint_action.device != values.device:
                raise ValueError("locomotion output and sensor frame must share a device")
            if not bool(torch.isfinite(joint_action).all().item()):
                raise ValueError("locomotion policy returned non-finite joint actions")
            self._held_supervisor = supervisor
            self._held_shield = shield_output
            self._held_command = safe_command
            self._held_joint_action = joint_action
            self._held_preference = preference
        else:
            assert self._held_supervisor is not None
            assert self._held_shield is not None
            assert self._held_command is not None
            assert self._held_joint_action is not None
            assert self._held_preference is not None
            supervisor = self._held_supervisor
            shield_output = self._held_shield
            safe_command = self._held_command
            joint_action = self._held_joint_action
            preference = self._held_preference

        self._tick += 1
        return EverestControllerOutput(
            joint_action=joint_action,
            safe_command=safe_command,
            estimator=estimate,
            supervisor=supervisor,
            shield=shield_output,
            supervisor_preference=preference,
            history=history,
            stale=effective_signals.stale,
            control_updated=control_updated,
        )
