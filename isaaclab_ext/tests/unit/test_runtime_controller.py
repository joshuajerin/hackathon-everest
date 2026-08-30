from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SOURCE = Path(__file__).parents[2] / "source/hackathon_everest_isaaclab"
sys.path.insert(0, str(SOURCE))

from hackathon_everest_isaaclab.data.schema import VisibleSensorBatch
from hackathon_everest_isaaclab.learning.models import EstimatorOutput, SupervisorOutput
from hackathon_everest_isaaclab.learning.shield import (
    SafetyShield,
    ShieldAction,
    ShieldConfig,
    ShieldSignals,
)
from hackathon_everest_isaaclab.runtime import (
    EverestController,
    EverestControllerConfig,
    RollingPacketHistory,
)


class RecordingEstimator:
    context_dim = 0

    def __init__(self) -> None:
        self.calls = 0
        self.histories: list[torch.Tensor] = []

    def __call__(self, values, mask, age, context=None) -> EstimatorOutput:
        self.calls += 1
        self.histories.append(values.clone())
        batch = values.shape[0]
        per_foot = values[:, -1, :, :4].mean(dim=-1, keepdim=True).expand(-1, -1, 4)
        bilateral = torch.cat((per_foot[:, 0], per_foot[:, 1]), dim=-1)
        zeros_11 = values.new_zeros(batch, 2, 11)
        return EstimatorOutput(
            regression_mean=zeros_11,
            regression_log_scale=zeros_11,
            event_logits=values.new_zeros(batch, 2, 3),
            per_foot_latent=per_foot,
            bilateral_latent=bilateral,
        )


class RecordingSupervisor:
    proposal_names = (
        "velocity_scale",
        "yaw_scale",
        "probe_load_n",
        "approach_speed_mps",
        "clearance_m",
        "transfer_rate_scale",
    )
    command_gait_context_dim = 0

    def __init__(self, preference: int = 0) -> None:
        self.calls = 0
        self.preference = preference

    def __call__(self, bilateral_latent, context=None) -> SupervisorOutput:
        self.calls += 1
        batch = bilateral_latent.shape[0]
        proposals = bilateral_latent.new_tensor([0.5, 0.25, 10.0, 0.1, 0.1, 0.5])
        logits = bilateral_latent.new_zeros(batch, 3)
        logits[:, self.preference] = 3.0
        return SupervisorOutput(
            action_logits=logits,
            continuous_proposals=proposals.expand(batch, -1),
        )


class RecordingLocomotion:
    def __init__(self) -> None:
        self.calls = 0
        self.inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, observation, command):
        self.calls += 1
        self.inputs.append((observation.clone(), command.clone()))
        # A recognizable stock-policy result. The bridge must return it unchanged.
        return observation[:, :1].expand(-1, 37) + float(self.calls)


def frame(values: torch.Tensor, timestamp: float, *, age: torch.Tensor | None = None):
    batch = values.shape[0]
    return VisibleSensorBatch(
        packet_values=values,
        valid_mask=torch.ones_like(values, dtype=torch.bool),
        timestamp_s=torch.full((batch, 2), timestamp, dtype=values.dtype, device=values.device),
        sample_age_s=torch.zeros_like(values) if age is None else age,
    )


def signals(
    batch: int,
    *,
    stale: bool = False,
    target_safe: bool = True,
    stance_safe: bool = True,
):
    full = lambda value: torch.full((batch,), value, dtype=torch.bool)
    return ShieldSignals(
        stale=full(stale),
        ood=full(False),
        target_safe=full(target_safe),
        stance_safe=full(stance_safe),
        settling=full(False),
    )


def make_controller(
    *, history_steps: int = 3, control_rate_hz: float = 50.0, supervisor_preference: int = 0
):
    estimator = RecordingEstimator()
    supervisor = RecordingSupervisor(supervisor_preference)
    locomotion = RecordingLocomotion()
    controller = EverestController(
        estimator=estimator,
        supervisor=supervisor,
        locomotion_policy=locomotion,
        shield=SafetyShield(ShieldConfig(min_dwell_steps=1, commit_hysteresis_steps=1)),
        config=EverestControllerConfig(
            history_steps=history_steps,
            control_rate_hz=control_rate_hz,
            stale_after_s=0.05,
            minimum_history_steps=1,
        ),
    )
    return controller, estimator, supervisor, locomotion


def test_exact_bilateral_history_preserves_independent_feet_and_rolls() -> None:
    controller, estimator, _, _ = make_controller(history_steps=2)
    observation = torch.zeros(2, 8)
    command = torch.ones(2, 3)
    first = torch.zeros(2, 2, 19)
    first[:, 0] = 11.0
    first[:, 1] = 29.0
    output = controller.step(
        frame(first, 0.0),
        stock_observation=observation,
        requested_velocity_yaw=command,
        shield_signals=signals(2),
    )
    assert output.history.values.shape == (2, 1, 2, 19)
    assert output.history.mask.shape == (2, 1, 2, 19)
    assert output.history.age_s.shape == (2, 1, 2, 19)
    assert torch.all(output.estimator.per_foot_latent[:, 0] == 11.0)
    assert torch.all(output.estimator.per_foot_latent[:, 1] == 29.0)

    for tick, value in ((1, 40.0), (2, 50.0)):
        current = torch.full_like(first, value)
        output = controller.step(
            frame(current, tick * 0.01),
            stock_observation=observation,
            requested_velocity_yaw=command,
            shield_signals=signals(2),
        )
    assert output.history.values.shape == (2, 2, 2, 19)
    assert torch.all(output.history.values[:, 0] == 40.0)
    assert torch.all(output.history.values[:, 1] == 50.0)
    assert estimator.histories[-1].shape == (2, 2, 2, 19)

    with pytest.raises(ValueError, match=r"exact shape \[B, 2, 19\]"):
        RollingPacketHistory(2).append(
            torch.zeros(2, 38), torch.ones(2, 38, dtype=torch.bool), torch.zeros(2, 38)
        )


def test_held_packet_values_mask_and_age_reach_estimator_unchanged() -> None:
    controller, estimator, _, _ = make_controller()
    values = torch.zeros(1, 2, 19)
    values[0, 0, 3] = 12.5
    valid = torch.ones_like(values, dtype=torch.bool)
    valid[0, 0, 3] = False
    age = torch.zeros_like(values)
    age[0, 0, 3] = 0.02
    sensor = VisibleSensorBatch(
        packet_values=values,
        valid_mask=valid,
        timestamp_s=torch.zeros(1, 2),
        sample_age_s=age,
    )
    output = controller.step(
        sensor,
        stock_observation=torch.zeros(1, 8),
        requested_velocity_yaw=torch.zeros(1, 3),
        shield_signals=signals(1),
    )
    assert output.history.values[0, 0, 0, 3].item() == 12.5
    assert not output.history.mask[0, 0, 0, 3].item()
    assert output.history.age_s[0, 0, 0, 3].item() == pytest.approx(0.02)
    assert estimator.histories[-1][0, 0, 0, 3].item() == 12.5


def test_command_adapter_uses_shield_final_authority_for_commit_hold_and_recovery() -> None:
    observation = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    requested = torch.tensor([[2.0, -4.0, 3.0]])

    safe_controller, _, _, safe_locomotion = make_controller()
    safe = safe_controller.step(
        frame(torch.zeros(1, 2, 19), 0.0),
        stock_observation=observation,
        requested_velocity_yaw=requested,
        shield_signals=signals(1),
    )
    assert safe.shield.action.item() == ShieldAction.COMMIT
    torch.testing.assert_close(safe.safe_command.velocity_yaw, torch.tensor([[1.0, -2.0, 0.75]]))
    torch.testing.assert_close(safe_locomotion.inputs[0][1], safe.safe_command.velocity_yaw)
    assert not safe.recovery_request.item()

    learned_hold_controller, _, _, learned_hold_locomotion = make_controller(
        supervisor_preference=1
    )
    learned_hold = learned_hold_controller.step(
        frame(torch.zeros(1, 2, 19), 0.0),
        stock_observation=observation,
        requested_velocity_yaw=requested,
        shield_signals=signals(1),
    )
    assert learned_hold.shield.action.item() == ShieldAction.HOLD_DOUBLE_SUPPORT
    assert torch.equal(learned_hold_locomotion.inputs[0][1], torch.zeros(1, 3))

    hold_controller, _, _, hold_locomotion = make_controller()
    hold = hold_controller.step(
        frame(torch.zeros(1, 2, 19), 0.0),
        stock_observation=observation,
        requested_velocity_yaw=requested,
        shield_signals=signals(1, stale=True),
    )
    assert hold.shield.action.item() == ShieldAction.HOLD_DOUBLE_SUPPORT
    assert torch.equal(hold_locomotion.inputs[0][1], torch.zeros(1, 3))
    assert not hold.recovery_request.item()

    recovery_controller, _, _, recovery_locomotion = make_controller(supervisor_preference=1)
    recovery = recovery_controller.step(
        frame(torch.zeros(1, 2, 19), 0.0),
        stock_observation=observation,
        requested_velocity_yaw=requested,
        shield_signals=signals(1, target_safe=False),
    )
    assert recovery.shield.action.item() == ShieldAction.REQUEST_RECOVERY
    assert torch.equal(recovery_locomotion.inputs[0][1], torch.zeros(1, 3))
    assert recovery.recovery_request.item()
    # The visible supervisor preferred HOLD, but it cannot replace the shield decision.
    assert recovery.supervisor_preference.item() == ShieldAction.HOLD_DOUBLE_SUPPORT


def test_estimator_is_100_hz_and_control_path_is_50_hz_sample_and_hold() -> None:
    controller, estimator, supervisor, locomotion = make_controller(control_rate_hz=50.0)
    observation = torch.full((1, 8), 7.0)
    requested = torch.ones(1, 3)
    outputs = []
    for tick in range(5):
        outputs.append(
            controller.step(
                frame(torch.full((1, 2, 19), float(tick)), tick * 0.01),
                stock_observation=observation + tick,
                requested_velocity_yaw=requested * (tick + 1),
                shield_signals=signals(1),
            )
        )
    assert estimator.calls == 5
    assert supervisor.calls == 3
    assert locomotion.calls == 3
    assert [item.control_updated for item in outputs] == [True, False, True, False, True]
    assert torch.equal(outputs[1].joint_action, outputs[0].joint_action)
    assert torch.equal(outputs[3].joint_action, outputs[2].joint_action)
    # The locomotion result is not mixed with sensor values or a joint residual.
    torch.testing.assert_close(outputs[2].joint_action, torch.full((1, 37), 11.0))


def test_packet_age_or_missed_100_hz_timebase_forces_stale_hold() -> None:
    controller, _, _, locomotion = make_controller()
    observation = torch.zeros(1, 8)
    requested = torch.ones(1, 3)
    controller.step(
        frame(torch.zeros(1, 2, 19), 0.0),
        stock_observation=observation,
        requested_velocity_yaw=requested,
        shield_signals=signals(1),
    )
    controller.step(
        frame(torch.zeros(1, 2, 19), 0.03),
        stock_observation=observation,
        requested_velocity_yaw=requested,
        shield_signals=signals(1),
    )
    missed_tick = controller.step(
        frame(torch.zeros(1, 2, 19), 0.04),
        stock_observation=observation,
        requested_velocity_yaw=requested,
        shield_signals=signals(1),
    )
    assert missed_tick.stale.item()
    assert missed_tick.shield.action.item() == ShieldAction.HOLD_DOUBLE_SUPPORT
    assert torch.equal(locomotion.inputs[-1][1], torch.zeros(1, 3))

    controller.reset()
    old_age = torch.zeros(1, 2, 19)
    old_age[:, 1, 7] = 0.051
    aged = controller.step(
        frame(torch.zeros(1, 2, 19), 1.0, age=old_age),
        stock_observation=observation,
        requested_velocity_yaw=requested,
        shield_signals=signals(1),
    )
    assert aged.stale.item()
    assert aged.shield.action.item() == ShieldAction.HOLD_DOUBLE_SUPPORT


def test_reset_clears_history_timebase_cadence_and_shield_state() -> None:
    controller, estimator, supervisor, locomotion = make_controller()
    first = controller.step(
        frame(torch.zeros(2, 2, 19), 2.0),
        stock_observation=torch.zeros(2, 8),
        requested_velocity_yaw=torch.ones(2, 3),
        shield_signals=signals(2, target_safe=False),
    )
    assert first.recovery_request.all()
    controller.reset()
    restarted = controller.step(
        frame(torch.ones(1, 2, 19), 0.0),
        stock_observation=torch.zeros(1, 8),
        requested_velocity_yaw=torch.ones(1, 3),
        shield_signals=signals(1),
    )
    assert restarted.history.values.shape == (1, 1, 2, 19)
    assert restarted.control_updated
    assert restarted.shield.action.item() == ShieldAction.COMMIT
    assert estimator.calls == 2
    assert supervisor.calls == 2
    assert locomotion.calls == 2


def test_outputs_are_finite_device_resident_and_locomotion_shape_is_exact() -> None:
    controller, _, _, _ = make_controller()
    output = controller.step(
        frame(torch.ones(2, 2, 19), 0.0),
        stock_observation=torch.ones(2, 8),
        requested_velocity_yaw=torch.ones(2, 3),
        shield_signals=signals(2),
    )
    tensors = (
        output.joint_action,
        output.safe_command.velocity_yaw,
        output.safe_command.recovery_request,
        output.estimator.bilateral_latent,
        output.supervisor.action_logits,
        output.shield.action,
        output.history.values,
        output.history.mask,
        output.history.age_s,
        output.stale,
    )
    assert all(value.device == output.joint_action.device for value in tensors)
    for value in tensors:
        if value.is_floating_point():
            assert torch.isfinite(value).all()
    assert output.joint_action.shape == (2, 37)
    assert not output.joint_action.requires_grad

    bad_controller, _, _, _ = make_controller()
    with pytest.raises(ValueError, match=r"exact shape \[B, 37\]"):
        bad_controller.locomotion_policy = lambda observation, command: torch.zeros(2, 36)
        bad_controller.step(
            frame(torch.ones(2, 2, 19), 0.0),
            stock_observation=torch.ones(2, 8),
            requested_velocity_yaw=torch.ones(2, 3),
            shield_signals=signals(2),
        )

    nonfinite, _, _, _ = make_controller()
    values = torch.zeros(1, 2, 19)
    values[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        nonfinite.step(
            frame(values, 0.0),
            stock_observation=torch.zeros(1, 8),
            requested_velocity_yaw=torch.zeros(1, 3),
            shield_signals=signals(1),
        )


def test_recovery_output_is_only_a_request_and_never_an_exact_replant_claim() -> None:
    assert not hasattr(ShieldAction, "REPLANT")
    controller, _, _, _ = make_controller()
    output = controller.step(
        frame(torch.zeros(1, 2, 19), 0.0),
        stock_observation=torch.zeros(1, 8),
        requested_velocity_yaw=torch.zeros(1, 3),
        shield_signals=signals(1, target_safe=False),
    )
    assert output.recovery_request.dtype == torch.bool
    assert output.recovery_request.item()
    assert not hasattr(output, "replant")


def test_selective_reset_clears_only_terminated_vector_rows() -> None:
    estimator = RecordingEstimator()
    supervisor = RecordingSupervisor()
    locomotion = RecordingLocomotion()
    controller = EverestController(
        estimator=estimator,
        supervisor=supervisor,
        locomotion_policy=locomotion,
        shield=SafetyShield(ShieldConfig(min_dwell_steps=1, commit_hysteresis_steps=1)),
        config=EverestControllerConfig(
            history_steps=3,
            control_rate_hz=25.0,
            stale_after_s=0.05,
            minimum_history_steps=3,
        ),
    )
    observation = torch.zeros(2, 8)
    command = torch.ones(2, 3)
    controller.step(
        frame(torch.ones(2, 2, 19), 0.0),
        stock_observation=observation,
        requested_velocity_yaw=command,
        shield_signals=signals(2),
    )
    controller.step(
        frame(torch.full((2, 2, 19), 2.0), 0.01),
        stock_observation=observation,
        requested_velocity_yaw=command,
        shield_signals=signals(2),
    )
    controller.reset_environments(torch.tensor([True, False]))
    values = torch.stack((torch.full((2, 19), 9.0), torch.full((2, 19), 3.0)), dim=0)
    restarted = controller.step(
        frame(values, 0.02),
        stock_observation=observation,
        requested_velocity_yaw=command,
        shield_signals=signals(2),
    )
    assert restarted.control_updated
    assert restarted.stale.tolist() == [True, True]
    assert restarted.shield.action.tolist() == [
        ShieldAction.HOLD_DOUBLE_SUPPORT,
        ShieldAction.HOLD_DOUBLE_SUPPORT,
    ]
    torch.testing.assert_close(restarted.history.values[0, :2], torch.zeros(2, 2, 19))
    torch.testing.assert_close(restarted.history.values[1, :, 0, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert not restarted.history.mask[0, :2].any()
    assert restarted.history.mask[1].all()
    controller.step(
        frame(values, 0.03),
        stock_observation=observation,
        requested_velocity_yaw=command,
        shield_signals=signals(2),
    )
    released = controller.step(
        frame(values, 0.04),
        stock_observation=observation,
        requested_velocity_yaw=command,
        shield_signals=signals(2),
    )
    assert released.shield.action.tolist() == [
        ShieldAction.HOLD_DOUBLE_SUPPORT,
        ShieldAction.COMMIT,
    ]
