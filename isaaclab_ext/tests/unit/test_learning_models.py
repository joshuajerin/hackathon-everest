from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SOURCE = Path(__file__).parents[2] / "source/hackathon_everest_isaaclab"
sys.path.insert(0, str(SOURCE))

from hackathon_everest_isaaclab.learning.models import (
    ACTION_NAMES,
    EVENT_DIM,
    REGRESSION_DIM,
    SENSOR_CHANNELS,
    CausalBilateralEstimator,
    VisibleEverestPolicy,
    VisibleOnlySupervisor,
    calibrated_commit_logit_subtraction,
    supervisor_selection_score,
)


def visible_inputs(batch: int, time: int, *, context_dim: int = 0):
    history = torch.randn(batch, time, 2, SENSOR_CHANNELS)
    valid = torch.ones_like(history, dtype=torch.bool)
    age = torch.zeros_like(history)
    context = None if context_dim == 0 else torch.randn(batch, time, 2, context_dim)
    return history, valid, age, context


def test_causal_shared_estimator_preserves_bilateral_contract_and_outputs() -> None:
    torch.manual_seed(3)
    model = CausalBilateralEstimator(hidden_size=12, bilateral_latent_size=7, context_dim=4)
    history, valid, age, context = visible_inputs(4, 30, context_dim=4)
    output = model(history, valid, age, context)
    assert output.regression_mean.shape == (4, 2, REGRESSION_DIM)
    assert output.regression_log_scale.shape == (4, 2, REGRESSION_DIM)
    assert output.event_logits.shape == (4, 2, EVENT_DIM)
    assert output.per_foot_latent.shape == (4, 2, 12)
    assert output.bilateral_latent.shape == (4, 7)
    assert model.foot_gru.input_size == 3 * SENSOR_CHANNELS + 4

    swapped = model(history.flip(2), valid.flip(2), age.flip(2), context.flip(2))
    torch.testing.assert_close(swapped.regression_mean, output.regression_mean.flip(1))
    torch.testing.assert_close(swapped.event_logits, output.event_logits.flip(1))
    assert model.foot_gru.bidirectional is False


def test_prefix_result_is_causal_and_normalization_covers_all_visible_features() -> None:
    torch.manual_seed(4)
    history, valid, age, context = visible_inputs(2, 14, context_dim=3)
    model = CausalBilateralEstimator(hidden_size=10, bilateral_latent_size=6, context_dim=3)
    prefix_a = model(history[:, :9], valid[:, :9], age[:, :9], context[:, :9])
    changed_future = history.clone()
    changed_future[:, 9:] = 100.0 * torch.randn_like(changed_future[:, 9:])
    prefix_b = model(changed_future[:, :9], valid[:, :9], age[:, :9], context[:, :9])
    torch.testing.assert_close(prefix_a.regression_mean, prefix_b.regression_mean)

    width = 3 * SENSOR_CHANNELS + 3
    normalized = CausalBilateralEstimator(
        hidden_size=10,
        bilateral_latent_size=6,
        context_dim=3,
        input_mean=torch.arange(width, dtype=torch.float32),
        input_std=torch.full((width,), 4.0),
    )
    captured = []
    handle = normalized.foot_gru.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    normalized(history, valid, age, context)
    handle.remove()
    raw = torch.cat((history, valid.float(), age, context), dim=-1)
    expected = ((raw - normalized.input_mean) / normalized.input_std).clamp(
        -normalized.maximum_normalized_input, normalized.maximum_normalized_input
    )
    expected = expected.permute(0, 2, 1, 3).reshape(4, 14, width)
    torch.testing.assert_close(captured[0], expected)
    normalized.set_input_normalization(torch.zeros(width), torch.ones(width))
    assert torch.equal(normalized.input_mean, torch.zeros(width))


def test_values_validity_age_and_context_each_influence_the_estimator() -> None:
    torch.manual_seed(9)
    model = CausalBilateralEstimator(hidden_size=16, bilateral_latent_size=8, context_dim=2)
    history, valid, age, context = visible_inputs(2, 8, context_dim=2)
    baseline = model(history, valid, age, context).per_foot_latent

    changed_values = history.clone()
    changed_values[:, -1, 0, 0] += 50.0
    changed_mask = valid.clone()
    changed_mask[:, -1, 0, 0] = False
    changed_age = age.clone()
    changed_age[:, -1, 0, 0] = 0.25
    changed_context = context.clone()
    changed_context[:, -1, 0, 0] += 10.0
    variants = (
        model(changed_values, valid, age, context).per_foot_latent,
        model(history, changed_mask, age, context).per_foot_latent,
        model(history, valid, changed_age, context).per_foot_latent,
        model(history, valid, age, changed_context).per_foot_latent,
    )
    assert all(not torch.equal(baseline, variant) for variant in variants)


def test_visible_metadata_and_configured_context_are_mandatory_and_exact() -> None:
    model = CausalBilateralEstimator(context_dim=2)
    history, valid, age, context = visible_inputs(2, 5, context_dim=2)
    with pytest.raises(TypeError, match="valid_mask"):
        model(history, age, age, context)
    with pytest.raises(ValueError, match="sample_age_s"):
        model(history, valid, age[..., :-1], context)
    with pytest.raises(ValueError, match="deployable_context is required"):
        model(history, valid, age)
    with pytest.raises(ValueError, match="deployable_context must have shape"):
        model(history, valid, age, torch.zeros(2, 5, 2, 1))


def test_mixed_dtype_visible_features_cannot_overflow_silently() -> None:
    estimator = CausalBilateralEstimator(context_dim=2)
    history, valid, age, _ = visible_inputs(1, 3, context_dim=2)
    huge_context = torch.full((1, 3, 2, 2), 1.0e100, dtype=torch.float64)
    with pytest.raises(ValueError, match="overflowed"):
        estimator(history, valid, age, huge_context)
    huge_age = torch.full((1, 3, 2, 19), 1.0e100, dtype=torch.float64)
    with pytest.raises(ValueError, match="overflowed"):
        estimator(history, valid, huge_age, torch.zeros(1, 3, 2, 2))

    supervisor = VisibleOnlySupervisor(bilateral_latent_size=4, command_gait_context_dim=2)
    with pytest.raises(ValueError, match="overflowed"):
        supervisor(torch.zeros(1, 4), torch.full((1, 2), 1.0e100, dtype=torch.float64))


def test_supervisor_outputs_three_logits_and_uses_deployable_command_gait_context() -> None:
    torch.manual_seed(11)
    supervisor = VisibleOnlySupervisor(
        bilateral_latent_size=7, hidden_size=9, command_gait_context_dim=5
    )
    latent = torch.randn(128, 7) * 100.0
    command_gait = torch.randn(128, 5)
    output = supervisor(latent, command_gait)
    assert ACTION_NAMES == ("COMMIT", "HOLD_DOUBLE_SUPPORT", "REQUEST_RECOVERY")
    assert output.action_logits.shape == (128, 3)
    assert output.continuous_proposals.shape == (128, len(supervisor.proposal_names))
    assert torch.all(output.continuous_proposals >= supervisor.proposal_lower)
    assert torch.all(output.continuous_proposals <= supervisor.proposal_upper)
    changed = command_gait.clone()
    changed[:, 0] += 20.0
    changed_output = supervisor(latent, changed)
    assert not torch.equal(output.action_logits, changed_output.action_logits)
    with pytest.raises(ValueError, match="is required"):
        supervisor(latent)


def test_supervisor_rejects_non_finite_or_float32_overflowing_bounds() -> None:
    with pytest.raises(ValueError, match="finite float32"):
        VisibleOnlySupervisor(proposal_bounds=(("unsafe", 0.0, float("inf")),))
    with pytest.raises(ValueError, match="finite float32"):
        VisibleOnlySupervisor(proposal_bounds=(("unsafe", 0.0, 1.0e100),))


def test_forward_apis_reject_flattening_and_have_no_privileged_parameters() -> None:
    estimator = CausalBilateralEstimator(hidden_size=8, bilateral_latent_size=5)
    supervisor = VisibleOnlySupervisor(bilateral_latent_size=5)
    policy = VisibleEverestPolicy(estimator, supervisor)
    history, valid, age, _ = visible_inputs(2, 10)
    with pytest.raises(ValueError, match=r"\[B, T, 2, 19\]"):
        estimator(torch.randn(2, 10, 38), valid, age)
    with pytest.raises(TypeError):
        estimator.forward(history, valid, age, truth_canary=torch.ones(2))
    with pytest.raises(TypeError):
        supervisor.forward(torch.randn(2, 5), privileged_truth=torch.ones(2))
    with pytest.raises(TypeError):
        policy.forward(history, valid, age, truth_canary=torch.ones(2))

    for forward in (estimator.forward, supervisor.forward, policy.forward):
        names = set(inspect.signature(forward).parameters)
        assert not names.intersection({"truth", "truth_canary", "privileged", "oracle"})


def test_supervisor_selection_balances_unsafe_commit_and_useful_commit_recall() -> None:
    useful_safe = supervisor_selection_score(
        {
            "unsafe_commit_rate": 0.0005,
            "teacher_action_accuracy": 0.85,
            "commit_recall": 0.80,
        }
    )
    unsafe_accurate = supervisor_selection_score(
        {
            "unsafe_commit_rate": 0.005,
            "teacher_action_accuracy": 0.95,
            "commit_recall": 0.95,
        }
    )
    never_commit = supervisor_selection_score(
        {
            "unsafe_commit_rate": 0.0,
            "teacher_action_accuracy": 0.65,
            "commit_recall": 0.0,
        }
    )
    assert useful_safe < unsafe_accurate
    assert useful_safe < never_commit


def test_commit_logit_calibration_meets_empirical_unsafe_rate_bound() -> None:
    logits = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    expected = torch.tensor([1, 2, 1, 2, 0], dtype=torch.int64)
    subtraction = calibrated_commit_logit_subtraction(
        logits, expected, maximum_unsafe_commit_rate=0.25
    )
    adjusted = logits.clone()
    adjusted[:, 0] -= subtraction
    predicted = adjusted.argmax(dim=-1)
    unsafe = expected != 0
    assert float((predicted[unsafe] == 0).float().mean()) <= 0.25
    assert predicted[-1].item() == 0
