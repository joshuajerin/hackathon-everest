from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SOURCE = Path(__file__).parents[2] / "source/hackathon_everest_isaaclab"
sys.path.insert(0, str(SOURCE))

from hackathon_everest_isaaclab.runtime.joint_residual import (
    SensorGatedJointResidual,
    SensorJointResidualConfig,
)

ACTION_JOINT_NAMES = (
    "left_ankle_pitch_joint",
    "left_shoulder_pitch_joint",
    "right_ankle_pitch_joint",
    "right_shoulder_pitch_joint",
    "torso_joint",
)


def _metadata(action_dim: int = 5) -> dict[str, torch.Tensor]:
    return {
        "action_scale": torch.tensor([0.5, 0.25, 2.0, 1.0, 1.0]),
        "action_offset": torch.tensor([0.1, -0.2, 0.0, 0.0, 0.0]),
        "joint_position_lower_rad": torch.tensor([-1.0, -0.25, -1.0, -1.0, -1.0]),
        "joint_position_upper_rad": torch.tensor([1.0, 0.25, 1.0, 1.0, 1.0]),
    }


def _layer() -> SensorGatedJointResidual:
    return SensorGatedJointResidual(
        SensorJointResidualConfig(
            joint_names=("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
            maximum_residual_rad=(0.08, 0.06),
            maximum_target_step_rad=0.03,
        )
    )


def test_named_sensor_residual_is_in_radians_and_leaves_other_actions_exact() -> None:
    layer = _layer()
    stock = torch.tensor([[0.2, 0.1, -0.3, 0.4, -0.5]])
    # tanh(20) is effectively one, so requested offsets are +0.08 and -0.06 rad.
    raw = torch.tensor([[20.0, -20.0]])
    output = layer.step(
        stock,
        raw,
        action_joint_names=ACTION_JOINT_NAMES,
        enabled=torch.tensor([True]),
        **_metadata(),
    )

    # The 0.03-rad per-tick limit applies in physical joint coordinates.
    torch.testing.assert_close(output.applied_residual_rad, torch.tensor([[0.03, -0.03]]))
    stock_targets = stock * _metadata()["action_scale"] + _metadata()["action_offset"]
    torch.testing.assert_close(
        output.joint_targets_rad[:, [0, 2]], stock_targets[:, [0, 2]] + output.applied_residual_rad
    )
    # Index 1, 3, and 4 are unchanged bit-for-bit.
    assert torch.equal(output.action[:, [1, 3, 4]], stock[:, [1, 3, 4]])


def test_unsafe_gate_removes_prior_joint_correction_immediately() -> None:
    layer = _layer()
    stock = torch.zeros((1, 5))
    active = layer.step(
        stock,
        torch.full((1, 2), 20.0),
        action_joint_names=ACTION_JOINT_NAMES,
        enabled=torch.tensor([True]),
        **_metadata(),
    )
    assert bool((active.applied_residual_rad > 0.0).all())

    held = layer.step(
        stock,
        torch.full((1, 2), 20.0),
        action_joint_names=ACTION_JOINT_NAMES,
        enabled=torch.tensor([False]),
        **_metadata(),
    )
    torch.testing.assert_close(held.action, stock)
    torch.testing.assert_close(held.applied_residual_rad, torch.zeros((1, 2)))


def test_joint_limit_clamps_target_and_back_projects_to_action_abi() -> None:
    layer = _layer()
    stock = torch.tensor([[1.9, 0.0, 0.0, 0.0, 0.0]])
    output = layer.step(
        stock,
        torch.tensor([[20.0, 0.0]]),
        action_joint_names=ACTION_JOINT_NAMES,
        enabled=torch.tensor([True]),
        **_metadata(),
    )

    # Stock target on joint 0 is 1.05 rad; physical upper limit wins.
    assert output.joint_targets_rad[0, 0].item() == pytest.approx(1.0)
    # The returned action is transformed back using q = scale * action + offset.
    assert output.action[0, 0].item() == pytest.approx((1.0 - 0.1) / 0.5)


def test_configured_names_are_resolved_from_reordered_action_abi() -> None:
    layer = _layer()
    action_names = (
        "right_ankle_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_shoulder_pitch_joint",
        "torso_joint",
    )
    output = layer.step(
        torch.zeros((1, 5)),
        torch.tensor([[20.0, -20.0]]),
        action_joint_names=action_names,
        enabled=torch.tensor([True]),
        **_metadata(),
    )

    # Config order is left, right; action order is right, left.
    assert output.joint_targets_rad[0, 2].item() > _metadata()["action_offset"][2].item()
    assert output.joint_targets_rad[0, 0].item() < _metadata()["action_offset"][0].item()
    assert torch.equal(output.action[:, [1, 3, 4]], torch.zeros((1, 3)))


def test_rejects_invalid_action_abi_or_zero_action_scale() -> None:
    layer = _layer()
    common = dict(
        stock_action=torch.zeros((1, 5)),
        raw_residual=torch.zeros((1, 2)),
        enabled=torch.tensor([True]),
        **_metadata(),
    )
    with pytest.raises(ValueError, match="unique"):
        layer.step(
            action_joint_names=(
                "left_ankle_pitch_joint",
                "left_shoulder_pitch_joint",
                "right_ankle_pitch_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_pitch_joint",
            ),
            **common,
        )
    with pytest.raises(ValueError, match="missing"):
        layer.step(
            action_joint_names=(
                "left_ankle_pitch_joint",
                "left_shoulder_pitch_joint",
                "left_elbow_joint",
                "right_shoulder_pitch_joint",
                "torso_joint",
            ),
            **common,
        )
    bad = _metadata()
    bad["action_scale"][2] = 0.0
    with pytest.raises(ValueError, match="must not contain zero"):
        layer.step(
            action_joint_names=ACTION_JOINT_NAMES,
            stock_action=torch.zeros((1, 5)),
            raw_residual=torch.zeros((1, 2)),
            enabled=torch.tensor([True]),
            **bad,
        )


def test_selective_reset_clears_only_selected_environment() -> None:
    layer = _layer()
    stock = torch.zeros((2, 5))
    layer.step(
        stock,
        torch.full((2, 2), 20.0),
        action_joint_names=ACTION_JOINT_NAMES,
        enabled=torch.tensor([True, True]),
        **_metadata(),
    )
    layer.reset(torch.tensor([True, False]))
    assert layer.applied_residual_rad is not None
    torch.testing.assert_close(layer.applied_residual_rad[0], torch.zeros(2))
    assert bool((layer.applied_residual_rad[1] > 0.0).all())
