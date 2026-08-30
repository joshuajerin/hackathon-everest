from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")

SOURCE = Path(__file__).parents[2] / "source/hackathon_everest_isaaclab"
sys.path.insert(0, str(SOURCE))

from hackathon_everest_isaaclab.tasks.manager_based.crampon_velocity.env_cfg import (
    EverestFrontPointLinearTrackingRewardsCfg,
    EverestFrontPointLoadCorrectedRewardsCfg,
    EverestFrontPointPositiveTrackingRewardsCfg,
    everest_contact_mode_rear_load_exp,
    everest_contact_mode_rear_load_l2,
    everest_contact_mode_rear_load_linear,
    everest_forward_velocity_error_l2,
)


class _Commands:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def get_command(self, _name: str) -> torch.Tensor:
        return self.value


def test_forward_velocity_penalty_is_direct_squared_error() -> None:
    velocity = torch.tensor([[0.15, 0.0, 0.0], [0.55, 0.0, 0.0]])
    command = torch.tensor([[0.15, 0.0, 0.0], [0.15, 0.0, 0.0]])
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_lin_vel_b=velocity))},
        command_manager=_Commands(command),
    )
    actual = everest_forward_velocity_error_l2(env, "base_velocity")
    assert torch.allclose(actual, torch.tensor([0.0, 0.16]))


def test_rear_load_penalty_targets_front_dominant_fraction() -> None:
    # Probe order is front pair then rear pair for each foot.
    force = torch.tensor(
        [
            [[40.0, 40.0, 10.0, 10.0], [40.0, 40.0, 10.0, 10.0]],
            [[25.0, 25.0, 25.0, 25.0], [25.0, 25.0, 25.0, 25.0]],
        ]
    )
    env = SimpleNamespace(
        everest_latest_wrench=SimpleNamespace(probe_normal_force_n=force),
        _everest_target_rear_load_fraction=torch.tensor([0.20, 0.20]),
    )
    actual = everest_contact_mode_rear_load_l2(env, 20.0)
    assert torch.allclose(actual, torch.tensor([0.0, 0.09]), atol=1.0e-7)


def test_corrected_reward_removes_sagittal_proxy_and_strengthens_direct_terms() -> None:
    rewards = EverestFrontPointLoadCorrectedRewardsCfg()
    assert rewards.everest_sagittal_toe_support is None
    assert rewards.everest_contact_mode_rear_load.weight == -40.0
    assert rewards.everest_forward_velocity_error.weight == -20.0


def test_positive_load_tracking_is_bounded_and_prefers_target() -> None:
    force = torch.tensor(
        [
            [[40.0, 40.0, 10.0, 10.0], [40.0, 40.0, 10.0, 10.0]],
            [[25.0, 25.0, 25.0, 25.0], [25.0, 25.0, 25.0, 25.0]],
        ]
    )
    env = SimpleNamespace(
        everest_latest_wrench=SimpleNamespace(probe_normal_force_n=force),
        _everest_target_rear_load_fraction=torch.tensor([0.20, 0.20]),
    )
    actual = everest_contact_mode_rear_load_exp(env, 20.0, 0.15)
    assert actual[0].item() == pytest.approx(1.0)
    assert 0.0 < actual[1].item() < 0.02

    rewards = EverestFrontPointPositiveTrackingRewardsCfg()
    assert rewards.everest_sagittal_toe_support is None
    assert rewards.everest_contact_mode_rear_load.weight == 10.0


def test_linear_load_tracking_has_dense_signal_from_stock_load() -> None:
    force = torch.tensor(
        [
            [[40.0, 40.0, 10.0, 10.0], [40.0, 40.0, 10.0, 10.0]],
            [[25.0, 25.0, 25.0, 25.0], [25.0, 25.0, 25.0, 25.0]],
        ]
    )
    env = SimpleNamespace(
        everest_latest_wrench=SimpleNamespace(probe_normal_force_n=force),
        _everest_target_rear_load_fraction=torch.tensor([0.20, 0.20]),
    )
    actual = everest_contact_mode_rear_load_linear(env, 20.0)
    assert actual[0].item() == pytest.approx(1.0)
    assert actual[1].item() == pytest.approx(0.70)

    rewards = EverestFrontPointLinearTrackingRewardsCfg()
    assert rewards.everest_contact_mode_rear_load.weight == 20.0
    assert rewards.everest_sagittal_toe_support is None
