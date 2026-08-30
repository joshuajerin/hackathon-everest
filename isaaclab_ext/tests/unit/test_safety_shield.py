from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SOURCE = Path(__file__).parents[2] / "source/hackathon_everest_isaaclab"
sys.path.insert(0, str(SOURCE))

from hackathon_everest_isaaclab.learning.shield import (
    SafetyShield,
    ShieldAction,
    ShieldConfig,
    ShieldReason,
    ShieldSignals,
    conservative_target_safe,
)


def signals(
    *,
    stale: bool = False,
    ood: bool = False,
    target_safe: bool = True,
    stance_safe: bool = True,
    settling: bool = False,
    batch: int = 1,
) -> ShieldSignals:
    full = lambda value: torch.full((batch,), value, dtype=torch.bool)
    return ShieldSignals(
        stale=full(stale),
        ood=full(ood),
        target_safe=full(target_safe),
        stance_safe=full(stance_safe),
        settling=full(settling),
    )


def test_memoryless_precedence_is_exact_and_combined_hazards_do_not_override_it() -> None:
    cases = (
        (
            signals(stale=True, target_safe=False, stance_safe=False),
            ShieldAction.HOLD_DOUBLE_SUPPORT,
            ShieldReason.STALE_OR_OOD,
        ),
        (
            signals(ood=True, target_safe=False),
            ShieldAction.HOLD_DOUBLE_SUPPORT,
            ShieldReason.STALE_OR_OOD,
        ),
        (
            signals(target_safe=False, stance_safe=False),
            ShieldAction.REQUEST_RECOVERY,
            ShieldReason.UNSAFE_TARGET,
        ),
        (
            signals(stance_safe=False),
            ShieldAction.HOLD_DOUBLE_SUPPORT,
            ShieldReason.UNSAFE_STANCE_OR_SETTLING,
        ),
        (
            signals(settling=True),
            ShieldAction.HOLD_DOUBLE_SUPPORT,
            ShieldReason.UNSAFE_STANCE_OR_SETTLING,
        ),
        (signals(), ShieldAction.COMMIT, ShieldReason.SAFE_TO_COMMIT),
    )
    for observation, expected_action, expected_reason in cases:
        action, reason = SafetyShield.precedence(observation)
        assert action.item() == int(expected_action)
        assert reason.item() == int(expected_reason)


def test_hazards_are_immediate_but_commit_waits_for_dwell_and_hysteresis() -> None:
    shield = SafetyShield(ShieldConfig(min_dwell_steps=2, commit_hysteresis_steps=2))
    assert shield.step(signals()).action.item() == ShieldAction.COMMIT
    unsafe = shield.step(signals(target_safe=False))
    assert unsafe.action.item() == ShieldAction.REQUEST_RECOVERY
    first_safe = shield.step(signals())
    assert first_safe.action.item() == ShieldAction.REQUEST_RECOVERY
    assert first_safe.reason.item() == ShieldReason.RELEASE_HYSTERESIS
    second_safe = shield.step(signals())
    assert second_safe.action.item() == ShieldAction.COMMIT

    # Higher-precedence hazards are never delayed by a previous minimum dwell.
    assert shield.step(signals(target_safe=False)).action.item() == ShieldAction.REQUEST_RECOVERY
    stale = shield.step(signals(stale=True, target_safe=False))
    assert stale.action.item() == ShieldAction.HOLD_DOUBLE_SUPPORT
    assert stale.reason.item() == ShieldReason.STALE_OR_OOD


def test_batched_state_is_isolated_and_batch_change_requires_reset() -> None:
    shield = SafetyShield(ShieldConfig(min_dwell_steps=1, commit_hysteresis_steps=2))
    observation = ShieldSignals(
        stale=torch.tensor([False, False]),
        ood=torch.tensor([False, False]),
        target_safe=torch.tensor([False, True]),
        stance_safe=torch.tensor([True, False]),
        settling=torch.tensor([False, False]),
    )
    output = shield.step(observation)
    assert output.action.tolist() == [
        int(ShieldAction.REQUEST_RECOVERY),
        int(ShieldAction.HOLD_DOUBLE_SUPPORT),
    ]
    with pytest.raises(ValueError, match="batch size changed"):
        shield.step(signals(batch=1))
    shield.reset()
    assert shield.step(signals()).action.item() == ShieldAction.COMMIT


def test_recovery_is_only_a_request_and_there_is_no_replant_action() -> None:
    assert set(ShieldAction.__members__) == {
        "COMMIT",
        "HOLD_DOUBLE_SUPPORT",
        "REQUEST_RECOVERY",
    }
    assert not hasattr(ShieldAction, "REPLANT")


def test_conservative_target_gate_requires_both_feet_and_no_hazard() -> None:
    mean = torch.full((2, 2, 4), 500.0)
    log_scale = torch.zeros_like(mean)
    event_logits = torch.full((2, 2, 3), -10.0)
    conformal = torch.full((4,), 2.0)
    safe = conservative_target_safe(
        mean,
        log_scale,
        event_logits,
        conformal,
        bearing_capacity_index=1,
        damage_index=2,
        slip_margin_index=3,
    )
    assert safe.tolist() == [True, True]

    mean[0, 1, 1] = 340.0
    event_logits[1, 0, 1] = 10.0
    safe = conservative_target_safe(
        mean,
        log_scale,
        event_logits,
        conformal,
        bearing_capacity_index=1,
        damage_index=2,
        slip_margin_index=3,
    )
    assert safe.tolist() == [False, False]


def test_conservative_target_gate_uses_conformal_lower_bound() -> None:
    mean = torch.full((1, 2, 4), 350.0)
    log_scale = torch.zeros_like(mean)
    event_logits = torch.full((1, 2, 3), -10.0)
    safe = conservative_target_safe(
        mean,
        log_scale,
        event_logits,
        torch.full((4,), 10.0),
        bearing_capacity_index=1,
        damage_index=2,
        slip_margin_index=3,
    )
    assert safe.tolist() == [False]


def test_target_gate_allows_supported_fracture_and_unilateral_transient_slip() -> None:
    mean = torch.zeros((2, 2, 4))
    mean[..., 1] = 500.0
    mean[..., 2] = 0.30
    mean[..., 3] = -150.0
    log_scale = torch.zeros_like(mean)
    events = torch.full((2, 2, 3), -10.0)
    events[0, 0, 1] = 10.0
    events[1, 0, 2] = 10.0
    safe = conservative_target_safe(
        mean,
        log_scale,
        events,
        torch.zeros(4),
        bearing_capacity_index=1,
        damage_index=2,
        slip_margin_index=3,
    )
    assert safe.tolist() == [True, True]

    events[1, 1, 2] = 10.0
    safe = conservative_target_safe(
        mean,
        log_scale,
        events,
        torch.zeros(4),
        bearing_capacity_index=1,
        damage_index=2,
        slip_margin_index=3,
    )
    assert safe.tolist() == [True, False]


def test_target_gate_can_use_absolute_bearing_conformal_radius() -> None:
    mean = torch.zeros((1, 2, 4))
    mean[..., 1] = 600.0
    log_scale = torch.full_like(mean, 5.0)
    events = torch.full((1, 2, 3), -10.0)
    safe = conservative_target_safe(
        mean,
        log_scale,
        events,
        torch.full((4,), 9.0),
        bearing_capacity_index=1,
        bearing_capacity_absolute_radius_n=220.0,
        damage_index=2,
        slip_margin_index=3,
    )
    assert safe.tolist() == [True]

    with pytest.raises(ValueError, match="non-negative"):
        conservative_target_safe(
            mean,
            log_scale,
            events,
            torch.full((4,), 9.0),
            bearing_capacity_index=1,
            bearing_capacity_absolute_radius_n=-1.0,
            damage_index=2,
            slip_margin_index=3,
        )
