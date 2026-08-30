from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SOURCE = Path(__file__).parents[2] / "source/hackathon_everest_isaaclab"
sys.path.insert(0, str(SOURCE))

from hackathon_everest_isaaclab.runtime.contact_correction import (
    ContactGatedPolicyCorrection,
    ContactGatedPolicyCorrectionConfig,
    visible_crampon_contact,
)


def test_correction_is_zero_before_contact_and_bounded_after_contact() -> None:
    correction = ContactGatedPolicyCorrection(
        ContactGatedPolicyCorrectionConfig(
            maximum_weight_step=1.0, maximum_action_residual=0.12
        )
    )
    stock = torch.zeros((2, 3))
    specialist = torch.full((2, 3), 1.0)

    output = correction.step(stock, specialist, torch.tensor([False, True]))

    torch.testing.assert_close(output[0], stock[0])
    torch.testing.assert_close(output[1], torch.full((3,), 0.12))


def test_contact_loss_slews_correction_back_to_stock() -> None:
    correction = ContactGatedPolicyCorrection(
        ContactGatedPolicyCorrectionConfig(
            maximum_weight_step=0.5, maximum_action_residual=0.12
        )
    )
    stock = torch.zeros((1, 2))
    specialist = torch.ones((1, 2))

    applied = correction.step(stock, specialist, torch.tensor([True]))
    released = correction.step(stock, specialist, torch.tensor([False]))

    assert bool((applied > released).all())
    torch.testing.assert_close(released, stock)


def test_correction_rejects_non_boolean_contact_mask() -> None:
    correction = ContactGatedPolicyCorrection()
    with pytest.raises(TypeError, match="bool"):
        correction.step(torch.zeros((1, 2)), torch.zeros((1, 2)), torch.ones(1))


def test_visible_contact_requires_fresh_valid_force_and_penetration() -> None:
    values = torch.zeros((2, 2, 19))
    values[0, 0, 0] = 10.0
    values[0, 0, 4] = 0.01
    values[1, 1, 0] = 10.0
    values[1, 1, 4] = 0.01
    valid = torch.ones((2, 2, 19), dtype=torch.bool)
    age = torch.zeros((2, 2, 19))
    valid[1, 1, 4] = False

    contact = visible_crampon_contact(values, valid, age, stale_after_s=0.05)

    assert contact.tolist() == [True, False]


def test_visible_contact_fails_closed_for_stale_channels() -> None:
    values = torch.zeros((1, 2, 19))
    values[0, 0, 0] = 10.0
    values[0, 0, 4] = 0.01
    valid = torch.ones((1, 2, 19), dtype=torch.bool)
    age = torch.zeros((1, 2, 19))
    age[0, 0, 0] = 0.051

    contact = visible_crampon_contact(values, valid, age, stale_after_s=0.05)

    assert contact.tolist() == [False]
