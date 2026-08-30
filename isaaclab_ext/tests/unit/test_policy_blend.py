from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SOURCE = Path(__file__).parents[2] / "source/hackathon_everest_isaaclab"
sys.path.insert(0, str(SOURCE))

from hackathon_everest_isaaclab.runtime.policy_blend import (
    SmoothPolicyBlend,
    SmoothPolicyBlendConfig,
)


def test_blend_slews_eases_and_bounds_specialist_residual() -> None:
    blend = SmoothPolicyBlend(
        SmoothPolicyBlendConfig(maximum_weight_step=0.25, maximum_action_residual=0.4)
    )
    stock = torch.zeros((2, 3))
    specialist = torch.full((2, 3), 2.0)
    target = torch.ones(2)
    outputs = [blend.step(stock, specialist, target) for _ in range(4)]
    assert torch.allclose(outputs[0], torch.full((2, 3), 0.0625))
    assert torch.allclose(outputs[-1], torch.full((2, 3), 0.4))
    assert all(bool((later >= earlier).all()) for earlier, later in pairwise(outputs))


def test_selective_reset_returns_only_selected_environment_to_stock() -> None:
    blend = SmoothPolicyBlend(SmoothPolicyBlendConfig(maximum_weight_step=1.0))
    stock = torch.zeros((2, 2))
    specialist = torch.ones((2, 2))
    blend.step(stock, specialist, torch.ones(2))
    blend.reset(torch.tensor([True, False]))
    output = blend.step(stock, specialist, torch.tensor([0.0, 1.0]))
    assert torch.allclose(output[0], stock[0])
    assert bool((output[1] > 0.0).all())


def test_blend_rejects_out_of_range_target() -> None:
    blend = SmoothPolicyBlend()
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        blend.step(torch.zeros((1, 2)), torch.zeros((1, 2)), torch.tensor([1.1]))
