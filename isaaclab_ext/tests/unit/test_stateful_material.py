from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SOURCE = Path(__file__).parents[2] / "source/hackathon_everest_isaaclab"
sys.path.insert(0, str(SOURCE))

from hackathon_everest_isaaclab.contact.stateful_material import (
    ICE,
    SNOW,
    BatchedMaterialParameters,
    BatchedStatefulMaterial,
)


def parameters(*, environments: int = 2, material: int = SNOW) -> BatchedMaterialParameters:
    shape = (environments, 2, 4)
    full = lambda value: torch.full(shape, value, dtype=torch.float32)
    return BatchedMaterialParameters(
        material_code=torch.full(shape, material, dtype=torch.int64),
        vertical_stiffness_n_per_m=full(18_000.0),
        damping_ns_per_m=full(45.0),
        bearing_capacity_n=full(720.0),
        shear_capacity_n=full(320.0),
        friction=full(0.24 if material == SNOW else 0.08),
        support_layer_depth_m=full(0.025),
        crust_thickness_m=full(0.004),
        fracture_strength_n=full(80.0),
        void_present=torch.zeros(shape, dtype=torch.bool),
        void_top_depth_m=full(0.03),
        void_height_m=full(0.20),
        ice_indentation_pressure_pa=full(40e6),
        ice_tip_radius_m=full(0.0004),
        ice_cone_half_angle_rad=full(0.5235987756),
        ice_shank_radius_m=full(0.003),
        ice_fracture_energy_j=full(0.02),
        ice_post_fracture_strength_ratio=full(0.4),
        breakout_displacement_m=full(0.003),
        maximum_probe_force_n=full(250.0),
    )


def step(material: BatchedStatefulMaterial, depth: float, load: float = 100.0):
    shape = material.parameters.shape
    return material.step(
        torch.full(shape, depth),
        torch.full(shape, 0.02),
        torch.zeros(shape),
        torch.full(shape, load),
        torch.full(shape, 10.0),
        dt_s=0.005,
    )


def test_snow_fracture_persists_across_lift_and_selected_reset_is_isolated() -> None:
    model = BatchedStatefulMaterial(parameters())
    result = step(model, 0.006, load=100.0)
    assert result.fractured.all()
    crater = model.residual_crater_depth_m.clone()
    model.lift(torch.tensor([0]))
    assert model.fractured.all()
    assert torch.equal(model.residual_crater_depth_m, crater)
    model.reset_worlds(torch.tensor([0]))
    assert not model.fractured[0].any()
    assert model.fractured[1].all()
    assert torch.all(model.residual_crater_depth_m[0] == 0.0)
    assert torch.equal(model.residual_crater_depth_m[1], crater[1])


def test_ice_response_is_bounded_finite_and_breakout_is_stateful() -> None:
    model = BatchedStatefulMaterial(parameters(material=ICE))
    shape = model.parameters.shape
    response = None
    for depth in torch.linspace(0.0, 0.006, 61):
        response = model.step(
            torch.full(shape, float(depth)),
            torch.full(shape, 0.02),
            torch.full(shape, 0.02),
            torch.full(shape, 250.0),
            torch.full(shape, 12.0),
            dt_s=0.005,
        )
    assert response is not None
    assert torch.isfinite(response.normal_force_n).all()
    assert torch.all(response.normal_force_n >= 0.0)
    assert torch.all(response.normal_force_n <= 180.0)  # 720 N / four probes
    assert response.fractured.all()
    assert response.broken_out.all()


def test_material_inputs_require_exact_environment_foot_probe_shape() -> None:
    model = BatchedStatefulMaterial(parameters())
    wrong = torch.zeros((2, 8))
    with pytest.raises(ValueError, match="penetration_m must have shape"):
        model.step(wrong, wrong, wrong, wrong, wrong, dt_s=0.005)
