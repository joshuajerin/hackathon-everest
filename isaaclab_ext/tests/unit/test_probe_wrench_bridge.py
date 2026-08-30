from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SOURCE = Path(__file__).parents[2] / "source/hackathon_everest_isaaclab"
sys.path.insert(0, str(SOURCE))

from hackathon_everest_isaaclab.contact.probe_wrench_bridge import (
    DEFAULT_PROBE_OFFSETS_LOCAL_M,
    BatchedCramponWrenchBridge,
    IsaacArticulationWrenchAdapter,
)
from hackathon_everest_isaaclab.contact.stateful_material import (
    SNOW,
    BatchedMaterialParameters,
    BatchedStatefulMaterial,
)


def parameters(
    environments: int = 2, device: torch.device | str = "cpu"
) -> BatchedMaterialParameters:
    shape = (environments, 2, 4)
    full = lambda value: torch.full(shape, value, dtype=torch.float32, device=device)
    return BatchedMaterialParameters(
        material_code=torch.full(shape, SNOW, dtype=torch.int64, device=device),
        vertical_stiffness_n_per_m=full(18_000.0),
        damping_ns_per_m=full(45.0),
        bearing_capacity_n=full(720.0),
        shear_capacity_n=full(320.0),
        friction=full(0.24),
        support_layer_depth_m=full(0.025),
        crust_thickness_m=full(0.004),
        fracture_strength_n=full(80.0),
        void_present=torch.zeros(shape, dtype=torch.bool, device=device),
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


def bridge(environments: int = 2) -> BatchedCramponWrenchBridge:
    return BatchedCramponWrenchBridge(
        BatchedStatefulMaterial(parameters(environments)),
        native_support_collisions_enabled=False,
    )


def inputs(environments: int = 2, *, ankle_z: float = 0.075) -> dict[str, torch.Tensor]:
    position = torch.zeros((environments, 2, 3))
    position[..., 2] = ankle_z
    orientation = torch.zeros((environments, 2, 4))
    orientation[..., 0] = 1.0
    return {
        "ankle_position_m": position,
        "ankle_orientation_wxyz": orientation,
        "ankle_linear_velocity_mps": torch.zeros((environments, 2, 3)),
        "ankle_angular_velocity_radps": torch.zeros((environments, 2, 3)),
        "terrain_origin_m": torch.zeros((environments, 3)),
        "terrain_normal": torch.tensor([[0.0, 0.0, 1.0]]).expand(environments, -1),
        "applied_load_n": torch.full((environments, 2), 400.0),
        "tangential_demand_n": torch.zeros((environments, 2)),
    }


def test_identity_transform_and_symmetric_contact_have_zero_torque() -> None:
    model = bridge(1)
    args = inputs(1)
    result = model.step(**args, dt_s=0.005)

    expected = torch.tensor(DEFAULT_PROBE_OFFSETS_LOCAL_M) + args["ankle_position_m"][0, 0]
    assert torch.allclose(result.probe_world_position_m[0, 0], expected)
    assert torch.allclose(
        result.probe_penetration_m,
        torch.full_like(result.probe_penetration_m, 0.010012),
        atol=1e-7,
    )
    assert torch.allclose(result.total_torque_nm, torch.zeros_like(result.total_torque_nm))
    assert result.sensor_force_n.shape == (1, 2, 4)
    assert result.sensor_penetration_m.shape == (1, 2, 4)


def test_rotated_transform_rotates_positions_and_angular_velocity() -> None:
    model = bridge(1)
    args = inputs(1, ankle_z=0.2)
    half = math.sqrt(0.5)
    args["ankle_orientation_wxyz"][..., 0] = half
    args["ankle_orientation_wxyz"][..., 3] = half
    args["ankle_angular_velocity_radps"][..., 2] = 2.0
    result = model.step(**args, dt_s=0.005)

    local = torch.tensor(DEFAULT_PROBE_OFFSETS_LOCAL_M)
    expected_offset = torch.stack((-local[:, 1], local[:, 0], local[:, 2]), dim=-1)
    assert torch.allclose(
        result.probe_world_position_m[0, 0] - args["ankle_position_m"][0, 0],
        expected_offset,
        atol=1e-6,
    )
    expected_velocity = torch.stack(
        (-2.0 * expected_offset[:, 1], 2.0 * expected_offset[:, 0], torch.zeros(4)),
        dim=-1,
    )
    assert torch.allclose(result.probe_world_velocity_mps[0, 0], expected_velocity, atol=1e-6)


def test_hard_stop_damping_is_inactive_before_virtual_travel_limit() -> None:
    static_model = bridge(1)
    static = static_model.step(**inputs(1), dt_s=0.005)
    moving_model = bridge(1)
    moving_inputs = inputs(1)
    moving_inputs["ankle_linear_velocity_mps"][..., 2] = -1.0
    moving = moving_model.step(**moving_inputs, dt_s=0.005)
    extra_per_probe = moving.probe_normal_force_n - static.probe_normal_force_n
    assert torch.all(extra_per_probe > 0.0)
    assert torch.all(extra_per_probe < 60.0)


def test_pitch_loaded_front_probes_produce_negative_y_torque() -> None:
    model = bridge(1)
    args = inputs(1, ankle_z=0.075)
    angle = 0.20
    args["ankle_orientation_wxyz"][..., 0] = math.cos(angle / 2.0)
    args["ankle_orientation_wxyz"][..., 2] = math.sin(angle / 2.0)
    result = model.step(**args, dt_s=0.005)

    assert torch.all(result.total_torque_nm[..., 1] < 0.0)


def test_spatial_void_only_removes_support_inside_configured_x_band() -> None:
    params = parameters(1)
    params.void_present[:] = True
    params.void_top_depth_m[:] = 0.0
    params.void_height_m[:] = 1.0
    model = BatchedCramponWrenchBridge(
        BatchedStatefulMaterial(params),
        native_support_collisions_enabled=False,
        spatial_void_x_bounds_m=torch.tensor([[1.0, 2.0]]),
    )
    outside_args = inputs(1)
    outside = model.step(**outside_args, dt_s=0.005)
    inside_args = inputs(1)
    inside_args["ankle_position_m"][..., 0] = 1.5
    inside = model.step(**inside_args, dt_s=0.005)
    assert torch.all(inside.probe_normal_force_n < 0.25 * outside.probe_normal_force_n)


def test_front_point_mode_disables_rear_analytical_probes() -> None:
    mask = torch.ones((1, 2, 4), dtype=torch.bool)
    mask[..., 2:] = False
    model = BatchedCramponWrenchBridge(
        BatchedStatefulMaterial(parameters(1)),
        native_support_collisions_enabled=False,
        probe_enabled_mask=mask,
    )
    result = model.step(**inputs(1), dt_s=0.005)
    assert torch.all(result.probe_normal_force_n[..., :2] > 0.0)
    assert torch.count_nonzero(result.probe_normal_force_n[..., 2:]) == 0
    assert torch.count_nonzero(result.probe_penetration_m[..., 2:]) == 0


def test_each_foot_requires_at_least_one_enabled_probe() -> None:
    with pytest.raises(ValueError, match="every foot"):
        BatchedCramponWrenchBridge(
            BatchedStatefulMaterial(parameters(1)),
            native_support_collisions_enabled=False,
            probe_enabled_mask=torch.zeros((1, 2, 4), dtype=torch.bool),
        )


def test_per_foot_load_is_distributed_over_only_active_probes() -> None:
    model = bridge(1)
    contact = torch.tensor([[[True, True, False, False], [True, False, False, False]]])
    load = model._probe_scalar(torch.tensor([[400.0, 320.0]]), "load", 1, contact)
    torch.testing.assert_close(load[0, 0, :2], torch.tensor([200.0, 200.0]))
    torch.testing.assert_close(load[0, 1, :1], torch.tensor([320.0]))
    torch.testing.assert_close((load * contact).sum(dim=-1), torch.tensor([[400.0, 320.0]]))


def test_tangential_force_is_finite_bounded_and_opposes_motion() -> None:
    model = bridge(1)
    args = inputs(1)
    args["ankle_linear_velocity_mps"][..., 0] = 0.5
    args["tangential_demand_n"][:] = 800.0
    result = model.step(**args, dt_s=0.005)

    assert torch.isfinite(result.probe_force_n).all()
    assert torch.all(result.probe_force_n[..., 0] <= 0.0)
    tangent_force = torch.linalg.vector_norm(result.probe_force_n[..., :2], dim=-1)
    assert torch.all(tangent_force <= result.material_response.shear_capacity_n + 1e-6)


def test_static_vector_demand_defines_tangential_opposition() -> None:
    model = bridge(1)
    args = inputs(1)
    args["tangential_demand_n"] = torch.tensor([[[80.0, 0.0, 0.0], [80.0, 0.0, 0.0]]])
    result = model.step(**args, dt_s=0.005)

    assert torch.all(result.probe_force_n[..., 0] < 0.0)
    assert torch.allclose(result.probe_force_n[..., 1], torch.zeros((1, 2, 4)))


def test_solid_surface_hard_stop_prevents_travel_bottom_out() -> None:
    model = bridge(1)
    args = inputs(1, ankle_z=0.0)
    result = model.step(**args, dt_s=0.005)
    assert torch.all(result.probe_penetration_m == model.virtual_travel_m)
    assert torch.all(result.probe_normal_force_n > result.material_response.normal_force_n)
    assert torch.all(result.probe_normal_force_n <= 180.0 + model.maximum_hard_stop_force_n)


def test_void_probes_never_receive_structural_hard_stop_support() -> None:
    params = parameters(1)
    params.void_present[:] = True
    model = BatchedCramponWrenchBridge(
        BatchedStatefulMaterial(params), native_support_collisions_enabled=False
    )
    args = inputs(1, ankle_z=0.0)
    result = model.step(**args, dt_s=0.005)
    torch.testing.assert_close(result.probe_normal_force_n, result.material_response.normal_force_n)


def test_separation_has_zero_reaction_even_with_downward_velocity_and_demand() -> None:
    model = bridge(1)
    args = inputs(1, ankle_z=0.3)
    args["ankle_linear_velocity_mps"][..., 2] = -5.0
    args["tangential_demand_n"][:] = 1000.0
    result = model.step(**args, dt_s=0.005)

    assert torch.count_nonzero(result.probe_penetration_m) == 0
    assert torch.count_nonzero(result.probe_force_n) == 0
    assert torch.count_nonzero(result.total_force_n) == 0
    assert torch.count_nonzero(result.total_torque_nm) == 0
    assert not result.material_response.slipping.any()
    assert torch.count_nonzero(result.material_response.friction_utilization) == 0


def test_material_is_called_once_and_selective_lift_reset_are_forwarded() -> None:
    model = bridge(2)
    calls = 0
    original_step = model.material.step

    def counted_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_step(*args, **kwargs)

    model.material.step = counted_step
    result = model.step(**inputs(2), dt_s=0.005)
    assert calls == 1
    assert result.material_response.fractured.all()

    crater = model.material.residual_crater_depth_m.clone()
    model.lift(torch.tensor([0]))
    assert model.material.fractured.all()
    assert torch.equal(model.material.residual_crater_depth_m, crater)
    model.reset_worlds(torch.tensor([0]))
    assert not model.material.fractured[0].any()
    assert model.material.fractured[1].all()


def test_accelerator_device_is_preserved_when_available() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        pytest.skip("no Torch accelerator is available")
    model = BatchedCramponWrenchBridge(
        BatchedStatefulMaterial(parameters(1, device)),
        native_support_collisions_enabled=False,
    )

    result = model.step(**inputs(1), dt_s=0.005)

    assert result.probe_force_n.device.type == device.type
    assert result.total_torque_nm.device.type == device.type
    assert result.material_response.normal_force_n.device.type == device.type


def test_double_force_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="double-force configuration rejected"):
        BatchedCramponWrenchBridge(
            BatchedStatefulMaterial(parameters(1)),
            native_support_collisions_enabled=True,
        )


def test_thin_articulation_adapter_targets_two_ankle_body_ids() -> None:
    class FakeArticulation:
        call = None

        def set_external_force_and_torque(self, forces, torques, positions, **kwargs):
            self.call = (forces, torques, positions, kwargs)

    result = bridge(1).step(**inputs(1), dt_s=0.005)
    articulation = FakeArticulation()
    adapter = IsaacArticulationWrenchAdapter(articulation, [7, 11])
    adapter.apply(result)

    forces, torques, positions, kwargs = articulation.call
    assert forces is result.total_force_n
    assert torques is result.total_torque_nm
    assert positions is result.ankle_position_m
    assert kwargs["body_ids"].tolist() == [7, 11]
    assert kwargs["is_global"] is True
