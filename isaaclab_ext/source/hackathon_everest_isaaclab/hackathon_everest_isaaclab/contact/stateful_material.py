from __future__ import annotations

from dataclasses import dataclass, fields

import torch

SNOW = 0
ICE = 1
LAYERED_SNOW_ICE = 2


@dataclass
class BatchedMaterialParameters:
    material_code: torch.Tensor
    vertical_stiffness_n_per_m: torch.Tensor
    damping_ns_per_m: torch.Tensor
    bearing_capacity_n: torch.Tensor
    shear_capacity_n: torch.Tensor
    friction: torch.Tensor
    support_layer_depth_m: torch.Tensor
    crust_thickness_m: torch.Tensor
    fracture_strength_n: torch.Tensor
    void_present: torch.Tensor
    void_top_depth_m: torch.Tensor
    void_height_m: torch.Tensor
    ice_indentation_pressure_pa: torch.Tensor
    ice_tip_radius_m: torch.Tensor
    ice_cone_half_angle_rad: torch.Tensor
    ice_shank_radius_m: torch.Tensor
    ice_fracture_energy_j: torch.Tensor
    ice_post_fracture_strength_ratio: torch.Tensor
    breakout_displacement_m: torch.Tensor
    maximum_probe_force_n: torch.Tensor

    def __post_init__(self) -> None:
        shapes = {tuple(getattr(self, field.name).shape) for field in fields(self)}
        if len(shapes) != 1:
            raise ValueError(f"All material tensors must have one common shape, got {shapes}")
        if len(next(iter(shapes))) != 3 or next(iter(shapes))[-1] != 4:
            raise ValueError("Material tensors must have shape [environment, foot, 4 probes]")

    @property
    def shape(self) -> torch.Size:
        return self.material_code.shape


@dataclass
class MaterialResponse:
    normal_force_n: torch.Tensor
    shear_capacity_n: torch.Tensor
    friction_utilization: torch.Tensor
    slipping: torch.Tensor
    fractured: torch.Tensor
    broken_out: torch.Tensor
    damage: torch.Tensor
    residual_crater_depth_m: torch.Tensor


class BatchedStatefulMaterial:
    """Persistent snow/ice law for four analytical probes per foot.

    This module owns material response only. Isaac integration must disable native
    support contact for the same probe/terrain pair before applying these forces.
    Visual crampon meshes never enter this law.
    """

    def __init__(self, parameters: BatchedMaterialParameters):
        self.parameters = parameters
        shape = parameters.shape
        options = {"device": parameters.material_code.device, "dtype": torch.float32}
        self.maximum_penetration_m = torch.zeros(shape, **options)
        self.loading_work_j = torch.zeros(shape, **options)
        self.lateral_displacement_m = torch.zeros(shape, **options)
        self.residual_crater_depth_m = torch.zeros(shape, **options)
        self.last_penetration_m = torch.zeros(shape, **options)
        self.compaction = torch.zeros(shape, **options)
        self.damage = torch.zeros(shape, **options)
        self.fractured = torch.zeros(shape, device=options["device"], dtype=torch.bool)
        self.broken_out = torch.zeros_like(self.fractured)

    def _assert_input(self, value: torch.Tensor, name: str) -> torch.Tensor:
        value = value.to(device=self.parameters.material_code.device, dtype=torch.float32)
        if value.shape != self.parameters.shape:
            raise ValueError(f"{name} must have shape {tuple(self.parameters.shape)}, got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
        return value

    def step(
        self,
        penetration_m: torch.Tensor,
        penetration_rate_mps: torch.Tensor,
        lateral_speed_mps: torch.Tensor,
        applied_load_n: torch.Tensor,
        tangential_demand_n: torch.Tensor,
        *,
        dt_s: float,
    ) -> MaterialResponse:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        depth = torch.clamp(self._assert_input(penetration_m, "penetration_m"), min=0.0)
        rate = self._assert_input(penetration_rate_mps, "penetration_rate_mps")
        lateral = torch.abs(self._assert_input(lateral_speed_mps, "lateral_speed_mps"))
        load = torch.clamp(self._assert_input(applied_load_n, "applied_load_n"), min=0.0)
        demand = torch.clamp(self._assert_input(tangential_demand_n, "tangential_demand_n"), min=0.0)
        p = self.parameters
        loading_increment = torch.clamp(depth - self.last_penetration_m, min=0.0)
        is_ice = p.material_code == ICE
        is_layered = p.material_code == LAYERED_SNOW_ICE
        uses_ice = is_ice | (is_layered & (depth >= p.support_layer_depth_m))

        # Stateful snow response, matching the simulator-neutral NumPy reference semantics.
        effective_k = p.vertical_stiffness_n_per_m * (1.0 + 1.8 * self.compaction)
        snow_depth = torch.clamp(depth - self.residual_crater_depth_m, min=0.0)
        snow_force = effective_k * torch.pow(snow_depth, 1.12)
        snow_force += p.damping_ns_per_m * torch.clamp(rate, min=0.0)
        snow_force += 3.0 * effective_k * torch.clamp(snow_depth - p.support_layer_depth_m, min=0.0)
        inside_void = p.void_present & (snow_depth >= p.void_top_depth_m) & (snow_depth <= p.void_top_depth_m + p.void_height_m)
        snow_force = torch.where(inside_void, 0.08 * snow_force, snow_force)
        crust_intact = (p.crust_thickness_m > 0.0) & ~self.fractured
        crust_progress = torch.clamp(snow_depth / torch.clamp(p.crust_thickness_m, min=1e-6), 0.0, 1.0)
        snow_force += torch.where(crust_intact, 0.25 * p.fracture_strength_n * torch.pow(crust_progress, 1.5), 0.0)
        snow_fracture = crust_intact & (snow_depth >= p.crust_thickness_m) & (4.0 * load >= p.fracture_strength_n)

        # Sharp-spike ice response, including irreversible fracture and breakout memory.
        ice_depth = torch.clamp(depth - self.residual_crater_depth_m, min=0.0)
        rounded_radius = torch.sqrt(torch.clamp(2.0 * p.ice_tip_radius_m * ice_depth, min=0.0))
        cone_radius = ice_depth * torch.tan(p.ice_cone_half_angle_rad)
        radius = torch.minimum(p.ice_shank_radius_m, torch.maximum(rounded_radius, cone_radius))
        area = torch.pi * radius.square()
        strength_ratio = torch.where(self.fractured, p.ice_post_fracture_strength_ratio, 1.0)
        ice_force = p.ice_indentation_pressure_pa * area * strength_ratio
        ice_force += p.damping_ns_per_m * torch.clamp(rate, min=0.0)
        ice_work_next = self.loading_work_j + torch.clamp(ice_force, min=0.0) * loading_increment
        ice_fracture = ~self.fractured & (ice_work_next >= p.ice_fracture_energy_j)

        just_fractured = torch.where(uses_ice, ice_fracture, snow_fracture)
        self.fractured |= just_fractured
        new_crater = torch.where(uses_ice, 0.55 * depth, 0.15 * depth)
        self.residual_crater_depth_m = torch.where(just_fractured, torch.maximum(self.residual_crater_depth_m, new_crater), self.residual_crater_depth_m)
        strength_ratio = torch.where(self.fractured, p.ice_post_fracture_strength_ratio, 1.0)
        ice_force = p.ice_indentation_pressure_pa * area * strength_ratio + p.damping_ns_per_m * torch.clamp(rate, min=0.0)

        raw_force = torch.where(uses_ice, ice_force, snow_force)
        bearing_per_probe = torch.clamp(p.bearing_capacity_n * (1.0 - 0.5 * self.damage) / 4.0, min=0.0)
        force_cap = torch.minimum(p.maximum_probe_force_n, load)
        normal_force = torch.minimum(torch.clamp(raw_force, min=0.0), torch.minimum(force_cap, bearing_per_probe))

        self.loading_work_j = torch.where(uses_ice, ice_work_next, self.loading_work_j + normal_force * loading_increment)
        self.maximum_penetration_m = torch.maximum(self.maximum_penetration_m, depth)
        self.lateral_displacement_m += lateral * float(dt_s)
        self.broken_out |= self.lateral_displacement_m >= p.breakout_displacement_m

        engagement = torch.clamp(depth / 0.012, 0.0, 1.0)
        snow_shear = p.friction * normal_force + 0.25 * p.shear_capacity_n * engagement
        frontal_area = 2.0 * radius * ice_depth
        ploughing = p.ice_indentation_pressure_pa * frontal_area * torch.where(self.broken_out, 0.25, 1.0)
        ice_shear = p.friction * normal_force + ploughing
        shear = torch.where(uses_ice, ice_shear, snow_shear)
        shear = torch.clamp(shear, min=0.0, max=2.0 * p.maximum_probe_force_n)
        utilization = demand / torch.clamp(shear, min=1e-6)
        slipping = demand > shear

        contact = depth > 0.0
        load_ratio = normal_force / torch.clamp(bearing_per_probe, min=1.0)
        compaction_delta = torch.clamp((0.12 * load_ratio + 2.0 * loading_increment) * contact, 0.0, 0.25)
        self.compaction = torch.clamp(self.compaction + compaction_delta, 0.0, 1.0)
        damage_delta = torch.where(just_fractured, 0.30, torch.clamp(0.20 * (load_ratio - 0.8), 0.0, 0.20))
        self.damage = torch.clamp(self.damage + damage_delta * contact, 0.0, 1.0)
        self.last_penetration_m = depth

        for name, value in {"normal_force_n": normal_force, "shear_capacity_n": shear, "friction_utilization": utilization}.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite {name}")
        return MaterialResponse(normal_force, shear, utilization, slipping, self.fractured.clone(), self.broken_out.clone(), self.damage.clone(), self.residual_crater_depth_m.clone())

    def lift(self, environment_ids: torch.Tensor | None = None) -> None:
        """Reset contact motion only; persistent terrain damage is deliberately retained."""
        ids = slice(None) if environment_ids is None else environment_ids.to(device=self.last_penetration_m.device, dtype=torch.long)
        self.last_penetration_m[ids] = 0.0
        self.lateral_displacement_m[ids] = 0.0
        self.broken_out[ids] = False

    def reset_worlds(self, environment_ids: torch.Tensor) -> None:
        ids = environment_ids.to(device=self.last_penetration_m.device, dtype=torch.long)
        for name in ("maximum_penetration_m", "loading_work_j", "lateral_displacement_m", "residual_crater_depth_m", "last_penetration_m", "compaction", "damage"):
            getattr(self, name)[ids] = 0.0
        self.fractured[ids] = False
        self.broken_out[ids] = False
