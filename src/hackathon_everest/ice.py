from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IceContactParameters:
    """One domain-randomized ice/spike realization.

    Ranges are broad priors from steel/ice friction and indentation literature.
    They must be replaced by posterior ranges from the actual spike calibration rig.
    """

    temperature_c: float
    indentation_pressure_pa: float
    sliding_friction: float
    tip_radius_m: float
    cone_half_angle_rad: float
    shank_radius_m: float
    normal_damping_ns_per_m: float
    fracture_energy_j: float
    post_fracture_strength_ratio: float
    breakout_displacement_m: float
    maximum_force_n: float = 250.0

    @classmethod
    def sample(cls, seed: int) -> IceContactParameters:
        rng = np.random.default_rng(seed)
        temperature_c = float(rng.uniform(-25.0, -1.0))
        # Interpolate 35 MPa at -3 C to 63 MPa at -25 C, then widen for
        # natural defects, salinity, frost, and uncertain sharp-tip transfer.
        hardness_mpa = np.interp(-temperature_c, [1.0, 3.0, 25.0], [28.0, 35.0, 63.0])
        indentation_pressure_pa = float(hardness_mpa * 1e6 * rng.uniform(0.45, 1.15))
        return cls(
            temperature_c=temperature_c,
            indentation_pressure_pa=indentation_pressure_pa,
            sliding_friction=float(np.exp(rng.uniform(np.log(0.03), np.log(0.18)))),
            tip_radius_m=float(rng.uniform(0.00025, 0.00075)),
            cone_half_angle_rad=float(np.deg2rad(rng.uniform(24.0, 36.0))),
            shank_radius_m=float(rng.uniform(0.0025, 0.0040)),
            normal_damping_ns_per_m=float(rng.uniform(20.0, 120.0)),
            fracture_energy_j=float(rng.uniform(0.035, 0.30)),
            post_fracture_strength_ratio=float(rng.uniform(0.30, 0.72)),
            breakout_displacement_m=float(rng.uniform(0.0015, 0.0060)),
        )


@dataclass
class IceContactState:
    maximum_penetration_m: float = 0.0
    loading_work_j: float = 0.0
    lateral_displacement_m: float = 0.0
    fractured: bool = False
    broken_out: bool = False
    residual_crater_depth_m: float = 0.0
    last_penetration_m: float = 0.0


@dataclass(frozen=True)
class IceContactResponse:
    normal_force_n: float
    shear_capacity_n: float
    projected_area_m2: float
    ploughing_force_n: float
    sliding_force_n: float
    fractured: bool
    broken_out: bool
    damage_fraction: float


class StatefulIceSpikeContact:
    """Reduced-order sharp-spike contact with irreversible fracture memory.

    MuJoCo supplies kinematics. This law supplies material forces that native
    recoverable soft contact cannot represent. The state is simulator-only.
    """

    def __init__(self, parameters: IceContactParameters):
        self.parameters = parameters
        self.state = IceContactState()

    def projected_area_m2(self, penetration_m: float) -> float:
        p = self.parameters
        depth = max(0.0, penetration_m)
        rounded_contact_radius = np.sqrt(max(0.0, 2.0 * p.tip_radius_m * depth))
        cone_contact_radius = depth * np.tan(p.cone_half_angle_rad)
        radius = min(p.shank_radius_m, max(rounded_contact_radius, cone_contact_radius))
        return float(np.pi * radius**2)

    def step(
        self,
        penetration_m: float,
        penetration_rate_mps: float,
        *,
        lateral_speed_mps: float = 0.0,
        dt_s: float,
    ) -> IceContactResponse:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        state, params = self.state, self.parameters
        depth = max(0.0, float(penetration_m))
        rate = float(penetration_rate_mps)
        loading_increment = max(0.0, depth - state.last_penetration_m)
        material_depth = max(0.0, depth - state.residual_crater_depth_m)
        area = self.projected_area_m2(material_depth)
        strength_ratio = params.post_fracture_strength_ratio if state.fractured else 1.0
        quasi_static_force = params.indentation_pressure_pa * area * strength_ratio
        damping_force = params.normal_damping_ns_per_m * max(0.0, rate)
        force = float(np.clip(quasi_static_force + damping_force, 0.0, params.maximum_force_n))

        state.loading_work_j += force * loading_increment
        state.maximum_penetration_m = max(state.maximum_penetration_m, depth)
        if not state.fractured and state.loading_work_j >= params.fracture_energy_j:
            state.fractured = True
            state.residual_crater_depth_m = max(state.residual_crater_depth_m, 0.55 * depth)
            strength_ratio = params.post_fracture_strength_ratio
            material_depth = max(0.0, depth - state.residual_crater_depth_m)
            area = self.projected_area_m2(material_depth)
            force = float(
                np.clip(params.indentation_pressure_pa * area * strength_ratio + damping_force, 0.0, params.maximum_force_n)
            )

        state.lateral_displacement_m += abs(float(lateral_speed_mps)) * dt_s
        if state.lateral_displacement_m >= params.breakout_displacement_m:
            state.broken_out = True
        sliding = params.sliding_friction * force
        frontal_area = 2.0 * min(params.shank_radius_m, np.sqrt(area / np.pi)) * material_depth
        ploughing = params.indentation_pressure_pa * frontal_area * (0.25 if state.broken_out else 1.0)
        shear_capacity = float(np.clip(sliding + ploughing, 0.0, 2.0 * params.maximum_force_n))
        state.last_penetration_m = depth
        damage_fraction = min(1.0, state.loading_work_j / params.fracture_energy_j)
        return IceContactResponse(
            normal_force_n=force,
            shear_capacity_n=shear_capacity,
            projected_area_m2=area,
            ploughing_force_n=float(ploughing),
            sliding_force_n=float(sliding),
            fractured=state.fractured,
            broken_out=state.broken_out,
            damage_fraction=float(damage_fraction),
        )

    def lift_and_reposition(self, *, preserve_terrain_memory: bool = True) -> None:
        """Reset contact motion while optionally preserving the damaged crater."""
        crater = self.state.residual_crater_depth_m if preserve_terrain_memory else 0.0
        fractured = self.state.fractured if preserve_terrain_memory else False
        self.state = IceContactState(residual_crater_depth_m=crater, fractured=fractured)
