from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .stateful_material import BatchedStatefulMaterial, MaterialResponse

if TYPE_CHECKING:
    from collections.abc import Sequence

# Authoritative layout from configs/isaaclab/g1_crampon_simulation.yaml.  Quaternion inputs
# use Isaac's scalar-first convention (w, x, y, z).
DEFAULT_PROBE_OFFSETS_LOCAL_M = (
    (0.081, 0.0486, -0.081012),
    (0.081, -0.0486, -0.081012),
    (-0.081, 0.0486, -0.081012),
    (-0.081, -0.0486, -0.081012),
)
DEFAULT_PROBE_AXIS_LOCAL = (0.0, 0.0, -1.0)
DEFAULT_PROBE_RADIUS_M = 0.004
DEFAULT_VIRTUAL_TRAVEL_M = 0.055
DEFAULT_HARD_STOP_STIFFNESS_N_PER_M = 10_000.0
DEFAULT_HARD_STOP_DAMPING_NS_PER_M = 100.0
DEFAULT_MAXIMUM_HARD_STOP_FORCE_N = 500.0


@dataclass(frozen=True)
class CramponWrench:
    """One batched contact result, with world-frame vectors unless noted otherwise."""

    total_force_n: torch.Tensor
    total_torque_nm: torch.Tensor
    ankle_position_m: torch.Tensor
    probe_force_n: torch.Tensor
    probe_normal_force_n: torch.Tensor
    probe_axial_force_n: torch.Tensor
    probe_penetration_m: torch.Tensor
    probe_world_position_m: torch.Tensor
    probe_world_velocity_mps: torch.Tensor
    probe_penetration_rate_mps: torch.Tensor
    probe_lateral_speed_mps: torch.Tensor
    material_response: MaterialResponse

    @property
    def sensor_force_n(self) -> torch.Tensor:
        """Four scalar force channels per foot, shaped ``[N, 2, 4]``."""
        return self.probe_axial_force_n

    @property
    def sensor_penetration_m(self) -> torch.Tensor:
        """Four penetration channels per foot, shaped ``[N, 2, 4]``."""
        return self.probe_penetration_m


class BatchedCramponWrenchBridge:
    """Map four analytical probes per foot to ankle forces and torques.

    The bridge is simulator-neutral.  It keeps every operation on the material tensor's
    device and invokes :class:`BatchedStatefulMaterial` once per call to :meth:`step`.
    ``applied_load_n`` is the supported gravity/other normal load.  It and scalar
    ``tangential_demand_n`` may be per-foot totals (``[N, 2]``) or already distributed
    per probe (``[N, 2, 4]``).  Tangential demand can also be a world-frame vector with
    a final dimension of three.  Gravity is not added to the returned external wrench;
    the simulator continues to integrate it normally.
    """

    def __init__(
        self,
        material: BatchedStatefulMaterial,
        *,
        native_support_collisions_enabled: bool,
        probe_offsets_local_m: torch.Tensor | Sequence[Sequence[float]] = (
            DEFAULT_PROBE_OFFSETS_LOCAL_M
        ),
        probe_axis_local: torch.Tensor | Sequence[float] = DEFAULT_PROBE_AXIS_LOCAL,
        probe_radius_m: float = DEFAULT_PROBE_RADIUS_M,
        virtual_travel_m: float = DEFAULT_VIRTUAL_TRAVEL_M,
        hard_stop_stiffness_n_per_m: float = DEFAULT_HARD_STOP_STIFFNESS_N_PER_M,
        hard_stop_damping_ns_per_m: float = DEFAULT_HARD_STOP_DAMPING_NS_PER_M,
        maximum_hard_stop_force_n: float = DEFAULT_MAXIMUM_HARD_STOP_FORCE_N,
        probe_enabled_mask: torch.Tensor | None = None,
        spatial_void_x_bounds_m: torch.Tensor | None = None,
    ) -> None:
        if native_support_collisions_enabled:
            raise ValueError(
                "native_support_collisions_enabled must be False when analytical "
                "crampon forces are enabled (double-force configuration rejected)"
            )
        if probe_radius_m <= 0.0:
            raise ValueError("probe_radius_m must be positive")
        if virtual_travel_m <= 0.0:
            raise ValueError("virtual_travel_m must be positive")
        if hard_stop_stiffness_n_per_m <= 0.0:
            raise ValueError("hard_stop_stiffness_n_per_m must be positive")
        if hard_stop_damping_ns_per_m < 0.0:
            raise ValueError("hard_stop_damping_ns_per_m must be non-negative")
        if maximum_hard_stop_force_n <= 0.0:
            raise ValueError("maximum_hard_stop_force_n must be positive")
        self.material = material
        device = material.parameters.material_code.device
        self.dtype = torch.float32
        self.probe_offsets_local_m = torch.as_tensor(
            probe_offsets_local_m, device=device, dtype=self.dtype
        )
        if self.probe_offsets_local_m.shape != (4, 3):
            raise ValueError("probe_offsets_local_m must have shape [4, 3]")
        if not torch.isfinite(self.probe_offsets_local_m).all():
            raise ValueError("probe_offsets_local_m contains non-finite values")
        self.probe_axis_local = torch.as_tensor(probe_axis_local, device=device, dtype=self.dtype)
        if self.probe_axis_local.shape != (3,):
            raise ValueError("probe_axis_local must have shape [3]")
        axis_norm = torch.linalg.vector_norm(self.probe_axis_local)
        if not torch.isfinite(axis_norm) or axis_norm <= 1e-8:
            raise ValueError("probe_axis_local must be finite and nonzero")
        self.probe_axis_local = self.probe_axis_local / axis_norm
        self.probe_radius_m = float(probe_radius_m)
        self.virtual_travel_m = float(virtual_travel_m)
        self.hard_stop_stiffness_n_per_m = float(hard_stop_stiffness_n_per_m)
        self.hard_stop_damping_ns_per_m = float(hard_stop_damping_ns_per_m)
        self.maximum_hard_stop_force_n = float(maximum_hard_stop_force_n)
        if probe_enabled_mask is None:
            self.probe_enabled_mask = torch.ones(
                material.parameters.shape, dtype=torch.bool, device=device
            )
        else:
            self.probe_enabled_mask = probe_enabled_mask.to(device=device, dtype=torch.bool)
            if tuple(self.probe_enabled_mask.shape) != tuple(material.parameters.shape):
                raise ValueError(
                    "probe_enabled_mask must match material shape "
                    f"{tuple(material.parameters.shape)}"
                )
            if not bool(self.probe_enabled_mask.any(dim=-1).all()):
                raise ValueError("every foot must keep at least one enabled analytical probe")
        self._base_void_present = material.parameters.void_present.clone()
        if spatial_void_x_bounds_m is None:
            self.spatial_void_x_bounds_m = torch.full(
                (material.parameters.shape[0], 2),
                float("nan"),
                dtype=self.dtype,
                device=device,
            )
        else:
            self.spatial_void_x_bounds_m = spatial_void_x_bounds_m.to(
                device=device, dtype=self.dtype
            )
            expected_bounds = (material.parameters.shape[0], 2)
            if tuple(self.spatial_void_x_bounds_m.shape) != expected_bounds:
                raise ValueError(f"spatial_void_x_bounds_m must have shape {expected_bounds}")

    @property
    def device(self) -> torch.device:
        return self.material.parameters.material_code.device

    def _input(self, value: torch.Tensor, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        value = value.to(device=self.device, dtype=self.dtype)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
        return value

    def _probe_scalar(
        self,
        value: torch.Tensor,
        name: str,
        environments: int,
        contact: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = value.to(device=self.device, dtype=self.dtype)
        if value.shape == (environments, 2):
            denominator = (
                torch.full_like(value, 4.0)
                if contact is None
                else contact.sum(dim=-1).clamp_min(1).to(dtype=self.dtype)
            )
            value = value.unsqueeze(-1).expand(-1, -1, 4) / denominator.unsqueeze(-1)
        elif value.shape != (environments, 2, 4):
            raise ValueError(
                f"{name} must have shape [{environments}, 2] or "
                f"[{environments}, 2, 4], got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
        return torch.clamp(value, min=0.0)

    def _tangential_demand(
        self,
        value: torch.Tensor,
        terrain_normal: torch.Tensor,
        lateral_velocity: torch.Tensor,
        contact: torch.Tensor,
        environments: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = value.to(device=self.device, dtype=self.dtype)
        scalar_shapes = {(environments, 2), (environments, 2, 4)}
        vector_shapes = {(environments, 2, 3), (environments, 2, 4, 3)}
        if value.shape in scalar_shapes:
            magnitude = self._probe_scalar(value, "tangential_demand_n", environments, contact)
            speed = torch.linalg.vector_norm(lateral_velocity, dim=-1)
            direction = lateral_velocity / torch.clamp(speed.unsqueeze(-1), min=1e-8)
            direction = torch.where(speed.unsqueeze(-1) > 1e-8, direction, 0.0)
            return magnitude, direction
        if value.shape not in vector_shapes:
            raise ValueError(
                "tangential_demand_n must be a per-foot or per-probe scalar/vector; "
                f"got {tuple(value.shape)}"
            )
        if value.shape == (environments, 2, 3):
            denominator = contact.sum(dim=-1).clamp_min(1).to(dtype=self.dtype)
            value = value.unsqueeze(-2).expand(-1, -1, 4, -1) / denominator[..., None, None]
        if not torch.isfinite(value).all():
            raise ValueError("tangential_demand_n contains non-finite values")
        tangent = value - torch.sum(value * terrain_normal, dim=-1, keepdim=True) * terrain_normal
        magnitude = torch.linalg.vector_norm(tangent, dim=-1)
        direction = tangent / torch.clamp(magnitude.unsqueeze(-1), min=1e-8)
        direction = torch.where(magnitude.unsqueeze(-1) > 1e-8, direction, 0.0)
        return magnitude, direction

    @staticmethod
    def _rotate_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        q_vector = quaternion[..., 1:]
        q_scalar = quaternion[..., :1]
        twice_cross = 2.0 * torch.linalg.cross(q_vector, vector, dim=-1)
        return vector + q_scalar * twice_cross + torch.linalg.cross(q_vector, twice_cross, dim=-1)

    def step(
        self,
        ankle_position_m: torch.Tensor,
        ankle_orientation_wxyz: torch.Tensor,
        ankle_linear_velocity_mps: torch.Tensor,
        ankle_angular_velocity_radps: torch.Tensor,
        terrain_origin_m: torch.Tensor,
        terrain_normal: torch.Tensor,
        applied_load_n: torch.Tensor,
        tangential_demand_n: torch.Tensor,
        *,
        dt_s: float,
    ) -> CramponWrench:
        """Compute probe kinematics and the resulting wrench about each ankle."""
        environments = self.material.parameters.shape[0]
        expected_material_shape = (environments, 2, 4)
        if tuple(self.material.parameters.shape) != expected_material_shape:
            raise ValueError(
                "material parameters must have shape "
                f"{expected_material_shape}, got {tuple(self.material.parameters.shape)}"
            )
        ankle_position = self._input(ankle_position_m, "ankle_position_m", (environments, 2, 3))
        orientation = self._input(
            ankle_orientation_wxyz,
            "ankle_orientation_wxyz",
            (environments, 2, 4),
        )
        norm = torch.linalg.vector_norm(orientation, dim=-1, keepdim=True)
        if torch.any(norm <= 1e-8):
            raise ValueError("ankle_orientation_wxyz contains a zero quaternion")
        orientation = orientation / norm
        linear_velocity = self._input(
            ankle_linear_velocity_mps,
            "ankle_linear_velocity_mps",
            (environments, 2, 3),
        )
        angular_velocity = self._input(
            ankle_angular_velocity_radps,
            "ankle_angular_velocity_radps",
            (environments, 2, 3),
        )
        origin = self._input(terrain_origin_m, "terrain_origin_m", (environments, 3))
        normal = self._input(terrain_normal, "terrain_normal", (environments, 3))
        normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        if torch.any(normal_norm <= 1e-8):
            raise ValueError("terrain_normal contains a zero vector")
        normal = normal / normal_norm

        local_offset = self.probe_offsets_local_m.view(1, 1, 4, 3).expand(environments, 2, -1, -1)
        orientation_per_probe = orientation.unsqueeze(-2).expand(-1, -1, 4, -1)
        world_offset = self._rotate_wxyz(orientation_per_probe, local_offset)
        probe_position = ankle_position.unsqueeze(-2) + world_offset
        probe_velocity = linear_velocity.unsqueeze(-2) + torch.linalg.cross(
            angular_velocity.unsqueeze(-2).expand_as(world_offset), world_offset, dim=-1
        )
        bounds = self.spatial_void_x_bounds_m
        has_spatial_void = torch.isfinite(bounds).all(dim=-1)
        relative_x = probe_position[..., 0] - origin[:, None, None, 0]
        inside_spatial_void = (relative_x >= bounds[:, None, None, 0]) & (
            relative_x <= bounds[:, None, None, 1]
        )
        dynamic_void = self._base_void_present & inside_spatial_void
        self.material.parameters.void_present.copy_(
            torch.where(
                has_spatial_void[:, None, None],
                dynamic_void,
                self._base_void_present,
            )
        )

        plane_normal = normal[:, None, None, :]
        signed_height = torch.sum(
            (probe_position - origin[:, None, None, :]) * plane_normal, dim=-1
        )
        physical_penetration = torch.clamp(self.probe_radius_m - signed_height, min=0.0)
        penetration = torch.clamp(physical_penetration, max=self.virtual_travel_m)
        excess_penetration = torch.clamp(physical_penetration - self.virtual_travel_m, min=0.0)
        penetration = torch.where(self.probe_enabled_mask, penetration, 0.0)
        material_penetration = torch.where(self.probe_enabled_mask, physical_penetration, 0.0)
        excess_penetration = torch.where(self.probe_enabled_mask, excess_penetration, 0.0)
        normal_velocity = torch.sum(probe_velocity * plane_normal, dim=-1)
        in_compliant_range = (signed_height < self.probe_radius_m) & (
            penetration < self.virtual_travel_m
        )
        penetration_rate = torch.where(in_compliant_range, -normal_velocity, 0.0)
        lateral_velocity = probe_velocity - normal_velocity.unsqueeze(-1) * plane_normal
        lateral_speed = torch.linalg.vector_norm(lateral_velocity, dim=-1)
        lateral_speed = torch.where(penetration > 0.0, lateral_speed, 0.0)

        contact = (penetration > 0.0) & self.probe_enabled_mask
        probe_load = self._probe_scalar(applied_load_n, "applied_load_n", environments, contact)
        demand_magnitude, opposition_direction = self._tangential_demand(
            tangential_demand_n,
            plane_normal,
            lateral_velocity,
            contact,
            environments,
        )
        # Free-space probes must not report load, utilization, or slip events.
        probe_load = torch.where(contact, probe_load, 0.0)
        demand_magnitude = torch.where(contact, demand_magnitude, 0.0)
        # The one and only stateful material evaluation for this physics step.
        response = self.material.step(
            material_penetration,
            penetration_rate,
            lateral_speed,
            probe_load,
            demand_magnitude,
            dt_s=dt_s,
        )

        closing_speed = torch.clamp(-normal_velocity, min=0.0)
        hard_stop_engaged = excess_penetration > 0.0
        hard_stop_force = (
            self.hard_stop_stiffness_n_per_m * excess_penetration
            + self.hard_stop_damping_ns_per_m * closing_speed
        ).clamp(max=self.maximum_hard_stop_force_n)
        void_bottom = (
            self.material.parameters.void_top_depth_m + self.material.parameters.void_height_m
        )
        hard_stop_allowed = ~self.material.parameters.void_present | (
            physical_penetration > void_bottom
        )
        hard_stop_force = torch.where(
            hard_stop_engaged & hard_stop_allowed,
            hard_stop_force,
            torch.zeros_like(hard_stop_force),
        )
        normal_force_magnitude = torch.where(
            contact, response.normal_force_n + hard_stop_force, 0.0
        )
        tangential_magnitude = torch.minimum(demand_magnitude, response.shear_capacity_n)
        tangential_magnitude = torch.where(contact, tangential_magnitude, 0.0)
        probe_force = normal_force_magnitude.unsqueeze(-1) * plane_normal
        probe_force -= tangential_magnitude.unsqueeze(-1) * opposition_direction
        probe_axis = self._rotate_wxyz(
            orientation_per_probe,
            self.probe_axis_local.view(1, 1, 1, 3).expand_as(world_offset),
        )
        axial_force = torch.clamp(torch.sum(probe_force * -probe_axis, dim=-1), min=0.0)
        total_force = torch.sum(probe_force, dim=-2)
        total_torque = torch.sum(torch.linalg.cross(world_offset, probe_force, dim=-1), dim=-2)

        for name, value in {
            "probe_force_n": probe_force,
            "total_force_n": total_force,
            "total_torque_nm": total_torque,
        }.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite {name}")

        return CramponWrench(
            total_force_n=total_force,
            total_torque_nm=total_torque,
            ankle_position_m=ankle_position,
            probe_force_n=probe_force,
            probe_normal_force_n=normal_force_magnitude,
            probe_axial_force_n=axial_force,
            probe_penetration_m=penetration,
            probe_world_position_m=probe_position,
            probe_world_velocity_mps=probe_velocity,
            probe_penetration_rate_mps=penetration_rate,
            probe_lateral_speed_mps=lateral_speed,
            material_response=response,
        )

    def lift(self, environment_ids: torch.Tensor | None = None) -> None:
        """Clear contact-motion memory while retaining terrain damage."""
        self.material.lift(environment_ids)

    def lift_probes(self, lift_mask: torch.Tensor) -> None:
        """Clear contact-motion state per separated analytical probe."""
        self.material.lift_probes(lift_mask)

    def lift_feet(self, lift_mask: torch.Tensor) -> None:
        """Clear contact-motion state for selected feet while retaining terrain damage."""
        self.material.lift_feet(lift_mask)

    def reset_worlds(self, environment_ids: torch.Tensor) -> None:
        """Reset all persistent material state for selected environments."""
        self.material.reset_worlds(environment_ids)


class IsaacArticulationWrenchAdapter:
    """Optional thin adapter for applying a bridge result to an Isaac articulation.

    The contact module never imports Isaac.  :meth:`from_articulation` performs a lazy
    API availability check when integration is requested, while direct construction is
    useful for duck-typed test doubles.
    """

    def __init__(self, articulation: Any, ankle_body_ids: torch.Tensor | Sequence[int]):
        if not callable(getattr(articulation, "set_external_force_and_torque", None)):
            raise TypeError("articulation must provide set_external_force_and_torque")
        self.articulation = articulation
        self.ankle_body_ids = torch.as_tensor(ankle_body_ids, dtype=torch.long)
        if self.ankle_body_ids.shape != (2,):
            raise ValueError("ankle_body_ids must contain the left and right body IDs")

    @classmethod
    def from_articulation(
        cls, articulation: Any, ankle_body_ids: torch.Tensor | Sequence[int]
    ) -> IsaacArticulationWrenchAdapter:
        try:
            importlib.import_module("isaaclab.assets")
        except ImportError as error:
            raise ImportError("Isaac Lab is required to construct this adapter") from error
        return cls(articulation, ankle_body_ids)

    def apply(self, wrench: CramponWrench) -> None:
        body_ids = self.ankle_body_ids.to(device=wrench.total_force_n.device)
        self.articulation.set_external_force_and_torque(
            forces=wrench.total_force_n,
            torques=wrench.total_torque_nm,
            positions=wrench.ankle_position_m,
            body_ids=body_ids,
            is_global=True,
        )


# Descriptive aliases for callers that use the generic probe terminology.
BatchedProbeWrenchBridge = BatchedCramponWrenchBridge
ProbeWrenchOutput = CramponWrench
