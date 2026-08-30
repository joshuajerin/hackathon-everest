from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .stateful_material import (
    ICE,
    LAYERED_SNOW_ICE,
    SNOW,
    BatchedMaterialParameters,
)


def _tensor(value: np.ndarray, device: torch.device, *, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def build_suite_material_parameters(
    config: dict[str, Any],
    cases: Sequence[Any],
    *,
    seed: int,
    device: torch.device | str,
) -> BatchedMaterialParameters:
    """Create deterministic GPU material tensors for a complete assigned terrain suite."""
    device = torch.device(device)
    rng = np.random.default_rng(seed)
    count = len(cases)
    if count < 1:
        raise ValueError("cases cannot be empty")
    shape = (count, 2, 4)
    surfaces = {item["id"]: item for item in config["surfaces"]}
    priors = config["physics_priors"]
    names = (
        "vertical_stiffness_n_per_m",
        "damping_ns_per_m",
        "bearing_capacity_n",
        "shear_capacity_n",
        "friction",
        "support_layer_depth_m",
        "crust_thickness_m",
        "fracture_strength_n",
        "void_top_depth_m",
        "void_height_m",
        "ice_indentation_pressure_pa",
        "ice_tip_radius_m",
        "ice_cone_half_angle_rad",
        "ice_shank_radius_m",
        "ice_fracture_energy_j",
        "ice_post_fracture_strength_ratio",
        "breakout_displacement_m",
        "maximum_probe_force_n",
    )
    arrays = {name: np.zeros(shape, dtype=np.float32) for name in names}
    material_code = np.zeros(shape, dtype=np.int64)
    void_present = np.zeros(shape, dtype=bool)
    for index, case in enumerate(cases):
        surface = surfaces[case.surface_id]
        prior = priors[surface["physics_prior"]]
        family = surface["family"]
        material_code[index] = (
            SNOW if family == "snow" else ICE if family == "ice" else LAYERED_SNOW_ICE
        )
        local = rng.uniform(0.92, 1.08, size=(2, 4)).astype(np.float32)

        def sampled(
            name: str,
            fallback: tuple[float, float],
            *,
            active_prior: dict[str, Any] = prior,
            multiplier: np.ndarray = local,
        ) -> np.ndarray:
            bounds = active_prior.get(name, fallback)
            return float(rng.uniform(float(bounds[0]), float(bounds[1]))) * multiplier

        arrays["vertical_stiffness_n_per_m"][index] = sampled("stiffness_n_per_m", (18_000, 42_000))
        arrays["damping_ns_per_m"][index] = sampled("damping_ns_per_m", (20, 120))
        arrays["bearing_capacity_n"][index] = sampled("bearing_n", (300, 900))
        arrays["shear_capacity_n"][index] = sampled("shear_n", (80, 500))
        arrays["friction"][index] = np.clip(sampled("friction", (0.03, 0.65)), 0.02, 0.9)
        depth = float(surface["configured_depth_m"])
        arrays["support_layer_depth_m"][index] = rng.uniform(
            max(0.003, 0.15 * depth), max(0.006, 0.9 * depth), size=(2, 4)
        )
        arrays["crust_thickness_m"][index] = sampled("crust_m", (0.0, 0.025))
        arrays["fracture_strength_n"][index] = sampled("fracture_strength_n", (50, 310))
        arrays["ice_indentation_pressure_pa"][index] = sampled(
            "indentation_pressure_pa", (28e6, 73e6)
        )
        arrays["ice_tip_radius_m"][index] = rng.uniform(0.00025, 0.00075, size=(2, 4))
        arrays["ice_cone_half_angle_rad"][index] = np.deg2rad(rng.uniform(24.0, 36.0, size=(2, 4)))
        arrays["ice_shank_radius_m"][index] = rng.uniform(0.0025, 0.0040, size=(2, 4))
        arrays["ice_fracture_energy_j"][index] = sampled("fracture_energy_j", (0.035, 0.30))
        arrays["ice_post_fracture_strength_ratio"][index] = rng.uniform(0.30, 0.72, size=(2, 4))
        arrays["breakout_displacement_m"][index] = rng.uniform(0.0015, 0.0060, size=(2, 4))
        arrays["maximum_probe_force_n"][index] = 250.0
        hazard = case.hazard_id
        if hazard in {
            "buried_shallow_void",
            "buried_deep_void",
            "thin_snow_bridge",
            "open_crevasse_gap",
            "edge_collapse",
        }:
            selected = np.ones((2, 4), dtype=bool)
            if hazard == "edge_collapse":
                selected[:] = False
                selected[:, rng.choice(4, size=2, replace=False)] = True
            void_present[index] = selected
            if hazard == "buried_shallow_void":
                top, height = rng.uniform(0.03, 0.12), rng.uniform(0.05, 0.25)
            elif hazard == "buried_deep_void":
                top, height = rng.uniform(0.12, 0.35), rng.uniform(0.20, 0.80)
            elif hazard == "thin_snow_bridge":
                top, height = rng.uniform(0.02, 0.12), rng.uniform(0.20, 0.80)
                arrays["crust_thickness_m"][index] = top
            elif hazard == "open_crevasse_gap":
                top, height = 0.0, 2.0
                arrays["bearing_capacity_n"][index] *= 0.08
                arrays["shear_capacity_n"][index] *= 0.08
            else:
                top, height = rng.uniform(0.01, 0.08), rng.uniform(0.10, 0.60)
                arrays["bearing_capacity_n"][index][selected] *= 0.30
            arrays["void_top_depth_m"][index] = top
            arrays["void_height_m"][index] = height
        else:
            arrays["void_top_depth_m"][index] = 0.5
            arrays["void_height_m"][index] = 0.0
        if hazard == "exposed_ice_patch" and family == "snow":
            material_code[index, :, :2] = ICE
    return BatchedMaterialParameters(
        material_code=_tensor(material_code, device, dtype=torch.int64),
        void_present=_tensor(void_present, device, dtype=torch.bool),
        **{name: _tensor(value, device) for name, value in arrays.items()},
    )


def suite_plane_normals(cases: Sequence[Any], device: torch.device | str) -> torch.Tensor:
    """Return upward normals for planes rising in +X at each case incline."""
    angle = torch.deg2rad(
        torch.as_tensor([case.incline_deg for case in cases], device=device, dtype=torch.float32)
    )
    return torch.stack((-torch.sin(angle), torch.zeros_like(angle), torch.cos(angle)), dim=-1)
