from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class TerrainCase:
    surface_id: str
    surface_family: str
    incline_deg: float
    hazard_id: str
    contact_mode_id: str
    repetition: int
    case_id: str


@dataclass(frozen=True)
class VectorLayout:
    num_envs: int
    rows: int
    cols: int
    cell_spacing_m: float
    patch_size_m: tuple[float, float]
    out_of_bounds_radius_m: float

    def origin(self, env_id: int) -> tuple[float, float, float]:
        if not 0 <= env_id < self.num_envs:
            raise IndexError(env_id)
        row, col = divmod(env_id, self.cols)
        x = (col - (self.cols - 1) / 2.0) * self.cell_spacing_m
        y = (row - (self.rows - 1) / 2.0) * self.cell_spacing_m
        return float(x), float(y), 0.0


def load_suite(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    _validate_suite(config)
    return config


def _ids(entries: list[dict[str, Any]], label: str) -> list[str]:
    values = [str(entry["id"]) for entry in entries]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} id")
    return values


def _validate_suite(config: dict[str, Any]) -> None:
    surfaces = config.get("surfaces", [])
    hazards = config.get("hazards", [])
    modes = config.get("contact_modes", [])
    inclines = [float(value) for value in config.get("inclines_deg", [])]
    if not surfaces or not hazards or not modes or not inclines:
        raise ValueError("Surface, incline, hazard, and contact-mode axes are required")
    _ids(surfaces, "surface")
    _ids(hazards, "hazard")
    _ids(modes, "contact mode")
    if len(inclines) != len(set(inclines)) or any(not 0.0 <= value <= 60.0 for value in inclines):
        raise ValueError("Inclines must be unique and between 0 and 60 degrees")
    vector = config["vectorization"]
    rows, cols, count = int(vector["grid_rows"]), int(vector["grid_cols"]), int(vector["default_num_envs"])
    if rows * cols != count:
        raise ValueError("grid_rows * grid_cols must equal default_num_envs")
    patch = tuple(float(value) for value in vector["patch_size_m"])
    spacing = float(vector["cell_spacing_m"])
    if spacing <= max(patch):
        raise ValueError("Cell spacing must exceed terrain patch extent")
    if not bool(vector.get("filter_cross_environment_collisions")):
        raise ValueError("Cross-environment collision filtering is mandatory")


def vector_layout(config: dict[str, Any]) -> VectorLayout:
    vector = config["vectorization"]
    return VectorLayout(
        num_envs=int(vector["default_num_envs"]),
        rows=int(vector["grid_rows"]),
        cols=int(vector["grid_cols"]),
        cell_spacing_m=float(vector["cell_spacing_m"]),
        patch_size_m=tuple(float(value) for value in vector["patch_size_m"]),
        out_of_bounds_radius_m=float(vector["out_of_bounds_radius_m"]),
    )


def required_cartesian_cases(config: dict[str, Any]) -> list[TerrainCase]:
    surfaces = [(str(item["id"]), str(item["family"])) for item in config["surfaces"]]
    inclines = [float(value) for value in config["inclines_deg"]]
    hazards = [str(item["id"]) for item in config["hazards"]]
    modes = [str(item["id"]) for item in config["contact_modes"]]
    cases: list[TerrainCase] = []
    for surface, incline, hazard, mode in product(surfaces, inclines, hazards, modes):
        surface_id, family = surface
        key = f"{surface_id}|{incline:.6g}|{hazard}|{mode}|0"
        cases.append(TerrainCase(surface_id, family, incline, hazard, mode, 0, sha256(key.encode()).hexdigest()[:16]))
    return cases


def assign_cases(config: dict[str, Any], *, num_envs: int | None = None, seed: int = 0) -> list[TerrainCase]:
    required = required_cartesian_cases(config)
    count = vector_layout(config).num_envs if num_envs is None else int(num_envs)
    if count < len(required):
        raise ValueError(f"{count} environments cannot cover {len(required)} required Cartesian cases")
    result = list(required)
    rng = np.random.default_rng(seed)
    repetition = 1
    while len(result) < count:
        order = rng.permutation(len(required))
        for index in order:
            base = required[int(index)]
            key = f"{base.surface_id}|{base.incline_deg:.6g}|{base.hazard_id}|{base.contact_mode_id}|{repetition}"
            result.append(TerrainCase(base.surface_id, base.surface_family, base.incline_deg, base.hazard_id, base.contact_mode_id, repetition, sha256(key.encode()).hexdigest()[:16]))
            if len(result) == count:
                break
        repetition += 1
    return result


def slope_normal(incline_deg: float) -> np.ndarray:
    angle = np.deg2rad(float(incline_deg))
    return np.asarray([-np.sin(angle), 0.0, np.cos(angle)], dtype=float)
