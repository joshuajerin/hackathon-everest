from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class TerrainPoint:
    surface_height_m: float
    support_layer_depth_m: float
    vertical_stiffness_n_per_m: float
    damping_ns_per_m: float
    bearing_capacity_n: float
    shear_capacity_n: float
    friction: float
    crust_thickness_m: float
    fracture_strength_n: float
    void_present: bool
    void_top_depth_m: float
    void_height_m: float
    temperature_c: float
    wetness: float
    compaction: float
    damage: float
    fractured: bool


@dataclass(frozen=True)
class TerrainGeneratorConfig:
    size_m: float = 2.0
    cell_size_m: float = 0.05
    density_correlation_length_m: tuple[float, float] = (0.2, 0.8)
    stiffness_correlation_length_m: tuple[float, float] = (0.2, 0.8)
    snow_depth_correlation_length_m: tuple[float, float] = (0.3, 1.5)
    max_voids: int = 3


class TerrainField:
    """A persistent, spatially correlated reduced-order layered terrain field."""

    def __init__(self, *, cell_size_m: float, arrays: dict[str, np.ndarray], seed: int):
        self.cell_size_m = float(cell_size_m)
        self.arrays = arrays
        self.seed = int(seed)
        shape = arrays["surface_height_m"].shape
        if any(value.shape != shape for value in arrays.values()):
            raise ValueError("All terrain arrays must have the same shape")
        self.shape = shape
        self.size_m = shape[0] * self.cell_size_m

    def copy(self) -> TerrainField:
        return TerrainField(
            cell_size_m=self.cell_size_m,
            arrays={key: value.copy() for key, value in self.arrays.items()},
            seed=self.seed,
        )

    def index(self, x_m: float, y_m: float) -> tuple[int, int]:
        half = self.size_m / 2
        ix = int(np.clip(np.floor((x_m + half) / self.cell_size_m), 0, self.shape[0] - 1))
        iy = int(np.clip(np.floor((y_m + half) / self.cell_size_m), 0, self.shape[1] - 1))
        return ix, iy

    def point(self, x_m: float, y_m: float) -> TerrainPoint:
        i, j = self.index(x_m, y_m)
        a = self.arrays
        return TerrainPoint(
            surface_height_m=float(a["surface_height_m"][i, j]),
            support_layer_depth_m=float(a["support_layer_depth_m"][i, j]),
            vertical_stiffness_n_per_m=float(a["vertical_stiffness_n_per_m"][i, j]),
            damping_ns_per_m=float(a["damping_ns_per_m"][i, j]),
            bearing_capacity_n=float(a["bearing_capacity_n"][i, j]),
            shear_capacity_n=float(a["shear_capacity_n"][i, j]),
            friction=float(a["friction"][i, j]),
            crust_thickness_m=float(a["crust_thickness_m"][i, j]),
            fracture_strength_n=float(a["fracture_strength_n"][i, j]),
            void_present=bool(a["void_present"][i, j]),
            void_top_depth_m=float(a["void_top_depth_m"][i, j]),
            void_height_m=float(a["void_height_m"][i, j]),
            temperature_c=float(a["temperature_c"][i, j]),
            wetness=float(a["wetness"][i, j]),
            compaction=float(a["compaction"][i, j]),
            damage=float(a["damage"][i, j]),
            fractured=bool(a["fractured"][i, j]),
        )

    def apply_contact(
        self,
        x_m: float,
        y_m: float,
        *,
        peak_load_n: float,
        penetration_m: float,
        fractured: bool,
    ) -> None:
        """Persist compaction, deformation, and fracture after the foot leaves."""
        i, j = self.index(x_m, y_m)
        capacity = max(float(self.arrays["bearing_capacity_n"][i, j]), 1.0)
        load_ratio = peak_load_n / capacity
        compaction_delta = np.clip(0.12 * load_ratio + 2.0 * penetration_m, 0.0, 0.25)
        self.arrays["compaction"][i, j] = np.clip(
            self.arrays["compaction"][i, j] + compaction_delta, 0.0, 1.0
        )
        self.arrays["surface_height_m"][i, j] -= penetration_m * (0.08 + 0.12 * load_ratio)
        # Moderate compaction hardens snow; damage removes support.
        self.arrays["vertical_stiffness_n_per_m"][i, j] *= 1.0 + 0.5 * compaction_delta
        if load_ratio > 1.0 or fractured:
            damage_delta = min(0.45, 0.2 * max(load_ratio - 0.8, 0.0) + (0.3 if fractured else 0.0))
            self.arrays["damage"][i, j] = np.clip(
                self.arrays["damage"][i, j] + damage_delta, 0.0, 1.0
            )
        if fractured:
            self.arrays["fractured"][i, j] = True
            self.arrays["crust_thickness_m"][i, j] = 0.0
            # Breaking a thin crust is not automatically loss of the whole foothold.
            # The remaining layer/void state controls how much capacity disappears.
            support_depth = self.arrays["support_layer_depth_m"][i, j]
            void_penalty = 0.16 if self.arrays["void_present"][i, j] else 0.0
            depth_penalty = 0.24 * np.clip(support_depth / 0.18, 0.0, 1.0)
            bearing_factor = np.clip(0.92 - depth_penalty - void_penalty, 0.52, 0.92)
            shear_factor = np.clip(bearing_factor - 0.06, 0.46, 0.86)
            self.arrays["bearing_capacity_n"][i, j] *= bearing_factor
            self.arrays["shear_capacity_n"][i, j] *= shear_factor


class TerrainGenerator:
    def __init__(self, config: TerrainGeneratorConfig | None = None):
        self.config = config or TerrainGeneratorConfig()

    @staticmethod
    def _field(
        rng: np.random.Generator,
        shape: tuple[int, int],
        sigma_cells: float,
        low: float,
        high: float,
    ) -> np.ndarray:
        raw = gaussian_filter(rng.normal(size=shape), sigma=max(sigma_cells, 0.5), mode="reflect")
        lo, hi = np.quantile(raw, [0.02, 0.98])
        normalized = np.clip((raw - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        return low + normalized * (high - low)

    def generate(self, seed: int) -> TerrainField:
        cfg = self.config
        rng = np.random.default_rng(seed)
        n = round(cfg.size_m / cfg.cell_size_m)
        shape = (n, n)

        depth_l = rng.uniform(*cfg.snow_depth_correlation_length_m) / cfg.cell_size_m
        stiff_l = rng.uniform(*cfg.stiffness_correlation_length_m) / cfg.cell_size_m
        density_l = rng.uniform(*cfg.density_correlation_length_m) / cfg.cell_size_m

        support_depth = self._field(rng, shape, depth_l, 0.003, 0.18)
        log_stiffness = self._field(rng, shape, stiff_l, np.log(2_500.0), np.log(45_000.0))
        stiffness = np.exp(log_stiffness)
        density = self._field(rng, shape, density_l, 180.0, 650.0)
        friction = self._field(rng, shape, stiff_l, 0.08, 0.75)
        wetness = self._field(rng, shape, density_l, 0.0, 0.55)
        surface = self._field(rng, shape, depth_l, -0.03, 0.04)
        # Add a continuous slope without creating named surface classes.
        slope_x, slope_y = rng.uniform(-0.25, 0.25, size=2)
        coords = np.linspace(-cfg.size_m / 2, cfg.size_m / 2, n, endpoint=False)
        surface += slope_x * coords[:, None] + slope_y * coords[None, :]

        # Most footholds should be load-bearing enough to make route planning meaningful,
        # while soft/wet zones and voids remain dangerous for single-leg support.
        bearing = np.clip(180.0 + 0.014 * stiffness + 0.28 * density - 180.0 * wetness, 100.0, 860.0)
        shear = np.clip(friction * bearing + 850.0 * support_depth, 35.0, 560.0)
        damping = np.clip(18.0 + 0.07 * density + 85.0 * wetness, 20.0, 140.0)

        crust_probability = self._field(rng, shape, depth_l / 2, 0.0, 1.0)
        crust = np.where(crust_probability > 0.58, rng.uniform(0.002, 0.03, shape), 0.0)
        fracture_strength = np.where(
            crust > 0,
            np.clip(70.0 + 7_000.0 * crust + rng.normal(0, 18, shape), 50.0, 310.0),
            1_000.0,
        )

        void_present = np.zeros(shape, dtype=bool)
        void_top = np.full(shape, 0.5)
        void_height = np.zeros(shape)
        xx, yy = np.meshgrid(coords, coords, indexing="ij")
        for _ in range(int(rng.integers(0, cfg.max_voids + 1))):
            cx, cy = rng.uniform(-0.75, 0.75, size=2)
            rx, ry = rng.uniform(0.08, 0.32, size=2)
            mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
            void_present |= mask
            void_top[mask] = rng.uniform(0.05, 0.35)
            void_height[mask] = rng.uniform(0.08, 0.6)
            bearing[mask] *= rng.uniform(0.28, 0.62)
            shear[mask] *= rng.uniform(0.35, 0.7)

        # Optional abrupt transition, still with continuous properties on either side.
        if rng.random() < 0.45:
            cut = int(rng.integers(n // 4, 3 * n // 4))
            factor = rng.uniform(0.45, 1.65)
            stiffness[cut:, :] *= factor
            bearing[cut:, :] *= np.sqrt(factor)

        arrays = {
            "surface_height_m": surface.astype(float),
            "support_layer_depth_m": support_depth.astype(float),
            "vertical_stiffness_n_per_m": stiffness.astype(float),
            "damping_ns_per_m": damping.astype(float),
            "bearing_capacity_n": np.clip(bearing, 60.0, 900.0).astype(float),
            "shear_capacity_n": np.clip(shear, 20.0, 600.0).astype(float),
            "friction": friction.astype(float),
            "crust_thickness_m": crust.astype(float),
            "fracture_strength_n": fracture_strength.astype(float),
            "void_present": void_present,
            "void_top_depth_m": void_top.astype(float),
            "void_height_m": void_height.astype(float),
            "temperature_c": self._field(rng, shape, depth_l, -32.0, -2.0),
            "wetness": wetness.astype(float),
            "compaction": np.zeros(shape, dtype=float),
            "damage": np.zeros(shape, dtype=float),
            "fractured": np.zeros(shape, dtype=bool),
        }
        return TerrainField(cell_size_m=cfg.cell_size_m, arrays=arrays, seed=seed)
