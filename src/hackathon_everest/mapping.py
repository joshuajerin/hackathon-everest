from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import FootTerrainEstimate

MAP_KEYS = (
    "support_layer_depth_m",
    "vertical_stiffness_n_per_m",
    "bearing_capacity_n",
    "shear_capacity_n",
    "void_probability",
    "damage_state",
)


@dataclass(frozen=True)
class CandidateScore:
    xy_m: np.ndarray
    score: float
    lower_confidence_support_n: float
    void_probability: float
    uncertainty_n: float


class TerrainBeliefMap:
    """Two-meter local map with Gaussian spatial updates and explicit uncertainty."""

    def __init__(
        self,
        *,
        size_m: float = 2.0,
        cell_size_m: float = 0.05,
        prior_means: dict[str, float] | None = None,
        prior_stds: dict[str, float] | None = None,
        correlation_length_m: float = 0.35,
    ):
        self.size_m = float(size_m)
        self.cell_size_m = float(cell_size_m)
        self.correlation_length_m = float(correlation_length_m)
        self.n = round(self.size_m / self.cell_size_m)
        self.coords = np.linspace(
            -self.size_m / 2 + self.cell_size_m / 2,
            self.size_m / 2 - self.cell_size_m / 2,
            self.n,
        )
        means = {
            "support_layer_depth_m": 0.08,
            "vertical_stiffness_n_per_m": 16_000.0,
            "bearing_capacity_n": 330.0,
            "shear_capacity_n": 150.0,
            "void_probability": 0.20,
            "damage_state": 0.10,
        }
        stds = {
            "support_layer_depth_m": 0.07,
            "vertical_stiffness_n_per_m": 14_000.0,
            "bearing_capacity_n": 190.0,
            "shear_capacity_n": 100.0,
            "void_probability": 0.25,
            "damage_state": 0.20,
        }
        means.update(prior_means or {})
        stds.update(prior_stds or {})
        self.mean = {key: np.full((self.n, self.n), means[key], dtype=float) for key in MAP_KEYS}
        self.variance = {key: np.full((self.n, self.n), stds[key] ** 2, dtype=float) for key in MAP_KEYS}
        self.observation_strength = np.zeros((self.n, self.n), dtype=float)

    def index(self, x_m: float, y_m: float) -> tuple[int, int]:
        half = self.size_m / 2
        i = int(np.clip(np.floor((x_m + half) / self.cell_size_m), 0, self.n - 1))
        j = int(np.clip(np.floor((y_m + half) / self.cell_size_m), 0, self.n - 1))
        return i, j

    def update(self, x_m: float, y_m: float, estimate: FootTerrainEstimate) -> None:
        xx, yy = np.meshgrid(self.coords, self.coords, indexing="ij")
        radius_sq = (xx - x_m) ** 2 + (yy - y_m) ** 2
        weight = np.exp(-radius_sq / (2.0 * self.correlation_length_m**2))
        mask = weight >= 0.01
        observed = {
            "support_layer_depth_m": estimate.support_layer_depth_m,
            "vertical_stiffness_n_per_m": estimate.effective_vertical_stiffness_n_per_m,
            "bearing_capacity_n": estimate.bearing_capacity_n,
            "shear_capacity_n": estimate.shear_capacity_n,
            "void_probability": estimate.void_probability,
            "damage_state": estimate.damage_state,
        }
        obs_std = {
            "support_layer_depth_m": estimate.uncertainty[0],
            "vertical_stiffness_n_per_m": estimate.uncertainty[1],
            "bearing_capacity_n": estimate.uncertainty[3],
            "shear_capacity_n": estimate.uncertainty[4],
            "void_probability": max(0.08, np.sqrt(estimate.void_probability * (1 - estimate.void_probability))),
            "damage_state": max(0.08, estimate.uncertainty[7]),
        }
        for key in MAP_KEYS:
            prior_var = self.variance[key]
            effective_obs_var = obs_std[key] ** 2 / np.maximum(weight, 1e-6)
            gain = np.where(mask, prior_var / (prior_var + effective_obs_var), 0.0)
            self.mean[key] += gain * (observed[key] - self.mean[key])
            self.variance[key] = np.where(mask, (1.0 - gain) * prior_var, prior_var)
        self.mean["void_probability"] = np.clip(self.mean["void_probability"], 0.0, 1.0)
        self.mean["damage_state"] = np.clip(self.mean["damage_state"], 0.0, 1.0)
        self.observation_strength = np.maximum(self.observation_strength, weight)

    def update_radar(self, x_m: float, y_m: float, radar_frontend: np.ndarray) -> None:
        """Fuse a pre-contact decoded radar scan without inventing contact support."""
        radar = np.asarray(radar_frontend, dtype=float)
        if radar.shape != (5,):
            raise ValueError(f"Expected five radar frontend values, got {radar.shape}")
        xx, yy = np.meshgrid(self.coords, self.coords, indexing="ij")
        radius_sq = (xx - x_m) ** 2 + (yy - y_m) ** 2
        radar_length = max(0.12, 0.65 * self.correlation_length_m)
        weight = np.exp(-radius_sq / (2.0 * radar_length**2))
        mask = weight >= 0.02
        observations = {
            "support_layer_depth_m": (max(0.0, radar[0]), max(0.02, radar[4])),
            "void_probability": (np.clip(radar[3], 0.0, 1.0), 0.22),
        }
        for key, (observed, observed_std) in observations.items():
            prior_var = self.variance[key]
            effective_obs_var = observed_std**2 / np.maximum(weight, 1e-6)
            gain = np.where(mask, prior_var / (prior_var + effective_obs_var), 0.0)
            self.mean[key] += gain * (observed - self.mean[key])
            self.variance[key] = np.where(mask, (1.0 - gain) * prior_var, prior_var)
        self.mean["void_probability"] = np.clip(self.mean["void_probability"], 0.0, 1.0)
        self.observation_strength = np.maximum(self.observation_strength, 0.5 * weight)

    def score_candidates(
        self,
        candidates_xy_m: np.ndarray,
        *,
        nominal_xy_m: np.ndarray,
        void_weight_n: float = 260.0,
        uncertainty_weight: float = 0.35,
        distance_weight_n_per_m: float = 75.0,
    ) -> list[CandidateScore]:
        scored: list[CandidateScore] = []
        for xy in np.asarray(candidates_xy_m, dtype=float):
            i, j = self.index(float(xy[0]), float(xy[1]))
            mean = self.mean["bearing_capacity_n"][i, j]
            std = np.sqrt(self.variance["bearing_capacity_n"][i, j])
            lower = mean - 2.0 * std
            void = self.mean["void_probability"][i, j]
            distance = float(np.linalg.norm(xy - nominal_xy_m))
            score = lower - void_weight_n * void - uncertainty_weight * std - distance_weight_n_per_m * distance
            scored.append(
                CandidateScore(
                    xy_m=xy.copy(),
                    score=float(score),
                    lower_confidence_support_n=float(lower),
                    void_probability=float(void),
                    uncertainty_n=float(std),
                )
            )
        return sorted(scored, key=lambda item: item.score, reverse=True)
