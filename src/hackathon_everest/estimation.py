from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, recall_score
from sklearn.model_selection import GroupShuffleSplit

from .dataset import EVENT_NAMES, TARGET_NAMES, ProbeDataset
from .features import extract_window_features, feature_names
from .models import FootTerrainEstimate, SynchronizedSensorPacket
from .physics import SPIKE_OFFSETS_M


@dataclass
class TrainingResult:
    estimator: TerrainStateEstimator
    metrics: dict[str, Any]
    train_indices: np.ndarray
    test_indices: np.ndarray


class TerrainStateEstimator:
    """CPU-friendly continuous estimator with ensemble-spread uncertainty."""

    def __init__(self, *, n_estimators: int = 128, random_state: int = 7):
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.regressor = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=2,
            max_features=0.82,
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.event_models: dict[str, ExtraTreesClassifier | float] = {}
        self.target_names = TARGET_NAMES.copy()
        self.event_names = EVENT_NAMES.copy()
        self.feature_names = feature_names()

    def fit(self, features: np.ndarray, targets: np.ndarray, events: np.ndarray) -> TerrainStateEstimator:
        self.regressor.fit(features, targets)
        self.event_models = {}
        for index, name in enumerate(self.event_names):
            labels = events[:, index]
            unique = np.unique(labels)
            if len(unique) == 1:
                self.event_models[name] = float(unique[0])
                continue
            model = ExtraTreesClassifier(
                n_estimators=max(64, self.n_estimators // 2),
                min_samples_leaf=2,
                max_features=0.82,
                n_jobs=-1,
                random_state=self.random_state + index + 1,
                class_weight="balanced",
            )
            model.fit(features, labels)
            self.event_models[name] = model
        return self

    def predict_continuous(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.atleast_2d(features)
        tree_predictions = np.stack([tree.predict(matrix) for tree in self.regressor.estimators_], axis=0)
        mean = tree_predictions.mean(axis=0)
        std = tree_predictions.std(axis=0)
        # Ensemble spread is approximate, so retain conservative physical floors.
        floors = np.array([0.004, 900.0, 4.0, 18.0, 10.0, 0.03, 0.03, 0.03, 12.0, 10.0, 0.015])
        return mean, np.maximum(std, floors)

    def predict_events(self, features: np.ndarray) -> dict[str, np.ndarray]:
        matrix = np.atleast_2d(features)
        output: dict[str, np.ndarray] = {}
        for name, model in self.event_models.items():
            if isinstance(model, float):
                output[name] = np.full(matrix.shape[0], model)
            else:
                classes = model.classes_.tolist()
                probabilities = model.predict_proba(matrix)
                output[name] = probabilities[:, classes.index(1)] if 1 in classes else np.zeros(matrix.shape[0])
        return output

    def estimate(self, packets: list[SynchronizedSensorPacket]) -> FootTerrainEstimate:
        features = extract_window_features(packets)
        mean, std = self.predict_continuous(features)
        values = dict(zip(self.target_names, mean[0], strict=True))
        uncertainty = std[0]
        event_probability = {key: value[0] for key, value in self.predict_events(features).items()}
        last = packets[-1]
        tail = packets[-min(5, len(packets)) :]
        duration = max(tail[-1].timestamp_s - tail[0].timestamp_s, 1e-6)
        sinkage = float(np.mean(last.penetration_m))
        sinkage_rate = float(
            (np.mean(tail[-1].penetration_m) - np.mean(tail[0].penetration_m)) / duration
        )
        engagement = ((last.penetration_m > 0.002) & (last.axial_force_n > 2.0)).astype(float)
        force_quality = last.axial_force_n / max(float(last.axial_force_n.max()), 1.0)
        support_quality = np.clip(engagement * force_quality, 0.0, 1.0)
        weights = np.clip(last.axial_force_n, 0.0, None)
        center = (
            np.average(SPIKE_OFFSETS_M, axis=0, weights=weights)
            if weights.sum() > 1e-6
            else np.zeros(2)
        )
        void_probability = float(event_probability.get("void_present", 0.0))
        void_depth = max(0.0, float(values["void_depth_m"])) if void_probability > 0.15 else 0.0
        return FootTerrainEstimate(
            support_layer_depth_m=max(0.0, float(values["support_layer_depth_m"])),
            void_probability=np.clip(void_probability, 0.0, 1.0),
            fracture_probability=np.clip(event_probability.get("fractured", 0.0), 0.0, 1.0),
            slip_probability=np.clip(event_probability.get("slipping", 0.0), 0.0, 1.0),
            void_depth_m=void_depth,
            effective_vertical_stiffness_n_per_m=max(
                1.0, float(values["effective_vertical_stiffness_n_per_m"])
            ),
            effective_vertical_damping_ns_per_m=max(
                0.0, float(values["effective_vertical_damping_ns_per_m"])
            ),
            bearing_capacity_n=max(0.0, float(values["bearing_capacity_n"])),
            current_sinkage_m=sinkage,
            sinkage_rate_mps=sinkage_rate,
            shear_capacity_n=max(0.0, float(values["shear_capacity_n"])),
            effective_friction=max(0.0, float(values["effective_friction"])),
            slip_margin_n=float(values["slip_margin_n"]),
            spike_engagement=engagement,
            spike_support_quality=support_quality,
            center_of_support_xy=np.asarray(center, dtype=float),
            compaction_state=np.clip(float(values["compaction_state"]), 0.0, 1.0),
            damage_state=np.clip(float(values["damage_state"]), 0.0, 1.0),
            fracture_margin_n=max(0.0, float(values["fracture_margin_n"])),
            uncertainty=uncertainty,
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> TerrainStateEstimator:
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}")
        return model


def train_estimator(
    dataset: ProbeDataset,
    *,
    n_estimators: int = 128,
    seed: int = 7,
    test_size: float = 0.2,
) -> TrainingResult:
    if dataset.feature_names != feature_names():
        raise ValueError("Dataset feature contract does not match this code version")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(splitter.split(dataset.features, groups=dataset.field_ids))
    estimator = TerrainStateEstimator(n_estimators=n_estimators, random_state=seed)
    estimator.fit(dataset.features[train_index], dataset.targets[train_index], dataset.events[train_index])

    predicted, uncertainty = estimator.predict_continuous(dataset.features[test_index])
    probabilities = estimator.predict_events(dataset.features[test_index])
    target_index = {name: idx for idx, name in enumerate(dataset.target_names)}
    event_index = {name: idx for idx, name in enumerate(dataset.event_names)}
    true = dataset.targets[test_index]

    bearing_idx = target_index["bearing_capacity_n"]
    support_idx = target_index["support_layer_depth_m"]
    shear_idx = target_index["shear_capacity_n"]
    fracture_idx = target_index["fracture_margin_n"]
    true_unsafe = true[:, bearing_idx] < 343.0
    predicted_safe = predicted[:, bearing_idx] - 2.0 * uncertainty[:, bearing_idx] >= 343.0
    false_safe_rate = float(np.mean(predicted_safe[true_unsafe])) if np.any(true_unsafe) else 0.0
    void_true = dataset.events[test_index, event_index["void_present"]]
    void_pred = probabilities["void_present"] >= 0.5

    metrics: dict[str, Any] = {
        "split": {
            "train_rows": len(train_index),
            "test_rows": len(test_index),
            "train_fields": len(np.unique(dataset.field_ids[train_index])),
            "test_fields": len(np.unique(dataset.field_ids[test_index])),
            "field_overlap": len(set(dataset.field_ids[train_index]).intersection(dataset.field_ids[test_index])),
        },
        "support_depth_mae_m": float(mean_absolute_error(true[:, support_idx], predicted[:, support_idx])),
        "bearing_capacity_mae_n": float(mean_absolute_error(true[:, bearing_idx], predicted[:, bearing_idx])),
        "bearing_capacity_relative_error": float(
            np.mean(np.abs(predicted[:, bearing_idx] - true[:, bearing_idx]) / np.maximum(true[:, bearing_idx], 1.0))
        ),
        "shear_capacity_relative_error": float(
            np.mean(np.abs(predicted[:, shear_idx] - true[:, shear_idx]) / np.maximum(true[:, shear_idx], 1.0))
        ),
        "fracture_margin_mae_n": float(
            mean_absolute_error(true[:, fracture_idx], predicted[:, fracture_idx])
        ),
        "void_recall": float(recall_score(void_true, void_pred, zero_division=0)),
        "false_safe_rate_on_truly_unsafe": false_safe_rate,
        "bearing_2sigma_coverage": float(
            np.mean(np.abs(predicted[:, bearing_idx] - true[:, bearing_idx]) <= 2.0 * uncertainty[:, bearing_idx])
        ),
        "uncertainty_note": "Tree-to-tree spread with physical floors; approximate, not a calibrated safety probability.",
    }
    return TrainingResult(estimator=estimator, metrics=metrics, train_indices=train_index, test_indices=test_index)
