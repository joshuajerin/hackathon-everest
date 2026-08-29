from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import extract_window_features, feature_names
from .physics import ProbeConfig, ReducedOrderContactBackend
from .sensors import SensorSimulator
from .terrain import TerrainGenerator

TARGET_NAMES = [
    "support_layer_depth_m",
    "effective_vertical_stiffness_n_per_m",
    "effective_vertical_damping_ns_per_m",
    "bearing_capacity_n",
    "shear_capacity_n",
    "effective_friction",
    "compaction_state",
    "damage_state",
    "fracture_margin_n",
    "slip_margin_n",
    "void_depth_m",
]
EVENT_NAMES = ["void_present", "fractured", "slipping"]
DEFAULT_PREFIXES_S = (0.05, 0.10, 0.15, 0.225, 0.30)


@dataclass
class ProbeDataset:
    features: np.ndarray
    targets: np.ndarray
    events: np.ndarray
    field_ids: np.ndarray
    episode_ids: np.ndarray
    prefix_s: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    event_names: list[str]

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            features=self.features,
            targets=self.targets,
            events=self.events,
            field_ids=self.field_ids,
            episode_ids=self.episode_ids,
            prefix_s=self.prefix_s,
            feature_names=np.asarray(self.feature_names),
            target_names=np.asarray(self.target_names),
            event_names=np.asarray(self.event_names),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> ProbeDataset:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                features=data["features"],
                targets=data["targets"],
                events=data["events"],
                field_ids=data["field_ids"],
                episode_ids=data["episode_ids"],
                prefix_s=data["prefix_s"],
                feature_names=data["feature_names"].tolist(),
                target_names=data["target_names"].tolist(),
                event_names=data["event_names"].tolist(),
            )


def generate_probe_dataset(
    *,
    episodes: int,
    seed: int = 7,
    fields: int | None = None,
    prefixes_s: tuple[float, ...] = DEFAULT_PREFIXES_S,
) -> ProbeDataset:
    if episodes < 10:
        raise ValueError("Use at least 10 probe episodes")
    fields = fields or max(20, episodes // 8)
    fields = min(fields, episodes)
    rng = np.random.default_rng(seed)
    generator = TerrainGenerator()
    backend = ReducedOrderContactBackend()
    sensor = SensorSimulator()

    rows: list[np.ndarray] = []
    targets: list[list[float]] = []
    events: list[list[int]] = []
    field_ids: list[int] = []
    episode_ids: list[int] = []
    prefixes: list[float] = []

    counts = np.full(fields, episodes // fields, dtype=int)
    counts[: episodes % fields] += 1
    episode_id = 0
    for field_id, episode_count in enumerate(counts):
        field_seed = int(seed * 100_003 + field_id)
        field = generator.generate(field_seed)
        for _ in range(int(episode_count)):
            x_m, y_m = rng.uniform(-0.82, 0.82, size=2)
            config = ProbeConfig(
                maximum_depth_m=float(rng.uniform(0.025, 0.065)),
                commanded_load_n=float(rng.uniform(90.0, 230.0)),
                approach_speed_mps=float(rng.uniform(0.08, 0.45)),
                tangential_demand_ratio=float(rng.uniform(0.05, 0.75)),
            )
            truth = backend.probe(
                field,
                float(x_m),
                float(y_m),
                seed=int(rng.integers(0, 2**31 - 1)),
                config=config,
                mutate=False,
            )
            packets = sensor.packets(truth, seed=int(rng.integers(0, 2**31 - 1)))
            for prefix in prefixes_s:
                selected = [packet for packet in packets if packet.timestamp_s <= prefix + 1e-9]
                if len(selected) < 2:
                    continue
                end_index = len(selected) - 1
                prefix_field = field.copy()
                backend.apply_episode_prefix(prefix_field, truth, end_index)
                prefix_labels, prefix_events = backend.labels_at_prefix(
                    prefix_field, truth, end_index
                )
                rows.append(extract_window_features(selected))
                targets.append([float(prefix_labels[name]) for name in TARGET_NAMES])
                events.append([int(bool(prefix_events[name])) for name in EVENT_NAMES])
                field_ids.append(field_seed)
                episode_ids.append(episode_id)
                prefixes.append(float(prefix))
            backend.apply_episode_prefix(field, truth, len(truth.timestamps_s) - 1)
            episode_id += 1

    return ProbeDataset(
        features=np.stack(rows),
        targets=np.asarray(targets, dtype=float),
        events=np.asarray(events, dtype=np.int8),
        field_ids=np.asarray(field_ids, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        prefix_s=np.asarray(prefixes, dtype=float),
        feature_names=feature_names(),
        target_names=TARGET_NAMES.copy(),
        event_names=EVENT_NAMES.copy(),
    )
