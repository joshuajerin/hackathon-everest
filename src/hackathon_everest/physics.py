from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .terrain import TerrainField, TerrainPoint

SPIKE_OFFSETS_M = np.array(
    [[0.075, 0.045], [0.075, -0.045], [-0.075, 0.045], [-0.075, -0.045]], dtype=float
)


@dataclass(frozen=True)
class ProbeConfig:
    duration_s: float = 0.30
    sample_rate_hz: int = 100
    maximum_depth_m: float = 0.055
    commanded_load_n: float = 150.0
    approach_speed_mps: float = 0.18
    tangential_demand_ratio: float = 0.18


@dataclass
class ProbeEpisodeTruth:
    timestamps_s: np.ndarray
    axial_force_n: np.ndarray
    penetration_m: np.ndarray
    contact_force_world_n: np.ndarray  # sim-only (T, 4, 3)
    accelerometer_mps2: np.ndarray
    gyroscope_rps: np.ndarray
    radar_frontend_truth: np.ndarray
    commanded_load_n: np.ndarray
    commanded_speed_mps: np.ndarray
    body_weight_on_foot_n: np.ndarray
    fracture_tick_by_spike: np.ndarray  # sim-only (4,), -1 means no fracture
    slipping_mask: np.ndarray  # sim-only (T, 4)
    labels: dict[str, float]
    events: dict[str, bool]
    x_m: float
    y_m: float


def foothold_points(field: TerrainField, x_m: float, y_m: float) -> list[TerrainPoint]:
    return [field.point(x_m + dx, y_m + dy) for dx, dy in SPIKE_OFFSETS_M]


def aggregate_bearing_capacity_n(field: TerrainField, x_m: float, y_m: float) -> float:
    """Whole-foot capacity from the same four cells used by contact simulation."""
    return float(np.mean([point.bearing_capacity_n for point in foothold_points(field, x_m, y_m)]))


class ReducedOrderContactBackend:
    """Fast persistent contact model used before MuJoCo/Newton integration.

    It models load-limited penetration, depth-dependent support, conditional crust
    failure, voids, and spike ploughing. Full 3-D force is simulator truth only.
    """

    @staticmethod
    def _resistance_n(
        point: TerrainPoint,
        depth_m: float,
        velocity_mps: float,
        *,
        fractured_during_probe: bool,
    ) -> float:
        effective_k = point.vertical_stiffness_n_per_m * (1.0 + 1.8 * point.compaction)
        resistance = effective_k * depth_m**1.12 + point.damping_ns_per_m * max(velocity_mps, 0.0)
        resistance += 3.0 * effective_k * max(depth_m - point.support_layer_depth_m, 0.0)

        if point.void_present and point.void_top_depth_m <= depth_m <= (
            point.void_top_depth_m + point.void_height_m
        ):
            resistance *= 0.08

        if point.crust_thickness_m > 0 and not point.fractured and not fractured_during_probe:
            progress = np.clip(depth_m / point.crust_thickness_m, 0.0, 1.0)
            resistance += 0.85 * point.fracture_strength_n / 4.0 * progress**1.5
        elif fractured_during_probe and depth_m > point.crust_thickness_m:
            resistance *= 0.55
        return max(0.0, float(resistance))

    def apply_episode_prefix(
        self,
        field: TerrainField,
        truth: ProbeEpisodeTruth,
        end_index: int,
    ) -> None:
        """Persist only contact history that has occurred by ``end_index``."""
        end_index = int(np.clip(end_index, 0, len(truth.timestamps_s) - 1))
        for spike, (dx, dy) in enumerate(SPIKE_OFFSETS_M):
            fracture_tick = int(truth.fracture_tick_by_spike[spike])
            field.apply_contact(
                truth.x_m + float(dx),
                truth.y_m + float(dy),
                peak_load_n=float(truth.axial_force_n[: end_index + 1, spike].max() * 4.0),
                penetration_m=float(truth.penetration_m[end_index, spike]),
                fractured=bool(0 <= fracture_tick <= end_index),
            )

    def labels_at_prefix(
        self,
        field: TerrainField,
        truth: ProbeEpisodeTruth,
        end_index: int,
    ) -> tuple[dict[str, float], dict[str, bool]]:
        """Return the terrain state available at a causal observation prefix."""
        end_index = int(np.clip(end_index, 0, len(truth.timestamps_s) - 1))
        points = foothold_points(field, truth.x_m, truth.y_m)
        bearing = float(np.mean([point.bearing_capacity_n for point in points]))
        shear = float(np.mean([point.shear_capacity_n for point in points]))
        tangential_peak = float(
            np.linalg.norm(
                truth.contact_force_world_n[: end_index + 1, :, :2].sum(axis=1), axis=1
            ).max()
        )
        fracture_margins: list[float] = []
        for spike, point in enumerate(points):
            fracture_tick = int(truth.fracture_tick_by_spike[spike])
            if 0 <= fracture_tick <= end_index:
                fracture_margins.append(0.0)
            elif point.fractured or point.crust_thickness_m <= 0:
                fracture_margins.append(1_000.0)
            else:
                peak_spike_equivalent = 4.0 * float(
                    truth.axial_force_n[: end_index + 1, spike].max()
                )
                fracture_margins.append(
                    max(0.0, point.fracture_strength_n - peak_spike_equivalent)
                )
        void_points = [point for point in points if point.void_present]
        labels = {
            "support_layer_depth_m": float(
                np.mean([point.support_layer_depth_m for point in points])
            ),
            "effective_vertical_stiffness_n_per_m": float(
                np.mean([point.vertical_stiffness_n_per_m for point in points])
            ),
            "effective_vertical_damping_ns_per_m": float(
                np.mean([point.damping_ns_per_m for point in points])
            ),
            "bearing_capacity_n": bearing,
            "shear_capacity_n": shear,
            "effective_friction": float(np.mean([point.friction for point in points])),
            "compaction_state": float(np.mean([point.compaction for point in points])),
            "damage_state": float(np.mean([point.damage for point in points])),
            "fracture_margin_n": float(min(fracture_margins)),
            "current_sinkage_m": float(truth.penetration_m[end_index].mean()),
            "slip_margin_n": shear - tangential_peak,
            "void_depth_m": float(
                min((point.void_top_depth_m for point in void_points), default=0.0)
            ),
        }
        events = {
            "void_present": bool(void_points),
            "fractured": bool(
                np.any(
                    (truth.fracture_tick_by_spike >= 0)
                    & (truth.fracture_tick_by_spike <= end_index)
                )
            ),
            "slipping": bool(np.any(truth.slipping_mask[: end_index + 1])),
        }
        return labels, events

    def probe(
        self,
        field: TerrainField,
        x_m: float,
        y_m: float,
        *,
        seed: int,
        config: ProbeConfig | None = None,
        mutate: bool = True,
    ) -> ProbeEpisodeTruth:
        cfg = config or ProbeConfig()
        rng = np.random.default_rng(seed)
        count = max(4, round(cfg.duration_s * cfg.sample_rate_hz) + 1)
        timestamps = np.linspace(0.0, cfg.duration_s, count)
        command_ramp = np.sin(np.linspace(0.0, np.pi / 2, count)) ** 1.35
        commanded_load = cfg.commanded_load_n * command_ramp
        dt = timestamps[1] - timestamps[0]

        axial = np.zeros((count, 4), dtype=float)
        depth = np.zeros((count, 4), dtype=float)
        contact_xyz = np.zeros((count, 4, 3), dtype=float)
        fractured_spikes = np.zeros(4, dtype=bool)
        fracture_tick_by_spike = np.full(4, -1, dtype=int)
        slipping_mask = np.zeros((count, 4), dtype=bool)
        fracture_tick: int | None = None
        slip_any = False
        local_points = foothold_points(field, x_m, y_m)

        for spike, point in enumerate(local_points):
            scale = float(rng.uniform(0.88, 1.12))
            offset_m = float(rng.uniform(0.0, 0.0015))
            direction = float(rng.choice([-1.0, 1.0]))
            lateral_skew = float(rng.uniform(-0.25, 0.25))
            previous_depth = 0.0

            for tick, timestamp in enumerate(timestamps):
                desired_depth = max(
                    0.0,
                    min(cfg.maximum_depth_m, cfg.approach_speed_mps * timestamp) * scale - offset_m,
                )
                load_cap = commanded_load[tick] / 4.0
                per_spike_capacity = point.bearing_capacity_n / 4.0 * (1.0 - 0.5 * point.damage)

                crust_intact = point.crust_thickness_m > 0 and not point.fractured and not fractured_spikes[spike]
                reaches_crust = crust_intact and desired_depth >= point.crust_thickness_m
                can_break_crust = load_cap * 4.0 >= point.fracture_strength_n
                if reaches_crust and can_break_crust:
                    fractured_spikes[spike] = True
                    fracture_tick_by_spike[spike] = tick
                    if fracture_tick is None:
                        fracture_tick = tick
                elif reaches_crust:
                    # An under-powered probe loads the crust but cannot pass through it.
                    desired_depth = min(desired_depth, point.crust_thickness_m * 0.98)

                velocity = max((desired_depth - previous_depth) / dt, 0.0)
                desired_resistance = self._resistance_n(
                    point,
                    desired_depth,
                    velocity,
                    fractured_during_probe=bool(fractured_spikes[spike]),
                )

                if desired_resistance > load_cap and load_cap < per_spike_capacity and desired_depth > 0:
                    # Load-controlled probe: solve for the depth supported by the current command.
                    lower, upper = 0.0, desired_depth
                    for _ in range(18):
                        middle = 0.5 * (lower + upper)
                        middle_velocity = max((middle - previous_depth) / dt, 0.0)
                        middle_resistance = self._resistance_n(
                            point,
                            middle,
                            middle_velocity,
                            fractured_during_probe=bool(fractured_spikes[spike]),
                        )
                        if middle_resistance > load_cap:
                            upper = middle
                        else:
                            lower = middle
                    actual_depth = lower
                else:
                    # Once the material yields at its capacity, penetration follows the commanded motion.
                    actual_depth = desired_depth

                actual_velocity = max((actual_depth - previous_depth) / dt, 0.0)
                resistance = self._resistance_n(
                    point,
                    actual_depth,
                    actual_velocity,
                    fractured_during_probe=bool(fractured_spikes[spike]),
                )
                force = min(resistance, load_cap, per_spike_capacity)
                depth[tick, spike] = actual_depth
                axial[tick, spike] = max(0.0, force)
                previous_depth = actual_depth

                tangential = cfg.tangential_demand_ratio * force
                engagement = np.clip(actual_depth / 0.012, 0.0, 1.0)
                tangential_capacity = point.friction * force + 0.25 * point.shear_capacity_n * engagement
                slipping_mask[tick, spike] = tangential > tangential_capacity
                slip_any |= bool(slipping_mask[tick, spike])
                contact_xyz[tick, spike, 0] = tangential * direction
                contact_xyz[tick, spike, 1] = tangential * lateral_skew
                contact_xyz[tick, spike, 2] = force

        total_force = axial.sum(axis=1)
        force_rate = np.gradient(total_force, dt)
        accel = np.zeros((count, 3), dtype=float)
        accel[:, 2] = 9.81 + 0.0028 * force_rate
        accel[:, 0] = cfg.tangential_demand_ratio * total_force / 35.0
        gyro = np.zeros((count, 3), dtype=float)
        asymmetry = (axial[:, 0] + axial[:, 1]) - (axial[:, 2] + axial[:, 3])
        gyro[:, 1] = 0.002 * asymmetry
        if fracture_tick is not None:
            event_idx = min(count - 2, max(2, fracture_tick))
            accel[event_idx, 2] -= 18.0
            gyro[event_idx, 1] += 2.4
        if slip_any:
            tail = max(2, count // 5)
            accel[-tail:, 0] += np.linspace(0.0, 5.5, tail)
            gyro[-tail:, 2] += np.linspace(0.0, 1.2, tail)

        # The frontend summarizes a small footprint scan around the candidate, not
        # only the mathematical foot center.
        pre_void_points = [point for point in local_points if point.void_present]
        first_depth = float(np.mean([point.support_layer_depth_m for point in local_points]))
        second_depth = (
            min(point.void_top_depth_m for point in pre_void_points)
            if pre_void_points
            else first_depth + float(rng.uniform(0.08, 0.35))
        )
        return_strength = np.clip(
            np.mean(
                [
                    0.25 + point.vertical_stiffness_n_per_m / 55_000.0 - 0.35 * point.wetness
                    for point in local_points
                ]
            ),
            0.02,
            1.0,
        )
        radar_truth = np.array(
            [first_depth, second_depth, return_strength, float(bool(pre_void_points)), 0.02],
            dtype=float,
        )

        truth = ProbeEpisodeTruth(
            timestamps_s=timestamps,
            axial_force_n=axial,
            penetration_m=depth,
            contact_force_world_n=contact_xyz,
            accelerometer_mps2=accel,
            gyroscope_rps=gyro,
            radar_frontend_truth=radar_truth,
            commanded_load_n=commanded_load,
            commanded_speed_mps=np.full(count, cfg.approach_speed_mps),
            body_weight_on_foot_n=np.clip(total_force, 0.0, 343.0),
            fracture_tick_by_spike=fracture_tick_by_spike,
            slipping_mask=slipping_mask,
            labels={},
            events={},
            x_m=x_m,
            y_m=y_m,
        )
        if mutate:
            self.apply_episode_prefix(field, truth, count - 1)
        truth.labels, truth.events = self.labels_at_prefix(field, truth, count - 1)
        return truth
