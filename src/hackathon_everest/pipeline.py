from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .control import AdaptiveStepScheduler, BilateralSupportManager
from .dataset import generate_probe_dataset
from .estimation import TerrainStateEstimator, train_estimator
from .mapping import TerrainBeliefMap
from .models import FootSide, FootTerrainEstimate
from .physics import ProbeConfig, ReducedOrderContactBackend, aggregate_bearing_capacity_n
from .sensors import SensorSimulator
from .terrain import TerrainField, TerrainGenerator


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Config root must be a mapping")
    return config


def _probe_estimate(
    estimator: TerrainStateEstimator,
    field: TerrainField,
    xy_m: np.ndarray,
    *,
    seed: int,
    probe_config: ProbeConfig | None = None,
    mutate: bool = True,
) -> tuple[FootTerrainEstimate, dict[str, Any]]:
    backend = ReducedOrderContactBackend()
    sensors = SensorSimulator()
    truth = backend.probe(
        field,
        float(xy_m[0]),
        float(xy_m[1]),
        seed=seed,
        config=probe_config or ProbeConfig(),
        mutate=mutate,
    )
    packets = sensors.packets(truth, seed=seed + 1)
    estimate = estimator.estimate(packets)
    diagnostics = {
        "truth": truth.labels,
        "events": truth.events,
        "estimate": estimate.to_dict(),
        "sensor_channels": int(packets[-1].vector().size),
        "valid_sample_fraction": float(np.stack([packet.valid_mask for packet in packets]).mean()),
        "executed_probe": {
            "commanded_load_n": float(truth.commanded_load_n[-1]),
            "approach_speed_mps": float(truth.commanded_speed_mps[-1]),
        },
    }
    return estimate, diagnostics


def _radar_scan(field: TerrainField, xy_m: np.ndarray, *, seed: int) -> np.ndarray:
    """Run the same noisy five-value frontend before contact, with zero probe load."""
    truth = ReducedOrderContactBackend().probe(
        field,
        float(xy_m[0]),
        float(xy_m[1]),
        seed=seed,
        config=ProbeConfig(
            maximum_depth_m=0.0,
            commanded_load_n=0.0,
            approach_speed_mps=0.0,
        ),
        mutate=False,
    )
    packets = SensorSimulator().packets(truth, seed=seed + 1)
    return packets[-1].radar_frontend.copy()


def _prepare_known_start_zone(field: TerrainField) -> None:
    """Create a known stable launch patch; the unknown route begins beyond x=-0.42 m."""
    n = field.shape[0]
    coords = np.linspace(-field.size_m / 2, field.size_m / 2, n, endpoint=False)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    mask = (xx < -0.42) & (np.abs(yy) < 0.32)
    field.arrays["support_layer_depth_m"][mask] = np.minimum(
        field.arrays["support_layer_depth_m"][mask], 0.018
    )
    field.arrays["vertical_stiffness_n_per_m"][mask] = np.maximum(
        field.arrays["vertical_stiffness_n_per_m"][mask], 38_000.0
    )
    field.arrays["bearing_capacity_n"][mask] = np.maximum(
        field.arrays["bearing_capacity_n"][mask], 580.0
    )
    field.arrays["shear_capacity_n"][mask] = np.maximum(
        field.arrays["shear_capacity_n"][mask], 290.0
    )
    field.arrays["void_present"][mask] = False
    field.arrays["void_height_m"][mask] = 0.0
    field.arrays["crust_thickness_m"][mask] = 0.0


def _candidate_grid(nominal: np.ndarray, foot: FootSide) -> np.ndarray:
    lateral = 1.0 if foot == FootSide.LEFT else -1.0
    offsets = np.array(
        [
            [0.00, 0.00],
            [0.04, 0.06 * lateral],
            [0.04, -0.06 * lateral],
            [-0.04, 0.08 * lateral],
            [-0.04, -0.08 * lateral],
            [0.08, 0.02 * lateral],
            [0.08, -0.10 * lateral],
        ]
    )
    return np.clip(nominal + offsets, -0.92, 0.92)


def run_bilateral_replay(
    estimator: TerrainStateEstimator,
    *,
    terrain_seed: int,
    steps: int = 6,
    mode: str = "full",
) -> tuple[dict[str, Any], TerrainField, TerrainBeliefMap]:
    if mode not in {"full", "current_only"}:
        raise ValueError("mode must be 'full' or 'current_only'")
    field = TerrainGenerator().generate(terrain_seed)
    _prepare_known_start_zone(field)
    belief = TerrainBeliefMap()
    support_manager = BilateralSupportManager()
    scheduler = AdaptiveStepScheduler(support_manager)
    witness_xy = np.array([-0.28, -0.11])
    witness_index = belief.index(*witness_xy)
    witness_before = {
        "bearing_mean_n": float(belief.mean["bearing_capacity_n"][witness_index]),
        "bearing_std_n": float(np.sqrt(belief.variance["bearing_capacity_n"][witness_index])),
    }
    witness_after_left = witness_before.copy()

    positions = {
        FootSide.LEFT: np.array([-0.68, 0.11]),
        FootSide.RIGHT: np.array([-0.55, -0.11]),
    }
    estimates: dict[FootSide, FootTerrainEstimate] = {}
    log: list[dict[str, Any]] = []
    for index, side in enumerate((FootSide.LEFT, FootSide.RIGHT)):
        estimate, diagnostics = _probe_estimate(
            estimator, field, positions[side], seed=terrain_seed * 100 + index
        )
        estimates[side] = estimate
        if mode == "full":
            belief.update(*positions[side], estimate)
            if side == FootSide.LEFT:
                witness_after_left = {
                    "bearing_mean_n": float(belief.mean["bearing_capacity_n"][witness_index]),
                    "bearing_std_n": float(np.sqrt(belief.variance["bearing_capacity_n"][witness_index])),
                }
        log.append(
            {
                "phase": "initial_contact",
                "foot": side.value,
                "position_xy_m": positions[side],
                **diagnostics,
            }
        )

    unsafe_transfers = 0
    replants = 0
    holds = 0
    commits = 0
    completed_steps = 0
    swing = FootSide.RIGHT
    for step in range(steps):
        attempted_swing = swing
        stance = FootSide.LEFT if swing == FootSide.RIGHT else FootSide.RIGHT
        nominal = positions[swing] + np.array([0.27, 0.0])
        candidates = _candidate_grid(nominal, swing)
        candidate_radar_scans: list[dict[str, Any]] = []
        if mode == "full":
            for candidate_index, candidate in enumerate(candidates):
                radar = _radar_scan(
                    field,
                    candidate,
                    seed=terrain_seed * 10_000 + step * 100 + candidate_index,
                )
                belief.update_radar(*candidate, radar)
                candidate_radar_scans.append({"xy_m": candidate, "radar_frontend": radar})
            plan = scheduler.select_candidate(belief, candidates, nominal_xy_m=nominal)
        else:
            target = candidates[0]
            plan = scheduler.select_candidate(
                TerrainBeliefMap(
                    prior_stds={
                        "support_layer_depth_m": 0.001,
                        "vertical_stiffness_n_per_m": 1.0,
                        "bearing_capacity_n": 1.0,
                        "shear_capacity_n": 1.0,
                        "void_probability": 0.01,
                        "damage_state": 0.01,
                    }
                ),
                np.asarray([target]),
                nominal_xy_m=nominal,
            )
        target_estimate, diagnostics = _probe_estimate(
            estimator,
            field,
            plan.target_xy_m,
            seed=terrain_seed * 1_000 + step * 10 + 1,
            probe_config=ProbeConfig(
                commanded_load_n=plan.probe_load_n,
                approach_speed_mps=plan.approach_velocity_mps,
            ),
        )
        if mode == "full":
            belief.update(*plan.target_xy_m, target_estimate)

        if swing == FootSide.RIGHT:
            left, right = estimates[FootSide.LEFT], target_estimate
            left_load, right_load = 220.0, 123.0
        else:
            left, right = target_estimate, estimates[FootSide.RIGHT]
            left_load, right_load = 123.0, 220.0
        bilateral = support_manager.evaluate(
            left,
            right,
            left_load_n=left_load,
            right_load_n=right_load,
        )
        if mode == "full":
            decision = scheduler.decide_after_probe(
                plan=plan, swing_side=swing, target=target_estimate, bilateral=bilateral
            )
        else:
            # Ablation: verify only the target contact, with no spatial memory or stance-foot gate.
            target_bad = target_estimate.lower_confidence_bearing_n() < 155.0 or target_estimate.void_probability > 0.55
            decision = scheduler.decide_after_probe(
                plan=plan, swing_side=swing, target=target_estimate, bilateral=bilateral
            )
            decision.action = "REPLANT" if target_bad else "COMMIT"
            decision.load_transfer_rate_nps = 120.0
            decision.reason = (
                "Current-contact target check rejected the foothold."
                if target_bad
                else "Current-contact target check passed; stance state and map are not used."
            )

        truth_capacity = float(diagnostics["truth"]["bearing_capacity_n"])
        stance_truth_capacity = aggregate_bearing_capacity_n(field, *positions[stance])
        dynamic_stance_demand_n = (
            support_manager.robot_weight_n + 0.04 * decision.load_transfer_rate_nps
        )
        dynamic_target_demand_n = 123.0 + 0.02 * decision.load_transfer_rate_nps
        source_height = field.point(*positions[attempted_swing]).surface_height_m
        target_height = field.point(*plan.target_xy_m).surface_height_m
        required_clearance_m = max(0.0, target_height - source_height) + 0.04
        clearance_ok = plan.swing_clearance_m >= required_clearance_m
        actual_unsafe = bool(
            diagnostics["events"]["void_present"]
            or truth_capacity < dynamic_target_demand_n
            or stance_truth_capacity < dynamic_stance_demand_n
            or not clearance_ok
        )
        if decision.action == "COMMIT":
            commits += 1
            completed_steps += 1
            unsafe_transfers += int(actual_unsafe)
            positions[swing] = plan.target_xy_m.copy()
            estimates[swing] = target_estimate
            swing = stance
        elif decision.action == "REPLANT":
            replants += 1
        else:
            holds += 1

        log.append(
            {
                "phase": "step",
                "step": step,
                "mode": mode,
                "swing_foot": attempted_swing.value,
                "stance_foot": stance.value,
                "nominal_xy_m": nominal,
                "candidate_count": len(candidates),
                "candidate_radar_scans": candidate_radar_scans,
                "plan": plan.to_dict(),
                "decision": decision.to_dict(),
                "bilateral_support": asdict(bilateral),
                "actual_unsafe_transfer_if_committed": actual_unsafe,
                "simulated_execution": {
                    "stance_capacity_n": stance_truth_capacity,
                    "stance_dynamic_demand_n": dynamic_stance_demand_n,
                    "target_capacity_n": truth_capacity,
                    "target_dynamic_demand_n": dynamic_target_demand_n,
                    "required_clearance_m": required_clearance_m,
                    "scheduled_clearance_m": plan.swing_clearance_m,
                    "clearance_ok": clearance_ok,
                },
                **diagnostics,
            }
        )
        if decision.action == "HOLD_DOUBLE_SUPPORT":
            break

    attempted_steps = sum(event.get("phase") == "step" for event in log)
    summary = {
        "mode": mode,
        "terrain_seed": terrain_seed,
        "requested_steps": steps,
        "attempted_steps": attempted_steps,
        "completed_steps": completed_steps,
        "commits": commits,
        "holds": holds,
        "replants": replants,
        "unsafe_transfers": unsafe_transfers,
        "known_start_zone": "Stable launch patch ending at x=-0.42 m; route terrain remains unknown.",
        "claim_limit": "Scripted reduced-order support scheduling; not whole-body G1 locomotion.",
    }
    cross_foot_evidence = {
        "source_foot": "left",
        "witness_xy_m": witness_xy,
        "before_left_contact": witness_before,
        "after_left_contact": witness_after_left,
        "meaning": "The left-foot contact changes the right-foot candidate prior before that candidate is touched.",
    }
    return {
        "summary": summary,
        "cross_foot_evidence": cross_foot_evidence,
        "events": log,
    }, field, belief


def benchmark_modes(
    estimator: TerrainStateEstimator,
    *,
    field_count: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    results: dict[str, list[dict[str, Any]]] = {"current_only": [], "full": []}
    terrain_seeds = [seed * 10_000 + 80_000 + index for index in range(field_count)]
    for terrain_seed in terrain_seeds:
        for mode, mode_rows in results.items():
            replay, _, _ = run_bilateral_replay(
                estimator, terrain_seed=terrain_seed, steps=steps, mode=mode
            )
            mode_rows.append(replay["summary"])
    summary: dict[str, Any] = {"field_seeds": terrain_seeds, "systems": {}}
    for mode, rows in results.items():
        completed = int(sum(row["completed_steps"] for row in rows))
        unsafe = int(sum(row["unsafe_transfers"] for row in rows))
        summary["systems"][mode] = {
            "completed_steps": completed,
            "unsafe_transfers": unsafe,
            "unsafe_transfer_rate_per_commit": float(unsafe / completed) if completed else 0.0,
            "holds": int(sum(row["holds"] for row in rows)),
            "replants": int(sum(row["replants"] for row in rows)),
            "episodes": len(rows),
        }
    summary["comparison_note"] = (
        "Both scripted systems use identical held-out terrain seeds. Results measure scheduler behavior "
        "inside the reduced-order simulator, not real-world fall probability."
    )
    return summary


def render_replay(
    replay: dict[str, Any],
    field: TerrainField,
    belief: TerrainBeliefMap,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    extent = [-1.0, 1.0, -1.0, 1.0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    images = [
        (field.arrays["bearing_capacity_n"].T, "Truth: bearing capacity (N)", "viridis"),
        (belief.mean["bearing_capacity_n"].T, "Controller belief: bearing (N)", "viridis"),
        (np.sqrt(belief.variance["bearing_capacity_n"]).T, "Belief uncertainty (N)", "magma"),
    ]
    for axis, (data, title, cmap) in zip(axes, images, strict=True):
        image = axis.imshow(data, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        axis.set_title(title)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        fig.colorbar(image, ax=axis, shrink=0.78)
    for event in replay["events"]:
        if event["phase"] == "initial_contact":
            xy = event["position_xy_m"]
            axes[1].plot(xy[0], xy[1], "wo", markeredgecolor="black")
        elif event["phase"] == "step":
            xy = event["plan"]["target_xy_m"]
            color = {"COMMIT": "lime", "REPLANT": "red", "HOLD_DOUBLE_SUPPORT": "orange"}[
                event["decision"]["action"]
            ]
            axes[1].plot(xy[0], xy[1], "o", color=color, markeredgecolor="black")
    fig.suptitle("Hackathon Everest — spatial memory and bilateral foothold assurance")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def render_html_report(
    *,
    output: Path,
    metrics: dict[str, Any],
    benchmark: dict[str, Any],
    replay: dict[str, Any],
    image_path: Path,
) -> None:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    rows = []
    for event in replay["events"]:
        if event["phase"] != "step":
            continue
        decision = event["decision"]
        rows.append(
            f"<tr><td>{event['step']}</td><td>{event['stance_foot']}</td>"
            f"<td>{decision['action']}</td><td>{decision['load_transfer_rate_nps']:.1f}</td>"
            f"<td>{decision['reason']}</td></tr>"
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Hackathon Everest Report</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#15202b}}
.hero{{background:#edf7ff;padding:1rem 1.4rem;border-radius:12px}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:.55rem;border-bottom:1px solid #ddd;text-align:left}}code,pre{{background:#f4f4f4;padding:.2rem}}
img{{width:100%;height:auto}}.limit{{color:#8a4b00}}</style></head><body>
<h1>Hackathon Everest</h1><div class="hero"><strong>Claim:</strong> The crampon estimates remaining support,
shares local terrain evidence between feet, and delays or redirects conservative load transfers.</div>
<p class="limit"><strong>Limit:</strong> This is a reduced-order scripted support scheduler. It is not a
validated snow model, autonomous G1 locomotion, or evidence of Everest readiness.</p>
<img alt="terrain belief visualization" src="data:image/png;base64,{encoded}">
<h2>Cross-foot spatial evidence</h2><pre>{json.dumps(replay["cross_foot_evidence"], indent=2, default=_json_default)}</pre>
<h2>Replay</h2><table><thead><tr><th>Step</th><th>Stance foot</th><th>Action</th><th>Transfer N/s</th><th>Reason</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Held-out estimator metrics</h2><pre>{json.dumps(metrics, indent=2)}</pre>
<h2>Scheduler ablation</h2><pre>{json.dumps(benchmark, indent=2)}</pre>
</body></html>"""
    output.write_text(html)


def _git_output(*args: str) -> str | None:
    project_root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_revision() -> str | None:
    return _git_output("rev-parse", "HEAD")


def git_is_dirty() -> bool | None:
    status = _git_output("status", "--porcelain")
    return None if status is None else bool(status)


def dependency_versions() -> dict[str, str]:
    names = ("numpy", "scipy", "scikit-learn", "joblib", "matplotlib", "PyYAML")
    return {name: importlib.metadata.version(name) for name in names}


def run_pipeline(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 7))
    episodes = int(config.get("episodes", 800))
    fields = int(config.get("fields", max(20, episodes // 8)))
    n_estimators = int(config.get("n_estimators", 128))
    replay_steps = int(config.get("replay_steps", 6))
    benchmark_fields = int(config.get("benchmark_fields", 8))

    dataset = generate_probe_dataset(episodes=episodes, fields=fields, seed=seed)
    dataset_path = dataset.save(out / "probe_dataset.npz")
    training = train_estimator(dataset, n_estimators=n_estimators, seed=seed)
    model_path = training.estimator.save(out / "terrain_estimator.joblib")
    metrics_path = out / "metrics.json"
    write_json(metrics_path, training.metrics)

    replay_seed = int(config.get("replay_terrain_seed", seed * 10_000 + 70_001))
    replay, field, belief = run_bilateral_replay(
        training.estimator, terrain_seed=replay_seed, steps=replay_steps, mode="full"
    )
    replay_path = out / "replay.json"
    write_json(replay_path, replay)
    benchmark = benchmark_modes(
        training.estimator,
        field_count=benchmark_fields,
        steps=replay_steps,
        seed=seed,
    )
    benchmark_path = out / "benchmark.json"
    write_json(benchmark_path, benchmark)
    image_path = out / "terrain_belief.png"
    render_replay(replay, field, belief, image_path)
    report_path = out / "report.html"
    render_html_report(
        output=report_path,
        metrics=training.metrics,
        benchmark=benchmark,
        replay=replay,
        image_path=image_path,
    )

    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    train_field_ids = sorted(np.unique(dataset.field_ids[training.train_indices]).astype(int).tolist())
    test_field_ids = sorted(np.unique(dataset.field_ids[training.test_indices]).astype(int).tolist())
    manifest = {
        "schema_version": "0.1.0",
        "config": config,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "seed": seed,
        "episodes": episodes,
        "fields": fields,
        "rows": len(dataset.features),
        "sensor_channels": 19,
        "sample_rate_hz": 100,
        "probe_duration_s": 0.30,
        "split_unit": "terrain_field_seed",
        "feature_names": dataset.feature_names,
        "target_names": dataset.target_names,
        "event_names": dataset.event_names,
        "git_revision_at_run": git_revision(),
        "git_dirty_at_run": git_is_dirty(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        },
        "split_field_ids": {
            "train": train_field_ids,
            "test": test_field_ids,
        },
        "artifacts": {
            "dataset": str(dataset_path),
            "model": str(model_path),
            "metrics": str(metrics_path),
            "replay": str(replay_path),
            "benchmark": str(benchmark_path),
            "visualization": str(image_path),
            "report": str(report_path),
        },
        "limitations": [
            "Synthetic priors have not yet been calibrated to SnowMicroPen or radar field data.",
            "Ensemble spread is approximate uncertainty, not a certified probability.",
            "Bilateral replay is scripted support reasoning, not whole-body locomotion.",
            "Penetration assumes a moving spike/probe or floating collar; a fixed spike cannot measure it directly.",
        ],
    }
    write_json(out / "manifest.json", manifest)
    return manifest
