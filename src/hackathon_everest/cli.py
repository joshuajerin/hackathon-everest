from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import ProbeDataset, generate_probe_dataset
from .estimation import TerrainStateEstimator, train_estimator
from .models import SENSOR_CHANNELS
from .pipeline import load_config, run_bilateral_replay, run_pipeline, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="everest",
        description="Hackathon Everest reduced-order foothold-assurance pipeline",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    pipeline = commands.add_parser("pipeline", help="generate, train, evaluate, and replay")
    pipeline.add_argument("--config", default="configs/smoke.yaml")
    pipeline.add_argument("--out", default="artifacts/smoke")

    generate = commands.add_parser("generate", help="generate a field-split probe dataset")
    generate.add_argument("--episodes", type=int, default=800)
    generate.add_argument("--fields", type=int, default=None)
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--out", default="artifacts/probe_dataset.npz")

    train = commands.add_parser("train", help="train the continuous terrain-state estimator")
    train.add_argument("--dataset", default="artifacts/probe_dataset.npz")
    train.add_argument("--model", default="artifacts/terrain_estimator.joblib")
    train.add_argument("--metrics", default="artifacts/metrics.json")
    train.add_argument("--estimators", type=int, default=128)
    train.add_argument("--seed", type=int, default=7)

    replay = commands.add_parser("replay", help="run an unseen scripted bilateral stepping replay")
    replay.add_argument("--model", default="artifacts/terrain_estimator.joblib")
    replay.add_argument("--terrain-seed", type=int, default=70001)
    replay.add_argument("--steps", type=int, default=6)
    replay.add_argument("--mode", choices=["full", "current_only"], default="full")
    replay.add_argument("--out", default="artifacts/replay.json")

    probe = commands.add_parser("mujoco-probe", help="run the single-foot MuJoCo sensor fixture")
    probe.add_argument("--model", default="mujoco/crampon_probe.xml")
    probe.add_argument("--duration", type=float, default=1.0)
    probe.add_argument("--ramp", type=float, default=0.30)
    probe.add_argument("--load", type=float, default=150.0)
    probe.add_argument("--slope-deg", type=float, default=0.0)
    probe.add_argument("--lateral-drive-force", type=float, default=0.0)
    probe.add_argument("--out", default="artifacts/mujoco_probe")

    ice_probe = commands.add_parser("mujoco-ice-probe", help="run the hybrid stateful ice probe")
    ice_probe.add_argument("--model", default="mujoco/crampon_probe.xml")
    ice_probe.add_argument("--duration", type=float, default=1.5)
    ice_probe.add_argument("--ramp", type=float, default=0.40)
    ice_probe.add_argument("--load", type=float, default=150.0)
    ice_probe.add_argument("--slope-deg", type=float, default=0.0)
    ice_probe.add_argument("--lateral-drive-force", type=float, default=0.0)
    ice_probe.add_argument("--seed", type=int, default=41)
    ice_probe.add_argument("--out", default="artifacts/mujoco_ice_probe")

    commands.add_parser("contract", help="print the hardware-shaped crampon channel contract")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pipeline":
        manifest = run_pipeline(load_config(args.config), args.out)
        print(json.dumps(manifest, indent=2))
    elif args.command == "generate":
        dataset = generate_probe_dataset(
            episodes=args.episodes,
            fields=args.fields,
            seed=args.seed,
        )
        path = dataset.save(args.out)
        print(f"wrote {len(dataset.features)} prefix rows to {path}")
    elif args.command == "train":
        result = train_estimator(
            ProbeDataset.load(args.dataset),
            n_estimators=args.estimators,
            seed=args.seed,
        )
        result.estimator.save(args.model)
        write_json(Path(args.metrics), result.metrics)
        print(json.dumps(result.metrics, indent=2))
    elif args.command == "replay":
        estimator = TerrainStateEstimator.load(args.model)
        replay, _, _ = run_bilateral_replay(
            estimator,
            terrain_seed=args.terrain_seed,
            steps=args.steps,
            mode=args.mode,
        )
        write_json(Path(args.out), replay)
        print(json.dumps(replay["summary"], indent=2))
    elif args.command == "mujoco-probe":
        from .mujoco_probe import run_mujoco_probe, save_mujoco_probe

        run = run_mujoco_probe(
            args.model,
            duration_s=args.duration,
            ramp_s=args.ramp,
            target_load_n=args.load,
            slope_deg=args.slope_deg,
            lateral_drive_force_n=args.lateral_drive_force,
        )
        report = save_mujoco_probe(run, args.out)
        print(json.dumps(report, indent=2))
    elif args.command == "mujoco-ice-probe":
        from .hybrid_ice_probe import run_hybrid_ice_probe, save_hybrid_ice_probe

        run = run_hybrid_ice_probe(
            args.model,
            duration_s=args.duration,
            ramp_s=args.ramp,
            target_load_n=args.load,
            seed=args.seed,
            slope_deg=args.slope_deg,
            lateral_drive_force_n=args.lateral_drive_force,
        )
        report = save_hybrid_ice_probe(run, args.out)
        print(json.dumps(report, indent=2))
    elif args.command == "contract":
        print(
            json.dumps(
                {
                    "total": SENSOR_CHANNELS,
                    "axial_force": 4,
                    "penetration": 4,
                    "accelerometer": 3,
                    "gyroscope": 3,
                    "radar_frontend": 5,
                    "note": "G1 proprioception is separate context, not extra crampon channels.",
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
