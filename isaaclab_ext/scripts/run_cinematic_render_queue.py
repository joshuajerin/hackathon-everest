#!/usr/bin/env python3
"""Run native Isaac cinematic render jobs serially and checkpoint each result."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--jobs", type=Path, required=True)
parser.add_argument("--runner", type=Path, required=True)
parser.add_argument("--launcher", type=Path, required=True)
parser.add_argument("--policy", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
args = parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def main() -> int:
    jobs = json.loads(args.jobs.read_text())["jobs"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "queue_status.json"
    status: dict[str, object] = {
        "schema_version": "1.0.0",
        "artifact_type": "native_isaac_cinematic_render_queue",
        "started_at": utc_now(),
        "total_jobs": len(jobs),
        "completed": [],
        "failed": [],
        "claim_boundary": "Project-authored visual-only alpine sets. Analytical stateful contact remains force authority.",
    }
    for job in jobs:
        shot_dir = args.output_root / f"shot_{int(job['id']):03d}_{job['surface']}_{job['camera']}"
        command = [
            str(args.launcher),
            str(args.runner),
            "--policy",
            str(args.policy),
            "--output-dir",
            str(shot_dir),
            "--surfaces",
            str(job["surface"]),
            "--incline-deg",
            str(job["incline_deg"]),
            "--contact-mode",
            str(job.get("contact_mode", "all_points_flat_foot")),
            "--scene-seed",
            str(job["scene_seed"]),
            "--steps-per-surface",
            str(job["steps"]),
            "--warmup-steps",
            str(job.get("warmup_steps", 100)),
            "--requested-vx",
            str(job.get("requested_vx", 0.15)),
            "--camera-eye",
            *(str(value) for value in job["camera_eye"]),
            "--camera-lookat",
            *(str(value) for value in job["camera_lookat"]),
            "--headless",
        ]
        started_at = utc_now()
        result = subprocess.run(command, check=False)
        videos = sorted(shot_dir.rglob("*.mp4"))
        report_path = shot_dir / "sensor_world.json"
        artifact_error = None
        locomotion = None
        if len(videos) != 1:
            artifact_error = f"expected exactly one MP4, found {len(videos)}"
        elif not report_path.is_file():
            artifact_error = "sensor_world.json is missing"
        else:
            try:
                report = json.loads(report_path.read_text())
                surfaces = report["surfaces"]
                if len(surfaces) != 1:
                    raise ValueError(f"expected one surface report, found {len(surfaces)}")
                surface = surfaces[0]
                locomotion = surface["locomotion"]
                progress_m = float(locomotion["forward_progress_m"])
                climb_m = float(locomotion["vertical_climb_m"])
                tracking_error_m = float(locomotion["climb_tracking_error_m"])
                normal = [float(value) for value in surface["terrain_normal_world"]]
                angle = math.radians(float(job["incline_deg"]))
                expected_normal = [-math.sin(angle), 0.0, math.cos(angle)]
                normal_error = max(
                    abs(actual - expected)
                    for actual, expected in zip(normal, expected_normal, strict=True)
                )
                if int(surface["terminations"]) != 0:
                    raise ValueError(f"terminations={surface['terminations']}")
                if not all(
                    math.isfinite(value) for value in (progress_m, climb_m, tracking_error_m)
                ):
                    raise ValueError("non-finite locomotion metric")
                if progress_m <= 0.20:
                    raise ValueError(f"insufficient uphill progress: {progress_m:.6f} m")
                expected_climb_m = float(locomotion["expected_climb_from_progress_m"])
                minimum_climb_m = max(0.01, 0.50 * expected_climb_m)
                if climb_m <= minimum_climb_m:
                    raise ValueError(
                        f"insufficient vertical climb: {climb_m:.6f} m; "
                        f"required {minimum_climb_m:.6f} m"
                    )
                if abs(tracking_error_m) > 0.20:
                    raise ValueError(
                        f"climb tracking error {tracking_error_m:.6f} m exceeds 0.20 m"
                    )
                if normal_error > 1.0e-5:
                    raise ValueError(f"terrain normal mismatch: {normal_error:.8f}")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                artifact_error = f"locomotion validation failed: {error}"
        effective_returncode = result.returncode or (86 if artifact_error else 0)
        entry = {
            "job": job,
            "output_dir": str(shot_dir),
            "started_at": started_at,
            "finished_at": utc_now(),
            "returncode": effective_returncode,
            "runner_returncode": result.returncode,
            "artifact_error": artifact_error,
            "video_paths": [str(path) for path in videos],
            "locomotion": locomotion,
        }
        if effective_returncode:
            status["failed"].append(entry)
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            return effective_returncode
        status["completed"].append(entry)
        status_path.write_text(json.dumps(status, indent=2) + "\n")
    status["finished_at"] = utc_now()
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
