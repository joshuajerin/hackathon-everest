#!/usr/bin/env python3
"""Run same-simulation crampon policy comparison render jobs serially."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--jobs", type=Path, required=True)
parser.add_argument(
    "--runner", type=Path, default=Path(__file__).with_name("record_policy_crampon_comparison.py")
)
parser.add_argument("--launcher", type=Path, required=True)
parser.add_argument(
    "--launcher-arg",
    action="append",
    default=[],
    help="Argument inserted after the launcher; repeat as needed (for Isaac Lab use --launcher-arg=-p).",
)
parser.add_argument("--crampon-policy", type=Path, required=True)
parser.add_argument("--baseline-policy", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
args = parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def build_command(job: dict, output_dir: Path) -> list[str]:
    command = [
        str(args.launcher),
        *args.launcher_arg,
        str(args.runner),
        "--crampon-policy",
        str(args.crampon_policy),
        "--baseline-policy",
        str(args.baseline_policy),
        "--output-dir",
        str(output_dir),
        "--surface",
        str(job["surface"]),
        "--incline-deg",
        str(job["incline_deg"]),
        "--scene-seed",
        str(job["scene_seed"]),
        "--steps",
        str(job["steps"]),
        "--warmup-steps",
        str(job.get("warmup_steps", 0)),
        "--requested-vx",
        str(job.get("requested_vx", 0.15)),
        "--baseline-grip-scale",
        str(job.get("baseline_grip_scale", 0.04)),
        "--lane-spacing",
        str(job.get("lane_spacing", 2.4)),
        "--camera-eye",
        *(str(value) for value in job["camera_eye"]),
        "--camera-lookat",
        *(str(value) for value in job["camera_lookat"]),
        "--headless",
    ]
    return command


def main() -> int:
    document = json.loads(args.jobs.read_text())
    jobs = document["jobs"]
    if not jobs:
        raise ValueError("jobs cannot be empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "queue_status.json"
    status: dict[str, object] = {
        "schema_version": "1.0.0",
        "artifact_type": "native_isaac_same_sim_policy_comparison_queue",
        "started_at": utc_now(),
        "total_jobs": len(jobs),
        "completed": [],
        "failed": [],
        "claim_boundary": document.get("claim_boundary"),
    }
    for job in jobs:
        shot_dir = args.output_root / (
            f"shot_{int(job['id']):03d}_{job['surface']}_{job['incline_deg']:g}deg_{job['camera']}"
        )
        command = build_command(job, shot_dir)
        started_at = utc_now()
        result = subprocess.run(command, check=False)
        videos = sorted(shot_dir.glob("*.mp4"))
        manifest = shot_dir / "comparison_manifest.json"
        artifact_error = None
        if len(videos) != 1:
            artifact_error = f"expected exactly one MP4, found {len(videos)}"
        elif not manifest.is_file():
            artifact_error = "comparison_manifest.json is missing"
        effective_returncode = result.returncode or (86 if artifact_error else 0)
        entry = {
            "job": job,
            "command": command,
            "output_dir": str(shot_dir),
            "started_at": started_at,
            "finished_at": utc_now(),
            "returncode": effective_returncode,
            "runner_returncode": result.returncode,
            "artifact_error": artifact_error,
            "video_paths": [str(path) for path in videos],
        }
        key = "failed" if effective_returncode else "completed"
        status[key].append(entry)
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        if effective_returncode:
            return effective_returncode
    status["finished_at"] = utc_now()
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
