from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
JOBS_PATH = REPO_ROOT / "configs/isaaclab/policy_crampon_comparison_render_jobs.json"
QUEUE_PATH = REPO_ROOT / "isaaclab_ext/scripts/run_policy_comparison_render_queue.py"


def test_comparison_jobs_cover_steep_distinct_ice_conditions() -> None:
    document = json.loads(JOBS_PATH.read_text())
    jobs = document["jobs"]

    assert document["schema_version"] == "1.0.0"
    assert len(jobs) == 2
    assert {job["surface"] for job in jobs} == {
        "fractured_blue_ice",
        "polished_wind_ice",
    }
    assert min(job["incline_deg"] for job in jobs) >= 25
    assert all(job["steps"] == 750 for job in jobs)
    assert all(job["lane_spacing"] <= 2.4 for job in jobs)
    assert all(job["baseline_grip_scale"] == 0.01 for job in jobs)
    assert all(job["warmup_steps"] == 0 for job in jobs)
    assert len({job["scene_seed"] for job in jobs}) == len(jobs)


def test_comparison_queue_help_is_available_without_isaac() -> None:
    result = subprocess.run(
        [sys.executable, str(QUEUE_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--crampon-policy" in result.stdout
    assert "--baseline-policy" in result.stdout
    assert "--launcher-arg" in result.stdout
