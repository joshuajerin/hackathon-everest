#!/usr/bin/env python3
"""Export and configure a contact-gated bounded-residual Isaac run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
EXPORT_SCRIPT = REPO_ROOT / "isaaclab_ext/scripts/export_bounded_residual_policy.py"
RUN_SCRIPT = REPO_ROOT / "isaaclab_ext/scripts/run_stateful_policy.py"
DEFAULT_VISIBLE_CHECKPOINT = (
    REPO_ROOT / "artifacts/lambda_prime/visible_policy_l0_v2/visible_policy.pt"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def runtime_command(
    *,
    runtime_python: str,
    stock_policy: Path,
    visible_checkpoint: Path,
    correction_policy: Path,
    output: Path,
    steps: int,
) -> list[str]:
    """Build the active-run command without executing an Isaac simulation."""

    return [
        runtime_python,
        str(RUN_SCRIPT),
        "--mode",
        "active",
        "--stock-policy",
        str(stock_policy),
        "--visible-checkpoint",
        str(visible_checkpoint),
        "--contact-correction-policy",
        str(correction_policy),
        "--output",
        str(output),
        "--steps",
        str(steps),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a trained bounded residual and prepare its active contact-correction run"
    )
    parser.add_argument("--residual-checkpoint", required=True, type=existing_file)
    parser.add_argument("--stock-rsl-checkpoint", required=True, type=existing_file)
    parser.add_argument("--stock-policy", required=True, type=existing_file)
    parser.add_argument(
        "--visible-checkpoint", type=existing_file, default=DEFAULT_VISIBLE_CHECKPOINT
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "artifacts/contact_correction"
    )
    parser.add_argument(
        "--runtime-python",
        default=sys.executable,
        help="Isaac Lab Python executable used for export and the generated active-run launcher.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if shutil.which(args.runtime_python) is None and not Path(args.runtime_python).is_file():
        raise FileNotFoundError(f"runtime Python executable not found: {args.runtime_python}")

    output_dir = args.output_dir.expanduser().resolve()
    policy_dir = output_dir / "policy"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_command = [
        args.runtime_python,
        str(EXPORT_SCRIPT),
        "--checkpoint",
        str(args.residual_checkpoint),
        "--stock-checkpoint",
        str(args.stock_rsl_checkpoint),
        "--output-dir",
        str(policy_dir),
    ]
    subprocess.run(export_command, check=True)
    correction_policy = policy_dir / "policy.pt"
    policy_manifest = policy_dir / "manifest.json"
    if not correction_policy.is_file() or not policy_manifest.is_file():
        raise RuntimeError("residual export did not produce policy.pt and manifest.json")

    run_output = output_dir / "active_run.json"
    command = runtime_command(
        runtime_python=args.runtime_python,
        stock_policy=args.stock_policy,
        visible_checkpoint=args.visible_checkpoint,
        correction_policy=correction_policy,
        output=run_output,
        steps=args.steps,
    )
    launcher = output_dir / "run_active_contact_correction.sh"
    launcher.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n")
    launcher.chmod(0o755)
    setup_manifest = {
        "schema_version": "1.0.0",
        "residual_checkpoint": str(args.residual_checkpoint),
        "residual_checkpoint_sha256": sha256(args.residual_checkpoint),
        "stock_rsl_checkpoint": str(args.stock_rsl_checkpoint),
        "stock_rsl_checkpoint_sha256": sha256(args.stock_rsl_checkpoint),
        "stock_policy": str(args.stock_policy),
        "stock_policy_sha256": sha256(args.stock_policy),
        "visible_checkpoint": str(args.visible_checkpoint),
        "visible_checkpoint_sha256": sha256(args.visible_checkpoint),
        "contact_correction_policy": str(correction_policy),
        "contact_correction_policy_sha256": sha256(correction_policy),
        "policy_export_manifest": str(policy_manifest),
        "runtime_python": args.runtime_python,
        "active_run_command": command,
        "active_run_output": str(run_output),
    }
    (output_dir / "setup_manifest.json").write_text(json.dumps(setup_manifest, indent=2) + "\n")
    print(f"Prepared {launcher}")
    print("Run it on the Isaac Lab machine after reviewing the generated command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
