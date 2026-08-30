#!/usr/bin/env python3
"""Fetch the pinned official Unitree G1 MuJoCo Menagerie asset."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie.git"
REVISION = "da76818e269b82289eba39808e2fb91d679d6994"


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=Path("vendor/mujoco_menagerie"))
    args = parser.parse_args()
    dest = args.dest.resolve()
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(dest))
    run("git", "sparse-checkout", "init", "--cone", cwd=dest)
    run("git", "sparse-checkout", "set", "unitree_g1", cwd=dest)
    run("git", "checkout", "--detach", REVISION, cwd=dest)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()
    if actual != REVISION:
        raise RuntimeError(f"Expected {REVISION}, got {actual}")
    print(dest / "unitree_g1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
