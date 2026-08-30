from __future__ import annotations

import os
from pathlib import Path

from isaaclab_assets import G1_MINIMAL_CFG


def _repo_root() -> Path:
    configured = os.environ.get("EVEREST_REPO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs/isaaclab/g1_crampon_asset.yaml").is_file():
            return parent
    raise RuntimeError("Set EVEREST_REPO_ROOT to the hackathon-everest checkout")


def resolved_composed_asset_path() -> Path:
    configured = os.environ.get("EVEREST_G1_CRAMPON_USD")
    path = (
        Path(configured).expanduser().resolve()
        if configured
        else _repo_root() / "build/isaaclab/g1_crampon.usdc"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Build the authoritative composed G1 asset first: {path}")
    return path


def resolved_stateful_asset_path() -> Path:
    configured = os.environ.get("EVEREST_G1_CRAMPON_STATEFUL_USD")
    path = (
        Path(configured).expanduser().resolve()
        if configured
        else _repo_root() / "build/isaaclab/g1_crampon_stateful.usdc"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Build the stateful-contact G1 asset first: {path}")
    return path


G1_CRAMPON_CFG = G1_MINIMAL_CFG.copy()
G1_CRAMPON_CFG.spawn.usd_path = str(resolved_composed_asset_path())

G1_CRAMPON_STATEFUL_CFG = G1_MINIMAL_CFG.copy()
G1_CRAMPON_STATEFUL_CFG.spawn.usd_path = str(resolved_stateful_asset_path())
