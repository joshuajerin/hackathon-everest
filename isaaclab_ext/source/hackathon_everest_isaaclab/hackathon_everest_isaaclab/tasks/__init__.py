from __future__ import annotations

import sys


def register_cli() -> list[str]:
    """Register Everest tasks and preserve all CLI tokens for Isaac Lab's preset parser."""
    from .manager_based import crampon_velocity  # noqa: F401

    return sys.argv[1:]
