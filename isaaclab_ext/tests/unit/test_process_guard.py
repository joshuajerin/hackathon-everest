from __future__ import annotations

import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[2] / "source/hackathon_everest_isaaclab"
sys.path.insert(0, str(SOURCE))

from hackathon_everest_isaaclab.runtime.process_guard import acquire_isaac_process_lock


def test_process_lock_rejects_overlap_and_releases_on_close(tmp_path: Path) -> None:
    path = tmp_path / "isaac.lock"
    first = acquire_isaac_process_lock(path)
    with pytest.raises(RuntimeError, match="Another Everest Isaac application"):
        acquire_isaac_process_lock(path)
    first.close()
    replacement = acquire_isaac_process_lock(path)
    assert "pid=" in path.read_text()
    replacement.close()
