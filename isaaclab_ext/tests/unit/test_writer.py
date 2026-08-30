from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("pyarrow")
SOURCE = Path(__file__).parents[2] / "source/hackathon_everest_isaaclab"
sys.path.insert(0, str(SOURCE))

from hackathon_everest_isaaclab.data.writer import write_immutable_shard


def visible_batch(episodes: int = 3, steps: int = 4):
    times = np.arange(steps, dtype=np.float32)[None, :, None] * 0.01
    times = np.broadcast_to(times, (episodes, steps, 2)).copy()
    shape = (episodes, steps, 2, 19)
    return {
        "packet_values": np.zeros(shape, dtype=np.float32),
        "valid_mask": np.ones(shape, dtype=bool),
        "timestamp_s": times,
        "sample_age_s": np.zeros(shape, dtype=np.float32),
        "context": {"commanded_probe_load_n": np.zeros((episodes, steps, 2), dtype=np.float32)},
        "commands": {"requested_vx_mps": np.zeros((episodes, steps), dtype=np.float32)},
    }


def test_writer_separates_visible_and_truth_and_is_immutable(tmp_path: Path):
    output = write_immutable_shard(
        tmp_path,
        dataset_id="test",
        shard_id="worker-0000",
        visible=visible_batch(),
        truth={"truth_canary": np.ones((3, 8), dtype=np.float32)},
        episode_rows=[{"episode_id": index, "group_hash": str(index)} for index in range(3)],
        provenance={"seed": 1},
    )
    assert (output / "visible.zarr").is_dir()
    assert (output / "truth.zarr").is_dir()
    assert (output / "_COMPLETE").is_file()
    assert not any("canary" in str(path).lower() for path in (output / "visible.zarr").rglob("*"))
    with pytest.raises(FileExistsError):
        write_immutable_shard(
            tmp_path,
            dataset_id="test",
            shard_id="worker-0000",
            visible=visible_batch(),
            truth={"truth_canary": np.ones((3, 8), dtype=np.float32)},
            episode_rows=[{"episode_id": index} for index in range(3)],
            provenance={},
        )


def test_writer_rejects_truth_named_visible_field(tmp_path: Path):
    visible = visible_batch()
    visible["truth_canary"] = np.zeros((3, 4))
    with pytest.raises(ValueError, match="Visible plane keys"):
        write_immutable_shard(
            tmp_path,
            dataset_id="bad",
            shard_id="worker-0000",
            visible=visible,
            truth={"labels": np.zeros((3, 1))},
            episode_rows=[{"episode_id": index} for index in range(3)],
            provenance={},
        )
