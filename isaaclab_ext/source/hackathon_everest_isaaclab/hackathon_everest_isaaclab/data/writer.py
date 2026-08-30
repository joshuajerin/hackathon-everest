from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .schema import (
    DEPLOYABLE_COMMAND_ALLOWLIST,
    DEPLOYABLE_CONTEXT_ALLOWLIST,
    FOOT_COUNT,
    SENSOR_CHANNELS,
)

_VISIBLE_ARRAY_KEYS = frozenset({"packet_values", "valid_mask", "timestamp_s", "sample_age_s"})
_FORBIDDEN_VISIBLE_TOKENS = frozenset(
    {
        "truth",
        "oracle",
        "canary",
        "material",
        "contact_force",
        "future",
        "fracture_strength",
        "bearing_capacity",
    }
)


def stable_group_hash(parts: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(sorted(parts.items())), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_visible(visible: Mapping[str, Any]) -> int:
    allowed = _VISIBLE_ARRAY_KEYS | {"context", "commands"}
    if set(visible) != allowed:
        raise ValueError(f"Visible plane keys must be exactly {sorted(allowed)}")
    lowered = {str(key).lower() for key in visible}
    if any(token in key for key in lowered for token in _FORBIDDEN_VISIBLE_TOKENS):
        raise ValueError("Truth-like field name in visible plane")
    context = visible["context"]
    commands = visible["commands"]
    if set(context).difference(DEPLOYABLE_CONTEXT_ALLOWLIST):
        raise ValueError("Non-deployable visible context field")
    if set(commands).difference(DEPLOYABLE_COMMAND_ALLOWLIST):
        raise ValueError("Non-deployable visible command field")
    values = np.asarray(visible["packet_values"])
    if values.ndim != 4 or values.shape[-2:] != (FOOT_COUNT, SENSOR_CHANNELS):
        raise ValueError("packet_values must have shape [episode, time, 2, 19]")
    if np.asarray(visible["valid_mask"]).shape != values.shape:
        raise ValueError("valid_mask shape mismatch")
    if np.asarray(visible["valid_mask"]).dtype != np.dtype(bool):
        raise TypeError("valid_mask must be boolean")
    if np.asarray(visible["sample_age_s"]).shape != values.shape:
        raise ValueError("sample_age_s shape mismatch")
    timestamps = np.asarray(visible["timestamp_s"])
    if timestamps.shape != values.shape[:-1]:
        raise ValueError("timestamp_s shape mismatch")
    if not np.all(np.diff(timestamps, axis=1) > 0.0):
        raise ValueError("Packet timestamps must be strictly monotonic")
    for group_name, group in (("context", context), ("commands", commands)):
        for name, value in group.items():
            if np.asarray(value).shape[0] != values.shape[0]:
                raise ValueError(f"{group_name}/{name} episode dimension mismatch")
    return int(values.shape[0])


def _write_group(path: Path, values: Mapping[str, Any]) -> None:
    import zarr

    root = zarr.open_group(str(path), mode="w")

    def write(group, mapping: Mapping[str, Any]) -> None:
        for name, value in mapping.items():
            if isinstance(value, Mapping):
                write(group.create_group(name), value)
                continue
            array = np.asarray(value)
            chunks = None if array.ndim == 0 else (min(256, array.shape[0]), *array.shape[1:])
            group.create_array(name, data=array, chunks=chunks, overwrite=False)

    write(root, values)


def _tree_checksums(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"SHA256SUMS", "_COMPLETE"}
    }


def write_immutable_shard(
    dataset_root: str | Path,
    *,
    dataset_id: str,
    shard_id: str,
    visible: Mapping[str, Any],
    truth: Mapping[str, Any],
    episode_rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    episode_count = _validate_visible(visible)
    if len(episode_rows) != episode_count:
        raise ValueError("Episode metadata count mismatch")
    if not truth:
        raise ValueError("Truth plane cannot be empty")
    for name, value in truth.items():
        if isinstance(value, Mapping):
            continue
        if np.asarray(value).shape[0] != episode_count:
            raise ValueError(f"Truth array {name} episode dimension mismatch")
    destination = Path(dataset_root) / dataset_id / "shards" / shard_id
    if (destination / "_COMPLETE").exists():
        raise FileExistsError(f"Immutable shard already complete: {destination}")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        _write_group(temporary / "visible.zarr", visible)
        _write_group(temporary / "truth.zarr", truth)
        table = pa.Table.from_pylist([dict(row) for row in episode_rows])
        pq.write_table(table, temporary / "episodes.parquet", compression="zstd")
        manifest = {
            "schema_version": "1.0.0",
            "dataset_id": dataset_id,
            "shard_id": shard_id,
            "episode_count": episode_count,
            "sensor_shape": list(np.asarray(visible["packet_values"]).shape),
            "visible_keys": sorted(_VISIBLE_ARRAY_KEYS),
            "visible_context_keys": sorted(visible["context"]),
            "visible_command_keys": sorted(visible["commands"]),
            "truth_keys": sorted(truth),
            "provenance": dict(provenance),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        checksums = _tree_checksums(temporary)
        (temporary / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in checksums.items())
        )
        (temporary / "_COMPLETE").write_text("complete\n")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
