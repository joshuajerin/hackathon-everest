#!/usr/bin/env python3
"""Normalize the user-supplied crampon STL into the Unitree G1 foot frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

RECORD_DTYPE = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)


def read_binary_stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path} is too small to be a binary STL")
    triangle_count = struct.unpack_from("<I", raw, 80)[0]
    expected = 84 + 50 * triangle_count
    if len(raw) != expected:
        raise ValueError(f"Expected {expected} bytes for {triangle_count} triangles, got {len(raw)}")
    records = np.frombuffer(raw, dtype=RECORD_DTYPE, count=triangle_count, offset=84)
    return records["vertices"].astype(np.float64)


def face_components(triangles: np.ndarray) -> np.ndarray:
    vertices = triangles.reshape(-1, 3)
    _, inverse = np.unique(np.round(vertices, 9), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    vertex_count = int(faces.max()) + 1
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    adjacency = coo_matrix(
        (
            np.ones(len(edges) * 2, dtype=np.uint8),
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    face_labels = labels[faces[:, 0]]
    if not np.all(labels[faces] == face_labels[:, None]):
        raise RuntimeError("A triangle spans connected-component labels")
    return face_labels


def transform_to_g1(triangles: np.ndarray, *, scale: float, x_offset_m: float) -> np.ndarray:
    result = np.empty_like(triangles)
    # Source: +Y toe, +X left/right, +Z up. G1: +X forward, +Y left, +Z up.
    result[:, :, 0] = triangles[:, :, 1] * scale + x_offset_m
    result[:, :, 1] = -triangles[:, :, 0] * scale
    result[:, :, 2] = triangles[:, :, 2] * scale
    return result


def write_binary_stl(path: Path, triangles: np.ndarray, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_a, edge_b)
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-12
    normals[valid] /= norms[valid, None]
    normals[~valid] = 0.0
    records = np.zeros(len(triangles), dtype=RECORD_DTYPE)
    records["normal"] = normals.astype(np.float32)
    records["vertices"] = triangles.astype(np.float32)
    header = f"Hackathon Everest normalized {label}".encode()[:80].ljust(80, b"\0")
    path.write_bytes(header + struct.pack("<I", len(records)) + records.tobytes())


def bounds(triangles: np.ndarray) -> dict[str, list[float]]:
    points = triangles.reshape(-1, 3)
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    return {
        "minimum_m": minimum.tolist(),
        "maximum_m": maximum.tolist(),
        "extents_m": (maximum - minimum).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=Path("assets/crampon"))
    parser.add_argument("--scale", type=float, default=108.0)
    parser.add_argument("--x-offset-m", type=float, default=0.0)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    triangles = read_binary_stl(args.source)
    labels = face_components(triangles)
    component_ids = np.unique(labels)
    if len(component_ids) != 2:
        raise ValueError(f"Expected two separated components, found {len(component_ids)}")
    transformed = transform_to_g1(triangles, scale=args.scale, x_offset_m=args.x_offset_m)

    components = []
    for component_id in component_ids:
        component = transformed[labels == component_id]
        component_bounds = bounds(component)
        mean_z = float(component[:, :, 2].mean())
        components.append((int(component_id), component, component_bounds, mean_z))
    components.sort(key=lambda item: item[3])
    named = [("crampon_frame", components[0]), ("mount_plate", components[1])]

    metadata = {
        "source_file": args.source.name,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_triangle_count": len(triangles),
        "source_bounds_unitless": bounds(triangles),
        "transform": {
            "uniform_scale": args.scale,
            "base_dimension_scale": 100.0,
            "g1_cavity_fit_factor": args.scale / 100.0,
            "rotation": "source +Y -> G1 +X; source +X -> G1 -Y; +Z unchanged",
            "x_offset_m": args.x_offset_m,
            "z_reference": "source z=0 preserved; final visual lowest point is about -0.0385 m",
        },
        "combined_bounds_m": bounds(transformed),
        "g1_fit_check": {
            "method": "XY bounding boxes at the cavity z=10 mm base-scale section; visual-fit check only",
            "official_left_ankle_roll_extents_xy_m": [0.208207, 0.075583],
            "scaled_cavity_section_extents_xy_m": [
                0.20948 * args.scale / 100.0,
                0.07576 * args.scale / 100.0,
            ],
            "estimated_clearance_per_side_xy_m": [
                (0.20948 * args.scale / 100.0 - 0.208207) / 2.0,
                (0.07576 * args.scale / 100.0 - 0.075583) / 2.0,
            ],
            "caveat": "Bounding-box clearance is not a fabrication tolerance or a collision-free CAD proof.",
        },
        "components": {},
        "note": (
            "STL has no units. Scale 100 gives plausible dimensions; a uniform 1.08 fit factor "
            "adds about 3 mm lateral clearance around the official G1 ankle-roll mesh."
        ),
    }
    for name, (component_id, component, component_bounds, _) in named:
        filename = f"{name}.stl"
        write_binary_stl(args.out / filename, component, name)
        metadata["components"][name] = {
            "source_component_id": component_id,
            "triangle_count": len(component),
            "file": filename,
            **component_bounds,
        }
    (args.out / "asset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
