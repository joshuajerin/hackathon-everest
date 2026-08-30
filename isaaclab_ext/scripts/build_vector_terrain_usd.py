#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from hackathon_everest.everest_suite import assign_cases, load_suite


def grid_origins(count: int, spacing: float) -> list[tuple[float, float, float]]:
    rows = math.ceil(count / math.sqrt(count))
    columns = math.ceil(count / rows)
    result = []
    for index in range(count):
        row, column = divmod(index, columns)
        x = -(row - (rows - 1) / 2) * spacing
        y = (column - (columns - 1) / 2) * spacing
        result.append((x, y, 0.0))
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_rectangle(points, counts, indices, *, origin, slope, x0, x1, half_width, anchor_x):
    base = len(points)
    ox, oy, oz = origin
    points.extend(
        (
            (ox + x0, oy - half_width, oz + slope * (x0 - anchor_x)),
            (ox + x1, oy - half_width, oz + slope * (x1 - anchor_x)),
            (ox + x1, oy + half_width, oz + slope * (x1 - anchor_x)),
            (ox + x0, oy + half_width, oz + slope * (x0 - anchor_x)),
        )
    )
    counts.append(4)
    indices.extend((base, base + 1, base + 2, base + 3))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--spacing", type=float, default=16.0)
    parser.add_argument("--patch-size", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--anchor-x", type=float, default=-4.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.num_envs < 1 or args.spacing <= args.patch_size or args.patch_size <= 0:
        raise ValueError("Require num-envs > 0 and spacing > patch-size > 0")
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    manifest = args.manifest or output.with_suffix(".manifest.json")
    if not manifest.is_absolute():
        manifest = root / manifest
    suite = load_suite(root / "configs/isaaclab/everest_terrain_suite.yaml")
    required = (
        len(suite["surfaces"])
        * len(suite["inclines_deg"])
        * len(suite["hazards"])
        * len(suite["contact_modes"])
    )
    cases = assign_cases(suite, num_envs=max(args.num_envs, required), seed=args.seed)[
        : args.num_envs
    ]
    origins = grid_origins(args.num_envs, args.spacing)
    geometry = defaultdict(lambda: ([], [], []))
    half = 0.5 * args.patch_size
    for case, origin in zip(cases, origins, strict=True):
        points, counts, indices = geometry[case.surface_id]
        slope = math.tan(math.radians(case.incline_deg))
        if case.hazard_id == "open_crevasse_gap":
            add_rectangle(
                points,
                counts,
                indices,
                origin=origin,
                slope=slope,
                x0=-half,
                x1=-0.65,
                half_width=half,
                anchor_x=args.anchor_x,
            )
            add_rectangle(
                points,
                counts,
                indices,
                origin=origin,
                slope=slope,
                x0=0.65,
                x1=half,
                half_width=half,
                anchor_x=args.anchor_x,
            )
        else:
            add_rectangle(
                points,
                counts,
                indices,
                origin=origin,
                slope=slope,
                x0=-half,
                x1=half,
                half_width=half,
                anchor_x=args.anchor_x,
            )
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.usdc")
    stage = Usd.Stage.CreateNew(str(temporary))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root_prim = UsdGeom.Xform.Define(stage, "/EverestTerrain")
    stage.SetDefaultPrim(root_prim.GetPrim())
    for surface_id, (points, counts, indices) in geometry.items():
        mesh = UsdGeom.Mesh.Define(stage, f"/EverestTerrain/{surface_id}")
        mesh.GetPointsAttr().Set([Gf.Vec3f(*point) for point in points])
        mesh.GetFaceVertexCountsAttr().Set(counts)
        mesh.GetFaceVertexIndicesAttr().Set(indices)
        mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.GetDoubleSidedAttr().Set(True)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).GetCollisionEnabledAttr().Set(True)
        mesh.GetPrim().CreateAttribute("everest:surfaceId", Sdf.ValueTypeNames.String).Set(
            surface_id
        )
    stage.GetRootLayer().Save()
    os.replace(temporary, output)
    report = {
        "schema_version": "1.0.0",
        "output": {
            "path": str(output),
            "sha256": sha256(output),
            "size_bytes": output.stat().st_size,
        },
        "num_environments": args.num_envs,
        "required_cartesian_cases": required,
        "spacing_m": args.spacing,
        "patch_size_m": args.patch_size,
        "terrain_anchor_x_m": args.anchor_x,
        "virtual_footprint_m": [
            max(value[0] for value in origins)
            - min(value[0] for value in origins)
            + args.patch_size,
            max(value[1] for value in origins)
            - min(value[1] for value in origins)
            + args.patch_size,
        ],
        "coverage": {
            "surface": dict(Counter(case.surface_id for case in cases)),
            "incline_deg": dict(Counter(str(case.incline_deg) for case in cases)),
            "hazard": dict(Counter(case.hazard_id for case in cases)),
            "contact_mode": dict(Counter(case.contact_mode_id for case in cases)),
        },
        "claim_boundary": "Analytical stress-test collider geometry. Route names and hazards are not surveyed Everest measurements.",
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
