#!/usr/bin/env python3
"""Build a provenance-preserving large USD world for parallel Everest evaluation.

The world is a direct visual representation of the deterministic stateful-suite
assignment. Analytical crampon contact remains authoritative; terrain meshes
supply placement, slopes, and scene navigation but never native support forces.
Flat colors are native USD preview materials, not generated textures or
material-calibration claims.
"""

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

SURFACE_COLORS = {
    "base_camp_patchy_snow": (0.74, 0.79, 0.84),
    "western_cwm_consolidated_snow": (0.88, 0.92, 0.96),
    "lhotse_boot_packed_snow": (0.63, 0.69, 0.75),
    "south_col_wind_pack": (0.82, 0.86, 0.91),
    "summit_ridge_drift": (0.96, 0.97, 1.00),
    "hard_glacier_ice": (0.24, 0.43, 0.63),
    "fractured_blue_ice": (0.13, 0.34, 0.56),
    "polished_wind_ice": (0.34, 0.53, 0.70),
    "thin_snow_over_ice": (0.71, 0.80, 0.89),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_origins(count: int, spacing: float) -> list[tuple[float, float, float]]:
    rows = math.ceil(count / math.sqrt(count))
    columns = math.ceil(count / rows)
    return [
        (
            -(row - (rows - 1) / 2) * spacing,
            (column - (columns - 1) / 2) * spacing,
            0.0,
        )
        for row, column in (divmod(index, columns) for index in range(count))
    ]


def add_patch(
    points: list[tuple[float, float, float]],
    counts: list[int],
    indices: list[int],
    *,
    origin: tuple[float, float, float],
    slope: float,
    x0: float,
    x1: float,
    half_width: float,
    anchor_x: float,
) -> None:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--num-envs", type=int, default=2160)
    parser.add_argument("--spacing", type=float, default=16.0)
    parser.add_argument("--patch-size", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--anchor-x", type=float, default=-4.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.num_envs < 1 or args.patch_size <= 0.0 or args.spacing <= args.patch_size:
        raise ValueError("Require num-envs > 0 and spacing > patch-size > 0")

    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    manifest = args.manifest or output.with_suffix(".manifest.json")
    if not manifest.is_absolute():
        manifest = root / manifest
    suite = load_suite(root / "configs/isaaclab/everest_terrain_suite.yaml")
    required_case_count = (
        len(suite["surfaces"])
        * len(suite["inclines_deg"])
        * len(suite["hazards"])
        * len(suite["contact_modes"])
    )
    cases = assign_cases(suite, num_envs=args.num_envs, seed=args.seed)
    origins = grid_origins(args.num_envs, args.spacing)
    geometry: dict[str, tuple[list, list, list]] = defaultdict(lambda: ([], [], []))
    half = 0.5 * args.patch_size
    for case, origin in zip(cases, origins, strict=True):
        points, counts, indices = geometry[case.surface_id]
        slope = math.tan(math.radians(case.incline_deg))
        if case.hazard_id == "open_crevasse_gap":
            add_patch(
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
            add_patch(
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
            add_patch(
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

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp.usdc")
    stage = Usd.Stage.CreateNew(str(temporary))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root_prim = UsdGeom.Xform.Define(stage, "/EverestParallelEvaluationWorld")
    stage.SetDefaultPrim(root_prim.GetPrim())
    materials_root = UsdGeom.Xform.Define(stage, "/EverestParallelEvaluationWorld/Materials")
    materials_root.GetPrim().SetMetadata(
        "documentation", "Flat native USD preview colors encode authored surface categories only."
    )
    UsdGeom.Xform.Define(stage, "/EverestParallelEvaluationWorld/Terrain")
    for surface_id, (points, counts, indices) in geometry.items():
        material = UsdShade.Material.Define(
            stage, f"/EverestParallelEvaluationWorld/Materials/{surface_id}"
        )
        shader = UsdShade.Shader.Define(
            stage, f"/EverestParallelEvaluationWorld/Materials/{surface_id}/PreviewSurface"
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*SURFACE_COLORS[surface_id])
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.82)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        mesh = UsdGeom.Mesh.Define(stage, f"/EverestParallelEvaluationWorld/Terrain/{surface_id}")
        mesh.GetPointsAttr().Set([Gf.Vec3f(*point) for point in points])
        mesh.GetFaceVertexCountsAttr().Set(counts)
        mesh.GetFaceVertexIndicesAttr().Set(indices)
        mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.GetDoubleSidedAttr().Set(True)
        # This mesh is a visual/provenance map only. Contact forces come solely
        # from the stateful analytical crampon model, so a USD collision schema
        # here would introduce invalid double support forces.
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        prim = mesh.GetPrim()
        prim.CreateAttribute("everest:surfaceId", Sdf.ValueTypeNames.String).Set(surface_id)
        prim.CreateAttribute("everest:visualOnlyCategoryColor", Sdf.ValueTypeNames.Bool).Set(True)
        prim.SetMetadata(
            "documentation",
            "Visual-only suite provenance mesh. Analytical stateful contact parameters remain authoritative; no USD collision is authored.",
        )
    stage.GetRootLayer().Save()
    os.replace(temporary, output)

    rows = math.ceil(args.num_envs / math.sqrt(args.num_envs))
    columns = math.ceil(args.num_envs / rows)
    report = {
        "schema_version": "1.0.0",
        "output": {
            "path": str(output),
            "sha256": sha256(output),
            "size_bytes": output.stat().st_size,
        },
        "layout": {
            "kind": "parallel_grid",
            "num_environments": args.num_envs,
            "rows": rows,
            "columns": columns,
            "spacing_m": args.spacing,
            "patch_size_m": args.patch_size,
            "terrain_anchor_x_m": args.anchor_x,
            "first_environment_origin_m": origins[0],
            "last_environment_origin_m": origins[-1],
        },
        "coverage": {
            "required_cartesian_cases": required_case_count,
            "surface": dict(Counter(case.surface_id for case in cases)),
            "incline_deg": dict(Counter(str(case.incline_deg) for case in cases)),
            "hazard": dict(Counter(case.hazard_id for case in cases)),
            "contact_mode": dict(Counter(case.contact_mode_id for case in cases)),
        },
        "areas": [
            {
                "environment_index": index,
                "origin_m": origin,
                "case_id": case.case_id,
                "surface_id": case.surface_id,
                "incline_deg": case.incline_deg,
                "hazard_id": case.hazard_id,
                "contact_mode_id": case.contact_mode_id,
            }
            for index, (case, origin) in enumerate(zip(cases, origins, strict=True))
        ],
        "claim_boundary": (
            "Native Isaac USD scene layout for parallel simulator evaluation. "
            "Surface colors are authored visual-only category aids, not material truth; stateful analytical contact remains authoritative and the terrain has no USD collision. "
            "Route names, hazards, and parameters are project-authored simulator priors, not surveyed Everest measurements."
        ),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": report["output"],
                "layout": report["layout"],
                "coverage": report["coverage"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
