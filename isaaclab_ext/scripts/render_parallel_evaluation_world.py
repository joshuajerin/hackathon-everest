#!/usr/bin/env python3
"""Render a native Isaac overview of the persisted parallel evaluation world."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from hackathon_everest_isaaclab.runtime import acquire_isaac_process_lock

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--world-usd", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--settle-frames", type=int, default=20)
parser.add_argument("--camera-position", nargs=3, type=float, default=(730.0, -800.0, 940.0))
parser.add_argument("--camera-lookat", nargs=3, type=float, default=(0.0, 0.0, 0.0))
args = parser.parse_args()
if args.width < 1 or args.height < 1 or args.settle_frames < 1:
    raise ValueError("width, height, and settle-frames must be positive")
world_usd = args.world_usd.expanduser().resolve()
if not world_usd.is_file():
    raise FileNotFoundError(world_usd)
output_dir = args.output_dir.expanduser().resolve()
output_dir.mkdir(parents=True, exist_ok=True)
_LOCK = acquire_isaac_process_lock()


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": args.width,
            "height": args.height,
            "renderer": "RayTracedLighting",
        }
    )
    try:
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux

        stage = omni.usd.get_context().get_stage()
        root = UsdGeom.Xform.Define(stage, "/World")
        root.GetPrim().GetReferences().AddReference(str(world_usd))
        dome = UsdLux.DomeLight.Define(stage, "/World/OverviewDome")
        dome.CreateIntensityAttr(900.0)
        dome.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
        dome.CreateTextureFormatAttr("latlong")
        distant = UsdLux.DistantLight.Define(stage, "/World/OverviewSun")
        distant.CreateIntensityAttr(1800.0)
        transform = UsdGeom.Xformable(distant)
        transform.AddRotateXYZOp().Set(Gf.Vec3f(25.0, -35.0, 20.0))
        camera = rep.create.camera(
            position=tuple(args.camera_position), look_at=tuple(args.camera_lookat)
        )
        render_product = rep.create.render_product(camera, (args.width, args.height))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(output_dir), rgb=True)
        writer.attach([render_product])
        for _ in range(args.settle_frames):
            app.update()
        rep.orchestrator.step()
        app.update()
        writer.detach()
        render_product.destroy()
        (output_dir / "render_manifest.json").write_text(
            "{\n"
            '  "artifact_type": "native_isaac_parallel_world_overview",\n'
            f'  "world_usd": "{world_usd}",\n'
            f'  "world_sha256": "{sha256(world_usd)}",\n'
            f'  "resolution": [{args.width}, {args.height}],\n'
            '  "claim_boundary": "Native Isaac overview of project-authored simulator terrain categories; colors are visual-only and analytical contact remains authoritative."\n'
            "}\n"
        )
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
