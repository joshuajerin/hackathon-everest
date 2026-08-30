#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BLENDER=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
cd "$ROOT"

if [[ ! -x "$BLENDER" ]]; then
  echo "Blender executable not found: $BLENDER" >&2
  exit 1
fi

"$BLENDER" --background blender/g1_crampon_fit.blend \
  --python blender/export_crampon_fit.py -- \
  --blend blender/g1_crampon_fit.blend \
  --out assets/crampon

uv run python scripts/sync_blender_fit.py
uv run python scripts/build_g1_crampon.py
uv run pytest tests/test_crampon_asset.py tests/test_mujoco_probe.py tests/test_hybrid_ice_probe.py

echo "Applied saved Blender fit to standalone and full-G1 MuJoCo models."
