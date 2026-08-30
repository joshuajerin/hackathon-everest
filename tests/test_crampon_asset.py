from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_crampon_asset_is_uniformly_scaled_and_has_g1_cavity_clearance() -> None:
    root = Path(__file__).parents[1]
    metadata = json.loads((root / "assets/crampon/asset_metadata.json").read_text())

    assert metadata["source_sha256"] == "07e4120e021883a42871019cdeaefe54f64126af8b99b7944f51d484b88e48bb"
    assert metadata["source_triangle_count"] == 85_951
    assert metadata["transform"]["uniform_scale"] == 108.0
    assert metadata["transform"]["g1_cavity_fit_factor"] == 1.08
    assert sum(component["triangle_count"] for component in metadata["components"].values()) == 85_951

    extents = np.asarray(metadata["combined_bounds_m"]["extents_m"])
    assert np.allclose(extents, [0.28276759, 0.10016099, 0.06521792], atol=1e-7)
    clearance = np.asarray(metadata["g1_fit_check"]["estimated_clearance_per_side_xy_m"])
    assert clearance[0] >= 0.009
    assert clearance[1] >= 0.003
