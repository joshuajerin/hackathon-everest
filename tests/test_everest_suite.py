from pathlib import Path

import numpy as np

from hackathon_everest.everest_suite import (
    assign_cases,
    load_suite,
    required_cartesian_cases,
    slope_normal,
    vector_layout,
)

CONFIG = Path(__file__).parents[1] / "configs/isaaclab/everest_terrain_suite.yaml"


def test_every_surface_is_crossed_with_every_incline() -> None:
    config = load_suite(CONFIG)
    cases = required_cartesian_cases(config)
    surfaces = {item["id"] for item in config["surfaces"]}
    inclines = {float(value) for value in config["inclines_deg"]}
    observed = {(case.surface_id, case.incline_deg) for case in cases}
    assert observed == {(surface, incline) for surface in surfaces for incline in inclines}
    assert len(cases) == len(surfaces) * len(inclines) * len(config["hazards"]) * len(config["contact_modes"])


def test_4096_environment_assignment_covers_required_suite_without_colliding_origins() -> None:
    config = load_suite(CONFIG)
    layout = vector_layout(config)
    assigned = assign_cases(config, seed=17)
    required_ids = {case.case_id for case in required_cartesian_cases(config)}
    assert len(assigned) == layout.num_envs == 4096
    assert required_ids <= {case.case_id for case in assigned}
    origins = np.asarray([layout.origin(index) for index in range(layout.num_envs)])
    assert len(np.unique(origins[:, :2], axis=0)) == layout.num_envs
    assert np.isclose(np.min(np.diff(np.unique(origins[:, 0]))), layout.cell_spacing_m)
    assert layout.cell_spacing_m > max(layout.patch_size_m)


def test_slope_normal_matches_configured_incline() -> None:
    assert np.allclose(slope_normal(0), [0.0, 0.0, 1.0])
    normal = slope_normal(45)
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert np.allclose(normal, [-2**-0.5, 0.0, 2**-0.5])
