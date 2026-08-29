from __future__ import annotations

import numpy as np
import pytest

from hackathon_everest.models import FootTerrainEstimate


@pytest.fixture
def safe_estimate() -> FootTerrainEstimate:
    return FootTerrainEstimate(
        support_layer_depth_m=0.03,
        void_probability=0.05,
        fracture_probability=0.05,
        slip_probability=0.05,
        void_depth_m=0.0,
        effective_vertical_stiffness_n_per_m=25_000.0,
        effective_vertical_damping_ns_per_m=55.0,
        bearing_capacity_n=620.0,
        current_sinkage_m=0.018,
        sinkage_rate_mps=0.01,
        shear_capacity_n=260.0,
        effective_friction=0.5,
        slip_margin_n=100.0,
        spike_engagement=np.ones(4),
        spike_support_quality=np.ones(4) * 0.8,
        center_of_support_xy=np.zeros(2),
        compaction_state=0.2,
        damage_state=0.05,
        fracture_margin_n=150.0,
        uncertainty=np.array([0.005, 1000.0, 4.0, 25.0, 12.0, 0.03, 0.03, 0.03, 15.0, 12.0, 0.01]),
    )
