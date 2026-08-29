import numpy as np

from hackathon_everest.mapping import TerrainBeliefMap


def test_contact_reduces_local_uncertainty_and_weakly_updates_neighbors(safe_estimate):
    belief = TerrainBeliefMap(correlation_length_m=0.25)
    before = belief.mean["bearing_capacity_n"].copy()
    before_var = belief.variance["bearing_capacity_n"].copy()
    belief.update(0.0, 0.0, safe_estimate)
    center = belief.index(0.0, 0.0)
    near = belief.index(0.15, 0.0)
    far = belief.index(0.9, 0.9)
    center_change = abs(belief.mean["bearing_capacity_n"][center] - before[center])
    near_change = abs(belief.mean["bearing_capacity_n"][near] - before[near])
    far_change = abs(belief.mean["bearing_capacity_n"][far] - before[far])
    assert center_change > near_change > far_change
    assert belief.variance["bearing_capacity_n"][center] < before_var[center]


def test_candidate_score_uses_lower_confidence_support(safe_estimate):
    belief = TerrainBeliefMap()
    belief.update(0.2, 0.1, safe_estimate)
    candidates = np.array([[0.2, 0.1], [-0.7, -0.7]])
    scored = belief.score_candidates(candidates, nominal_xy_m=np.array([0.2, 0.1]))
    assert np.allclose(scored[0].xy_m, [0.2, 0.1])



def test_precontact_radar_changes_candidate_prior_without_contact():
    belief = TerrainBeliefMap()
    index = belief.index(0.25, -0.1)
    before_void = belief.mean["void_probability"][index]
    before_depth = belief.mean["support_layer_depth_m"][index]
    belief.update_radar(0.25, -0.1, np.array([0.16, 0.24, 0.4, 0.9, 0.03]))
    assert belief.mean["void_probability"][index] > before_void
    assert belief.mean["support_layer_depth_m"][index] > before_depth
