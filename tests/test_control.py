import copy

from hackathon_everest.control import AdaptiveStepScheduler, BilateralSupportManager
from hackathon_everest.mapping import TerrainBeliefMap
from hackathon_everest.models import FootSide


def test_low_stance_reserve_blocks_transfer(safe_estimate):
    manager = BilateralSupportManager()
    unsafe_stance = copy.deepcopy(safe_estimate)
    unsafe_stance.bearing_capacity_n = 250.0
    unsafe_stance.uncertainty[3] = 35.0
    state = manager.evaluate(
        unsafe_stance,
        safe_estimate,
        left_load_n=220.0,
        right_load_n=123.0,
    )
    scheduler = AdaptiveStepScheduler(manager)
    plan = scheduler.select_candidate(
        TerrainBeliefMap(),
        candidates_xy_m=[[0.2, -0.1]],
        nominal_xy_m=[0.2, -0.1],
    )
    decision = scheduler.decide_after_probe(
        plan=plan,
        swing_side=FootSide.RIGHT,
        target=safe_estimate,
        bilateral=state,
    )
    assert decision.action == "HOLD_DOUBLE_SUPPORT"


def test_safe_bilateral_state_commits(safe_estimate):
    manager = BilateralSupportManager()
    state = manager.evaluate(
        safe_estimate,
        safe_estimate,
        left_load_n=220.0,
        right_load_n=123.0,
    )
    scheduler = AdaptiveStepScheduler(manager)
    plan = scheduler.select_candidate(
        TerrainBeliefMap(),
        candidates_xy_m=[[0.2, -0.1]],
        nominal_xy_m=[0.2, -0.1],
    )
    decision = scheduler.decide_after_probe(
        plan=plan,
        swing_side=FootSide.RIGHT,
        target=safe_estimate,
        bilateral=state,
    )
    assert decision.action == "COMMIT"
