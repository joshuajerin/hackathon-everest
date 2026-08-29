from __future__ import annotations

import numpy as np

from .mapping import TerrainBeliefMap
from .models import BilateralSupportState, FootSide, FootTerrainEstimate, StepDecision


class BilateralSupportManager:
    def __init__(self, *, robot_weight_n: float = 343.0, safety_buffer_n: float = 10.0):
        self.robot_weight_n = float(robot_weight_n)
        self.safety_buffer_n = float(safety_buffer_n)

    def evaluate(
        self,
        left: FootTerrainEstimate,
        right: FootTerrainEstimate,
        *,
        left_load_n: float,
        right_load_n: float,
        support_polygon_margin_m: float = 0.035,
    ) -> BilateralSupportState:
        left_reserve = left.lower_confidence_bearing_n() - left_load_n
        right_reserve = right.lower_confidence_bearing_n() - right_load_n
        settling = max(abs(left.sinkage_rate_mps), abs(right.sinkage_rate_mps))
        uncertainty_penalty = left.uncertainty[3] + right.uncertainty[3]
        safe_rate = np.clip(280.0 - 2_200.0 * settling - 0.25 * uncertainty_penalty, 20.0, 280.0)
        return BilateralSupportState(
            left_current_load_n=float(left_load_n),
            right_current_load_n=float(right_load_n),
            left_support_reserve_n=float(left_reserve),
            right_support_reserve_n=float(right_reserve),
            left_sinkage_m=left.current_sinkage_m,
            right_sinkage_m=right.current_sinkage_m,
            height_difference_m=left.current_sinkage_m - right.current_sinkage_m,
            total_support_margin_n=float(left_reserve + right_reserve),
            support_polygon_margin_m=float(support_polygon_margin_m),
            maximum_safe_transfer_rate_nps=float(safe_rate),
        )

    def transfer_is_safe(self, state: BilateralSupportState, swing_side: FootSide) -> bool:
        if swing_side == FootSide.RIGHT:
            stance_reserve = state.left_support_reserve_n
            transfer = state.right_current_load_n
        else:
            stance_reserve = state.right_support_reserve_n
            transfer = state.left_current_load_n
        return bool(
            stance_reserve >= transfer + self.safety_buffer_n
            and state.support_polygon_margin_m >= 0.015
            and state.maximum_safe_transfer_rate_nps > 25.0
        )


class AdaptiveStepScheduler:
    def __init__(self, support_manager: BilateralSupportManager | None = None):
        self.support_manager = support_manager or BilateralSupportManager()

    def select_candidate(
        self,
        belief: TerrainBeliefMap,
        candidates_xy_m: np.ndarray,
        *,
        nominal_xy_m: np.ndarray,
    ) -> StepDecision:
        best = belief.score_candidates(candidates_xy_m, nominal_xy_m=nominal_xy_m)[0]
        depth_i, depth_j = belief.index(float(best.xy_m[0]), float(best.xy_m[1]))
        expected_depth = belief.mean["support_layer_depth_m"][depth_i, depth_j]
        uncertainty = np.sqrt(belief.variance["support_layer_depth_m"][depth_i, depth_j])
        clearance = np.clip(0.06 + expected_depth + 1.5 * uncertainty, 0.08, 0.28)
        approach = np.clip(0.24 - 0.45 * uncertainty - 0.08 * best.void_probability, 0.07, 0.22)
        return StepDecision(
            action="PROBE",
            target_xy_m=best.xy_m,
            conservative_score=best.score,
            swing_clearance_m=float(clearance),
            approach_velocity_mps=float(approach),
            probe_load_n=132.0,
            load_transfer_rate_nps=60.0,
            reason="Highest lower-confidence support after void, uncertainty, and distance penalties.",
        )

    def decide_after_probe(
        self,
        *,
        plan: StepDecision,
        swing_side: FootSide,
        target: FootTerrainEstimate,
        bilateral: BilateralSupportState,
    ) -> StepDecision:
        lower_fracture_margin = target.fracture_margin_n - 2.0 * target.uncertainty[8]
        fracture_risk = (
            target.fracture_probability > 0.65
            and lower_fracture_margin < 0.0
            and target.lower_confidence_bearing_n() < 0.9 * self.support_manager.robot_weight_n
        )
        target_is_bad = (
            target.lower_confidence_bearing_n() < 0.45 * self.support_manager.robot_weight_n
            or target.void_probability > 0.55
            or target.damage_state > 0.72
            or target.slip_margin_n < 0.0
            or target.slip_probability > 0.75
            or fracture_risk
        )
        if target_is_bad:
            action = "REPLANT"
            reason = "Target support, void, damage, slip, or lower-confidence fracture margin is unsafe."
        elif not self.support_manager.transfer_is_safe(bilateral, swing_side):
            action = "HOLD_DOUBLE_SUPPORT"
            reason = "Stance foot cannot conservatively accept the planned load transfer yet."
        elif abs(target.sinkage_rate_mps) > 0.22:
            action = "HOLD_DOUBLE_SUPPORT"
            reason = "Target is still settling too quickly to commit body weight."
        else:
            action = "COMMIT"
            reason = "Target and stance support reserves cover the bounded transfer."
        return StepDecision(
            action=action,
            target_xy_m=plan.target_xy_m,
            conservative_score=plan.conservative_score,
            swing_clearance_m=plan.swing_clearance_m,
            approach_velocity_mps=plan.approach_velocity_mps,
            probe_load_n=plan.probe_load_n,
            load_transfer_rate_nps=min(plan.load_transfer_rate_nps, bilateral.maximum_safe_transfer_rate_nps),
            reason=reason,
        )
