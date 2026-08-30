"""Project-authored simulator policy thresholds, never hardware calibration."""

MINIMUM_BEARING_CAPACITY_N = 343.0
FRACTURE_DAMAGE_CAUTION = 0.45
DEGRADED_SLIP_MARGIN_N = -40.0
SEVERE_SLIP_MARGIN_N = -100.0
SUPERVISOR_VALIDATION_MAXIMUM_UNSAFE_COMMIT_RATE = 0.0015

CLAIM_BOUNDARY = (
    "Project-authored simulator-policy priors. Replace or validate with instrumented "
    "hardware evidence before physical deployment."
)
