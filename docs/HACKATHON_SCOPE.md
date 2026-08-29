# Hackathon scope

## Demo claim

> The crampon estimates remaining support under each foot, remembers what nearby terrain is likely to
> be like, and delays or redirects the next weight transfer when the two-foot support state is unsafe.

This is a Track 1 movement project. The two-minute demo should show one deterministic 4–8 step replay,
not a slide deck of future components.

## What the live demo proves

1. A single terrain field persists across both feet and every step.
2. A left-foot observation changes the prior for nearby right-foot candidates before contact.
3. The estimator returns continuous state and uncertainty instead of a snow/ice class.
4. Candidate ranking uses lower-confidence bearing support, void probability, uncertainty, and distance.
5. A good target is not enough when the stance foot cannot carry the transfer.
6. The full scheduler can hold or replant where the current-contact-only ablation commits unsafely.

## What it does not prove

- stable whole-body G1 walking;
- dynamic balance, fall recovery, or support-polygon control;
- validated snow, ice, radar, or hardware behavior;
- physical Everest readiness;
- sim-to-real transfer.

Use the phrase **“reduced-order scripted foothold scheduler”** in the pitch.

## Suggested 2-minute flow

- **0:00–0:20 — Problem:** a visually plausible snow foothold may not support the next weight transfer.
- **0:20–0:40 — Hardware contract:** four axial forces, four probe depths, foot IMU, and radar frontend.
- **0:40–1:20 — Replay:** show the left observation updating the map, the next candidate score, and the
  bilateral reserve panel changing to `COMMIT`, `HOLD_DOUBLE_SUPPORT`, or `REPLANT`.
- **1:20–1:45 — A/B:** show unsafe transfers for current-contact-only and the conservative full system
  on identical held-out field seeds. Report the actual run; do not promise the full system always wins.
- **1:45–2:00 — Path forward:** real snow calibration, hybrid MuJoCo G1 kinematics, then Newton replay.

## Acceptance gates

- `uv run pytest -q` passes.
- `uv run everest pipeline --config configs/smoke.yaml --out artifacts/smoke` succeeds from a clean checkout.
- Sensor vectors have exactly 19 hardware-shaped values.
- Train/test field IDs have zero overlap.
- The saved replay contains `cross_foot_evidence`, pre-contact candidate radar scans, executed probe
  commands, and bilateral support for each attempted step.
- Contact changes compaction/damage persistently.
- The report states the synthetic and whole-body-control limitations.
