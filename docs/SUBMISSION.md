# Submission kit

## Track

**Track 1 — Movement**

## Short description

Hackathon Everest is a reduced-order foothold-assurance system for a sensorized Unitree G1 crampon.
Four axial spike-force channels, four moving-probe depth channels, a foot IMU, and a decoded radar
frontend estimate continuous vertical and lateral support rather than a named terrain class. Contact from
one foot updates a shared local terrain belief, and a bilateral manager commits, holds, or replants based on
lower-confidence support reserves.

## 90-second pitch outline

1. **Problem:** visual terrain does not reveal whether a snow/ice foothold can accept a full transfer.
2. **Insight:** human climbers test crampon purchase before committing; the G1 needs an instrumented version.
3. **System:** 19 sensor values → continuous estimate → spatial belief → bilateral support gate.
4. **Demo:** identical held-out terrain seeds; compare current-contact-only with the full conservative scheduler.
5. **Honest next step:** calibrate priors, integrate rigid G1 kinematics in MuJoCo, then replay in Newton MPM.

## Demo command

```bash
uv run everest pipeline --config configs/hackathon.yaml --out artifacts/hackathon
open artifacts/hackathon/report.html
```

Show the controller view first. Keep truth in the diagnostics panel and label it as simulator-only.

## Required links

- Repository: https://github.com/joshuajerin/hackathon-everest
- Demo video: add before submission
