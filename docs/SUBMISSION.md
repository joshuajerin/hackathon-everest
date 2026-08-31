# Submission kit

## Track

**Track 1 — Movement**

## Short description

Hackathon Everest is a reduced-order foothold-assurance system for a sensorized Unitree G1 crampon.
Four axial spike-force channels, four moving-probe depth channels, a foot IMU, and a decoded radar
frontend estimate continuous vertical and lateral support rather than a named terrain class. Contact from
one foot updates a shared local terrain belief, and a bilateral manager emits `COMMIT`, `HOLD_DOUBLE_SUPPORT`, or `REPLANT` recommendations from
lower-confidence support reserves. The reduced-order replay does not command a physical G1.

## 90-second pitch outline

1. **Problem:** visual terrain does not reveal whether a snow/ice foothold can accept a full transfer.
2. **Insight:** human climbers test crampon purchase before committing; the G1 needs an instrumented version.
3. **System:** 19 sensor values → continuous estimate → spatial belief → bilateral support gate.
4. **Demo:** identical held-out terrain seeds; compare current-contact-only with the full conservative scheduler.
5. **Honest next step:** calibrate the priors on a physical contact rig, run matched-policy Isaac ablations, and validate the control ladder on hardware.

## Demo command

```bash
uv run everest pipeline --config configs/hackathon.yaml --out artifacts/hackathon
open artifacts/hackathon/report.html
```

Show the controller view first. Keep truth in the diagnostics panel and label it as simulator-only.

## Required links

- Repository: https://github.com/joshuajerin/hackathon-everest
- Simulator demo: [`docs/media/bilateral-sensor-packet-demo.mp4`](media/bilateral-sensor-packet-demo.mp4)
