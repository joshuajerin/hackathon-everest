# Hackathon Everest

**A continuous foothold-assurance system for a sensorized Unitree G1 crampon.**

The core idea is not to classify a patch as “snow” or “ice.” The crampon estimates how much
vertical and lateral support each foot has left, shares that evidence through a local terrain map,
and delays or redirects the next body-weight transfer when the bilateral support state is unsafe.

> Current status: a runnable reduced-order synthetic pipeline for the hackathon. It is not autonomous
> G1 locomotion, validated snow physics, tested hardware, or evidence that a robot is ready for Everest.

## Why this is the hackathon scope

The full concept includes MuJoCo, radar waveform simulation, SnowMicroPen calibration, Newton MPM,
and eventual G1 control. Doing all of those first would hide the main innovation and make the live demo
fragile. This repository implements one complete vertical slice:

```text
persistent correlated terrain field
        ↓
19-channel crampon packets + separate G1 context
        ↓
continuous per-foot estimator + approximate uncertainty
        ↓
shared 2 m × 2 m terrain belief map
        ↓
bilateral support reserve
        ↓
COMMIT / HOLD_DOUBLE_SUPPORT / REPLANT
```

The terrain is continuous. Scenario names such as hard ice, firn, crust, or snow bridge are not model
outputs and are not training labels.

## Run it

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
uv run pytest -q
uv run everest pipeline --config configs/smoke.yaml --out artifacts/smoke
open artifacts/smoke/report.html
```

The larger Mac CPU demo is:

```bash
uv run everest pipeline --config configs/hackathon.yaml --out artifacts/hackathon
open artifacts/hackathon/report.html
```

On the development Mac, the hackathon config generates 1,600 probes / 8,000 causal prefix samples,
trains the estimator, benchmarks both schedulers, and renders the report in tens of seconds. Runtime
will vary by machine.

### Individual commands

```bash
uv run everest contract
uv run everest generate --episodes 800 --fields 100 --out artifacts/probes.npz
uv run everest train --dataset artifacts/probes.npz --model artifacts/model.joblib
uv run everest replay --model artifacts/model.joblib --steps 6 --out artifacts/replay.json
```

## The exact sensor contract

Each synchronized crampon sample has **19 values**, not 22:

| Input | Values | Hardware story |
|---|---:|---|
| Four spike axial forces | 4 | One scalar strain/load measurement along each spike axis |
| Four penetration readings | 4 | Requires moving probes, telescoping spikes, or a floating surface collar |
| Foot accelerometer | 3 | Crampon-mounted IMU |
| Foot gyroscope | 3 | Crampon-mounted IMU |
| Radar frontend | 5 | Decoded interfaces/strength/void/uncertainty, not raw radar bins |
| **Total** | **19** | Synchronized at one timestamp |

G1 foot pose/velocity, pelvis orientation, probe command, approach speed, and current load are paired
context. They are not extra crampon sensors. A separate 19-bit validity mask marks held or missing samples;
it is metadata, not another sensor reading. Full 3-D simulator contact forces remain labels and diagnostics
only; the estimator never receives lateral-force truth unavailable to the planned v1 hardware.

The radar frontend is quantized to 40 mm range resolution in the fast simulator. It therefore does not
pretend to resolve a 2–20 mm crust. Force, penetration, and IMU contact signatures handle that case.

## What the pipeline produces

`everest pipeline` writes:

- `probe_dataset.npz` — causal windows at 50, 100, 150, 225, and 300 ms;
- `terrain_estimator.joblib` — ExtraTrees continuous/event estimator;
- `metrics.json` — held-out field-seed metrics and false-safe rate;
- `replay.json` — per-step sensor estimates, map evidence, support state, and decisions;
- `benchmark.json` — current-contact-only versus map + bilateral reasoning;
- `terrain_belief.png` — truth, controller belief, and uncertainty views;
- `report.html` — a self-contained demo report;
- `manifest.json` — config hash/content, feature order, split field IDs, dependency versions, Git state,
  seeds, schema, and limitations.

The smoke replay begins on an explicitly known stable launch patch. Unknown terrain starts at
`x = -0.42 m`. This avoids confusing “cannot leave the starting block” with the foothold problem.

## Learning setup

- **Regressions:** support depth, stiffness, damping, bearing capacity, shear capacity, friction,
  compaction, damage, fracture margin, slip margin, and void depth.
- **Events:** void, fracture, and slip probabilities.
- **Uncertainty:** tree-to-tree ensemble spread with conservative floors. It is approximate and is not
  presented as a calibrated safety probability.
- **Split:** whole terrain field seeds, never neighboring strokes from one field.
- **Pre-contact scan:** the noisy radar frontend updates every candidate cell before ranking.
- **Controller:** deterministic conservative scoring and support gates. Planned probe load and approach speed
  drive contact; bounded transfer rate and swing clearance enter execution checks. No RL is required for v0.

## Repository map

```text
src/hackathon_everest/
  terrain.py       correlated fields + persistent compaction/fracture
  physics.py       reduced-order per-spike contact truth
  sensors.py       noise, quantization, dropouts, 19-channel packets
  features.py      causal prefix-window features
  dataset.py       field-seed dataset generation and storage
  estimation.py    continuous/event models and held-out metrics
  mapping.py       40 × 40 mean/uncertainty belief map
  control.py       bilateral reserves and step decisions
  pipeline.py      replay, ablation benchmark, plot, and HTML report
  cli.py           command-line entry point
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
[docs/HACKATHON_SCOPE.md](docs/HACKATHON_SCOPE.md),
[docs/COMPUTE.md](docs/COMPUTE.md), and the
[submission kit](docs/SUBMISSION.md) for the design boundaries and demo flow.

## Safety and claim boundary

This project is experimental simulation software. It does not command a physical robot. Its synthetic
material priors are not field calibrated. Any hardware or mountain deployment would need mechanical,
environmental, electrical, control, and human safety review well beyond this repository.
