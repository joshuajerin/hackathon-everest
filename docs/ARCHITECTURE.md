# Architecture

## 1. Persistent terrain truth

`TerrainGenerator` produces a 2 m × 2 m grid with 5 cm cells. Spatially correlated arrays store surface
height, support depth, vertical stiffness/damping, bearing and shear capacity, friction, crust thickness,
fracture strength, temperature, wetness, and void geometry. Compaction, damage, fracture, and surface
deformation persist after contact.

The reduced contact backend uses a continuous depth-dependent law per spike:

```text
normal force = k * depth^p + c * depth_rate + deeper-layer resistance
```

It then applies capacity, crust-fracture, void, damage, and spike-engagement effects. Tangential simulator
truth follows friction plus a penetration-dependent ploughing term. This is an empirical fast model, not
continuum snow physics.

## 2. Truth boundary and sensor boundary

The backend retains `(Fx, Fy, Fz)` for each spike as simulator-only diagnostics. `SensorSimulator`
projects the planned hardware path into:

```text
4 axial force + 4 penetration + 3 accel + 3 gyro + 5 radar frontend = 19
```

Bias, white noise, filtering, quantization, saturation, sample drops, and radar resolution are applied after
truth generation. A 19-bit validity mask identifies held samples without increasing the sensor count.
G1-like context is stored in `Proprioception`; it is intentionally separate from the crampon channel count.
It is context for the synthetic probe, not a claim that Unitree telemetry is wired.

## 3. Causal dataset

A probe lasts 300 ms at 100 Hz. Each episode becomes causal feature samples after 50, 100, 150, 225,
and 300 ms. For every prefix, a copy of the pre-contact field applies only the load, penetration, slip, and
fracture history observed by that timestamp before its label is calculated. Future fracture or damage is not
backfilled into early rows. The persistent episode field is mutated once by the full stroke. The dataset saves
`field_ids`, and `GroupShuffleSplit` keeps entire terrain seeds out of training.

## 4. Estimator

One `ExtraTreesRegressor` predicts continuous quantities. Separate tree classifiers estimate void,
fracture, and slip events. Tree-to-tree variation supplies approximate uncertainty with minimum physical
floors. Future work should calibrate intervals on a separate calibration split.

The controller uses margins, not scenario labels:

```text
vertical reserve = lower-confidence bearing capacity - current vertical load
slip reserve = estimated shear capacity - current tangential demand
```

## 5. Shared belief

`TerrainBeliefMap` stores means and variances in a 40 × 40 local grid. A contact updates its cell strongly
and neighboring cells using:

```text
w(r) = exp(-r² / (2 * correlation_length²))
```

Observation variance increases as spatial weight falls. This makes uncertainty reduction local and lets one
foot affect the other without pretending to predict a future video.

## 6. Bilateral scheduler

The support manager tracks current loads, lower-confidence reserves, sinkage mismatch, a simplified support
polygon margin, and a bounded transfer rate. Before contact, the same noisy five-value radar frontend scans
every candidate and fuses support depth and void probability into the map. The scheduler:

1. ranks candidate footholds;
2. chooses clearance and approach speed from expected depth/uncertainty;
3. requests a probe;
4. checks target support and stance-foot reserve;
5. returns `COMMIT`, `HOLD_DOUBLE_SUPPORT`, or `REPLANT` with a reason.

The reduced replay feeds planned probe load and approach speed into the load-controlled contact model. It
also evaluates planned swing clearance and transfer-rate-dependent load demand. It still recommends a step;
it does not output actuator commands or simulate whole-body dynamics.

## 7. Backend boundary

The stable contracts are the sensor packet, estimator, belief map, bilateral state, and step decision.
A later hybrid MuJoCo backend can replace reduced kinematics/contact generation while keeping these layers.
Newton MPM can later replace only the deformable terrain backend for replay. See
[MUJOCO_NEWTON_ROADMAP.md](MUJOCO_NEWTON_ROADMAP.md).
