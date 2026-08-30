# DeepSense technical pitch, workflow, and evaluation playbook

This document fills the technical gaps in `HimalayaHack.pdf`. It is both a speaking script and an
engineering plan. Statements marked **measured** come from checked-in artifacts. Statements marked
**proposed** are hypotheses or future work.

## 1. The one-sentence concept

**DeepSense is a sensorized crampon and a fail-closed foothold supervisor that estimates the support
reserve under both feet before it allows the walking controller to transfer body weight.**

Do not pitch it as a snow classifier or as a new end-to-end walking policy. The core contribution is a
contact-level safety layer between terrain sensing and locomotion.

## 2. The precise problem

A normal humanoid locomotion policy usually observes joint state, body orientation, commanded velocity,
and sometimes exteroception such as a height scan. It can learn balance and foot placement, but a visually
flat patch can hide weak crust, a void, low shear strength, or previously damaged snow. Geometry alone does
not answer the two questions that matter during weight transfer:

1. Can the target foot accept the additional vertical load?
2. Can the stance foot retain enough vertical and lateral support while that transfer occurs?

A target can be locally strong while the stance foot is already slipping. That is why DeepSense reasons
about both feet. The phrase **bilateral assurance** means that the target and stance constraints must pass
at the same time.

## 3. What is in the boot

One foot emits one synchronized 19-value packet at 100 Hz:

| Signal | Count | Information | Hardware caveat |
|---|---:|---|---|
| Spike axial force | 4 | Load sharing, stiffness, fracture force drop | Scalar force along each spike axis, not simulator 3-D contact truth |
| Spike/probe penetration | 4 | Sinkage, layer engagement, asymmetry | Needs a moving probe, telescoping element, or floating collar |
| Accelerometer | 3 | Touchdown impulse, collapse, vibration | Bias, clipping, mount pose, and vibration must be calibrated |
| Gyroscope | 3 | Foot rotation, chatter, incipient loss of purchase | Not a direct measurement of friction |
| Decoded radar frontend | 5 | Interface depth, second interface, return strength, void likelihood, uncertainty | These are decoded features, not raw radar bins |
| **Total** | **19** | | A separate 19-bit validity mask is metadata, not 19 more sensors |

G1 pose, velocity, pelvis attitude, requested load, approach speed, and body load are separate robot
context. Exact material parameters and 3-D contact forces are training labels and diagnostics only. If any
exported actor receives those privileged values, the sim-to-real claim is invalid.

## 4. End-to-end data and control workflow

```text
four spike contacts + IMU + radar
              │  100 Hz; timestamps, validity, sample age
              ▼
causal 300 ms packet history per foot
              │
              ▼
continuous material/support estimator
  bearing, shear, support depth, stiffness, fracture/slip/void risk
  plus uncertainty and out-of-distribution status
              │
              ├──────────► persistent 2 m × 2 m local belief map
              │             contact updates its cell and correlated neighbors
              ▼
bilateral support computation
  target reserve + stance reserve + sinkage mismatch + transfer demand
              │
              ▼
deterministic safety authority
  COMMIT | HOLD_DOUBLE_SUPPORT | REPLANT/RETREAT
              │
              ▼
bounded high-level command or contact-gated residual
              │
              ▼
frozen/generic G1 walking policy and post-policy joint safety limits
```

The current CPU demo uses a transparent ExtraTrees estimator over causal prefix features. The Isaac stack
adds a causal bilateral estimator, visible-only supervisor, deterministic shield, and an optional bounded
joint residual that is enabled only after a fresh visible contact packet. A velocity-only locomotion policy
can slow or stop; it cannot guarantee an exact `REPLANT`. Exact replant execution needs a footstep-conditioned
policy or whole-body controller.

## 5. Physical quantities and decision rule

Let estimated bearing and shear capacities for foot `i` be `B_i` and `S_i`, with uncertainty
`σ_Bi` and `σ_Si`. A conservative lower confidence bound is

```text
B_i^- = B_i - k_B σ_Bi
S_i^- = S_i - k_S σ_Si
```

where `k` is selected on a separate calibration set. It is not automatically a certified probability.
For expected normal load `N_i` and tangential demand `T_i`, define

```text
vertical reserve R_Bi = B_i^- - N_i
lateral reserve  R_Si = S_i^- - T_i
```

A transfer may commit only when the target foot has positive vertical and lateral reserve, the stance foot
retains reserve through the planned load trajectory, packets are fresh, uncertainty is below its gate, and
the robot remains inside validated pose and transfer-rate limits. Otherwise the system holds, replants, or
enters a stop/recovery path. Hysteresis and minimum dwell time prevent mode chatter.

For a bare foot on a slope, the simplified tangential requirement is

```text
F_parallel = m g sin(θ)
N          = m g cos(θ)
no-slip condition: m g sin(θ) <= μ m g cos(θ)
                  tan(θ) <= μ
```

A spike can add penetration-dependent ploughing and geometric interlock, so its capacity is better written

```text
S_crampon = μ_residual N + F_plough(depth, geometry, material, damage, speed)
```

This explains the hypothesis, but it does not prove effectiveness. `F_plough` and fracture behavior must be
identified with the actual spike and prepared snow/ice rig.

## 6. Why the architecture is useful

- **Auditable:** outputs are capacities, margins, and uncertainty rather than an opaque terrain name.
- **Bilateral:** a good target cannot hide a degraded stance foot.
- **Persistent:** compaction, fracture, craters, and prior contacts do not disappear between steps.
- **Non-invasive:** the generic locomotion actor can remain frozen while the supervisor changes a bounded
  command envelope.
- **Fail-closed:** stale, invalid, uncertain, or out-of-distribution evidence causes hold/stop, not optimism.
- **Replaceable backends:** reduced-order, MuJoCo, Isaac Lab, and hardware logs share the same packet boundary.

The cost is conservatism. A safe system that never moves is not useful, so safety and progress must always be
reported together.

## 7. What is working now, and what is not

### Working in this repository

- **Measured:** a runnable reduced-order terrain, sensor, estimator, map, and bilateral scheduler pipeline.
- **Measured:** exact 19-channel packet contracts, causal prefixes, and field-seed train/test separation.
- **Measured:** persistent compaction/damage in the reduced model.
- **Measured:** a sensorized single-foot MuJoCo fixture and hybrid irreversible ice prior.
- **Implemented:** complete G1/crampon USD composition, Isaac Lab stateful contact/sensor environment, visible
  estimator/supervisor runtime, shield, render queues, and a side-by-side policy/crampon comparison runner.
- **Implemented but GPU-host dependent:** native Isaac runs and policy comparison require the pinned Linux,
  NVIDIA, Isaac Lab, policy, and checkpoint artifacts. They do not run on this Mac.

### Not proved

- Natural snow/ice model accuracy or Everest environmental validity.
- Physical sensor calibration, waterproofing, thermal behavior, or mechanical survival.
- Reliable whole-body replant, dynamic fall recovery, or real-G1 deployment.
- A measured stock-foot versus crampon effectiveness number. The current no-crampon Isaac lane is a
  deliberately low-grip proxy, not a validated stock G1 foot model.

The electronics slide is concept art, not the locked interface: it labels **3 Futek load buttons and 2 linear
probes**, while the software/hardware contract requires **4 force + 4 penetration** sites. It also labels gyro
output in degrees rather than angular rate (the code uses rad/s). Fix the slide or change the ABI and models;
do not present both as one implemented design.

## 8. Current measured results and their correct interpretation

The locally generated, reproducible `artifacts/hackathon` run used 200 fields, 1,600 probe episodes, 8,000
causal samples, and 16 held-out benchmark fields. The `artifacts/` directory is ignored by Git, so preserve a
hash-stamped result bundle separately if these exact numbers will appear in the final submission.

Estimator results:

- bearing-capacity MAE: **47.53 N**;
- bearing relative error: **14.25%**;
- shear relative error: **36.52%**;
- support-depth MAE: **34.1 mm**;
- void recall: **1.00**;
- false-safe rate on truly unsafe samples: **0.50%**;
- approximate two-sigma bearing coverage: **90.31%**.

Scheduler A/B on identical held-out seeds:

| System | Commits/completed steps | Unsafe transfers | Unsafe per commit | Holds | Replants |
|---|---:|---:|---:|---:|---:|
| Current-contact-only ablation | 79 | 7 | 8.86% | 0 | 49 |
| Full map + bilateral scheduler | 22 | 0 | 0% | 13 | 22 |

This is evidence that the safety logic can reject unsafe synthetic transfers. It is also evidence of a large
throughput cost: completed steps fell from 79 to 22 in this small run. Do not say “100% safer” or imply a real
fall probability. Say: **“In 16 synthetic held-out fields, the full scheduler eliminated 7 unsafe commits but
completed 57 fewer steps. The next goal is to recover progress while preserving the false-safe gate.”**

There is not yet a checked-in, statistically powered Isaac result for bare foot versus crampon. Any slide that
shows such a number before the experiment below runs must be labeled **proposed** or **illustrative**.

## 9. The correct crampon-effectiveness experiment

A two-lane video is a demonstration, not enough evidence. Use a paired 2 × 2 factorial test:

| Arm | Contact hardware model | Controller | What it isolates |
|---|---|---|---|
| A | Validated bare/stock foot | Frozen stock locomotion | Baseline |
| B | Crampon mechanics | Same frozen stock locomotion | Mechanical traction effect |
| C | Crampon mechanics | DeepSense supervisor + same locomotion core | Full product effect |
| D | Bare/stock foot | DeepSense supervisor | Controller effect without spike traction; useful negative/control arm |

The current `record_policy_crampon_comparison.py` compares separate policy artifacts and changes grip scale at
the same time. That is acceptable for a visual “full system versus baseline” demo, but it confounds hardware and
policy effects. Arm A versus B must use the **same policy bytes**. Arm B versus C must use the **same crampon
contact model**. Report both comparisons.

### Test conditions

Stratify, do not average away failures:

- surfaces: hard glacier ice, fractured blue ice, polished wind ice, thin snow over ice;
- inclines: 0°, 10°, 20°, 25°, 30°, then adaptive testing near failure boundaries;
- commands: 0.0, 0.1, 0.2, and 0.3 m/s plus controlled starts/stops;
- contact modes: all spikes, edge/two-spike, toe-only, heel-only;
- hazards: hidden void, thin crust, repeated footprint, one-foot weak patch;
- sensor faults for C/D: dropout, stale packet, bias, saturation, time skew;
- at least 30 paired seeds per condition for an initial result; use 100+ near the claimed operating boundary.

Each seed must create the same terrain/material draw, robot reset, command trace, and disturbance across arms.
Randomize lane/arm ordering to remove systematic environment-index effects. Exclude warm-up using a predeclared
rule. Never delete failed seeds.

### Primary metrics

1. **Time to first termination/fall**, with a fixed horizon, censoring, and restricted mean survival time;
   also report fall risk by the horizon. Do not treat post-reset cycles as independent episodes.
2. **Material slip fraction** under load.
3. **Force-weighted stance lateral speed** in m/s.
4. **Route completion without a safety violation**.
5. **Unsafe transfer rate per commit**, with the oracle definition frozen before evaluation.

### Secondary metrics

- forward command tracking error and net progress;
- minimum base height, roll/pitch excursions, and recovery count;
- mean stride advance and swing clearance;
- energy/cost of transport when trustworthy torque data is available;
- `HOLD`, `REPLANT`, and stop dwell time;
- estimator false-safe rate, event recall, calibration, and OOD rate;
- action correction magnitude and slew.

Do not use forward displacement alone. Auto-reset can make start-to-final displacement misleading after a
termination. Accumulate progress per episode and report survival-weighted progress.

### Statistics and pass criteria

- Analyze paired per-seed differences and show every point, not only bars.
- Report median, mean, bootstrap 95% confidence interval, and paired win fraction.
- Report each surface/incline stratum and an overall macro-average with equal stratum weight.
- Use a predefined primary metric and correct for multiple secondary comparisons.
- Treat zero observed failures as an upper bound, not proof of zero risk. With zero failures in `n` independent
  trials, the rough 95% “rule of three” upper bound is `3/n`.

Suggested initial engineering gates, to be frozen before the final run:

- B reduces force-weighted stance lateral speed by at least 30% relative to A on ice, with a paired 95% interval
  excluding zero degradation.
- B does not increase termination rate or command-tracking error by more than the declared non-inferiority margin.
- C reduces unsafe transfers or terminations relative to B while retaining at least 70% of B's safe progress.
- C fails closed under stale/invalid packets and never uses privileged truth.

These are project targets, not safety certification thresholds.

## 10. How to run the evidence stack

### Mac: estimator and bilateral-control evidence

```bash
uv sync --group dev
uv run pytest -q
uv run everest pipeline --config configs/hackathon.yaml --out artifacts/hackathon
open artifacts/hackathon/report.html
```

Show `metrics.json`, `benchmark.json`, `replay.json`, and `terrain_belief.png`. Label simulator truth as
simulator-only.

### Pinned Isaac Linux host: visual policy + traction-proxy comparison

The host needs Isaac Lab, the extension, the exact stock and crampon policy artifacts, `ffmpeg`, and the asset
hashes in `isaaclab_ext/stack.lock.json`.

```bash
/home/ubuntu/everest/IsaacLab3/isaaclab.sh -p -m pip install -e \
  isaaclab_ext/source/hackathon_everest_isaaclab

/home/ubuntu/everest/IsaacLab3/isaaclab.sh -p \
  isaaclab_ext/scripts/record_policy_crampon_comparison.py \
  --crampon-policy "$CRAMPON_POLICY" \
  --baseline-policy "$BASELINE_POLICY" \
  --output-dir artifacts/comparison/hard_ice_seed_311 \
  --surface hard_glacier_ice --incline-deg 20 --scene-seed 311 \
  --steps 500 --requested-vx 0.15 --headless
```

For all presentation shots:

```bash
/home/ubuntu/everest/IsaacLab3/isaaclab.sh -p \
  isaaclab_ext/scripts/run_policy_comparison_render_queue.py \
  --jobs configs/isaaclab/policy_crampon_comparison_render_jobs.json \
  --launcher /home/ubuntu/everest/IsaacLab3/isaaclab.sh \
  --launcher-arg=-p \
  --crampon-policy "$CRAMPON_POLICY" \
  --baseline-policy "$BASELINE_POLICY" \
  --output-root artifacts/comparison/render_queue
```

Aggregate completed manifests with:

```bash
uv run python scripts/analyze_policy_crampon_comparison.py \
  artifacts/comparison --output artifacts/comparison/summary.json
```

Do not present the four render jobs as a powered experiment. This renderer loads two plain TorchScript
locomotion actors; it does not load or hash the visible DeepSense estimator/supervisor checkpoint. Call its
output a policy + traction-proxy comparison unless the runner is extended to prove that full controller
provenance. The analyzer summarizes only the four fields present in these manifests; a formal science run must
also record first-fall/censoring, slip fraction, per-episode progress, route completion, and unsafe commits.
Generate repeated paired seeds without cameras and keep the render queue only for judge-facing footage.

## 11. Slide-by-slide speaking script

### Slide 1 — DeepSense (10 seconds)

“Humanoid robots can balance, but they cannot feel whether a foothold will hold. DeepSense gives a Unitree G1
an instrumented crampon and a safety layer that checks terrain support before the robot transfers its weight.”

### Slide 2 — Snow collapse under load (20 seconds)

“The failure is not always visible. A crust can look flat and remain stable under a light probe, then fracture
under full body load. A normal vision or height-map policy can place the foot correctly and still make the
wrong material decision.”

### Slide 3 — Boot with integrated sensors (25 seconds)

“Each boot produces exactly 19 synchronized values: four axial spike forces, four penetration measurements, a
six-axis IMU, and five decoded radar features. We intentionally do not feed exact simulator contact forces or
material labels to the deployable estimator.”

### Slide 4 — System overview (35 seconds)

“We keep a 300-millisecond causal history per foot. The estimator predicts continuous bearing and shear
capacity, support depth, fracture, slip, void risk, and uncertainty. Contact updates a persistent local map.
Then a bilateral controller asks whether both the target and stance foot retain reserve through the planned
weight transfer. The final action is commit, hold, or replant.”

### Slide 5 — Electronics (20 seconds)

“Force tells us load and stiffness. Penetration tells us sinkage and layer engagement. IMU transients reveal
collapse and rotation. Radar gives pre-contact layer and void evidence. A validity mask and timestamp make
missing or stale data explicit, and the controller fails closed.”

### Slide 6 — Simulation methods (40 seconds)

“We use three fidelity levels. A fast persistent terrain model generates causal training and lets us inspect
every margin. MuJoCo checks the actual spike geometry and a stateful ice law. Isaac Lab runs the full G1 with
four analytical contacts per foot, the hardware-shaped sensor stream, and the frozen locomotion policy. None
of these is called ground truth; physical rig calibration is the next gate.”

### Slide 7 — Simulation results (35 seconds)

“In the checked-in reduced-order run, the current-contact baseline made 79 commits and seven were unsafe. The
bilateral system made 22 commits and zero were unsafe. That is a strong safety signal and also a clear utility
cost. We do not hide it. Our next optimization target is safe progress, not just rejection. The Isaac crampon
comparison is reported separately because mechanical grip and sensing must be isolated.”

### Slide 8 — Core innovation (25 seconds)

“The innovation is bilateral assurance: sensing, physical estimation, behavior recommendation, and a
deterministic safety authority. A learned network may recommend, but it cannot override stale-data, support,
or actuator safety gates.”

### Slide 9 — What works (20 seconds)

“Today we have the end-to-end reduced pipeline, exact sensor contract, persistent terrain state, a fitted G1
crampon asset, MuJoCo fixtures, and an Isaac runtime. We do not yet have calibrated natural snow and ice or a
field-tested replant controller.”

### Slide 10 — Safety without opacity (20 seconds)

“We expose the estimated capacity, uncertainty, reserve, and exact reason for every hold. Conservative behavior
is calibrated and measurable. The system can be attached above a generic locomotion policy instead of hiding
all decisions inside one network.”

### Slide 11 — Feasibility (25 seconds)

“Next we calibrate each sensor and the actual spike on prepared snow and ice, fit posterior material ranges,
replay those trajectories in simulation, and run hardware-in-the-loop. Only after low-load, tethered tests pass
do we increase slope or autonomy.”

### Slide 12 — Why now (15 seconds)

“Extreme locomotion is a material-judgment problem under uncertainty. Everest is the stress test, but the same
architecture applies to avalanche response, mines, glaciers, rubble, and any robot that must decide whether a
contact is trustworthy.”

### Close (10 seconds)

“DeepSense does not promise that a robot can climb Everest today. It adds the missing primitive: measure the
foothold, bound uncertainty, and refuse the transfer when the evidence is not strong enough.”

## 12. Judge questions and direct answers

### “Is this just a terrain classifier?”

No. Named classes are too coarse for control. The output is continuous bearing/shear capacity, margins, event
risk, and uncertainty. The controller acts on support reserve.

### “Why radar if you already touch the ground?”

Radar gives a noisy pre-contact prior for layers and voids. Force, penetration, and IMU dominate thin-crust
contact signatures that the 40 mm radar range resolution cannot resolve. The system fuses them rather than
claiming radar solves everything.

### “How do four force sensors measure lateral grip?”

They do not directly measure a full 3-D wrench. Lateral capacity is inferred from axial load distribution,
penetration, foot motion, commanded demand, and calibrated contact histories. A rig with a six-axis reference
load cell supplies training labels. If that estimate is unreliable, the design should add lateral sensing.

### “What is novel relative to a force-torque sensor in the ankle?”

An ankle sensor measures the aggregate reaction after contact. DeepSense resolves four local spike loads,
penetration, subsurface evidence, persistent local damage, and bilateral transfer reserve. The novelty is the
contact-to-safety workflow, not merely adding a load cell.

### “Why not train one RL policy end to end?”

A monolithic policy makes truth leakage, uncertainty, and safety authority hard to audit. This stack keeps the
walking policy replaceable and gives a deterministic shield precedence. End-to-end fine-tuning can come later,
but not before contracts and safety gates work.

### “How is uncertainty calibrated?”

The CPU baseline uses tree-to-tree spread with physical floors and is explicitly approximate. The final system
needs a separate calibration split, one-sided interval coverage tests, OOD tests, and real rig traces. Until
then we do not call it a certified probability.

### “Does HOLD_DOUBLE_SUPPORT really guarantee double support?”

Not with a velocity-only walking actor. Today it is a scheduler recommendation or bounded stop request. A real
guarantee needs gait-phase or footstep-conditioned control and must be tested. We state that limitation.

### “Why does your full system complete fewer steps?”

Because it is conservative and its uncertainty is broad. Safety without progress is not sufficient. We report
safe progress and tune/calibrate only on development data while preserving a sealed false-safe test.

### “Is your no-crampon baseline fair?”

The current visual runner uses a low-grip proxy, so it is not yet a validated stock-foot model. The defensible
experiment uses a calibrated bare-foot contact model, identical policy bytes for A/B, matched seeds, and a
factorial B/C comparison to separate mechanics from sensing.

### “Does MuJoCo or Isaac simulate real ice?”

Not by default. The project adds an irreversible prior for indentation, fracture, ploughing, and crater state.
Its ranges are literature-grounded but not identified for this crampon. The physical rig is the authority.

### “What happens when a sensor fails?”

Every channel has validity and age. Stale/invalid/OOD evidence triggers a hold or stop path. Evaluation injects
single- and multi-sensor faults and checks that privileged simulator truth cannot affect the actor.

### “What prevents one foot from damaging terrain for the next foot?”

Nothing prevents the damage; the model records it. Compaction, fracture, and craters persist in a world cell.
That history updates the map and changes later decisions.

### “Can this run in real time?”

The packet rate is 100 Hz and walking policy rate is 50 Hz. The current ExtraTrees pipeline is lightweight.
The exported neural estimator must be benchmarked on the target onboard CPU with end-to-end timestamp and
watchdog tests before deployment.

### “What is the first hardware experiment?”

Instrument one actual spike and then a complete foot above a six-axis reference load cell. Sweep temperature,
vertical rate/load, lateral speed/direction, fresh/reused locations, and measure full curves, fracture timing,
hysteresis, penetration, and residual craters.

### “What is the business/use-case beyond Everest?”

The architecture applies wherever geometry is not enough: glacier inspection, avalanche response, mining,
polar logistics, mud, rubble, roofs, and disaster sites. The crampon is the first contact tool, while the
support-reserve interface is general.

## 13. Words to use and words to avoid

Use:

- “reduced-order scripted foothold scheduler”;
- “controlled simulator ablation”;
- “hardware-shaped 19-channel packet”;
- “lower-confidence support reserve”;
- “project-authored priors pending rig calibration”;
- “full system versus low-grip proxy” for the current visual comparison.

Avoid:

- “Everest-ready,” “proven safe,” or “digital twin”;
- “ground truth” for a second simulator;
- “the robot replants” until an execution controller does it;
- “the crampon is X% better” until the paired hardware-only experiment runs;
- “zero risk” when a finite simulation observed zero failures.
