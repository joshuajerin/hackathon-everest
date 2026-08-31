# Isaac Lab implementation and synthetic-data plan

## Status

This document is the implementation authority for moving Hackathon Everest into Isaac Lab.

Current state:

- The complete USD crampon assembly, Blender fit, and reproducible composition scripts are implemented.
- The reduced-order dataset, estimator, terrain map, and bilateral decision system are runnable on CPU.
- The standalone MuJoCo geometry and hybrid ice fixtures are implemented.
- An external Isaac Lab extension now registers stateful G1/crampon tasks, visible sensor packets, contact
  mechanics, estimator/supervisor runtime, safety shield, training/evaluation scripts, and render queues.
- Native Isaac recordings were produced on the pinned Linux/A100 stack in `isaaclab_ext/stack.lock.json` and
  a small reviewed media bundle is published under `docs/media/`.
- A clean checkout still requires external official G1 assets and compatible stock/residual/visible policy
  artifacts to reproduce full native runs. Default CI does not run Isaac Sim.
- No claim of calibrated natural snow/ice, physical crampon effectiveness, exact replant execution, real G1
  balance, mountain readiness, or hardware safety is made.

## Locked decisions

1. **Use the complete USD component assembly for Isaac Lab.**
   - Source: `assets/crampon/g1_crampon_components_source.usdc`
   - SHA-256: `53703057dff7ea5b2e7e468164289d6c0aba629400952c9a8a9a5f7048f2a660`
   - Editable fit: `blender/g1_crampon_components_fit.blend`
   - Generator: `blender/setup_usd_component_fit.py`
   - Alignment metadata: `assets/crampon/usd_component_fit_metadata.json`
   - Machine-readable authority: `configs/isaaclab/g1_crampon_asset.yaml`
   - Never use the old two-mesh STL fit for Isaac Lab.

2. **Keep locomotion and foothold assurance separate.**
   - A generic walking policy produces G1 joint targets.
   - The crampon system estimates support and supervises velocity, probing, load transfer, and stop/replant behavior.
   - A deterministic safety shield has final authority over `COMMIT`, `REPLANT`, and `HOLD_DOUBLE_SUPPORT`.

3. **Keep the exact sensor ABI.**
   - One foot produces exactly 19 values: 4 axial forces, 4 penetrations, 6 IMU, and 5 decoded radar values.
   - Each foot has its own packet and validity mask.
   - Bilateral code consumes two packet histories; it does not redefine the per-foot packet as 38 values.
   - G1 proprioception and commands remain separate context.
   - Exact contact vectors, terrain state, and material truth never enter a deployable actor.

4. **Do not begin with one end-to-end RL policy.**
   - First validate mechanics and sensors.
   - Then train and calibrate a causal estimator.
   - Then run it in shadow mode with a frozen walking policy.
   - Only then train a visible-only supervisory policy.

5. **Use visual meshes and analytical contact separately.**
   - Preserve all 26 authored USD component transforms for visuals and mass/inertia work.
   - Use four auditable analytical probes per foot for contact and sensing.
   - Do not make all high-detail meshes terrain collision geometry.

## End-to-end architecture

```text
complete fitted crampon USDC
        +
official G1 USD and named joint contract
        ↓
Isaac Lab vector environments
        ↓
stateful GPU snow/ice contact at 4 probes per foot
        ↓
hardware-shaped 19-value packet per foot at 100 Hz
        ↓
causal recurrent estimator + calibrated uncertainty
        ↓
shared persistent terrain belief map
        ↓
foothold-assurance supervisor and deterministic shield
        ↓
safe velocity/mode/footstep envelope
        ↓
generic G1 walking policy at 50 Hz
        ↓
post-policy limits, watchdog, damping/stop path
```

The simulator also creates label and privileged data planes. Those planes are available for losses, teacher/critic training, and evaluation only.

## 1. Reproducible Isaac Lab platform

### Primary version target

Pin a complete tested stack. Do not follow moving branches.

Current NVIDIA reference target:

- Isaac Lab: `v3.0.0-beta2.patch1`
- Isaac Sim: `6.0.1`
- Python: `3.12`
- RL library: RSL-RL
- OS: Ubuntu 24.04 preferred

Official references:

- <https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2.patch1>
- <https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/setup/installation/index.html>
- <https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/overview/core-concepts/task_workflows.html>

Do not mix Isaac Lab 2.3 code or documentation with 3.0 manager, Warp, or launcher APIs.

### Hardware gate

Full Isaac Sim does not run on the development Mac. Use the NVIDIA workstation or a Linux GPU host.

Published baseline requirements are approximately:

- 32 GB RAM or more;
- 16 GB VRAM or more;
- current production NVIDIA driver;
- GLIBC 2.35 or newer for the pip installation path.

An RTX 5070 with 12 GB VRAM is below the published 16 GB recommendation. It may work headless with reduced environment counts and no RTX sensors, but this must be profiled rather than assumed. Start at 128 environments and scale through 256, 512, and 1,024 while measuring VRAM and simulation rate. Use a larger GPU if the foot/contact state or recorder cannot fit reliably.

### Repository integration style

Create an external manager-based Isaac Lab extension. Do not patch Isaac Lab core.

Proposed layout:

```text
isaaclab_ext/
  source/hackathon_everest_isaaclab/
    config/extension.toml
    pyproject.toml
    setup.py
    hackathon_everest_isaaclab/
      assets/g1_crampon_cfg.py
      contact/stateful_material.py
      sensors/crampon_sensor.py
      data/schema.py
      data/writer.py
      tasks/manager_based/crampon_velocity/
        __init__.py
        env_cfg.py
        mdp/observations.py
        mdp/events.py
        mdp/rewards.py
        mdp/terminations.py
        agents/rsl_rl_ppo_cfg.py
  scripts/
    build_g1_crampon_usd.py
    collect_probe_farm.py
    collect_walking_rollouts.py
    train_crampon_estimator.py
    evaluate_crampon_stack.py
```

Use `ManagerBasedRLEnv` because official G1 locomotion is manager-based and its observation, action, event, reward, curriculum, and recorder terms are reusable. Switch a fused inner loop to `DirectRLEnv` only after profiling proves the manager layer is the bottleneck.

## 2. Generic G1 walking policy

### Simulation baseline

Isaac Lab provides official tasks:

- `Isaac-Velocity-Flat-G1-v0`
- `Isaac-Velocity-Flat-G1-Play-v0`
- `Isaac-Velocity-Rough-G1-v0`
- `Isaac-Velocity-Rough-G1-Play-v0`

Use the official rough RSL-RL checkpoint as the first simulator baseline. Preserve its exact robot asset, joint ordering, observation ordering, action scaling, physics rate, and checkpoint version.

The current stock policy contract is important:

- physics timestep: 0.005 s;
- policy timestep: 0.02 s / 50 Hz;
- action: 37 joint-position residuals;
- rough actor includes a simulator height scan;
- it is not automatically compatible with a physical 23- or 29-DoF G1.

Reference task paths:

- <https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-beta2.patch1/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py>
- <https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-beta2.patch1/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py>
- <https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-beta2.patch1/docs/source/_static/policy_deployment/01_io_descriptors/isaac_velocity_flat_g1_v0_IO_descriptors.yaml>

### Real-G1 deployment gate

Before hardware deployment, identify the exact G1 revision and joint contract.

If the robot is the Unitree 29-DoF configuration, prefer the Unitree-maintained `unitree_rl_lab` 29-DoF policy and deployment mapping rather than truncating or remapping the NVIDIA 37-action checkpoint:

- <https://github.com/unitreerobotics/unitree_rl_lab>
- Task: `Unitree-G1-29dof-Velocity`
- Bundled ONNX and exact `deploy.yaml` must remain paired.

Never pad, truncate, reorder, or guess joint actions between 37-, 29-, 23-, or 12-DoF policies.

### Locomotion interface

Create a replaceable interface:

```text
LocomotionInput:
  timestamp
  requested vx, vy, wz
  mode: STAND | WALK | STOP
  optional allowed footstep corridor

LocomotionOutput:
  timestamp
  named joint targets
  policy health
```

The foothold system must not rewrite the walking policy's internal observation vector or individual joint actions.

A velocity-only walking policy cannot guarantee an exact foothold or execute a true `REPLANT`. Version one can:

- continue at a bounded speed;
- reduce speed;
- stop and hold;
- command a controlled retreat if validated.

Exact target relocation requires a later footstep-conditioned locomotion policy, gait-phase interface, or model-predictive controller. Do not claim that setting velocity to zero guarantees double support until it is tested.

## 3. Isaac asset composition

1. Reference the official G1 USD used by the selected checkpoint.
2. Compose `assets/crampon/g1_crampon_components_source.usdc` into a generated G1-crampon USD.
3. Apply `USD_ASSET_POSITION_CONTROL` and per-foot fine-tune transforms from the Blender/metadata authority.
4. Attach one complete assembly under each `.*_ankle_roll_link` frame.
5. Preserve all 26 component-relative transforms.
6. Calculate updated foot mass, inertia, center of mass, and ground clearance.
7. Add named analytical probe points/bodies for the four contacts per foot.
8. Prevent stock sole collisions from bypassing those contacts.
9. Keep the generated USD in `build/`; keep source assets and build scripts in version control.

Asset acceptance gates:

- exact visual transform match against the Blender reference;
- correct left/right placement and axis signs;
- no mesh-unit change;
- no stock collision path carries terrain load;
- stable standing pose with finite state;
- center-of-mass and inertia changes recorded;
- walking checkpoint observation and action dimensions unchanged.

If physical prismatic probes add articulation DoFs, exclude them by explicit joint names from the generic policy contract. Never use `joint_names=[".*"]` after adding probe joints. The first high-throughput version should model probe deflection in the custom contact state so the walking articulation remains unchanged.

## 4. Three-fidelity simulation stack

### L0: vectorized probe farm

Purpose: produce most estimator data cheaply.

- One or two foot fixtures per environment.
- GPU tensor/Warp stateful material law.
- Random load, penetration, approach speed, lateral demand, temperature, slope, layer, void, and damage history.
- Thousands of environments if memory allows.
- No full G1 rendering.

### L1: complete G1 closed-loop environment

Purpose: capture realistic gait, impacts, bilateral support, and policy distribution.

- Complete G1 and crampon assemblies.
- Frozen generic walking policy.
- Two feet, persistent terrain, actuator error, body-load transfer, and falls.
- Crampon estimator first runs in shadow mode.
- Later, supervisor changes high-level commands.

### L2: adjudication and discrepancy suite

Purpose: find L0/L1 blind spots.

- Small high-fidelity subsets;
- alternate timestep/substep settings;
- alternate backend where supported;
- MuJoCo hybrid replay;
- eventually rig-calibrated real packet replay.

Do not describe Newton, deformables, MPM, or a second simulator as ground truth. Use them as disagreement tests.

## 5. Stateful snow and ice contact

PhysX rigid friction alone cannot represent crust fracture, snow compaction, penetration, breakout, or crater memory.

Implement a batched custom material model that:

1. reads each probe's world position and velocity;
2. queries a per-environment terrain/material state field;
3. integrates normal indentation and tangential shear response;
4. updates irreversible state;
5. sums forces and moments onto the ankle/foot body;
6. applies equal and opposite terrain bookkeeping;
7. disables native contact for the same spike/terrain pair to avoid double force.

Per-cell state should include:

- surface height and residual crater;
- maximum penetration and work;
- plastic sinkage;
- density/compaction;
- damage;
- crust integrity and fractured flag;
- tangential run-in and shear displacement;
- breakout state;
- temperature and wetness context.

Terrain state resets only when that world resets. It must not heal when a foot lifts or an individual step ends.

Initial physics rate: 500-1,000 Hz for the custom contact integration, subject to convergence tests. Device-shaped sensor packets remain 100 Hz. Walking policy remains 50 Hz.

Convergence gate:

- halving the physics timestep changes impulse, peak force, and maximum penetration by less than 5%;
- fracture/slip event timing changes by less than 10 ms;
- no native/custom force double counting;
- force and energy accounting remain finite.

## 6. Sensor simulation

Each foot emits a `SynchronizedSensorPacket` compatible with `src/hackathon_everest/models.py`.

Randomize correlated hardware effects, not only independent white noise:

- force gain, offset, misalignment, cross-talk, hysteresis, saturation, quantization, and drift;
- probe compliance, backlash, zero offset, travel limit, and temperature dependence;
- IMU bias, colored noise, vibration, clipping, and mounting error;
- radar clutter, multipath-like decoded errors, interface merging, range resolution, uncertainty, and failure;
- packet latency, jitter, clock phase, clock drift, burst dropout, and held values.

Use Warp/raycast terrain queries for high-volume decoded radar training. Isaac RTX Radar is experimental and too expensive for thousands of environments; use it only in small perception/evaluation scenes. It is not an FMCW snow radar truth generator.

## 7. Strict data planes

Never construct one giant observation tensor and slice it casually.

### Sensor plane — deployable

- left and right 19-value histories;
- per-foot validity masks and timestamps;
- explicitly named deployable G1 context and commands.

### Label plane — supervised losses

- causal support depth, stiffness, damping, bearing, shear, friction;
- compaction, damage, fracture/slip margins, and void depth;
- void, fracture, and slip events;
- outcomes at the current prefix only.

### Privileged plane — teacher/critic and diagnostics

- exact material parameters;
- full contact vectors;
- exact map and cell state;
- future capacity and oracle safety;
- terrain-family and simulator internals.

Add a random truth-canary to the privileged plane. CI must fail if changing the canary changes exported student actions. Export and test the student in a clean process that never allocates privileged observations.

## 8. Synthetic dataset program

Dataset size is useful only if coverage and grouping are correct.

### Initial target scale

- 2 million four-spike 300 ms probe episodes;
- approximately 62 million raw packet timestamps;
- 250,000 repeated-contact sequences;
- 100,000 complete bilateral route episodes;
- paired L1/L2 discrepancy subset;
- increase volume only while sealed-test learning curves improve.

### Probe interventions

Randomize and log:

- XY target and local slope;
- load trajectory and maximum load;
- approach speed and maximum travel;
- tangential demand and direction;
- swing clearance;
- transfer rate;
- repeated contact count and time since previous contact;
- neighboring and alternating-foot contacts.

Fork paired counterfactual branches from identical saved world states with different probe actions. This distinguishes action effects from material correlation.

### Required difficult cases

- thin crust over a void;
- delayed collapse after initially strong support;
- one or two spikes engaged;
- load imbalance and edge contact;
- breakout and snagging;
- high bearing but low shear;
- stance degradation while the target appears safe;
- previously damaged or compacted footprints;
- radar clutter and incorrect interface pairing;
- single-sensor and multi-sensor failures;
- stale packets, time skew, and latency bursts;
- parameter combinations outside the core calibration distribution.

Oversample rare unsafe events for training but record sampling probabilities. Final safety metrics must be recomputed on a frozen natural-prior distribution.

### Causal records

Store raw sequences and reproduce compatibility prefixes at 50, 100, 150, 225, and 300 ms. Every label at time `t` must equal replay stopped at `t`; later fracture or damage cannot be backfilled.

Canonical record:

```text
world/group identity
current time
left/right packet[19]
left/right validity masks
separate deployable context
proposed and applied command
shield action
next visible packet
causal labels at current time
terminal outcome
sampling weight
```

### Split policy

Use a stable group hash over:

- terrain generator family;
- parent field/world seed;
- spatial block;
- material batch/site/day;
- geometry revision;
- all counterfactual and repeated-contact descendants.

Never split frames, prefixes, neighboring probes, repeated craters, or counterfactual branches independently.

Frozen allocation:

- 65% train;
- 10% calibration;
- 10% validation;
- 15% sealed test.

Maintain separate challenge sets for unseen terrain generators, backend shift, parameter corners, sensor faults, and later real batches/days/sites/routes.

### Storage

```text
datasets/isaac/<dataset_id>/
  manifest.json
  episodes.parquet
  visible.zarr/
  truth.zarr/
  completion-markers/
```

`manifest.json` records schema, units, channel order, code/asset/checkpoint hashes, simulator versions, seed lineage, randomization posterior, split hashes, calibration provenance, and licenses.

Use immutable shards per worker/GPU process, checksums, atomic completion markers, and monotonic timestamp validation. Do not share one HDF5 writer across GPU processes.

## 9. System identification and randomization

Do not use an unstructured independent uniform distribution for every parameter.

1. Calibrate every load sensor, probe, IMU, clock, and radar frontend.
2. Run actual spike and whole-foot prepared snow/ice tests across load, speed, temperature, and lateral direction.
3. Fit correlated parameter distributions using complete force-depth-time curves, impulse, hysteresis, fracture timing/drop, residual crater, and shear curves.
4. Sample posterior-like correlated draws for the main training distribution.
5. Add a separately labeled 5-10% stress tail.
6. Update the distribution from new real trials rather than hand-tuning one simulator constant.

MOSAiC/SnowMicroPen data constrains plausible snow behavior but does not replace crampon calibration.

## 10. Estimator

Keep ExtraTrees as a transparent baseline. Train a causal sequence estimator for the main model.

Recommended first neural model:

- shared per-foot temporal CNN or GRU;
- last 300 ms at 100 Hz;
- packet values, validity, sample age, commands, and deployable context;
- shared weights for left and right;
- bilateral fusion after per-foot encoding;
- small enough for target CPU inference.

Outputs remain physical and auditable:

- support depth;
- stiffness and damping;
- bearing and shear capacity/reserve;
- friction/slip margin;
- compaction and damage;
- fracture margin;
- void depth;
- void/fracture/slip probabilities;
- uncertainty/OOD score.

Use deep ensembles or equivalent epistemic uncertainty, heteroscedastic/quantile heads for sensor/material variation, held-out probability calibration, and one-sided lower confidence bounds for bearing/shear/fracture margins. High uncertainty or unsupported OOD input must produce `HOLD_DOUBLE_SUPPORT`, not a confident guess.

## 11. Crampon supervisory policy

Train this only after the estimator and shield pass independent gates.

### Inputs

- two visible-only packet histories or frozen estimator latents;
- calibrated uncertainty and OOD flags;
- terrain belief map;
- deployable G1 context;
- walking command and gait/contact phase.

### Proposed outputs

- allowed velocity scale;
- allowed yaw/sideways envelope;
- probe load and approach speed;
- clearance and transfer-rate proposal;
- foothold/corridor preference where supported;
- preference over `COMMIT`, `REPLANT`, and `HOLD_DOUBLE_SUPPORT`.

### Final action

The deterministic shield retains precedence:

1. bad target -> `REPLANT` or stop/recovery;
2. unsafe stance reserve -> `HOLD_DOUBLE_SUPPORT`;
3. fast settling/stale/OOD -> `HOLD_DOUBLE_SUPPORT`;
4. only safe target and stance -> `COMMIT`.

Add hysteresis, minimum dwell time, stale-data timeout, and a bounded recovery path to prevent chatter or indefinite hold.

### Teacher/student process

1. Train a privileged teacher with exact terrain/contact/map truth.
2. Distill its risk/value and proposals into a recurrent visible-only student.
3. Collect DAgger-style rollouts under student behavior.
4. Optionally fine-tune with asymmetric actor-critic: visible-only actor, privileged critic.
5. Freeze and export the student.
6. Evaluate in a clean runtime with no privileged buffers.

The teacher is an upper-bound research tool, not deployable software.

## 12. Closed-loop training sequence

### Phase A — stock walking baseline

- Play the pinned official checkpoint on its unchanged robot and terrain.
- Export checkpoint hash and I/O descriptor.
- Validate zero-command standing behavior.
- Record fall rate and command tracking.

### Phase B — crampon asset, no custom terrain

- Add complete visual/mass asset and analytical rigid contacts.
- Run the frozen checkpoint.
- Measure gait degradation caused by mass, inertia, ground clearance, and contact changes.
- Fine-tune locomotion only if needed, without crampon truth in actor observations.

### Phase C — probe farm

- Validate one spike, four spikes, one foot, and bilateral fixtures.
- Generate causal sensor data and train estimator.

### Phase D — shadow walking

- Frozen walking policy crosses stateful terrain.
- Estimator and scheduler log decisions but cannot change commands.
- Compare predicted decisions with oracle outcomes and collect failure distribution.

### Phase E — shielded walking

- Allow only speed reduction, stop, and validated mode transitions.
- No exact-replant claim.
- Require safe standing/stop transitions before proceeding.

### Phase F — foothold-aware locomotion

- Introduce a footstep-conditioned or mode-conditioned walking controller.
- Add target corridor, swing clearance, probe, hold, and replant semantics.
- Keep the same estimator and shield interface.

### Phase G — sim-to-sim and HIL

- Replay the frozen exported stack in MuJoCo and recorded-packet/HIL runtimes.
- Verify named joints, rates, normalization, timestamps, and limits.

## 13. Independent safety layer

The learned policies do not own final hardware safety.

After policy inference enforce:

- named joint position, velocity, and torque limits;
- command slew limits;
- watchdog and stale timestamp limits;
- base height and orientation limits;
- unexpected contact and fall detection;
- packet validity and estimator OOD checks;
- validated damping/stand/stop state;
- hardware E-stop outside the neural stack.

No learned controller may bypass these checks.

## 14. Evaluation gates

These are engineering targets, not certification.

### P0 — contract and fixture

- exact per-foot 19-value order, units, signs, and 100 Hz timestamps;
- validity mask remains metadata;
- no truth-canary influence;
- no force double counting;
- timestep convergence targets pass.

### P1 — state and writer

- lift does not heal terrain;
- selective environment reset affects only selected worlds;
- causal prefix labels equal stopped replay;
- paired branches are identical before intervention;
- 10 million-frame stress run has no environment/shard mixing, no non-monotonic timestamps, and more than 99.99% complete rows.

### P2 — system identification

On held-out rig conditions, proposed targets are:

- median normalized force-curve RMSE <=10%;
- 90th percentile <=20%;
- penetration and impulse error <=10%;
- fracture/slip recall >=0.95;
- event timing error <=20 ms;
- real traces have appropriate coverage inside simulator predictive bands.

If these fail, revise the contact model before more RL.

### P3 — estimator

On frozen natural-prior and per-stratum tests:

- support-depth MAE <=5 mm;
- bearing/shear median relative error <=10%, p90 <=20%;
- void recall >=0.98;
- fracture/slip recall >=0.95;
- event ECE and Brier <=0.03;
- nominal one-sided interval coverage achieved;
- false-safe rate <=0.5% with 95% upper bound <=1%;
- OOD cases abstain rather than confidently commit.

### P4 — bilateral supervisor

- unit tests preserve decision precedence;
- zero oracle-unsafe commits across at least 3,000 independent hidden natural-prior commit opportunities;
- zero commits with stale/OOD critical sensors;
- route completion >=95% on traversable routes;
- progress >=90% of teacher;
- unnecessary hold/replant <=10%;
- decision latency p99 <=10 ms on the target CPU.

A policy that always holds fails the progress gate.

### P5 — backend/HIL

- same frozen adapter and normalization across L0, L1, L2, MuJoCo, rig replay, and HIL;
- no model-specific hidden remapping;
- decision disagreement <=5% overall;
- zero cases where the student commits while the higher-fidelity adjudicator is unsafe;
- packet replay deterministic after preprocessing.

### P6 — real staged validation

Proceed only in this order:

1. individual sensors;
2. one-spike fixture;
3. four-spike foot fixture;
4. suspended one-foot low load;
5. low-load double support;
6. tethered flat prepared material;
7. tethered controlled slopes;
8. controlled outdoor route.

Every stage retains independent force/vision instrumentation, hard load/velocity limits, and E-stop. Increase load or slope only after predeclared clean trials.

## 15. Major risks

- The generic walking policy may fail after crampon mass/contact changes.
- A 37-action simulator policy may not match the physical G1 revision.
- A velocity-only actor cannot guarantee a foothold or true replant.
- Synthetic radar may become an unrealistically perfect shortcut.
- Privileged simulator truth may leak through observations, normalization, or replay buffers.
- Random frame-level splits may make results look much better than they are.
- Native PhysX and custom snow forces may be double counted.
- Terrain may accidentally reset/heal between steps.
- Large uniform randomization may generate impossible worlds and an unusable estimator.
- Dataset scale may hide poor coverage of rare unsafe boundaries.
- RTX 5070 VRAM may limit environment count.
- Success in PhysX does not prove MuJoCo or hardware behavior.

Each risk has a corresponding gate above. Do not bypass a failed gate by generating more data.

## 16. Immediate implementation order

1. Provision and record the pinned Linux/Isaac/CUDA/RSL-RL environment.
2. Run and hash the unchanged official G1 rough checkpoint.
3. Export its exact I/O descriptor.
4. Generate an external manager-based extension.
5. Build the complete G1-crampon composed USD from the authoritative USDC.
6. Validate asset pose, mass/inertia, collision ownership, and policy dimensions.
7. Implement the one-foot L0 analytical contact fixture.
8. Port the existing stateful ice/material logic to batched Torch/Warp.
9. Implement the exact 19-channel sensor adapter and truth boundary tests.
10. Implement immutable dataset shards and group splits.
11. Run a small 10,000-probe dataset and recover known labels.
12. Scale to the first probe farm only after convergence and leakage tests pass.
13. Train ExtraTrees baseline and recurrent estimator.
14. Run the frozen walking policy with the estimator in shadow mode.
15. Add the deterministic supervisor and safe stop/hold behavior.
16. Train a visible-only supervisory student only after estimator gates pass.
17. Perform MuJoCo, recorded-rig, HIL, and staged hardware checks.

## Definition of the first credible demo

The first credible Isaac Lab demo is not unrestricted snow walking. It is:

- a frozen generic G1 policy walking slowly through a controlled generated course;
- complete authoritative crampon geometry on both feet;
- stateful terrain that remembers previous contacts;
- exact per-foot 19-channel hardware-shaped packets;
- a causal estimator running without simulator truth;
- a visible terrain belief map;
- a supervisor that safely continues, slows/stops, or requests recovery;
- an oracle overlay used only for post-run scoring;
- a report comparing frozen walking alone against estimator + map + bilateral shield.

A later footstep-conditioned controller is required before claiming exact `REPLANT` execution.
