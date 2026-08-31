# Contact-corrected Isaac run setup

Run this on the pinned Isaac Lab machine, not on the Mac. The local checkout does
not include the trained bounded-residual checkpoint or either stock-policy artifact.

Required inputs:

- a trained bounded-residual RSL checkpoint (`model_<N>.pt`);
- the exact official stock RSL checkpoint used to train that residual;
- a matching TorchScript stock actor with a `[B, 310] -> [B, 37]` ABI;
- the visible checkpoint. The default is
  `artifacts/lambda_prime/visible_policy_l0_v2/visible_policy.pt`.

Install the extension into Isaac Lab first:

```bash
/home/ubuntu/everest/IsaacLab3/isaaclab.sh -p -m pip install -e \
  isaaclab_ext/source/hackathon_everest_isaaclab
```

Train the bounded residual if its checkpoint is unavailable:

```bash
export EVEREST_STOCK_RSL_CHECKPOINT="$STOCK_RSL_CHECKPOINT"
/home/ubuntu/everest/IsaacLab3/isaaclab.sh -p \
  isaaclab_ext/scripts/launch_registered_rsl_train.py \
  --task Everest-Velocity-Flat-G1-Crampon-Stateful-FrontPoint-BoundedResidual-v0 \
  --num_envs 128 --max_iterations 30 --headless
```

Prepare the exported policy and active launcher. Run the setup command through
Isaac Lab so its Python can import Torch, RSL-RL, and the extension:

```bash
/home/ubuntu/everest/IsaacLab3/isaaclab.sh -p \
  isaaclab_ext/scripts/setup_contact_correction.py \
  --residual-checkpoint logs/rsl_rl/everest_g1_crampon_front_point_bounded_residual/<run>/model_<N>.pt \
  --stock-rsl-checkpoint "$STOCK_RSL_CHECKPOINT" \
  --stock-policy "$STOCK_TORCHSCRIPT" \
  --output-dir artifacts/contact_correction/model_<N> \
  --steps 500
```

The setup script exports and validates `policy/policy.pt`, writes hash manifests,
and generates `run_active_contact_correction.sh`. Review and run that launcher.
The correction is active only after fresh visible axial-force and penetration packets
show contact, and only while the shield permits `COMMIT`.


## Sensor-driven joint-target residual path

`runtime.SensorGatedJointResidual` is the direct joint-target adapter for a
separately trained sensor policy. That policy must take the visible estimator
latent and emit one raw residual per named joint. The default ABI is nine
lower-body targets: bilateral hip pitch, knee, ankle pitch, ankle roll, and
torso.

It applies each policy output as a bounded physical correction in radians:

```text
q_target = clamp(q_stock + limit * tanh(raw_residual), joint_limits)
```

The correction is rate-limited to `0.015 rad` per control tick by default and
is removed immediately whenever its `enabled` gate is false. The adapter then
converts the physical target back through the exact deployed `joint_pos` action
transform. Its caller must supply that action term's named joint order, scales,
offsets, and position limits. The adapter resolves and validates configured joint names internally; it must
never guess a G1 action map or trust anonymous joint indices.

This repository does not yet contain a trained/exported sensor-to-joint
residual checkpoint or an Isaac runner flag for it. It is deliberately not
connected to the existing `--contact-correction-policy`, whose ABI is a
stock-observation-only 37-action specialist. Train and validate the new
sensor-conditioned policy first, then wire it to this adapter using the exact
pinned Isaac Lab action metadata.
