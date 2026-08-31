# Isaac Lab quickstart

This guide runs the native DeepSense extension on the pinned Linux/NVIDIA stack. The CPU pipeline and
Isaac-neutral unit tests run on macOS and Linux; Isaac Sim itself does not run on the development Mac.

## 1. Pinned platform

The verified simulator lock is [`isaaclab_ext/stack.lock.json`](../isaaclab_ext/stack.lock.json):

| Component | Pin |
|---|---|
| OS used for recorded artifacts | Ubuntu 22.04.5 LTS |
| GPU used for recorded artifacts | NVIDIA A100-SXM4 40 GB |
| Driver | 580.105.08 |
| Isaac Sim | 6.0.1 |
| Isaac Lab | `v3.0.0-beta2.patch1` / `ffff603e...` |
| Python | 3.12.14 |
| Torch | 2.10.0 + CUDA 12.8 |
| RSL-RL | 5.0.1 |

Those pins describe the recorded simulator environment. They do not certify physical behavior.

## 2. Install the repository and extension

```bash
git clone https://github.com/joshuajerin/hackathon-everest.git
cd hackathon-everest
uv sync --group dev --locked

export ISAACLAB_ROOT=/home/ubuntu/everest/IsaacLab3
# Protect Isaac's pinned Torch/NumPy stack: install both editable packages without dependency resolution.
"$ISAACLAB_ROOT/isaaclab.sh" -p -m pip install --no-deps \
  -e . \
  -e isaaclab_ext/source/hackathon_everest_isaaclab

# The immutable L0 writer needs these separately pinned, non-simulator packages.
"$ISAACLAB_ROOT/isaaclab.sh" -p -m pip install \
  "pyarrow==21.0.0" "zarr==3.1.5"

# Preflight required imports inside Isaac's interpreter. Stop and resolve a missing package explicitly;
# do not run an unconstrained upgrade over the simulator environment.
"$ISAACLAB_ROOT/isaaclab.sh" -p -c \
  "import torch, numpy, scipy, sklearn, yaml, joblib, zarr, pyarrow, hackathon_everest, hackathon_everest_isaaclab; print(torch.__version__)"
```

Confirm the stack and repository hashes before producing evidence:

```bash
nvidia-smi
cat isaaclab_ext/stack.lock.json
sha256sum assets/crampon/g1_crampon_components_source.usdc
```

## 3. Build the composed G1 + crampon assets

The Isaac asset authority is the complete 26-object USDC assembly, not the older fitted STL path.
`$OFFICIAL_G1_USD` must be the exact official G1 asset paired with the chosen locomotion checkpoint.

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  isaaclab_ext/scripts/build_clean_crampon_payload.py \
  --repo-root .

"$ISAACLAB_ROOT/isaaclab.sh" -p \
  isaaclab_ext/scripts/build_g1_crampon_usd.py \
  --repo-root . \
  --official-g1 "$OFFICIAL_G1_USD" \
  --contact-model rigid_baseline \
  --output build/isaaclab/g1_crampon.usdc \
  --manifest build/isaaclab/g1_crampon.manifest.json

"$ISAACLAB_ROOT/isaaclab.sh" -p \
  isaaclab_ext/scripts/build_g1_crampon_usd.py \
  --repo-root . \
  --official-g1 "$OFFICIAL_G1_USD" \
  --contact-model stateful_material \
  --output build/isaaclab/g1_crampon_stateful.usdc \
  --manifest build/isaaclab/g1_crampon_stateful.manifest.json
```

Do not copy a generated USDC between incompatible G1 assets or checkpoint ABIs. Review every generated
manifest before training or evaluation.

Start with a one-environment, no-checkpoint native smoke test. Increase to 128, 256, or more environments only
after measuring VRAM and step time on the actual GPU:

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  isaaclab_ext/scripts/benchmark_vector_envs.py \
  --task Everest-Velocity-Flat-G1-Crampon-Stateful-v0 \
  --num_envs 1 \
  --warmup_steps 10 \
  --steps 50 \
  --output artifacts/isaaclab/smoke.json \
  --headless
```

## 4. Collect the L0 probe dataset

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  isaaclab_ext/scripts/collect_probe_farm.py \
  --repo-root . \
  --episodes 10000 \
  --dataset-root artifacts/datasets \
  --dataset-id everest_l0_probe_10k_v1 \
  --device cuda:0
```

The per-foot deployable packet is always:

```text
4 axial force + 4 penetration + 3 accelerometer + 3 gyro + 5 radar frontend = 19 values
```

Validity, sample age, commands, and robot context remain separate. Material state and exact 3-D contact
forces are labels or diagnostics only.

## 5. Evaluate a frozen locomotion policy

The policy must match the exact observation/action ABI recorded in the lock file. Never pad, truncate, or
reorder actions between G1 variants.

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  isaaclab_ext/scripts/evaluate_locomotion_policy.py \
  --task Everest-Velocity-Flat-G1-Crampon-Stateful-Play-v0 \
  --policy "$STOCK_TORCHSCRIPT" \
  --num-envs 8 \
  --steps 2000 \
  --output artifacts/eval/frozen-policy.json \
  --headless
```

## 6. Run DeepSense in shadow mode, then active mode

Shadow mode logs estimator, supervisor, shield, and gait behavior without allowing DeepSense to alter the
stock locomotion command. Start there.

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  isaaclab_ext/scripts/run_stateful_policy.py \
  --task Everest-Velocity-Flat-G1-Crampon-Stateful-Play-v0 \
  --stock-policy "$STOCK_TORCHSCRIPT" \
  --visible-checkpoint "$VISIBLE_CHECKPOINT" \
  --mode shadow \
  --num-envs 8 \
  --steps 2000 \
  --output artifacts/eval/deepsense-shadow.json \
  --headless
```

After the shadow gates pass, repeat with `--mode active`. The current velocity-level interface can scale,
hold, stop, or request recovery. It does not guarantee an exact `REPLANT`; that requires a validated
footstep-conditioned or whole-body controller.

The optional contact-corrected residual workflow is documented in
[`CONTACT_CORRECTION_SETUP.md`](CONTACT_CORRECTION_SETUP.md).

## 7. Record a controlled comparison

```bash
"$ISAACLAB_ROOT/isaaclab.sh" -p \
  isaaclab_ext/scripts/record_policy_crampon_comparison.py \
  --crampon-policy "$CRAMPON_POLICY" \
  --baseline-policy "$BASELINE_POLICY" \
  --output-dir artifacts/comparison/polished-25-seed-411 \
  --surface polished_wind_ice \
  --incline-deg 25 \
  --scene-seed 411 \
  --steps 750 \
  --requested-vx 0.12 \
  --baseline-grip-scale 0.01 \
  --headless
```

This runner is a same-process visual ablation. The baseline is a low-grip proxy and can use different policy
bytes, so it is not an isolated or calibrated hardware-effect estimate. Use the same policy in both arms to
isolate contact mechanics, then compare shadow versus active DeepSense separately to isolate supervision.

## 8. Sync evidence from a GPU host

Generated artifacts stay out of Git. Pull only the review bundle you need, then publish small, reviewed media
with hashes:

```bash
mkdir -p artifacts/gpu-import
rsync -av --progress \
  --include='*/' --include='*.json' --include='*.mp4' --include='*.png' --exclude='*' \
  ubuntu@GPU_HOST:/home/ubuntu/everest/hackathon-everest/artifacts/ \
  artifacts/gpu-import/
```

Before publishing:

1. decode each video fully;
2. verify the manifest and file SHA-256;
3. remove absolute host paths or secrets from public metadata;
4. keep the simulator-only claim boundary next to the artifact;
5. never commit checkpoints, tokens, simulator installations, or proprietary assets.

The README media bundle follows this rule and records provenance in
[`docs/media/manifest.json`](media/manifest.json).

## 9. Required interpretation boundary

The Everest suite contains 2,160 project-authored physical cases across nine surface families, ten inclines,
eight hazards, and three contact modes. Six fault cycles create 12,960 case/fault exposures. These are stress
tests—not surveyed mountain conditions, field frequencies, a digital twin, or evidence that a robot is ready
for Mount Everest.
