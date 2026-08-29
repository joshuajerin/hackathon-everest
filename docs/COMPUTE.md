# Compute setup

## Current pipeline: use the Mac

The implemented estimator is scikit-learn ExtraTrees. It is CPU-bound and already uses all CPU cores with
`n_jobs=-1`. The RTX 5070 will not accelerate this model. The tested hackathon config finishes in tens of
seconds on the development Mac, so remote compute would add more setup risk than value right now.

```bash
uv sync --group dev
uv run everest pipeline --config configs/hackathon.yaml --out artifacts/hackathon
```

For the larger synthetic set:

```bash
uv run everest pipeline --config configs/workstation.yaml --out artifacts/workstation
```

Run that on either machine. Copy only source/config changes through Git. Do not commit generated model or
dataset artifacts.

## RTX 5070 readiness

The GPU becomes useful when a raw-window TCN/GRU or a vectorized MuJoCo/Newton backend is added. On the
Linux RTX machine:

```bash
git clone https://github.com/joshuajerin/hackathon-everest.git
cd hackathon-everest
uv sync --group dev
bash scripts/bootstrap_rtx5070.sh
```

The bootstrap installs an official CUDA 12.8 PyTorch wheel into the project environment and runs an actual
matrix multiplication on CUDA. A successful check must print:

- `torch.cuda.is_available() == True`;
- the RTX 5070 device name;
- PyTorch and CUDA versions;
- a finite tensor checksum.

This is only an environment check. The current ExtraTrees path remains CPU-only. Do not say the model was
GPU trained until a GPU estimator exists and the training manifest records it.

## Artifact transfer

Generated files can be large. Prefer `rsync` or a release/object store rather than Git:

```bash
rsync -av --progress artifacts/workstation/ user@mac:~/hackathon-everest-artifacts/
```

The `manifest.json` records seeds, schema, feature order, and Git revision so a remote run can be reproduced.
