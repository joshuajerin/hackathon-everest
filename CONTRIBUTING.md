# Contributing

DeepSense is experimental robotics software. Contributions should preserve its strict boundary between
**deployable sensor data**, **training labels**, and **simulator-only privileged truth**.

## Development setup

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/joshuajerin/hackathon-everest.git
cd hackathon-everest
make setup
make verify
```

`make verify` runs Ruff, the reduced-order tests, and the CPU smoke pipeline. Isaac-neutral extension tests
need Torch plus the immutable-writer dependencies. Run them separately with `make test-isaac-unit` or
together with `make verify-all`.
Native Isaac Lab tests require the pinned Linux/NVIDIA stack documented in
[`docs/ISAACLAB_QUICKSTART.md`](docs/ISAACLAB_QUICKSTART.md).

## Pull request checklist

- Keep one logical change per pull request.
- Add focused tests for changed contracts or behavior.
- Keep each crampon packet at exactly 19 values per foot; validity and age remain metadata.
- Never feed material truth, exact 3-D contact forces, labels, or future state into a deployable actor.
- Record units, rates, seeds, asset/checkpoint hashes, and claim boundaries in generated manifests.
- Split datasets by world/field/site lineage, not neighboring frames or contacts.
- Report safety and progress together; a controller that only stops is not a successful result.
- Do not present simulator priors as field calibration, hardware validation, or an Everest digital twin.
- Run `make verify` and `git diff --check` before requesting review.

## Generated artifacts

`artifacts/`, `build/`, checkpoints, datasets, and simulator caches are intentionally ignored. Small,
reviewed README media lives in `docs/media/` with hashes and provenance in `docs/media/manifest.json`.
Do not commit large generated datasets, simulator installations, credentials, or proprietary policy files.

## Isaac Lab changes

Use the external extension under `isaaclab_ext/`; do not patch Isaac Lab core. Preserve the stack pin in
`isaaclab_ext/stack.lock.json`. Native GPU evidence should include:

- simulator, driver, asset, policy, and checkpoint hashes;
- task ID, seed, surface, slope, hazard, contact mode, and sensor-fault mode;
- exact command and output schema;
- a statement of what the run does and does not validate.
