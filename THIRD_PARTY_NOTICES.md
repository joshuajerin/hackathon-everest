# Third-party notices and asset provenance

The repository-level [MIT license](LICENSE) applies to original Hackathon Everest source code unless a file or
directory states otherwise. It does **not** automatically relicense third-party robot models, simulator
packages, policies, checkpoints, textures, or user-supplied CAD.

## Unitree G1 model

The pinned MuJoCo Menagerie G1 files under `vendor/mujoco_menagerie/unitree_g1/` retain the Unitree Robotics
license reproduced at [`vendor/mujoco_menagerie/unitree_g1/LICENSE`](vendor/mujoco_menagerie/unitree_g1/LICENSE).
The Unitree name is used only to identify model compatibility and does not imply endorsement.

## User-supplied crampon geometry

The crampon source was supplied to the project owner. Its provenance, source hash, inferred units, and
transforms are documented in [`assets/crampon/README.md`](assets/crampon/README.md). No separate public
redistribution license is recorded in this repository. The geometry and its rendered/derived media are not
asserted to be covered by the repository MIT license. Anyone redistributing or commercializing those files
must first confirm that they have the necessary rights.

## Isaac Sim, Isaac Lab, RSL-RL, and pretrained policies

These are not distributed as part of the Python package. Install and use them under their respective NVIDIA,
upstream, model-provider, and checkpoint terms. The stack lock records compatibility and hashes; it does not
grant redistribution rights.

## MuJoCo and MuJoCo Menagerie

MuJoCo is an optional external dependency. Menagerie assets retain their per-model licenses. See
`vendor/mujoco_menagerie/LICENSE` and the license within each model directory.

## Visual and terrain references

Configuration entries that name Poly Haven, ambientCG, or other visual sources are identifiers and provenance
references. Their content retains its original license. Verify the applicable terms before redistributing a
generated scene or render.

## README media

Files in `docs/media/` are simulator recordings or derived excerpts. Their hashes and derivation are recorded
in `docs/media/manifest.json`. The media is published as simulator-only project evidence and inherits the
licenses and restrictions of the robot, crampon, simulator, and any visual assets visible in the recording.
