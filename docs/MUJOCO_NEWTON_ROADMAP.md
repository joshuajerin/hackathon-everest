# MuJoCo and Newton roadmap

The current repository intentionally does not claim MuJoCo or G1 results. There is no verified G1 asset or
whole-body controller in this v0.

## Phase 1 — hybrid MuJoCo adapter

- Import the 29-DOF MuJoCo Menagerie G1 asset.
- Add a rigid crampon body and four named spike geoms per foot.
- Use a floating or planar pelvis with a no-force safety catch during valid motion.
- Let MuJoCo provide rigid-body kinematics and contact geometry.
- Apply reduced terrain forces through an explicit hybrid contact adapter.
- Use `mj_contactForce` only for contacts MuJoCo solved. Do not call it deformable-snow truth.
- Preserve the same 19-channel sensor packet and simulator-only diagnostics split.

Validation order:

1. single spike;
2. four-spike static probe;
3. one full foot;
4. two feet in double support;
5. one bounded load transfer;
6. alternating quasi-static steps.

## Phase 2 — real-data calibration

- Import SnowMicroPen force-versus-depth profiles.
- Fit stiffness, yield, collapse depth, and force-drop distributions.
- Calibrate radar noise/interface widths using SnowEx GPR.
- Generate detailed A-scans offline with gprMax and fit the five-value frontend surrogate.
- Version all priors and preserve dataset licenses/citations.

## Phase 3 — Newton MPM replay

Replace only the reduced deformable-terrain backend. Keep the sensor contract, estimator, map, bilateral
manager, and scheduler unchanged. Validate the same sequence from a single spike through alternating steps.
Only after replay agreement should a locomotion/residual policy be considered.
