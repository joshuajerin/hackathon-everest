# Selected Newton MPM discrepancy test

This directory is intentionally separate from the Isaac production environment. It pins
Newton `1.5.1` (`17c82b57c0cf369ee23baa776636fc633b82ccfa`) for one offline,
snow-only single-spike indentation, unload, and recontact comparison.

## Claim boundary

This is not ground truth, an Everest digital twin, an ice validation, hardware calibration,
or a runtime policy input. The continuum parameters are exploratory Newton example priors.
The analytical contact law uses effective `N/m`, `N*s/m`, and force caps, while Newton uses
continuum stresses in Pa. There is no defensible direct mapping without rig data. Fitting one
simulator to the other would be circular.

Newton v1.5.1 exposes elastoplastic MPM state such as `particle_Jp`, but it does not implement
the project's brittle crust/ice fracture energy, void, post-fracture strength drop, or
breakout state. Therefore this test is limited to intact consolidated snow.

## Required comparison

Run the same prescribed `0 -> 12 mm -> 0 -> 12 mm` trajectory at `0.02 m/s` with 2 mm and
1 mm voxels. Collect collider impulse reaction, depth, work, hysteresis, second/first peak
ratio, `particle_Jp`, and residual crater. Report voxel disagreement before comparing the
analytical trace. A result is a discrepancy measurement, not validation of either model.

Planned native command:

```bash
uv run --project validation/newton_mpm --locked \
  python validation/newton_mpm/compare_selected_case.py \
  --case western_cwm_consolidated_snow_single_spike_recontact \
  --device cuda:0 \
  --output artifacts/validation/newton_mpm/v1.5.1/consolidated_snow
```

Do not add Newton to the root project or Isaac Lab environment. `collect_collider_impulses`
outputs remain offline and must never enter the visible `[B,T,2,19]` ABI.
