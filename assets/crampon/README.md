# User-supplied crampon asset

Source received locally as `Shoe_with_crampons_separated.stl`. The original file is not modified by the build.
The normalized derivatives in this directory are used by MuJoCo.

## Provenance and inferred units

- source SHA-256: `07e4120e021883a42871019cdeaefe54f64126af8b99b7944f51d484b88e48bb`
- binary STL triangles: `85,951`
- connected components: `2`
- raw extents: `0.0009274 × 0.0026182 × 0.0006039` unitless STL units
- base dimension-inference scale: `100`
- G1 cavity fit factor: `1.08`
- final uniform scale: `108`
- normalized extents: `0.10016 × 0.28277 × 0.06522 m`
- normalized axes: G1 `+X` forward/toe, `+Y` left, `+Z` up
- source transform: source `+Y -> G1 +X`, source `+X -> G1 -Y`, `+Z` unchanged

STL does not encode units. Scale 100 yields plausible shoe dimensions but leaves less than 0.1 mm lateral
bounding-box clearance around the official ankle-roll mesh. The final model uniformly enlarges the assembly
by 1.08. At the measured shallow-cavity section this gives approximately `226.24 × 81.82 mm` versus the
ankle-roll mesh's `208.21 × 75.58 mm`: about 9.0 mm fore-aft and 3.1 mm lateral clearance per side. This is a
visual bounding-box fit, not a collision-free CAD or fabrication-tolerance proof. Confirm a physical dimension
before fabrication or control. Do not fit axes independently.

The two components are written as:

- `mount_plate.stl`: 17,337 triangles;
- `crampon_frame.stl`: 68,614 triangles.

Regenerate and verify metadata with:

```bash
uv run python scripts/prepare_crampon_asset.py \
  /path/to/Shoe_with_crampons_separated.stl --out assets/crampon
```

## Simulation use

Both meshes are visual-only (`contype=0`, `conaffinity=0`, `density=0`). Triangle-mesh contact would hide
which spike carried load and would be slow and fragile. Four named analytical probes follow the estimator's
fixed order:

```text
0 (+0.081, +0.0486) m   1 (+0.081, -0.0486) m
2 (-0.081, +0.0486) m   3 (-0.081, -0.0486) m
```

These are sensor/load proxies, not a claim that their centers exactly match four tips in the CAD. Each is a
3 mm radius capsule on a 20 mm spring-loaded slide. The visual lowest point and unloaded analytical tips both
lie near ankle-local `z=-0.0385 m`.

The assembly attaches directly to official bodies `left_ankle_roll_link` and `right_ankle_roll_link`. The
stock four foot spheres and ankle-pitch collision mesh are removed from the generated instrumented model so
they cannot bypass the named probes. Official source files remain unmodified in the pinned sparse checkout.

The asset was supplied by the project owner. No public redistribution license was provided; confirm rights
before making a repository containing the derivative meshes public.

## Current saved Blender fit

`blender/g1_crampon_fit.blend` is now the fit authority. The current saved shared control is:

```text
location = (+0.020428015, +0.000219768, -0.040249769) m
rotation = (0, 0, 0) rad
scale    = (1.080000043, 1.080000043, 1.080000043)
```

`crampon_frame_fitted.stl` and `mount_plate_fitted.stl` contain the evaluated result in G1 ankle-roll local
coordinates. `blender_fit_metadata.json` records the source `.blend` SHA-256, final bounds, sensor-site
locations, and analytical probe transforms. Run `./scripts/apply_blender_fit.sh` after every saved Blender
change.
