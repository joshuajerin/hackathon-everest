# Editable Blender crampon fit

Open the original two-mesh fit scene:

```bash
open blender/g1_crampon_fit.blend
```

## Complete USD component assembly

The complete USD shoe/crampon/component assembly is available in a separate editable scene:

```bash
open blender/g1_crampon_components_fit.blend
```

Select `USD_ASSET_POSITION_CONTROL` and use normal `G`, `R`, and `S` transforms to move both complete
assemblies. Use `LEFT_USD_FINE_TUNE` or `RIGHT_USD_FINE_TUNE` for one-foot corrections. Every imported
component is selectable under `04_EDITABLE_USD_COMPONENT_ASSEMBLY`.

The source USDC contains 26 mesh objects but no humanoid or ankle prim. The generator aligns its 34,265-vertex
main frame to the saved G1 fit, preserving every USD component's relative transform. Alignment provenance and
error are recorded in `assets/crampon/usd_component_fit_metadata.json`.

Regenerate it with:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python blender/setup_usd_component_fit.py -- \
  --base-blend blender/g1_crampon_fit.blend \
  --source-usd assets/crampon/g1_crampon_components_source.usdc \
  --output blender/g1_crampon_components_fit.blend \
  --render blender/g1_crampon_components_fit_preview.png \
  --metadata assets/crampon/usd_component_fit_metadata.json
```

This USD-component scene and `assets/crampon/g1_crampon_components_source.usdc` are the asset authority for
future Isaac Lab deployment. Isaac Lab must not import the old two-mesh fitted STL asset. The old
`g1_crampon_fit.blend` path remains only for current MuJoCo compatibility until that exporter is replaced.
`configs/isaaclab/g1_crampon_asset.yaml` records this boundary in a machine-readable form.

## Easiest adjustment

`CRAMPON_FIT_CONTROL` is selected when the file opens. It is a normal Blender transform object with no
drivers or custom-property UI.

- Press `G` to move both crampons.
- Press `R` to rotate both crampons.
- Press `S` to scale both crampons uniformly.
- Press `N` and use the **Item → Transform** fields for exact values.

Its initial Scale is `1.08, 1.08, 1.08`. Both crampons follow this control around their own ankle origin, so
scaling does not change the distance between the robot's legs.

For one-foot corrections, select `LEFT_FINE_TUNE` or `RIGHT_FINE_TUNE` and use the same normal transform tools.

## Edit the actual mesh

The actual parts are selectable and have clear names:

- `EDITABLE__LEFT_MOUNT_PLATE`
- `EDITABLE__LEFT_CRAMPON_FRAME`
- `EDITABLE__RIGHT_MOUNT_PLATE`
- `EDITABLE__RIGHT_CRAMPON_FRAME`

Select one and press `Tab` for Edit Mode if you need to change vertices. The gray G1 reference geometry and
fixed source-axis conversion are locked to prevent accidental edits.

The red probes and side instruction text were removed. Cameras, lights, and the floor do not intercept
selection.

## Scene layout

- `01_EDITABLE_FIT`: shared control, per-foot fine controls, mount, and frame meshes;
- `02_G1_REFERENCE`: locked official knee, ankle-pitch, and ankle-roll meshes;
- `03_GUIDES_CAMERAS`: locked tip plane, lights, and cameras;
- `99_INTERNAL_SOURCE_TEMPLATE`: hidden original two-component source mesh.

Cameras are `CAMERA__FIT_PERSPECTIVE`, `CAMERA__TOP`, and `CAMERA__SIDE`.

## Regenerate

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  --python blender/setup_crampon_fit.py -- \
  --source-stl /Users/joshuajerin/Downloads/Shoe_with_crampons_separated.stl \
  --menagerie vendor/mujoco_menagerie/unitree_g1 \
  --output blender/g1_crampon_fit.blend \
  --render blender/g1_crampon_fit_preview.png
```

## Apply the saved fit to MuJoCo

Save the Blender file, close or leave Blender open, then run:

```bash
./scripts/apply_blender_fit.sh
```

This exports evaluated fitted meshes in G1 ankle-local metres, writes
`assets/crampon/blender_fit_metadata.json`, synchronizes the standalone fixture, rebuilds the official G1
model, and runs the focused MuJoCo tests. It never modifies the source STL.
