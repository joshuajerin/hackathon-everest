# Ice simulation and sim-to-real funnel

## Claim boundary

MuJoCo does not contain a brittle ice material. A low Coulomb coefficient and a soft contact plane are
not enough. This project uses a hybrid boundary:

1. MuJoCo computes the G1/crampon rigid-body kinematics and the compliant probe mechanism.
2. `StatefulIceSpikeContact` computes per-spike indentation, damping, fracture force drop, ploughing,
   residual sliding, breakout, and persistent crater state.
3. Only the hardware-shaped 19 values enter the estimator. Exact penetration into ice, contact state,
   material parameters, and 3-D applied forces remain simulator diagnostics or labels.
4. The same packet and command boundary will be used by Isaac Sim and hardware adapters.

The current law is an evidence-grounded prior. It is not a validated model of the supplied spike CAD or
natural mountain ice. The actual spike/ice calibration rig is required before physical control.

## Evidence used for broad priors

| Quantity | Broad prior | Evidence and limitation |
|---|---:|---|
| Residual steel/ice sliding friction | `0.03–0.18`; randomization tail `0.005–0.24` | Marmo et al. measured walking-relevant speeds over `-27` to `-0.5 °C`; surface preparation can move results substantially. This is not total spike traction. |
| Ice indentation/hardness pressure | average `15–60 MPa`; hardness `35–65 MPa`; local pressure up to about `120 MPa` | Wells, Barnes/Tabor values reported by Marmo, and Liefferink. Natural defects and sharp geometry require wider uncertainty. |
| G1 static load | about `343 N` total for a nominal 35 kg robot | Unitree specification. Equal double support is about `43 N/spike`; unequal single support is about `86 N/spike` before dynamics. |
| Engineering rig envelope | `20–250 N/spike`, `0–1 kN/foot` | Deliberately wider than nominal static load. It is a test envelope, not a measured gait distribution. |
| Snow density | roughly `100–560 kg/m³` | Literature review plus the downloaded MOSAiC subset. Snow is modeled separately from hard ice. |

Primary references:

- Tusima (1977), steel ball on ice: <https://doi.org/10.3189/S0022143000029300>
- Marmo et al. (2005), temperature/speed/load effects: <https://doi.org/10.3189/172756505781829304>
- Liefferink et al. (2021), ice friction and ploughing: <https://doi.org/10.1103/PhysRevX.11.011025>
- Wells et al. (2010), ice crushing pressure: <https://doi.org/10.1016/j.coldregions.2010.11.002>
- Nakao et al. (2021), rate/shape-dependent ice indentation fracture: <https://doi.org/10.1299/mej.21-00083>
- Velkavrh et al. (2019), ice roughness and run-in effects: <https://doi.org/10.3390/lubricants7120106>
- Shoop et al. (2022), snow mechanics review: <https://doi.org/10.1016/j.jterra.2021.11.006>
- Schneebeli and Johnson (1998), 5 mm snow cone: <https://doi.org/10.3189/1998AoG26-1-107-111>
- MuJoCo contact model: <https://mujoco.readthedocs.io/en/stable/computation/index.html#contact>

The cited experiments use different geometry, speed, load, ice preparation, and temperature. Their values
set randomization bounds. They do not identify the final model.

## Implemented hard-ice law

For a rounded conical spike, the loading response uses

```text
contact radius = min(shank radius,
                     max(sqrt(2 * tip radius * depth), depth * tan(cone half-angle)))
normal force = indentation pressure * projected area + damping * positive depth rate
```

The response is capped by the chosen calibration envelope. Loading work accumulates. When sampled fracture
energy is exceeded, strength drops and a residual crater is stored. Tangential capacity is the sum of the
residual sliding term `mu * normal force` and a geometry-dependent ploughing term. Lateral displacement can
trigger breakout. Reusing a cell preserves the crater and damage.

This is intentionally stateful and stochastic. Native MuJoCo soft overlap is recoverable and cannot represent
that history.

## Calibration rig

Use the actual spike and complete foot, not a generic cone.

- Six-axis load cell, at least 1 kN vertical capacity.
- Vertical and lateral stages with at least 1 kHz acquisition; 5 kHz is preferable for fracture.
- Displacement resolution at or below `0.02 mm`.
- Record ice temperature, spike temperature, tip radius/angle/roughness, frost, water, salinity, bubbles,
  and whether the track is fresh or reused.
- For snow, use a bed at least five times deeper than maximum planned sinkage.

Normal test matrix:

- temperatures `-5, -10, -20 °C`;
- rates `0.002, 0.02, 0.2, 2 mm/s`, plus measured touchdown speed;
- loads `50, 100, 250, 500 N`;
- at least five fresh-location repeats and repeated loading at the same location.

Tangential test matrix:

- insertion depths `0, 2, 5, 10 mm` where geometry permits;
- loads `50, 100, 250, 500 N`;
- speeds `0.01, 0.05, 0.2, 0.5 m/s` in fore-aft and lateral directions;
- fresh and run-in tracks.

Fit complete force-depth and force-displacement curves, impulses, force drops, residual holes, peak and steady
shear, not only peak force. Hold out temperature, speed, load, and whole-foot trials. Keep posterior intervals
and randomize them in simulation.

## Sim-to-real stages and gates

1. **Geometry and packet gate — implemented.** Verify dimensions, axes, named contact ownership, sensor order,
   signs, units, rate, saturation, quantization, latency, and validity masks.
2. **Component calibration.** Identify load-cell, moving-probe, IMU, and radar bias/noise/latency independently.
3. **Material identification.** Fit ice and snow contact posterior ranges with the above rig.
4. **Trajectory matching.** Replay identical displacement/load commands in the rig and simulation. Compare full
   curves, event timing, penetration, impulse, shear, and hysteresis.
5. **Domain randomization.** Sample posterior material parameters plus mount pose, mass/CoM, actuator mapping,
   controller latency, temperature, roughness, defects, and sensor degradation.
6. **Estimator replay.** Train only from packet-visible histories. Split real evaluation by block/day/site/route,
   not neighboring traces.
7. **Hardware-in-the-loop.** Feed recorded packets through the production estimator and scheduler clock before
   any robot load transfer.
8. **Progressive robot tests.** Bench fixture, suspended one-foot contact, low-load double support, tethered flat
   ice, then slopes. Use independent safety limits at every stage.

## Isaac Sim boundary

Isaac Sim may replace MuJoCo kinematics and later provide richer rendering or deformable backends. It does not
change these contracts:

- `SimulatorObservation -> SynchronizedSensorPacket`;
- `ProbeCommand(load, speed, pose) -> simulator or robot`;
- material truth remains outside estimator input;
- calibration manifests retain units, geometry revision, distribution, and provenance;
- the exact same trained estimator is evaluated on logged packets from both simulators and hardware.

Moving to Isaac before these gates pass would change tools without fixing material identification.
