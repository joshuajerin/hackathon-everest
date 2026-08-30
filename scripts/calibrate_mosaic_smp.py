#!/usr/bin/env python3
"""Summarize a MOSAiC SMP subset into auditable snow calibration priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import median_filter


def quantiles(values: np.ndarray, probabilities: tuple[float, ...]) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {f"p{int(probability * 100):02d}": float(np.quantile(values, probability)) for probability in probabilities}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, nargs="?", default=Path("data/external/mosaic_smp"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/calibration/mosaic_smp"))
    parser.add_argument("--crampon-tip-mm", type=float, default=6.0)
    args = parser.parse_args()
    try:
        from snowmicropyn import Profile
    except ImportError as error:
        raise RuntimeError("Run with: uv run --extra calibration python scripts/calibrate_mosaic_smp.py") from error

    manifest = json.loads((args.data / "manifest.json").read_text())
    force_samples = []
    density_samples = []
    positive_slopes = []
    profiles = []
    failures = []
    for path in sorted(args.data.glob("*.pnt")):
        try:
            profile = Profile.load(path)
            surface_mm = float(profile.detect_surface())
            ground_mm = float(profile.detect_ground())
            snow = profile.samples_within_snowpack()
            distance_mm = snow["distance"].to_numpy(dtype=float)
            force_n = np.clip(snow["force"].to_numpy(dtype=float), 0.0, None)
            if len(force_n) < 500:
                raise ValueError(f"only {len(force_n)} snowpack samples")
            smoothed = median_filter(force_n, size=min(251, len(force_n) // 2 * 2 + 1), mode="nearest")
            slope_n_per_m = np.gradient(smoothed, distance_mm / 1000.0)
            positive_slopes.append(slope_n_per_m[slope_n_per_m > 0])
            force_samples.append(force_n)
            derivatives = profile.calc_derivatives(snowpack_only=True)
            density_column = next(column for column in derivatives.columns if "density" in column)
            density = derivatives[density_column].to_numpy(dtype=float)
            density_samples.append(density[(density > 10.0) & (density < 1000.0)])
            profiles.append(
                {
                    "filename": path.name,
                    "timestamp": str(profile.timestamp),
                    "surface_mm": surface_mm,
                    "ground_mm": ground_mm,
                    "snow_depth_mm": ground_mm - surface_mm,
                    "sample_count": len(force_n),
                    "force_quantiles_n": quantiles(force_n, (0.5, 0.75, 0.9, 0.95, 0.99)),
                    "penetration_work_j": float(np.trapezoid(force_n, distance_mm / 1000.0)),
                }
            )
        except (IndexError, KeyError, OSError, RuntimeError, ValueError) as error:
            failures.append({"filename": path.name, "error": str(error)})

    if not profiles:
        raise RuntimeError(f"No profiles could be calibrated: {failures}")
    forces = np.concatenate(force_samples)
    densities = np.concatenate(density_samples)
    slopes = np.concatenate(positive_slopes)
    force_q = quantiles(forces, (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
    density_q = quantiles(densities, (0.05, 0.25, 0.5, 0.75, 0.95))
    slope_q = quantiles(slopes, (0.5, 0.75, 0.9, 0.95))
    area_ratio = (args.crampon_tip_mm / 5.0) ** 2
    summary = {
        "source": {
            "dataset": manifest["dataset"],
            "doi": manifest["doi"],
            "license": manifest["license"],
            "citation": manifest["citation"],
            "downloaded_profiles": len(manifest["files"]),
            "usable_profiles": len(profiles),
            "download_failures": len(manifest["failures"]),
            "parse_failures": failures,
        },
        "observed_smp": {
            "tip_diameter_mm": 5.0,
            "measurement_speed_mm_per_s": 20.0,
            "force_quantiles_n": force_q,
            "positive_smoothed_gradient_quantiles_n_per_m": slope_q,
            "P2015_density_quantiles_kg_per_m3": density_q,
            "snow_depth_quantiles_mm": quantiles(
                np.asarray([profile["snow_depth_mm"] for profile in profiles]), (0.05, 0.5, 0.95)
            ),
        },
        "engineering_translation": {
            "crampon_probe_tip_diameter_mm": args.crampon_tip_mm,
            "projected_area_ratio_vs_smp": area_ratio,
            "area_scaled_force_quantiles_n": {key: value * area_ratio for key, value in force_q.items()},
            "use": "Prior/noise calibration and domain randomization only, not synchronized traction labels.",
            "caveat": (
                "Area scaling is an explicit weak assumption. Crampon angle, speed, compaction zone, "
                "shear, and fracture differ from the 5 mm SMP cone."
            ),
        },
        "profiles": profiles,
    }
    priors = {
        "schema_version": 1,
        "source_doi": manifest["doi"],
        "evidence_level": "weak_material_prior_not_crampon_ground_truth",
        "snow": {
            "density_kg_per_m3": [density_q["p05"], density_q["p95"]],
            "single_probe_resistance_n": [
                force_q["p05"] * area_ratio,
                force_q["p95"] * area_ratio,
            ],
            "positive_stiffness_n_per_m": [slope_q["p50"] * area_ratio, slope_q["p95"] * area_ratio],
            "probe_speed_mps": 0.020,
        },
        "required_randomization": [
            "temperature",
            "density_and_microstructure",
            "crust_and_layer_depth",
            "tip_geometry",
            "approach_speed",
            "force_depth_scale",
            "hysteresis_and_repeated_footfall",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "material_priors.yaml").write_text(yaml.safe_dump(priors, sort_keys=False))
    print(json.dumps({"usable_profiles": len(profiles), "force_quantiles_n": force_q, "density_quantiles": density_q, "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
