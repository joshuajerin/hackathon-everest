from __future__ import annotations

import numpy as np

from hackathon_everest.ice import IceContactParameters, StatefulIceSpikeContact


def test_ice_parameter_sampling_is_reproducible_and_inside_evidence_priors() -> None:
    first = IceContactParameters.sample(19)
    second = IceContactParameters.sample(19)
    assert first == second
    assert -25.0 <= first.temperature_c <= -1.0
    assert 0.03 <= first.sliding_friction <= 0.18
    assert 12.0e6 <= first.indentation_pressure_pa <= 73.0e6


def test_ice_contact_is_stateful_bounded_and_fractures() -> None:
    params = IceContactParameters(
        temperature_c=-10.0,
        indentation_pressure_pa=40e6,
        sliding_friction=0.08,
        tip_radius_m=0.0004,
        cone_half_angle_rad=np.deg2rad(30.0),
        shank_radius_m=0.003,
        normal_damping_ns_per_m=40.0,
        fracture_energy_j=0.02,
        post_fracture_strength_ratio=0.4,
        breakout_displacement_m=0.003,
    )
    contact = StatefulIceSpikeContact(params)
    forces = []
    for depth in np.linspace(0.0, 0.006, 61):
        response = contact.step(depth, 0.02, lateral_speed_mps=0.0, dt_s=0.005)
        forces.append(response.normal_force_n)
    assert np.isfinite(forces).all()
    assert min(forces) >= 0.0
    assert max(forces) <= params.maximum_force_n
    assert response.fractured
    assert contact.state.residual_crater_depth_m > 0.0

    crater = contact.state.residual_crater_depth_m
    contact.lift_and_reposition(preserve_terrain_memory=True)
    repeated = contact.step(crater * 0.5, 0.01, dt_s=0.005)
    assert repeated.fractured
    assert repeated.normal_force_n < max(forces)
