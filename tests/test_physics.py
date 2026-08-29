import numpy as np

from hackathon_everest.physics import (
    SPIKE_OFFSETS_M,
    ProbeConfig,
    ReducedOrderContactBackend,
    aggregate_bearing_capacity_n,
)
from hackathon_everest.terrain import TerrainGenerator


def test_recontact_observes_persistent_spike_cell_changes():
    field = TerrainGenerator().generate(31)
    backend = ReducedOrderContactBackend()
    first = backend.probe(field, 0.0, 0.0, seed=9, mutate=True)
    second = backend.probe(field, 0.0, 0.0, seed=9, mutate=False)
    assert np.max(np.abs(first.penetration_m - second.penetration_m)) > 0.00025
    compacted = [field.point(dx, dy).compaction for dx, dy in SPIKE_OFFSETS_M]
    assert min(compacted) > 0.0


def test_probe_load_and_speed_change_contact_history():
    backend = ReducedOrderContactBackend()
    low_field = TerrainGenerator().generate(33)
    high_field = TerrainGenerator().generate(33)
    low = backend.probe(
        low_field,
        0.1,
        0.1,
        seed=12,
        config=ProbeConfig(commanded_load_n=70.0, approach_speed_mps=0.08),
        mutate=False,
    )
    high = backend.probe(
        high_field,
        0.1,
        0.1,
        seed=12,
        config=ProbeConfig(commanded_load_n=220.0, approach_speed_mps=0.30),
        mutate=False,
    )
    assert high.axial_force_n.max() > low.axial_force_n.max()
    assert high.penetration_m[-1].mean() > low.penetration_m[-1].mean()


def test_crust_fracture_depends_on_commanded_load():
    backend = ReducedOrderContactBackend()
    low_field = TerrainGenerator().generate(35)
    high_field = TerrainGenerator().generate(35)
    for field in (low_field, high_field):
        for dx, dy in SPIKE_OFFSETS_M:
            i, j = field.index(dx, dy)
            field.arrays["crust_thickness_m"][i, j] = 0.006
            field.arrays["fracture_strength_n"][i, j] = 260.0
            field.arrays["bearing_capacity_n"][i, j] = 650.0
            field.arrays["vertical_stiffness_n_per_m"][i, j] = 25_000.0
            field.arrays["fractured"][i, j] = False
    low = backend.probe(
        low_field,
        0.0,
        0.0,
        seed=14,
        config=ProbeConfig(commanded_load_n=120.0, approach_speed_mps=0.2),
        mutate=False,
    )
    high = backend.probe(
        high_field,
        0.0,
        0.0,
        seed=14,
        config=ProbeConfig(commanded_load_n=320.0, approach_speed_mps=0.2),
        mutate=False,
    )
    assert not low.events["fractured"]
    assert high.events["fractured"]



def test_post_probe_label_matches_four_spike_truth():
    field = TerrainGenerator().generate(37)
    truth = ReducedOrderContactBackend().probe(field, 0.2, -0.1, seed=16, mutate=True)
    expected = aggregate_bearing_capacity_n(field, 0.2, -0.1)
    assert np.isclose(truth.labels["bearing_capacity_n"], expected)



def test_prefix_state_does_not_include_a_future_fracture():
    field = TerrainGenerator().generate(39)
    for dx, dy in SPIKE_OFFSETS_M:
        i, j = field.index(dx, dy)
        field.arrays["crust_thickness_m"][i, j] = 0.006
        field.arrays["fracture_strength_n"][i, j] = 260.0
        field.arrays["bearing_capacity_n"][i, j] = 650.0
        field.arrays["vertical_stiffness_n_per_m"][i, j] = 25_000.0
        field.arrays["fractured"][i, j] = False
    backend = ReducedOrderContactBackend()
    truth = backend.probe(
        field,
        0.0,
        0.0,
        seed=18,
        config=ProbeConfig(commanded_load_n=320.0, approach_speed_mps=0.2),
        mutate=False,
    )
    fracture_ticks = truth.fracture_tick_by_spike[truth.fracture_tick_by_spike >= 0]
    first_fracture = int(fracture_ticks.min())
    assert first_fracture > 1

    early_field = field.copy()
    backend.apply_episode_prefix(early_field, truth, first_fracture - 1)
    early_labels, early_events = backend.labels_at_prefix(
        early_field, truth, first_fracture - 1
    )
    final_field = field.copy()
    backend.apply_episode_prefix(final_field, truth, len(truth.timestamps_s) - 1)
    final_labels, final_events = backend.labels_at_prefix(
        final_field, truth, len(truth.timestamps_s) - 1
    )

    assert not early_events["fractured"]
    assert final_events["fractured"]
    assert early_labels["damage_state"] < final_labels["damage_state"]
    assert early_labels["fracture_margin_n"] > final_labels["fracture_margin_n"]
