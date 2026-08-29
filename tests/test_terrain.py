import numpy as np

from hackathon_everest.terrain import TerrainGenerator


def test_field_is_deterministic_and_spatially_correlated():
    first = TerrainGenerator().generate(11)
    second = TerrainGenerator().generate(11)
    values = first.arrays["vertical_stiffness_n_per_m"]
    np.testing.assert_array_equal(values, second.arrays["vertical_stiffness_n_per_m"])
    neighbor_difference = np.mean(np.abs(values[1:, :] - values[:-1, :]))
    far_difference = np.mean(np.abs(values[10:, :] - values[:-10, :]))
    assert neighbor_difference < far_difference


def test_contact_state_persists():
    field = TerrainGenerator().generate(13)
    i, j = field.index(0.0, 0.0)
    before_height = field.arrays["surface_height_m"][i, j]
    field.apply_contact(0.0, 0.0, peak_load_n=500.0, penetration_m=0.04, fractured=True)
    assert field.arrays["compaction"][i, j] > 0.0
    assert field.arrays["damage"][i, j] > 0.0
    assert field.arrays["fractured"][i, j]
    assert field.arrays["surface_height_m"][i, j] < before_height
