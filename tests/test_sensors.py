import numpy as np

from hackathon_everest.physics import ReducedOrderContactBackend
from hackathon_everest.sensors import SensorNoiseConfig, SensorSimulator
from hackathon_everest.terrain import TerrainGenerator


def test_sensor_values_are_hardware_shaped_and_radar_is_quantized():
    field = TerrainGenerator().generate(17)
    truth = ReducedOrderContactBackend().probe(field, 0.0, 0.0, seed=2, mutate=False)
    config = SensorNoiseConfig(radar_resolution_m=0.04)
    packets = SensorSimulator(config).packets(truth, seed=3)
    assert all(packet.vector().shape == (19,) for packet in packets)
    radar_depth = packets[-1].radar_frontend[0]
    assert np.isclose(radar_depth / 0.04, round(radar_depth / 0.04))
    # Full 3-D contact force remains diagnostics and is not in the packet.
    assert truth.contact_force_world_n.shape[-1] == 3
    assert not hasattr(packets[-1], "contact_force_world_n")



def test_dropouts_are_explicit_in_validity_mask():
    field = TerrainGenerator().generate(21)
    truth = ReducedOrderContactBackend().probe(field, 0.0, 0.0, seed=4, mutate=False)
    packets = SensorSimulator(SensorNoiseConfig(sample_drop_probability=1.0)).packets(truth, seed=5)
    assert packets[0].valid_mask.all()
    assert all(not packet.valid_mask[:8].any() for packet in packets[1:])
    assert all(packet.valid_mask[8:].all() for packet in packets)
