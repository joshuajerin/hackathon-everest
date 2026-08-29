import numpy as np

from hackathon_everest.models import SENSOR_CHANNELS, Proprioception, SynchronizedSensorPacket


def test_sensor_packet_is_exactly_19_channels():
    packet = SynchronizedSensorPacket(
        timestamp_s=0.0,
        axial_force_n=np.ones(4),
        penetration_m=np.ones(4) * 0.01,
        accelerometer_mps2=np.zeros(3),
        gyroscope_rps=np.zeros(3),
        radar_frontend=np.zeros(5),
        valid_mask=np.ones(19, dtype=bool),
        proprioception=Proprioception(
            foot_position_xyz_m=np.zeros(3),
            foot_velocity_xyz_mps=np.zeros(3),
            pelvis_roll_pitch_yaw_rad=np.zeros(3),
            commanded_probe_load_n=132.0,
            commanded_foot_speed_mps=0.2,
            body_weight_on_foot_n=120.0,
        ),
    )
    assert SENSOR_CHANNELS == 19
    assert packet.vector().shape == (19,)
