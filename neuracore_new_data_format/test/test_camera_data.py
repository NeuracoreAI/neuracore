import os

import numpy as np

import neuracore_new_data_format as nc


def test_log_and_read_camera_data():
    dataset_name = "test_camera_data"
    recording_name = f"{dataset_name}-0"
    db_file = f"{recording_name}.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    camera_id = "test_camera"
    data_points = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
    try:
        nc.create_dataset(name=dataset_name)
        recording = nc.start_recording()
        for data_point in data_points:
            nc.log_rgb(camera_id=camera_id, image=data_point)
        nc.stop_recording()

        # Reopen the recording to read data
        recording = nc.recording_factory.create_recording(name=recording_name)
        read_data = list(recording.read_data(data_type="rgb"))

        assert len(read_data) == len(data_points)
        for i, data_point in enumerate(data_points):
            np.testing.assert_array_equal(read_data[i].frame, data_point)
            assert read_data[i].camera_id == camera_id
            assert read_data[i].timestamp is not None
            assert isinstance(read_data[i].timestamp, float)
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)


def test_log_and_read_camera_data_rgb():
    dataset_name = "test_camera_data"
    recording_name = f"{dataset_name}-0"
    db_file = f"{recording_name}.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    camera_id = "test_camera"
    np.random.seed(42)
    data_points = [
        np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8) for _ in range(100)
    ]

    try:
        nc.create_dataset(name=dataset_name)
        recording = nc.start_recording()
        for i, data_point in enumerate(data_points):
            nc.log_rgb(timestamp=i, camera_id=camera_id, image=data_point)
        nc.stop_recording()

        # Reopen the recording to read data
        recording = nc.recording_factory.create_recording(name=recording_name)
        read_data = list(recording.read_data(data_type="rgb"))

        assert len(read_data) == len(data_points)
        for i, data_point in enumerate(data_points):
            np.testing.assert_array_equal(read_data[i].frame, data_point)
            assert read_data[i].camera_id == camera_id
            assert read_data[i].timestamp is not None
            assert isinstance(read_data[i].timestamp, float)
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)
