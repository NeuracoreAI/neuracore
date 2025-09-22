import os

import neuracore_new_data_format as nc


def test_log_and_read_joint_positions():
    dataset_name = "test_joint_positions"
    recording_name = f"{dataset_name}-0"
    db_file = f"{recording_name}.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    data_points = [{"j1": 1, "j2": 2}, {"j1": 3, "j2": 4}]
    try:
        nc.create_dataset(name=dataset_name)
        recording = nc.start_recording()
        for data_point in data_points:
            nc.log_joint_positions(positions=data_point)
        nc.stop_recording()

        # Reopen the recording to read data
        recording = nc.recording_factory.create_recording(name=recording_name)
        read_data = list(recording.read_data(data_type="joint_positions"))

        assert len(read_data) == len(data_points)
        for i, data_point in enumerate(data_points):
            assert read_data[i].values == data_point
            assert read_data[i].timestamp is not None
            assert isinstance(read_data[i].timestamp, float)
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)


def test_log_and_read_joint_velocities():
    dataset_name = "test_joint_velocities"
    recording_name = f"{dataset_name}-0"
    db_file = f"{recording_name}.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    data_points = [{"j1": 0.1, "j2": 0.2}, {"j1": 0.3, "j2": 0.4}]
    try:
        nc.create_dataset(name=dataset_name)
        recording = nc.start_recording()
        for data_point in data_points:
            nc.log_joint_velocities(velocities=data_point)
        nc.stop_recording()

        # Reopen the recording to read data
        recording = nc.recording_factory.create_recording(name=recording_name)
        read_data = list(recording.read_data(data_type="joint_velocities"))

        assert len(read_data) == len(data_points)
        for i, data_point in enumerate(data_points):
            assert read_data[i].values == data_point
            assert read_data[i].timestamp is not None
            assert isinstance(read_data[i].timestamp, float)
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)


def test_log_and_read_joint_target_positions():
    dataset_name = "test_joint_target_positions"
    recording_name = f"{dataset_name}-0"
    db_file = f"{recording_name}.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    data_points = [{"j1": 1, "j2": 2}, {"j1": 3, "j2": 4}]
    try:
        nc.create_dataset(name=dataset_name)
        recording = nc.start_recording()
        for data_point in data_points:
            nc.log_joint_target_positions(target_positions=data_point)
        nc.stop_recording()

        # Reopen the recording to read data
        recording = nc.recording_factory.create_recording(name=recording_name)
        read_data = list(recording.read_data(data_type="joint_target_positions"))

        assert len(read_data) == len(data_points)
        for i, data_point in enumerate(data_points):
            assert read_data[i].values == data_point
            assert read_data[i].timestamp is not None
            assert isinstance(read_data[i].timestamp, float)
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)
