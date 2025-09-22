import os

import neuracore_new_data_format as nc


def test_log_and_read_custom_data():
    dataset_name = "test_custom_data"
    recording_name = f"{dataset_name}-0"
    db_file = f"{recording_name}.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    data_points = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6, "c": 7}]
    try:

        nc.create_dataset(name=dataset_name)
        recording = nc.start_recording()
        for data_point in data_points:
            nc.log_custom_data(name="test_data", data=data_point)
        nc.stop_recording()

        # Reopen the recording to read data
        recording = nc.recording_factory.create_recording(name=recording_name)
        read_data = list(recording.read_data(data_type="custom"))

        assert len(read_data) == len(data_points)
        for i, data_point in enumerate(data_points):
            assert read_data[i].data == data_point
            assert read_data[i].timestamp is not None
            assert isinstance(read_data[i].timestamp, float)

    finally:
        if os.path.exists(db_file):
            os.remove(db_file)
