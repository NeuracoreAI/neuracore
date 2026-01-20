import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from neuracore_types import DataType
from neuracore_types.nc_data import DATA_TYPE_TO_NC_DATA_CLASS

import neuracore as nc
from neuracore.api.logging import ExperimentalPointCloudWarning
from neuracore.core.const import API_URL


@dataclass(frozen=True)
class TestConfigItem:
    __test__ = False  # Prevent pytest from collecting this as a test class.
    data: object
    logging_function: Callable
    expected_value: Callable
    suppress_warning: bool = False


class DummyData:
    joint_data = {"joint1": 0.1, "joint2": 0.2, "joint3": 0.3}
    parallel_gripper = {"name": "test_parallel_gripper", "value": 0.5}
    rgb_image = {
        "name": "test_rgb_camera",
        "image": np.zeros((100, 100, 3), dtype=np.uint8),
        "extrinsics": np.eye(4, dtype=np.float32),
        "intrinsics": np.eye(3, dtype=np.float32),
    }
    depth_image = {
        "name": "test_depth_camera",
        "image": np.ones((100, 100), dtype=np.float32),
        "extrinsics": np.eye(4, dtype=np.float32),
        "intrinsics": np.eye(3, dtype=np.float32),
    }
    end_effector_pose = {
        "name": "test_end_effector",
        "pose": np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    pose = {
        "name": "test_pose",
        "pose": np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    language = {"name": "test_language", "text": "a dummy statement"}
    custom_1d = {
        "name": "test_custom_1d",
        "data": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }
    point_cloud = {
        "name": "test_point_cloud",
        "points": np.zeros((10, 3), dtype=np.float16),
        "rgb_points": np.zeros((10, 3), dtype=np.uint8),
        "extrinsics": np.eye(4, dtype=np.float32),
        "intrinsics": np.eye(3, dtype=np.float32),
    }


def setup_test_config():

    return {
        DataType.JOINT_POSITIONS: TestConfigItem(
            data=DummyData.joint_data,
            logging_function=lambda data: nc.log_joint_positions(positions=data),
            expected_value=lambda data: {
                name: {"value": value} for name, value in data.items()
            },
        ),
        DataType.JOINT_TARGET_POSITIONS: TestConfigItem(
            data=DummyData.joint_data,
            logging_function=lambda data: nc.log_joint_target_positions(
                target_positions=data
            ),
            expected_value=lambda data: {
                name: {"value": value} for name, value in data.items()
            },
        ),
        DataType.JOINT_VELOCITIES: TestConfigItem(
            data=DummyData.joint_data,
            logging_function=lambda data: nc.log_joint_velocities(velocities=data),
            expected_value=lambda data: {
                name: {"value": value} for name, value in data.items()
            },
        ),
        DataType.JOINT_TORQUES: TestConfigItem(
            data=DummyData.joint_data,
            logging_function=lambda data: nc.log_joint_torques(torques=data),
            expected_value=lambda data: {
                name: {"value": value} for name, value in data.items()
            },
        ),
        DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: TestConfigItem(
            data=DummyData.parallel_gripper,
            logging_function=lambda data: nc.log_parallel_gripper_open_amount(
                name=data["name"],
                value=data["value"],
            ),
            expected_value=lambda data: {data["name"]: {"open_amount": data["value"]}},
        ),
        DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: TestConfigItem(
            data=DummyData.parallel_gripper,
            logging_function=lambda data: nc.log_parallel_gripper_target_open_amount(
                name=data["name"],
                value=data["value"],
            ),
            expected_value=lambda data: {data["name"]: {"open_amount": data["value"]}},
        ),
        DataType.RGB_IMAGES: TestConfigItem(
            data=DummyData.rgb_image,
            logging_function=lambda data: nc.log_rgb(
                name=data["name"],
                rgb=data["image"],
                extrinsics=data["extrinsics"],
                intrinsics=data["intrinsics"],
            ),
            expected_value=lambda data: {
                data["name"]: {
                    "frame": data["image"],
                    "extrinsics": data["extrinsics"].astype(np.float16),
                    "intrinsics": data["intrinsics"].astype(np.float16),
                }
            },
        ),
        DataType.DEPTH_IMAGES: TestConfigItem(
            data=DummyData.depth_image,
            logging_function=lambda data: nc.log_depth(
                name=data["name"],
                depth=data["image"],
                extrinsics=data["extrinsics"],
                intrinsics=data["intrinsics"],
            ),
            expected_value=lambda data: {
                data["name"]: {
                    "frame": data["image"],
                    "extrinsics": data["extrinsics"].astype(np.float16),
                    "intrinsics": data["intrinsics"].astype(np.float16),
                }
            },
        ),
        DataType.END_EFFECTOR_POSES: TestConfigItem(
            data=DummyData.end_effector_pose,
            logging_function=lambda data: nc.log_end_effector_pose(
                name=data["name"],
                pose=data["pose"],
            ),
            expected_value=lambda data: {data["name"]: {"pose": data["pose"]}},
        ),
        DataType.POSES: TestConfigItem(
            data=DummyData.pose,
            logging_function=lambda data: nc.log_pose(
                name=data["name"],
                pose=data["pose"],
            ),
            expected_value=lambda data: {data["name"]: {"pose": data["pose"]}},
        ),
        DataType.LANGUAGE: TestConfigItem(
            data=DummyData.language,
            logging_function=lambda data: nc.log_language(
                name=data["name"], language=data["text"]
            ),
            expected_value=lambda data: {data["name"]: {"text": data["text"]}},
        ),
        DataType.CUSTOM_1D: TestConfigItem(
            data=DummyData.custom_1d,
            logging_function=lambda data: nc.log_custom_1d(
                name=data["name"], data=data["data"]
            ),
            expected_value=lambda data: {data["name"]: {"data": data["data"]}},
        ),
        DataType.POINT_CLOUDS: TestConfigItem(
            data=DummyData.point_cloud,
            logging_function=lambda data: nc.log_point_cloud(
                name=data["name"],
                points=data["points"],
                rgb_points=data["rgb_points"],
                extrinsics=data["extrinsics"],
                intrinsics=data["intrinsics"],
            ),
            expected_value=lambda data: {
                data["name"]: {
                    "points": data["points"],
                    "rgb_points": data["rgb_points"],
                    "extrinsics": data["extrinsics"].astype(np.float16),
                    "intrinsics": data["intrinsics"].astype(np.float16),
                }
            },
            suppress_warning=True,
        ),
    }


test_configs = setup_test_config()
# Keep track of any datatypes that do not yet have a test config
MISSING_DATA_TYPES = set(DataType) - set(test_configs)


def log_mock_data():
    for test_config in test_configs.values():
        if test_config.suppress_warning:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ExperimentalPointCloudWarning)
                test_config.logging_function(test_config.data)
        else:
            test_config.logging_function(test_config.data)


def assert_field_equal(actual, expected) -> None:
    if expected is None:
        assert actual is None
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, (list, tuple)):
        np.testing.assert_array_equal(np.array(actual), np.array(expected))
    elif isinstance(expected, (float, int, np.floating, np.integer)):
        assert abs(float(actual) - float(expected)) < 1e-6
    else:
        assert actual == expected


def build_expected_data():
    return {
        data_type: test_config.expected_value(test_config.data)
        for data_type, test_config in test_configs.items()
    }


def test_log_and_retrieve_sync_point(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    mocked_org_id,
):
    """Test logging random data and retrieving via sync point."""
    if MISSING_DATA_TYPES:
        pytest.skip(
            f"Missing test configs for: {sorted(dt.value for dt in MISSING_DATA_TYPES)}"
        )

    nc.login("test_api_key")

    mock_auth_requests.post(
        re.compile(f"{API_URL}/org/[^/]+/robots(\\?.*)?"),
        json={"robot_id": "mock_robot_id", "has_urdf": False, "archived": False},
        status_code=200,
    )
    nc.connect_robot("test-robot")

    log_mock_data()

    expected_data = build_expected_data()

    assert set(expected_data.keys()) == set(
        DataType
    ), "Not all data types being tested."

    sync_point = nc.get_latest_sync_point(include_remote=False)

    for data_type in DataType:
        assert data_type in sync_point.data, f"Sync point missing {data_type} datatype"

        expected_data_for_type = expected_data[data_type]
        for name, expected_fields in expected_data_for_type.items():
            assert (
                name in sync_point.data[data_type]
            ), f"Sync point missing 'sensor label {name}' for {data_type}"

            expected_nc_data_class = DATA_TYPE_TO_NC_DATA_CLASS[data_type]
            nc_data = expected_nc_data_class.model_validate(
                sync_point.data[data_type][name]
            )

            for field_name, expected_value in expected_fields.items():
                assert hasattr(nc_data, field_name), (
                    f"Sync point missing field '{field_name}' "
                    f"for {expected_nc_data_class.__name__}"
                )
                assert_field_equal(getattr(nc_data, field_name), expected_value)
