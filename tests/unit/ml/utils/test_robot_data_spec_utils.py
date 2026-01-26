import pytest
from neuracore_types import DataType

from neuracore.core.utils.robot_data_spec_utils import (
    convert_robot_data_spec_names_to_ids,
)
from neuracore.core.utils.robot_mapping import RobotMapping


def test_convert_robot_data_spec_names_to_ids_accepts_name_or_id():
    mapping = RobotMapping("test-org")
    mapping._id_to_name = {"robot_id": "robot_name"}
    mapping._name_to_ids = {"robot_name": ["robot_id"]}
    mapping._initialized = True
    spec = {
        "robot_name": {DataType.RGB_IMAGES: ["cam"]},
        "robot_id": {DataType.JOINT_POSITIONS: ["j0"]},
    }
    resolved = convert_robot_data_spec_names_to_ids(spec, mapping)

    assert set(resolved) == {"robot_id"}
    assert DataType.RGB_IMAGES in resolved["robot_id"]
    assert DataType.JOINT_POSITIONS in resolved["robot_id"]


def test_convert_robot_data_spec_names_to_ids_merges_and_dedupes():
    mapping = RobotMapping("test-org")
    mapping._id_to_name = {"robot_id": "robot_name"}
    mapping._name_to_ids = {"robot_name": ["robot_id"]}
    mapping._initialized = True
    spec = {
        "robot_name": {DataType.RGB_IMAGES: ["cam", "cam2"]},
        "robot_id": {DataType.RGB_IMAGES: ["cam2", "cam3"]},
    }

    resolved = convert_robot_data_spec_names_to_ids(spec, mapping)

    assert set(resolved) == {"robot_id"}
    assert resolved["robot_id"][DataType.RGB_IMAGES] == ["cam", "cam2", "cam3"]


def test_convert_robot_data_spec_names_to_ids_allows_unknown_robot():
    mapping = RobotMapping("test-org")
    mapping._id_to_name = {"robot_id": "robot_name"}
    mapping._name_to_ids = {"robot_name": ["robot_id"]}
    mapping._initialized = True
    spec = {"unknown": {DataType.RGB_IMAGES: ["cam"]}}

    resolved = convert_robot_data_spec_names_to_ids(spec, mapping)
    assert resolved == spec


def test_convert_robot_data_spec_names_to_ids_allows_missing():
    mapping = RobotMapping("test-org")
    mapping._id_to_name = {"robot_id": "robot_name"}
    mapping._name_to_ids = {"robot_name": ["robot_id"]}
    mapping._initialized = True
    spec = {"missing": {DataType.RGB_IMAGES: ["cam"]}}

    resolved = convert_robot_data_spec_names_to_ids(spec, mapping)
    assert resolved == spec


def test_convert_robot_data_spec_names_to_ids_raises_on_ambiguous_name():
    mapping = RobotMapping("test-org")
    mapping._id_to_name = {"robot_id_1": "dup_name", "robot_id_2": "dup_name"}
    mapping._name_to_ids = {"dup_name": ["robot_id_1", "robot_id_2"]}
    mapping._initialized = True
    spec = {"dup_name": {DataType.RGB_IMAGES: ["cam"]}}

    with pytest.raises(Exception):
        convert_robot_data_spec_names_to_ids(
            spec,
            mapping,
        )


def test_convert_robot_data_spec_names_to_ids_raises_on_name_id_collision():
    mapping = RobotMapping("test-org")
    mapping._id_to_name = {"robot_id_1": "robot_name", "robot_id_2": "robot_id_1"}
    mapping._name_to_ids = {
        "robot_name": ["robot_id_1"],
        "robot_id_1": ["robot_id_2"],
    }
    mapping._initialized = True
    spec = {"robot_id_1": {DataType.RGB_IMAGES: ["cam"]}}

    with pytest.raises(Exception):
        convert_robot_data_spec_names_to_ids(
            spec,
            mapping,
        )
