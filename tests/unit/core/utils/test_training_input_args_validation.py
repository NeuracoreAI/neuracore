import pytest
from neuracore_types import DataType

from neuracore.core.utils.robot_mapping import RobotMapping
from neuracore.core.utils.training_input_args_validation import (
    validate_algorithm_exists,
    validate_data_specs,
    validate_robot_existence,
)


class FakeDataset:
    def __init__(self, data_types, full_specs, robot_ids, robot_name_to_id=None):
        self.data_types = data_types
        self._full_specs = full_specs
        self.robot_ids = robot_ids
        mapping = RobotMapping("test-org")
        name_to_id = robot_name_to_id or {}
        mapping._id_to_name = {rid: name for name, rid in name_to_id.items()}
        mapping._name_to_ids = {}
        for name, rid in name_to_id.items():
            mapping._name_to_ids.setdefault(name, []).append(rid)
        mapping._initialized = True
        self.robot_mapping = mapping

    def get_full_data_spec(self, robot_id):
        return self._full_specs[robot_id]


def test_validate_data_specs_rejects_missing_data_values():
    dataset = FakeDataset(
        data_types={DataType.RGB_IMAGES},
        full_specs={"robot_1": {DataType.RGB_IMAGES: ["front"]}},
        robot_ids=["robot_1"],
    )
    robot_data_spec = {"robot_1": {DataType.RGB_IMAGES: ["front", "side"]}}

    with pytest.raises(ValueError, match="data values .* not present in dataset"):
        validate_data_specs(
            dataset=dataset,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            robot_data_spec=robot_data_spec,
            supported_data_types={DataType.RGB_IMAGES},
            spec_kind="input",
        )


def test_validate_data_specs_allows_subset_of_dataset_values():
    dataset = FakeDataset(
        data_types={DataType.RGB_IMAGES},
        full_specs={"robot_1": {DataType.RGB_IMAGES: ["front", "side"]}},
        robot_ids=["robot_1"],
    )
    robot_data_spec = {"robot_1": {DataType.RGB_IMAGES: ["front"]}}

    validate_data_specs(
        dataset=dataset,
        dataset_name="test-dataset",
        algorithm_name="test-algorithm",
        robot_data_spec=robot_data_spec,
        supported_data_types={DataType.RGB_IMAGES},
        spec_kind="input",
    )


def test_validate_algorithm_exists_raises_when_missing():
    with pytest.raises(ValueError, match="Algorithm .* not found"):
        validate_algorithm_exists(None, "MissingAlgorithm")


def test_validate_robot_existence_rejects_missing_robot_ids():
    dataset = FakeDataset(
        data_types={DataType.RGB_IMAGES},
        full_specs={"robot_1": {DataType.RGB_IMAGES: ["front"]}},
        robot_ids=["robot_1"],
    )
    input_robot_data_spec = {"robot_2": {DataType.RGB_IMAGES: ["front"]}}
    output_robot_data_spec = {"robot_1": {DataType.RGB_IMAGES: ["front"]}}

    with pytest.raises(ValueError, match="Robot .*not found in dataset"):
        validate_robot_existence(
            dataset=dataset,
            dataset_name="test-dataset",
            input_robot_data_spec=input_robot_data_spec,
            output_robot_data_spec=output_robot_data_spec,
        )


def test_validate_robot_existence_accepts_robot_id_and_name():
    dataset = FakeDataset(
        data_types={DataType.RGB_IMAGES},
        full_specs={
            "robot_1": {DataType.RGB_IMAGES: ["front"]},
        },
        robot_ids=["robot_1"],
        robot_name_to_id={"robot_name": "robot_1"},
    )
    # IDs only
    validate_robot_existence(
        dataset=dataset,
        dataset_name="test-dataset",
        input_robot_data_spec={"robot_1": {DataType.RGB_IMAGES: ["front"]}},
        output_robot_data_spec={},
    )
    # Names only
    validate_robot_existence(
        dataset=dataset,
        dataset_name="test-dataset",
        input_robot_data_spec={"robot_name": {DataType.RGB_IMAGES: ["front"]}},
        output_robot_data_spec={},
    )


def test_validate_data_specs_rejects_unsupported_data_type():
    dataset = FakeDataset(
        data_types={DataType.RGB_IMAGES},
        full_specs={"robot_1": {DataType.RGB_IMAGES: ["front"]}},
        robot_ids=["robot_1"],
    )
    robot_data_spec = {"robot_1": {DataType.JOINT_POSITIONS: ["j0"]}}

    with pytest.raises(ValueError, match="data type .* is not supported"):
        validate_data_specs(
            dataset=dataset,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            robot_data_spec=robot_data_spec,
            supported_data_types={DataType.RGB_IMAGES},
            spec_kind="input",
        )


def test_validate_data_specs_rejects_missing_data_type_in_dataset():
    dataset = FakeDataset(
        data_types={DataType.RGB_IMAGES},
        full_specs={"robot_1": {DataType.RGB_IMAGES: ["front"]}},
        robot_ids=["robot_1"],
    )
    robot_data_spec = {"robot_1": {DataType.JOINT_POSITIONS: ["j0"]}}

    with pytest.raises(ValueError, match="data type .* is not present in dataset"):
        validate_data_specs(
            dataset=dataset,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            robot_data_spec=robot_data_spec,
            supported_data_types={DataType.JOINT_POSITIONS},
            spec_kind="input",
        )


def test_validate_data_specs_rejects_missing_data_type_in_full_spec():
    dataset = FakeDataset(
        data_types={DataType.RGB_IMAGES},
        full_specs={"robot_1": {}},
        robot_ids=["robot_1"],
    )
    robot_data_spec = {"robot_1": {DataType.RGB_IMAGES: ["front"]}}

    with pytest.raises(ValueError, match="data values .* not present in dataset"):
        validate_data_specs(
            dataset=dataset,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            robot_data_spec=robot_data_spec,
            supported_data_types={DataType.RGB_IMAGES},
            spec_kind="input",
        )


def test_validate_data_specs_accepts_robot_name():
    dataset = FakeDataset(
        data_types={DataType.RGB_IMAGES},
        full_specs={"robot_1": {DataType.RGB_IMAGES: ["front"]}},
        robot_ids=["robot_1"],
        robot_name_to_id={"robot_name": "robot_1"},
    )
    robot_data_spec = {"robot_name": {DataType.RGB_IMAGES: ["front"]}}

    validate_data_specs(
        dataset=dataset,
        dataset_name="test-dataset",
        algorithm_name="test-algorithm",
        robot_data_spec=robot_data_spec,
        supported_data_types={DataType.RGB_IMAGES},
        spec_kind="input",
    )
