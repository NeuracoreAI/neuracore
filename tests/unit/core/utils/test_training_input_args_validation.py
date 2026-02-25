from unittest.mock import MagicMock

import pytest
from neuracore_types import DataType

from neuracore.core.data.dataset import Dataset
from neuracore.core.utils.training_input_args_validation import (
    _validate_algorithm_exists,
    _validate_data_specs,
)

TEST_ROBOT_ID = "20a621b7-2f9b-4699-a08e-7d080488a5a1"


@pytest.fixture
def dataset() -> Dataset:
    dataset = MagicMock(spec=Dataset)
    return dataset


def test_validate_data_specs_rejects_missing_data_values(dataset: Dataset):
    dataset.data_types = [DataType.RGB_IMAGES]
    robot_data_spec = {TEST_ROBOT_ID: {DataType.RGB_IMAGES: ["front", "side"]}}
    with pytest.raises(ValueError, match="data values .* not present in dataset"):
        _validate_data_specs(
            dataset=dataset,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            robot_data_spec=robot_data_spec,
            supported_data_types={DataType.RGB_IMAGES},
            spec_kind="input",
        )


def test_validate_data_specs_rejects_robot_name_rather_than_id(dataset: Dataset):
    dataset.data_types = [DataType.RGB_IMAGES]
    dataset.get_full_data_spec = MagicMock(
        return_value={DataType.RGB_IMAGES: ["front"]}
    )
    robot_data_spec = {"robot_name": {DataType.RGB_IMAGES: ["front"]}}

    with pytest.raises(AssertionError, match="Expected robot_id format for robot_name"):
        _validate_data_specs(
            dataset=dataset,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            robot_data_spec=robot_data_spec,
            supported_data_types={DataType.RGB_IMAGES},
            spec_kind="input",
        )


def test_validate_data_specs_allows_subset_of_dataset_values(dataset: Dataset):
    dataset.data_types = [DataType.RGB_IMAGES]
    dataset.get_full_data_spec = MagicMock(
        return_value={DataType.RGB_IMAGES: ["front", "side"]}
    )
    robot_data_spec = {TEST_ROBOT_ID: {DataType.RGB_IMAGES: ["front"]}}

    _validate_data_specs(
        dataset=dataset,
        dataset_name="test-dataset",
        algorithm_name="test-algorithm",
        robot_data_spec=robot_data_spec,
        supported_data_types={DataType.RGB_IMAGES},
        spec_kind="input",
    )


def test_validate_algorithm_exists_raises_when_missing():
    with pytest.raises(ValueError, match="Algorithm .* not found"):
        _validate_algorithm_exists(None, "MissingAlgorithm")


def test_validate_data_specs_rejects_unsupported_data_type(dataset: Dataset):
    dataset.data_types = [DataType.RGB_IMAGES]
    dataset.get_full_data_spec = MagicMock(
        return_value={DataType.RGB_IMAGES: ["front"]}
    )
    robot_data_spec = {TEST_ROBOT_ID: {DataType.JOINT_POSITIONS: ["j0"]}}

    with pytest.raises(ValueError, match="data type .* is not supported"):
        _validate_data_specs(
            dataset=dataset,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            robot_data_spec=robot_data_spec,
            supported_data_types={DataType.RGB_IMAGES},
            spec_kind="input",
        )


def test_validate_data_specs_rejects_missing_data_type_in_dataset(dataset: Dataset):
    dataset.data_types = [DataType.RGB_IMAGES]
    dataset.get_full_data_spec = MagicMock(
        return_value={DataType.RGB_IMAGES: ["front"]}
    )
    robot_data_spec = {TEST_ROBOT_ID: {DataType.JOINT_POSITIONS: ["j0"]}}

    with pytest.raises(ValueError, match="data type .* is not present in dataset"):
        _validate_data_specs(
            dataset=dataset,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            robot_data_spec=robot_data_spec,
            supported_data_types={DataType.JOINT_POSITIONS},
            spec_kind="input",
        )


def test_validate_data_specs_rejects_missing_data_type_in_full_spec(dataset: Dataset):
    dataset.data_types = [DataType.RGB_IMAGES]
    dataset.get_full_data_spec = MagicMock(return_value={})
    robot_data_spec = {TEST_ROBOT_ID: {DataType.RGB_IMAGES: ["front"]}}

    with pytest.raises(ValueError, match="data values .* not present in dataset"):
        _validate_data_specs(
            dataset=dataset,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            robot_data_spec=robot_data_spec,
            supported_data_types={DataType.RGB_IMAGES},
            spec_kind="input",
        )
