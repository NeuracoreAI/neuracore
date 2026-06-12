from unittest.mock import MagicMock

import pytest
from neuracore_types import DataType

from neuracore.core.data.dataset import Dataset
from neuracore.core.data.recording import Recording
from neuracore.core.utils.training_input_args_validation import (
    _validate_algorithm_exists,
    _validate_cross_embodiment_description,
    _validate_per_recording_data_types,
    validate_training_params,
)

TEST_ROBOT_ID = "20a621b7-2f9b-4699-a08e-7d080488a5a1"


def _indexed_names(*names: str) -> dict[int, str]:
    return dict(enumerate(names))


@pytest.fixture
def dataset() -> Dataset:
    dataset = MagicMock(spec=Dataset)
    return dataset


def test_validate_cross_embodiment_description_rejects_missing_data_values(
    dataset: Dataset,
):
    dataset.data_types = [DataType.RGB_IMAGES]
    cross_embodiment_description = {
        TEST_ROBOT_ID: {DataType.RGB_IMAGES: _indexed_names("front", "side")}
    }
    with pytest.raises(ValueError, match="data values .* not present in dataset"):
        _validate_cross_embodiment_description(
            dataset=dataset,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            cross_embodiment_description=cross_embodiment_description,
            supported_data_types={DataType.RGB_IMAGES},
            description_kind="input",
        )


def test_validate_cross_embodiment_description_rejects_robot_name_rather_than_id(
    dataset: Dataset,
):
    dataset.data_types = [DataType.RGB_IMAGES]
    dataset.get_full_embodiment_description = MagicMock(
        return_value={DataType.RGB_IMAGES: _indexed_names("front")}
    )
    cross_embodiment_description = {
        "robot_name": {DataType.RGB_IMAGES: _indexed_names("front")}
    }

    with pytest.raises(AssertionError, match="Expected robot_id format for robot_name"):
        _validate_cross_embodiment_description(
            dataset=dataset,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            cross_embodiment_description=cross_embodiment_description,
            supported_data_types={DataType.RGB_IMAGES},
            description_kind="input",
        )


def test_validate_cross_embodiment_description_allows_subset_of_dataset_values(
    dataset: Dataset,
):
    dataset.data_types = [DataType.RGB_IMAGES]
    dataset.get_full_embodiment_description = MagicMock(
        return_value={DataType.RGB_IMAGES: _indexed_names("front", "side")}
    )
    cross_embodiment_description = {
        TEST_ROBOT_ID: {DataType.RGB_IMAGES: _indexed_names("front")}
    }

    _validate_cross_embodiment_description(
        dataset=dataset,
        dataset_name="test-dataset",
        algorithm_name="test-algorithm",
        cross_embodiment_description=cross_embodiment_description,
        supported_data_types={DataType.RGB_IMAGES},
        description_kind="input",
    )


def test_validate_algorithm_exists_raises_when_missing():
    with pytest.raises(ValueError, match="Algorithm .* not found"):
        _validate_algorithm_exists(None, "MissingAlgorithm")


def test_validate_cross_embodiment_description_rejects_unsupported_data_type(
    dataset: Dataset,
):
    dataset.data_types = [DataType.RGB_IMAGES]
    dataset.get_full_embodiment_description = MagicMock(
        return_value={DataType.RGB_IMAGES: _indexed_names("front")}
    )
    cross_embodiment_description = {
        TEST_ROBOT_ID: {DataType.JOINT_POSITIONS: _indexed_names("j0")}
    }

    with pytest.raises(ValueError, match="data type .* is not present in dataset"):
        _validate_cross_embodiment_description(
            dataset=dataset,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            cross_embodiment_description=cross_embodiment_description,
            supported_data_types={DataType.RGB_IMAGES},
            description_kind="input",
        )


def test_validate_cross_embodiment_description_rejects_missing_data_type_in_dataset(
    dataset: Dataset,
):
    dataset.data_types = [DataType.RGB_IMAGES]
    dataset.get_full_embodiment_description = MagicMock(
        return_value={DataType.RGB_IMAGES: _indexed_names("front")}
    )
    cross_embodiment_description = {
        TEST_ROBOT_ID: {DataType.JOINT_POSITIONS: _indexed_names("j0")}
    }

    with pytest.raises(ValueError, match="data type .* is not present in dataset"):
        _validate_cross_embodiment_description(
            dataset=dataset,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            cross_embodiment_description=cross_embodiment_description,
            supported_data_types={DataType.JOINT_POSITIONS},
            description_kind="input",
        )


def test_validate_cross_embodiment_description_rejects_missing_data_type(
    dataset: Dataset,
):
    dataset.data_types = [DataType.RGB_IMAGES]
    dataset.get_full_embodiment_description = MagicMock(return_value={})
    cross_embodiment_description = {
        TEST_ROBOT_ID: {DataType.RGB_IMAGES: _indexed_names("front")}
    }

    with pytest.raises(ValueError, match="data values .* not present in dataset"):
        _validate_cross_embodiment_description(
            dataset=dataset,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            cross_embodiment_description=cross_embodiment_description,
            supported_data_types={DataType.RGB_IMAGES},
            description_kind="input",
        )


# ---------------------------------------------------------------------------
# _validate_per_recording_data_types
# ---------------------------------------------------------------------------


def _make_recording(name: str, data_types: set[DataType]) -> Recording:
    """Return a minimal mock Recording with the given name and data_types."""
    rec = MagicMock(spec=Recording)
    rec.name = name
    rec.data_types = data_types
    return rec


def _dataset_with_recordings(recordings: list) -> Dataset:
    ds = MagicMock(spec=Dataset)
    ds.__iter__ = MagicMock(return_value=iter(recordings))
    return ds


def test_validate_per_recording_data_types_passes_when_all_present():
    ds = _dataset_with_recordings([
        _make_recording("rec_a", {DataType.JOINT_POSITIONS, DataType.RGB_IMAGES}),
        _make_recording("rec_b", {DataType.JOINT_POSITIONS, DataType.RGB_IMAGES}),
    ])
    _validate_per_recording_data_types(
        ds, {DataType.JOINT_POSITIONS, DataType.RGB_IMAGES}
    )


def test_validate_per_recording_data_types_passes_with_superset():
    """Recording with extra datatypes beyond requested must not fail."""
    ds = _dataset_with_recordings([
        _make_recording(
            "rec_a",
            {DataType.JOINT_POSITIONS, DataType.RGB_IMAGES, DataType.JOINT_VELOCITIES},
        ),
    ])
    _validate_per_recording_data_types(ds, {DataType.JOINT_POSITIONS})


def test_validate_per_recording_data_types_passes_empty_requested():
    ds = _dataset_with_recordings([_make_recording("rec_a", set())])
    _validate_per_recording_data_types(ds, set())


def test_validate_per_recording_data_types_raises_grouped_error():
    ds = _dataset_with_recordings([
        _make_recording("joint_rec", {DataType.JOINT_POSITIONS}),
        _make_recording("rgb_rec", {DataType.RGB_IMAGES}),
        _make_recording("all_rec", {DataType.JOINT_POSITIONS, DataType.RGB_IMAGES}),
    ])
    with pytest.raises(ValueError) as exc_info:
        _validate_per_recording_data_types(
            ds, {DataType.JOINT_POSITIONS, DataType.RGB_IMAGES}
        )
    msg = str(exc_info.value)
    assert "Failed to start training run" in msg
    assert "some recordings are missing requested datatypes" in msg
    assert "Missing RGB_IMAGES" in msg
    assert "- joint_rec" in msg
    assert "Missing JOINT_POSITIONS" in msg
    assert "- rgb_rec" in msg
    assert "all_rec" not in msg


def test_validate_per_recording_data_types_error_lists_all_missing_types():
    """A recording missing two types must appear under both type sections."""
    ds = _dataset_with_recordings([
        _make_recording("partial", {DataType.RGB_IMAGES}),
    ])
    with pytest.raises(ValueError) as exc_info:
        _validate_per_recording_data_types(
            ds,
            {DataType.JOINT_POSITIONS, DataType.JOINT_VELOCITIES, DataType.RGB_IMAGES},
        )
    msg = str(exc_info.value)
    assert "Missing JOINT_POSITIONS" in msg
    assert "Missing JOINT_VELOCITIES" in msg
    assert msg.count("- partial") == 2


# ---------------------------------------------------------------------------
# validate_training_params wiring
# ---------------------------------------------------------------------------


def test_validate_training_params_raises_per_recording_error_before_algo_check():
    """validate_training_params must fire per-recording check before algorithm check."""
    ds = MagicMock(spec=Dataset)
    ds.data_types = [DataType.JOINT_POSITIONS, DataType.RGB_IMAGES]
    ds.get_full_embodiment_description = MagicMock(
        return_value={
            DataType.JOINT_POSITIONS: _indexed_names("j0"),
            DataType.RGB_IMAGES: _indexed_names("front"),
        }
    )
    ds.__iter__ = MagicMock(
        return_value=iter([
            _make_recording("partial_rec", {DataType.JOINT_POSITIONS}),
            _make_recording(
                "full_rec", {DataType.JOINT_POSITIONS, DataType.RGB_IMAGES}
            ),
        ])
    )

    with pytest.raises(ValueError) as exc_info:
        validate_training_params(
            dataset=ds,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            input_cross_embodiment_description={
                TEST_ROBOT_ID: {
                    DataType.JOINT_POSITIONS: _indexed_names("j0"),
                    DataType.RGB_IMAGES: _indexed_names("front"),
                }
            },
            output_cross_embodiment_description={},
            supported_input_data_types={DataType.JOINT_POSITIONS, DataType.RGB_IMAGES},
            supported_output_data_types=set(),
        )

    msg = str(exc_info.value)
    assert "Failed to start training run" in msg
    assert "Missing RGB_IMAGES" in msg
    assert "- partial_rec" in msg


def test_validate_training_params_raises_when_no_input_types():
    """validate_training_params raises if no input datatypes specified."""
    ds = MagicMock(spec=Dataset)
    ds.data_types = []

    with pytest.raises(ValueError, match="no input datatypes specified"):
        validate_training_params(
            dataset=ds,
            dataset_name="test-dataset",
            algorithm_name="test-algorithm",
            input_cross_embodiment_description={},
            output_cross_embodiment_description={},
            supported_input_data_types=set(),
            supported_output_data_types=set(),
        )
