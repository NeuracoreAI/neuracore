import pytest
from neuracore_types import BatchedNCData, DataType, ModelInitDescription

from neuracore.ml import (
    BatchedInferenceInputs,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)


class DummyModel(NeuracoreModel):
    @staticmethod
    def get_supported_input_data_types() -> set[DataType]:
        return {DataType.JOINT_POSITIONS}

    @staticmethod
    def get_supported_output_data_types() -> set[DataType]:
        return {
            DataType.JOINT_TARGET_POSITIONS,
            DataType.JOINT_POSITIONS,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        }

    def forward(
        self, batch: BatchedInferenceInputs
    ) -> dict[DataType, list[BatchedNCData]]:
        return {}

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        raise NotImplementedError

    def configure_optimizers(self) -> list:
        return []


class CustomOrderDummyModel(DummyModel):
    CANONICAL_OUTPUT_DATA_TYPE_ORDER = (
        DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        DataType.JOINT_POSITIONS,
        DataType.JOINT_TARGET_POSITIONS,
    )


class UnsupportedCanonicalOrderDummyModel(DummyModel):
    @staticmethod
    def get_supported_output_data_types() -> set[DataType]:
        return DummyModel.get_supported_output_data_types().union(
            {DataType.JOINT_TORQUES}
        )


def _make_model_init_description(
    output_data_types: set[DataType],
) -> ModelInitDescription:
    return ModelInitDescription(
        input_data_types={DataType.JOINT_POSITIONS},
        output_data_types=output_data_types,
        input_dataset_statistics={DataType.JOINT_POSITIONS: []},
        output_dataset_statistics={data_type: [] for data_type in output_data_types},
        output_prediction_horizon=1,
    )


def test_ordered_output_data_types_use_shared_canonical_order():
    model = DummyModel(
        _make_model_init_description({
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.JOINT_TARGET_POSITIONS,
            DataType.JOINT_POSITIONS,
        })
    )

    assert model.ordered_output_data_types == [
        DataType.JOINT_TARGET_POSITIONS,
        DataType.JOINT_POSITIONS,
        DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
    ]


def test_ordered_output_data_types_allow_model_specific_override():
    model = CustomOrderDummyModel(
        _make_model_init_description({
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.JOINT_TARGET_POSITIONS,
            DataType.JOINT_POSITIONS,
        })
    )

    assert model.ordered_output_data_types == [
        DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        DataType.JOINT_POSITIONS,
        DataType.JOINT_TARGET_POSITIONS,
    ]


def test_missing_canonical_order_entry_raises_value_error():
    with pytest.raises(
        ValueError,
        match="Encountered output data types without canonical order entries",
    ):
        UnsupportedCanonicalOrderDummyModel(
            _make_model_init_description({DataType.JOINT_TORQUES})
        )
