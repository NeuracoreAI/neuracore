import pytest
import torch
from neuracore_types import BatchedNCData, BatchedRGBData, DataType

from neuracore.ml.core.ml_types import BatchedTrainingSamples
from neuracore.ml.preprocessing.base import (
    PreprocessingConfiguration,
    PreprocessingMethod,
)
from neuracore.ml.utils.preprocessing import (
    apply_device_preprocessing,
    apply_preprocessing_methods,
    validate_preprocessing_configuration,
)


def _sample_rgb(height: int = 100, width: int = 200) -> BatchedRGBData:
    return BatchedRGBData(
        frame=torch.zeros((1, 1, 3, height, width), dtype=torch.float32),
        extrinsics=torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        intrinsics=torch.zeros((1, 1, 3, 3), dtype=torch.float32),
    )


class _RecordStep(PreprocessingMethod):
    def __init__(
        self,
        call_order: list[str],
        label: str,
    ) -> None:
        self._call_order = call_order
        self._label = label

    @staticmethod
    def allowed_data_types() -> frozenset[DataType]:
        return frozenset({DataType.RGB_IMAGES})

    def __call__(self, data: BatchedNCData) -> BatchedNCData:
        self._call_order.append(self._label)
        return data


def test_apply_methods_for_data_type_rejects_unsupported_data_type():
    methods = [_RecordStep(call_order=[], label="depth-step")]

    with pytest.raises(ValueError, match="not allowed for data type"):
        validate_preprocessing_configuration(
            preprocessing_config={DataType.DEPTH_IMAGES: methods},
        )


def test_apply_methods_for_data_type_executes_handlers_in_order():
    call_order: list[str] = []
    methods = [_RecordStep(call_order, "first"), _RecordStep(call_order, "second")]

    result = apply_preprocessing_methods(
        batched_data=_sample_rgb(),
        methods=methods,
    )

    assert isinstance(result, BatchedRGBData)
    assert call_order == ["first", "second"]


class _StagedStep(_RecordStep):
    """Records its label, and reports whichever stage it was built for."""

    def __init__(self, call_order: list[str], label: str, on_cpu: bool) -> None:
        super().__init__(call_order, label)
        self.on_cpu = on_cpu


def test_split_by_stage_partitions_and_preserves_order():
    order: list[str] = []
    config = PreprocessingConfiguration({
        DataType.RGB_IMAGES: [
            _StagedStep(order, "resize", on_cpu=True),
            _StagedStep(order, "jitter", on_cpu=False),
            _StagedStep(order, "noise", on_cpu=False),
        ]
    })

    on_cpu, on_device = config.split_by_stage()

    assert [m._label for m in on_cpu[DataType.RGB_IMAGES]] == ["resize"]
    assert [m._label for m in on_device[DataType.RGB_IMAGES]] == ["jitter", "noise"]


def test_split_by_stage_omits_data_types_with_nothing_to_do():
    order: list[str] = []
    config = PreprocessingConfiguration({
        DataType.RGB_IMAGES: [_StagedStep(order, "jitter", on_cpu=False)],
        DataType.DEPTH_IMAGES: [_StagedStep(order, "resize", on_cpu=True)],
    })

    on_cpu, on_device = config.split_by_stage()

    assert DataType.RGB_IMAGES not in on_cpu
    assert DataType.DEPTH_IMAGES not in on_device


def test_methods_run_on_the_worker_unless_they_opt_out():
    """The safe default: a method with no opinion runs where it always did."""
    assert _RecordStep(call_order=[], label="x").on_cpu is True


def test_apply_device_preprocessing_transforms_inputs_and_outputs():
    order: list[str] = []
    batch = BatchedTrainingSamples(
        inputs={DataType.RGB_IMAGES: [_sample_rgb()]},
        inputs_mask={DataType.RGB_IMAGES: torch.ones(1, 1)},
        outputs={DataType.RGB_IMAGES: [_sample_rgb()]},
        outputs_mask={DataType.RGB_IMAGES: torch.ones(1, 1)},
        batch_size=1,
    )

    apply_device_preprocessing(
        batch,
        PreprocessingConfiguration(
            {DataType.RGB_IMAGES: [_StagedStep(order, "in", on_cpu=False)]}
        ),
        PreprocessingConfiguration(
            {DataType.RGB_IMAGES: [_StagedStep(order, "out", on_cpu=False)]}
        ),
    )

    assert order == ["in", "out"]


def test_apply_device_preprocessing_skips_absent_data_types():
    """A configured data type the batch does not carry is simply not applied."""
    order: list[str] = []
    batch = BatchedTrainingSamples(
        inputs={}, inputs_mask={}, outputs={}, outputs_mask={}, batch_size=1
    )

    apply_device_preprocessing(
        batch,
        PreprocessingConfiguration(
            {DataType.RGB_IMAGES: [_StagedStep(order, "in", on_cpu=False)]}
        ),
        PreprocessingConfiguration(),
    )

    assert order == []
