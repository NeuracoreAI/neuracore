import pytest
import torch
from neuracore_types import BatchedNCData, BatchedRGBData, DataType

from neuracore.ml.preprocessing.base import PreprocessingMethod
from neuracore.ml.utils.preprocessing_utils import (
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
    def __init__(self, call_order: list[str], label: str) -> None:
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
