import pytest
import torch
from neuracore_types import BatchedDepthData, BatchedNCData, BatchedRGBData, DataType

from neuracore.ml.preprocessing.base import PreprocessingMethod
from neuracore.ml.utils.preprocessing_utils import apply_preprocessing_configs


def _sample_depth(height: int = 100, width: int = 200) -> BatchedDepthData:
    return BatchedDepthData(
        frame=torch.zeros((1, 1, 1, height, width), dtype=torch.float32),
        extrinsics=torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        intrinsics=torch.zeros((1, 1, 3, 3), dtype=torch.float32),
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


class _AddBias(PreprocessingMethod):
    def __init__(self, bias: float) -> None:
        self._bias = float(bias)

    @staticmethod
    def allowed_data_types() -> frozenset[DataType]:
        return frozenset({DataType.RGB_IMAGES})

    def __call__(self, data: BatchedNCData) -> BatchedNCData:
        data.frame = data.frame + self._bias
        return data


def test_apply_methods_for_data_type_rejects_unsupported_data_type():
    depth = _sample_depth()
    methods = [_RecordStep(call_order=[], label="depth-step")]

    with pytest.raises(ValueError, match="not allowed for data type"):
        apply_preprocessing_configs(
            data_type=DataType.DEPTH_IMAGES,
            batched_data=depth,
            preprocessing_configs={DataType.DEPTH_IMAGES: methods},
        )


def test_apply_methods_for_data_type_executes_handlers_in_order():
    call_order: list[str] = []
    methods = [_RecordStep(call_order, "first"), _RecordStep(call_order, "second")]

    result = apply_preprocessing_configs(
        data_type=DataType.RGB_IMAGES,
        batched_data=_sample_rgb(),
        preprocessing_configs={DataType.RGB_IMAGES: methods},
    )

    assert isinstance(result, BatchedRGBData)
    assert call_order == ["first", "second"]


def test_apply_methods_for_data_type_custom_method_uses_constructor_args():
    methods = [_AddBias(bias=2.5)]
    result = apply_preprocessing_configs(
        data_type=DataType.RGB_IMAGES,
        batched_data=_sample_rgb(),
        preprocessing_configs={DataType.RGB_IMAGES: methods},
    )

    assert isinstance(result, BatchedRGBData)
    assert torch.allclose(result.frame, torch.full_like(result.frame, 2.5))
