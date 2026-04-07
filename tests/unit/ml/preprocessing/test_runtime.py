from unittest.mock import patch

import pytest
import torch
from neuracore_types import BatchedRGBData, DataType
from neuracore_types.preprocessing import (
    MethodSpec,
    PreProcessingConfiguration,
    PreProcessingMethod,
)

from neuracore.ml.datasets.pytorch_synchronized_dataset import (
    PytorchSynchronizedDataset,
)
from neuracore.ml.preprocessing.runtime import apply_methods_for_slot


def _custom_add_bias_from_dotted_path(
    batched_data: BatchedRGBData, bias: float = 1.0
) -> BatchedRGBData:
    batched_data.frame = batched_data.frame + float(bias)
    return batched_data


def _sample_rgb(height: int = 100, width: int = 200) -> BatchedRGBData:
    return BatchedRGBData(
        frame=torch.zeros((1, 1, 3, height, width), dtype=torch.float32),
        extrinsics=torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        intrinsics=torch.zeros((1, 1, 3, 3), dtype=torch.float32),
    )


def test_apply_methods_for_slot_rejects_unsupported_data_type_for_resize_pad():
    rgb = _sample_rgb()
    methods = [PreProcessingMethod(name="resize_pad", args={"size": [64, 64]})]

    with pytest.raises(ValueError, match="not allowed for data type"):
        apply_methods_for_slot(
            data_type=DataType.JOINT_POSITIONS, batched_data=rgb, methods=methods
        )


def test_apply_methods_for_slot_rejects_unknown_method_name():
    rgb = _sample_rgb()
    methods = [PreProcessingMethod(name="does_not_exist", args={})]

    with pytest.raises(ValueError, match="Unsupported preprocessing method"):
        apply_methods_for_slot(
            data_type=DataType.RGB_IMAGES, batched_data=rgb, methods=methods
        )


def test_apply_methods_for_slot_executes_handlers_in_order():
    call_order: list[str] = []

    def step_one(batched_data, **kwargs):
        call_order.append("first")
        return batched_data

    def step_two(batched_data, **kwargs):
        call_order.append("second")
        return batched_data

    custom_registry = {
        "first": MethodSpec(handler=step_one, allowed_data_types={DataType.RGB_IMAGES}),
        "second": MethodSpec(
            handler=step_two, allowed_data_types={DataType.RGB_IMAGES}
        ),
    }
    methods = [PreProcessingMethod(name="first"), PreProcessingMethod(name="second")]

    with patch("neuracore.ml.preprocessing.runtime._METHOD_REGISTRY", custom_registry):
        result = apply_methods_for_slot(
            data_type=DataType.RGB_IMAGES, batched_data=_sample_rgb(), methods=methods
        )

    assert isinstance(result, BatchedRGBData)
    assert call_order == ["first", "second"]


def test_apply_methods_for_slot_custom_method_uses_args():
    def add_bias(batched_data, **kwargs):
        bias = float(kwargs["bias"])
        batched_data.frame = batched_data.frame + bias
        return batched_data

    custom_registry = {
        "add_bias": MethodSpec(
            handler=add_bias, allowed_data_types={DataType.RGB_IMAGES}
        ),
    }
    methods = [PreProcessingMethod(name="add_bias", args={"bias": 2.5})]

    with patch("neuracore.ml.preprocessing.runtime._METHOD_REGISTRY", custom_registry):
        result = apply_methods_for_slot(
            data_type=DataType.RGB_IMAGES, batched_data=_sample_rgb(), methods=methods
        )

    assert isinstance(result, BatchedRGBData)
    assert torch.allclose(result.frame, torch.full_like(result.frame, 2.5))


def test_apply_methods_for_slot_custom_callable_runs_from_dotted_path():
    rgb = _sample_rgb()
    methods = [
        PreProcessingMethod(
            name="custom_add_bias",
            custom_callable="tests.unit.ml.preprocessing.test_runtime._custom_add_bias_from_dotted_path",
            args={"bias": 3.0},
        )
    ]

    result = apply_methods_for_slot(
        data_type=DataType.RGB_IMAGES, batched_data=rgb, methods=methods
    )

    assert isinstance(result, BatchedRGBData)
    assert torch.allclose(result.frame, torch.full_like(result.frame, 3.0))


def test_apply_methods_for_slot_custom_callable_requires_field():
    rgb = _sample_rgb()
    methods = [PreProcessingMethod(name="custom_my_step", args={"k": 1})]

    with pytest.raises(ValueError, match="requires `custom_callable`"):
        apply_methods_for_slot(
            data_type=DataType.RGB_IMAGES, batched_data=rgb, methods=methods
        )


def test_dataset_slot_not_configured_is_no_op():
    rgb = _sample_rgb()
    config = PreProcessingConfiguration(
        steps={
            DataType.RGB_IMAGES: {
                1: [PreProcessingMethod(name="resize_pad", args={"size": [64, 64]})]
            }
        }
    )

    untouched = PytorchSynchronizedDataset._apply_slot_preprocessing(
        data_type=DataType.RGB_IMAGES,
        slot_idx=0,
        batched_nc_data=rgb,
        preprocessing_config=config,
    )

    assert untouched.frame.shape[-2:] == (100, 200)
