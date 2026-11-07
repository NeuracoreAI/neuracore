"""Preprocessing helpers for config resolution and runtime application."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuracore_types import (
    DataType,
    PreprocessingMethodConfig,
    PreprocessingPipeline,
    PreprocessingStage,
)
from omegaconf import DictConfig

if TYPE_CHECKING:
    from neuracore_types import BatchedNCData

    from neuracore.ml.core.ml_types import BatchedTrainingSamples

from neuracore.ml.preprocessing.base import (
    PreprocessingConfiguration,
    PreprocessingMethod,
)
from neuracore.ml.preprocessing.methods import ColorJitter, GaussianNoise, ResizePad

PREPROCESSING_METHOD_CLASSES_BY_NAME: dict[str, type[PreprocessingMethod]] = {
    "resize_pad": ResizePad,
    "color_jitter": ColorJitter,
    "gaussian_noise": GaussianNoise,
}


def validate_preprocessing_configuration(
    preprocessing_config: PreprocessingConfiguration,
) -> None:
    """Validate preprocessing methods are allowed for configured data types."""
    for data_type, methods in preprocessing_config.items():
        for method in methods:
            allowed_types = method.allowed_data_types()
            if data_type not in allowed_types:
                allowed_list = ", ".join(sorted(dt.value for dt in allowed_types))
                raise ValueError(
                    f"Preprocessing method '{type(method).__name__}' "
                    "is not allowed for data type "
                    f"{data_type.value}. Allowed data types: [{allowed_list}]"
                )


def resolve_preprocessing_config(
    config_dict: DictConfig,
) -> PreprocessingConfiguration:
    """Resolve one preprocessing role to serialized and runtime forms.

    Args:
        config_dict: Dictionary containing the preprocessing
            configuration.
                  Example:
                      {
                         "RGB_IMAGES": [
                          {
                              "_target_":
                                  "neuracore.ml.preprocessing.methods.ResizePad",
                              "size": [224, 224]
                          }
                         ]
                      }

    Returns:
        A preprocessing configuration in the runtime form.
    """
    from hydra.utils import instantiate

    preprocessing_methods = instantiate(config_dict, _convert_="all")
    resolved_config = PreprocessingConfiguration({
        DataType(data_type): methods
        for data_type, methods in preprocessing_methods.items()
    })
    validate_preprocessing_configuration(preprocessing_config=resolved_config)
    return resolved_config


def build_preprocessing_method(
    method_config: PreprocessingMethodConfig,
) -> PreprocessingMethod:
    """Build the preprocessing method a validated method config names.

    Args:
        method_config: Settings of one preprocessing step, including its
            method name.

    Returns:
        The preprocessing method instance.
    """
    method_class = PREPROCESSING_METHOD_CLASSES_BY_NAME[method_config.method]
    return method_class(**method_config.model_dump(exclude={"method", "stages"}))


def _build_preprocessing_configuration(
    methods_by_data_type: dict[DataType, list[PreprocessingMethodConfig]],
    stage: PreprocessingStage,
) -> PreprocessingConfiguration:
    methods_for_stage_by_data_type = {
        data_type: [
            method_config
            for method_config in method_configs
            if stage in method_config.stages
        ]
        for data_type, method_configs in methods_by_data_type.items()
    }
    preprocessing_config = PreprocessingConfiguration({
        data_type: [
            build_preprocessing_method(method_config)
            for method_config in method_configs
        ]
        for data_type, method_configs in methods_for_stage_by_data_type.items()
        if method_configs
    })
    validate_preprocessing_configuration(preprocessing_config=preprocessing_config)
    return preprocessing_config


def resolve_input_output_preprocessing(
    preprocessing_pipeline: PreprocessingPipeline,
    *,
    stage: PreprocessingStage,
) -> tuple[PreprocessingConfiguration, PreprocessingConfiguration]:
    """Build the input and output preprocessing one stage of a run applies.

    Args:
        preprocessing_pipeline: Validated pipeline with input and output keys,
            each mapping a data type to its ordered method configs.
        stage: Stage whose methods to build. Methods keep their pipeline order.

    Returns:
        The input and output preprocessing configurations.
    """
    input_methods_by_data_type = preprocessing_pipeline.get("input")
    if not input_methods_by_data_type:
        raise ValueError(
            "Preprocessing input configuration is missing! Please provide an "
            "input preprocessing configuration under training.preprocessing.input."
        )
    output_methods_by_data_type = preprocessing_pipeline.get("output")
    if not output_methods_by_data_type:
        raise ValueError(
            "Preprocessing output configuration is missing! Please provide an "
            "output preprocessing configuration under training.preprocessing.output."
        )
    return (
        _build_preprocessing_configuration(input_methods_by_data_type, stage),
        _build_preprocessing_configuration(output_methods_by_data_type, stage),
    )


def apply_preprocessing_methods(
    batched_data: BatchedNCData,
    methods: list[PreprocessingMethod],
) -> BatchedNCData:
    """Apply preprocessing methods to a batch of data.

    Args:
        batched_data: Batch to transform.
        methods: Ordered preprocessing methods to apply.

    Returns:
        The transformed batch.
    """
    for method in methods:
        batched_data = method(batched_data)
    return batched_data


def apply_device_preprocessing(
    batch: BatchedTrainingSamples,
    input_config: PreprocessingConfiguration,
    output_config: PreprocessingConfiguration,
) -> None:
    """Apply device-side preprocessing to a batch already on the device.

    Mutates ``batch`` in place: each method transforms an item and returns it,
    and the batch's per-slot lists are rebuilt from the results.

    Args:
        batch: Batch whose tensors are already on the target device.
        input_config: Device-side methods for input slots.
        output_config: Device-side methods for output slots.
    """
    for role_data, role_config in (
        (batch.inputs, input_config),
        (batch.outputs, output_config),
    ):
        for data_type, methods in role_config.items():
            items = role_data.get(data_type)
            if not items:
                continue
            role_data[data_type] = [
                apply_preprocessing_methods(batched_data=item, methods=methods)
                for item in items
            ]
