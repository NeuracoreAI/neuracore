"""Preprocessing helpers for config resolution and runtime application."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuracore_types import DataType
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from neuracore_types import BatchedNCData

    from neuracore.ml.core.ml_types import BatchedTrainingSamples

from neuracore.ml.preprocessing.base import (
    PreprocessingConfiguration,
    PreprocessingMethod,
)


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


def resolve_input_output_preprocessing(
    role_config: DictConfig | None,
    *,
    role_name: str,
) -> tuple[PreprocessingConfiguration, PreprocessingConfiguration]:
    """Resolve ``input`` / ``output`` preprocessing for a train or inference role.

    Args:
        role_config: Mapping with ``input`` and ``output`` keys, each a
            ``Dict[DataType, List[MethodConfig]]``.
        role_name: Name used in error messages (e.g. ``train_preprocessing``).

    Returns:
        ``(input_preprocessing_config, output_preprocessing_config)``.
    """
    if not role_config:
        raise ValueError(
            f"{role_name} configuration is missing! Please provide a "
            f"{role_name} configuration."
        )
    input_cfg = role_config.get("input", OmegaConf.create({}))
    if not input_cfg:
        raise ValueError(
            f"{role_name} input configuration is missing! Please provide an "
            f"input preprocessing configuration under {role_name}.input."
        )
    output_cfg = role_config.get("output", OmegaConf.create({}))
    if not output_cfg:
        raise ValueError(
            f"{role_name} output configuration is missing! Please provide an "
            f"output preprocessing configuration under {role_name}.output."
        )
    return (
        resolve_preprocessing_config(input_cfg),
        resolve_preprocessing_config(output_cfg),
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
