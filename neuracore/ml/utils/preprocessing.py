"""Preprocessing helpers for config resolution and runtime application."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from neuracore_types import DataType
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from neuracore_types import BatchedNCData

    from neuracore.ml.core.ml_types import BatchedTrainingSamples

from neuracore.ml.preprocessing.base import (
    PreprocessingConfiguration,
    PreprocessingMethod,
    PreprocessingType,
)
from neuracore.ml.preprocessing.methods import PREPROCESSING_METHODS


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


def _build_preprocessing_step(
    step: Mapping[str, Any],
    *,
    data_type: DataType,
    preprocessing_type: PreprocessingType,
) -> PreprocessingMethod:
    """Build one method from a one-key name to params mapping.

    Raises:
        ValueError: If the step is malformed, names an unknown method, or
            names a method that belongs to the other group.
    """
    location = f"{data_type.value}.{preprocessing_type.value}"
    if not isinstance(step, Mapping) or len(step) != 1:
        raise ValueError(
            f"Each preprocessing step under {location} must be a mapping with "
            f"exactly one key, the method name. Got: {step!r}"
        )
    ((method_name, method_params),) = step.items()
    method_class = PREPROCESSING_METHODS.get(method_name)
    if method_class is None:
        known_method_names = ", ".join(sorted(PREPROCESSING_METHODS))
        raise ValueError(
            f"Unknown preprocessing method '{method_name}' under {location}. "
            f"Known methods: [{known_method_names}]"
        )
    if method_class.preprocessing_type is not preprocessing_type:
        method_group_name = method_class.preprocessing_type.value
        raise ValueError(
            f"Preprocessing method '{method_name}' is {method_group_name} "
            f"and cannot run under {location}. Move it to "
            f"{data_type.value}.{method_group_name}."
        )
    return method_class(**dict(method_params or {}))


def _resolve_data_type_methods(
    data_type: DataType, groups: Any
) -> list[PreprocessingMethod]:
    """Resolve one data type's steps, geometric first, in the order written.

    A flat list of target entries, as stored in older model archives, is
    instantiated and sorted into geometric then photometric order.
    """
    from hydra.utils import instantiate

    preprocessing_type_names = [
        preprocessing_type.value for preprocessing_type in PreprocessingType
    ]
    if isinstance(groups, list):
        legacy_methods: list[PreprocessingMethod] = [
            instantiate(method_config, _convert_="all") for method_config in groups
        ]
        return sorted(
            legacy_methods,
            key=lambda method: preprocessing_type_names.index(
                method.preprocessing_type.value
            ),
        )
    if not isinstance(groups, Mapping):
        raise ValueError(
            f"Preprocessing for {data_type.value} must map "
            f"{' and '.join(preprocessing_type_names)} to step lists. "
            f"Got: {groups!r}"
        )
    unknown_group_names = set(groups) - set(preprocessing_type_names)
    if unknown_group_names:
        raise ValueError(
            f"Unknown preprocessing group(s) {sorted(unknown_group_names)} "
            f"under {data_type.value}. Allowed groups: {preprocessing_type_names}"
        )
    methods: list[PreprocessingMethod] = []
    for preprocessing_type in PreprocessingType:
        steps = groups.get(preprocessing_type.value) or []
        if not isinstance(steps, list):
            raise ValueError(
                f"{data_type.value}.{preprocessing_type.value} must be a list "
                f"of steps. Got: {steps!r}"
            )
        methods.extend(
            _build_preprocessing_step(
                step, data_type=data_type, preprocessing_type=preprocessing_type
            )
            for step in steps
        )
    return methods


def resolve_preprocessing_config(
    config_dict: DictConfig | Mapping[str, Any],
) -> PreprocessingConfiguration:
    """Resolve one preprocessing role into its runtime form.

    Args:
        config_dict: Mapping from data type to geometric and photometric
            step lists. Example:
                {
                    "RGB_IMAGES": {
                        "geometric": [{"resize_pad": {"size": [224, 224]}}],
                        "photometric": [{"gaussian_noise": {"std": 5.0}}],
                    }
                }
            A flat list of target entries per data type, as stored in older
            model archives, also resolves.

    Returns:
        Methods per data type, geometric first, each group in the order written.
    """
    if isinstance(config_dict, DictConfig):
        config_dict = OmegaConf.to_container(config_dict, resolve=True)
    resolved_config = PreprocessingConfiguration()
    for data_type_name, groups in config_dict.items():
        data_type = DataType(data_type_name)
        methods = _resolve_data_type_methods(data_type, groups)
        if methods:
            resolved_config[data_type] = methods
    validate_preprocessing_configuration(preprocessing_config=resolved_config)
    return resolved_config


def resolve_input_output_preprocessing(
    role_config: DictConfig | None,
    *,
    role_name: str,
) -> tuple[PreprocessingConfiguration, PreprocessingConfiguration]:
    """Resolve ``input`` / ``output`` preprocessing for a train or inference role.

    Args:
        role_config: Mapping with ``input`` and ``output`` keys, each mapping
            a data type to its geometric and photometric step lists.
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
