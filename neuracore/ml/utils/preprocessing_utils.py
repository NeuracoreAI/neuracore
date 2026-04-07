"""Preprocessing helpers for config resolution and runtime application."""

from __future__ import annotations

from typing import Any

from neuracore_types import BatchedNCData, DataType

from neuracore.ml.preprocessing.base import PreprocessingMethod

PreprocessingConfiguration = dict[DataType, list[PreprocessingMethod]]


def resolve_preprocessing_config(
    preprocessing_config_dict: dict[str, list[dict[str, Any]]],
) -> PreprocessingConfiguration:
    """Resolve one preprocessing role to serialized and runtime forms.

    Args:
        preprocessing_config_dict: Dictionary containing the preprocessing
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

    if not preprocessing_config_dict:
        raise ValueError(
            "Missing preprocessing configuration. "
            "Expected preprocessing_config_dict to be defined."
        )

    resolved_preprocessing_dict: PreprocessingConfiguration = {}
    for data_type_key, method_entries in preprocessing_config_dict.items():
        data_type = DataType(str(data_type_key).split(".")[-1].strip())
        if not isinstance(method_entries, list):
            raise ValueError(f"preprocessing[{data_type}] must be a list of methods.")
        methods_for_type: list[PreprocessingMethod] = []
        for method_entry in method_entries:
            if not isinstance(method_entry, dict):
                raise ValueError(
                    f"Method entry at preprocessing[{data_type}] must be a mapping."
                )
            if "_target_" not in method_entry:
                raise ValueError(
                    f"Method entry at preprocessing[{data_type}] requires "
                    "a '_target_' field."
                )
            methods_for_type.append(instantiate(method_entry, _convert_="object"))
        resolved_preprocessing_dict[data_type] = methods_for_type
    return resolved_preprocessing_dict


def serialize_preprocessing_config(
    preprocessing_config: PreprocessingConfiguration,
) -> dict[str, list[dict[str, Any]]]:
    """Serialize a preprocessing configuration to a dictionary."""
    serialized = {}
    for data_type, methods in preprocessing_config.items():
        serialized[str(data_type)] = [
            m if isinstance(m, dict) else m.to_dict() for m in methods
        ]
    return serialized


def apply_preprocessing_configs(
    data_type: DataType,
    batched_data: BatchedNCData,
    preprocessing_configs: PreprocessingConfiguration,
) -> BatchedNCData:
    """Apply preprocessing configs for one explicit data type."""
    methods = preprocessing_configs.get(data_type, [])
    for method in methods:
        allowed_types = method.allowed_data_types()
        if data_type not in allowed_types:
            allowed_list = ", ".join(sorted(dt.value for dt in allowed_types))
            raise ValueError(
                f"Preprocessing method '{type(method).__name__}' "
                "is not allowed for data type "
                f"{data_type.value}. Allowed data types: [{allowed_list}]"
            )
        batched_data = method(batched_data)
    return batched_data
