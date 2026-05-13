"""Preprocessing helpers for config resolution and runtime application."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuracore_types import DataType
from omegaconf import DictConfig

if TYPE_CHECKING:
    from neuracore_types import BatchedNCData

from neuracore.ml.preprocessing.base import PreprocessingMethod

PreprocessingConfiguration = dict[DataType, list[PreprocessingMethod]]


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
    resolved_config = {
        DataType(data_type): methods
        for data_type, methods in preprocessing_methods.items()
    }
    validate_preprocessing_configuration(preprocessing_config=resolved_config)
    return resolved_config


def apply_preprocessing_methods(
    batched_data: BatchedNCData,
    methods: list[PreprocessingMethod],
) -> BatchedNCData:
    """Apply preprocessing methods to a batch of data."""
    for method in methods:
        batched_data = method(batched_data)
    return batched_data
