"""Training parameter validation utilities.

This module provides validation functions for training parameters, including
algorithm existence, robot existence, and data spec compatibility.
"""

from __future__ import annotations

from neuracore_types import CrossEmbodimentDescription, DataType
from omegaconf import DictConfig

from neuracore.core.data.dataset import Dataset, EmbodimentDescription
from neuracore.core.utils.robot_data_spec_utils import is_robot_id


def _validate_algorithm_exists(algorithm_id: str | None, algorithm_name: str) -> None:
    """Validate that the requested algorithm exists.

    Args:
        algorithm_id: Resolved algorithm ID, or None if not found.
        algorithm_name: Algorithm name requested by the user.

    Raises:
        ValueError: If the algorithm ID is None.
    """
    if algorithm_id is None:
        raise ValueError(f"Algorithm {algorithm_name} not found.")


def _validate_data_specs(
    dataset: Dataset,
    dataset_name: str,
    algorithm_name: str,
    cross_embodiment_description: CrossEmbodimentDescription,
    supported_data_types: set[DataType],
    spec_kind: str,
) -> None:
    """Validate that a robot data spec is compatible with the dataset and algorithm."""
    _validate_data_specs_against_dataset(
        dataset=dataset,
        dataset_name=dataset_name,
        cross_embodiment_description=cross_embodiment_description,
        spec_kind=spec_kind,
    )
    _validate_data_specs_against_algorithm(
        algorithm_name=algorithm_name,
        cross_embodiment_description=cross_embodiment_description,
        supported_data_types=supported_data_types,
        spec_kind=spec_kind,
    )


def _validate_data_specs_against_dataset(
    dataset: Dataset,
    dataset_name: str,
    cross_embodiment_description: CrossEmbodimentDescription,
    spec_kind: str,
) -> None:
    """Validate data spec robot IDs, types, and values against dataset metadata."""
    for robot_id, robot_data in cross_embodiment_description.items():
        assert is_robot_id(robot_id), f"Expected robot_id format for {robot_id}"
        dataset_spec: EmbodimentDescription = dataset.get_full_data_spec(robot_id)

        for data_type, data_value in robot_data.items():
            if data_type not in dataset.data_types:
                raise ValueError(
                    f"{spec_kind} data type {data_type} is not present in dataset "
                    f"{dataset_name}. Please check the dataset contents."
                )

            dataset_values = dataset_spec.get(data_type)
            if dataset_values is None:
                raise ValueError(
                    f"{spec_kind} data values {sorted(data_value)} for "
                    f"{data_type} are not present in dataset {dataset_name}."
                )

            if isinstance(data_value, (dict, DictConfig)):
                requested_values = set(data_value.values())
            elif isinstance(data_value, (list, set, tuple)):
                requested_values = set(data_value)
            else:
                raise ValueError(
                    f"Expected {spec_kind} data value for {data_type} to be a dict "
                    f"of index to string, but got {data_value}."
                )

            if isinstance(dataset_values, dict):
                available_values = set(dataset_values.values())
            elif isinstance(dataset_values, (list, set, tuple)):
                available_values = set(dataset_values)
            else:
                available_values = set()

            missing_values = requested_values - available_values
            if missing_values:
                raise ValueError(
                    f"{spec_kind} data values {sorted(missing_values)} for "
                    f"{data_type} are not present in dataset {dataset_name}."
                )


def _validate_data_specs_against_algorithm(
    algorithm_name: str,
    cross_embodiment_description: CrossEmbodimentDescription,
    supported_data_types: set[DataType],
    spec_kind: str,
) -> None:
    """Validate that requested data types are supported by the algorithm."""
    for robot_data in cross_embodiment_description.values():
        for data_type in robot_data:
            if data_type not in supported_data_types:
                raise ValueError(
                    f"{spec_kind} data type {data_type} is not supported by algorithm "
                    f"{algorithm_name}. Please check the training job requirements."
                )


def _get_data_types_for_algorithms(
    algorithm_name: str,
    algorithm_jsons: list[dict],
) -> tuple[set[DataType], set[DataType]]:
    """Resolve supported input and output data types for an algorithm.

    Args:
        algorithm_name: Algorithm name to look up.
        algorithm_jsons: List of algorithm metadata dictionaries.

    Returns:
        A tuple containing:
          - Supported input data types.
          - Supported output data types.

        If the algorithm name is not found, both sets are empty.
    """
    input_data_types: list[DataType] = []
    output_data_types: list[DataType] = []

    for algorithm_json in algorithm_jsons:
        if algorithm_json.get("name") != algorithm_name:
            continue

        input_data_types = [
            DataType(v) for v in algorithm_json.get("supported_input_data_types", [])
        ]
        output_data_types = [
            DataType(v) for v in algorithm_json.get("supported_output_data_types", [])
        ]
        break

    return set(input_data_types), set(output_data_types)


def get_algorithm_name(algorithm_id: str, algorithm_jsons: list[dict]) -> str:
    """Get algorithm name from its ID.

    Args:
        algorithm_id (str): The ID of the algorithm.
        algorithm_jsons (list[dict]): List of algorithm metadata dictionaries.

    Returns:
        str: The name of the algorithm.

    Raises:
        ValueError: If the algorithm ID is not found.
    """
    for algorithm in algorithm_jsons:
        if algorithm["id"] == algorithm_id:
            return algorithm["name"]
    raise ValueError(f"Algorithm with ID {algorithm_id} not found.")


def get_algorithm_id(algorithm_name: str, algorithm_jsons: list[dict]) -> str | None:
    """Resolve an algorithm ID from its name.

    Args:
        algorithm_name: Algorithm name to look up.
        algorithm_jsons: List of algorithm metadata dictionaries.

    Returns:
        The algorithm ID if found; otherwise None.
    """
    for algorithm_json in algorithm_jsons:
        if algorithm_json.get("name") == algorithm_name:
            return algorithm_json.get("id")
    return None


def validate_training_params(
    dataset: Dataset,
    dataset_name: str,
    algorithm_name: str,
    input_cross_embodiment_description: CrossEmbodimentDescription,
    output_cross_embodiment_description: CrossEmbodimentDescription,
    algorithm_jsons: list[dict],
) -> None:
    """Validate all training parameters.

    This performs the following checks:
      1) The algorithm name resolves to a known algorithm ID.
      2) All robots referenced in input/output specs exist in the dataset.
      3) All requested input data types are supported by the algorithm and present
         in the dataset.
      4) All requested output data types are supported by the algorithm and present
         in the dataset.

    Args:
        dataset: Dataset metadata object.
        dataset_name: Human-readable dataset name (used for error messages).
        algorithm_name: Algorithm name.
        input_cross_embodiment_description: Input robot data specification
            keyed by robot ID.
        output_cross_embodiment_description: Output robot data specification
            keyed by robot ID.
        algorithm_jsons: List of algorithm metadata dictionaries.

    Raises:
        ValueError: If any validation check fails.
    """
    algorithm_id = get_algorithm_id(algorithm_name, algorithm_jsons)
    _validate_algorithm_exists(algorithm_id, algorithm_name)

    supported_inputs, supported_outputs = _get_data_types_for_algorithms(
        algorithm_name,
        algorithm_jsons,
    )

    _validate_data_specs(
        dataset=dataset,
        dataset_name=dataset_name,
        algorithm_name=algorithm_name,
        cross_embodiment_description=input_cross_embodiment_description,
        supported_data_types=supported_inputs,
        spec_kind="input",
    )

    _validate_data_specs(
        dataset=dataset,
        dataset_name=dataset_name,
        algorithm_name=algorithm_name,
        cross_embodiment_description=output_cross_embodiment_description,
        supported_data_types=supported_outputs,
        spec_kind="output",
    )
