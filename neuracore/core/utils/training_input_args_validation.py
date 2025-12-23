"""Training parameter validation utilities.

This module provides validation functions for training parameters, including
algorithm existence, robot existence, and data spec compatibility.
"""

from __future__ import annotations

from neuracore_types import DataType, RobotDataSpec

from neuracore.core.data.dataset import Dataset


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


def validate_robot_existence(
    dataset: Dataset,
    dataset_name: str,
    input_robot_data_spec: RobotDataSpec,
    output_robot_data_spec: RobotDataSpec,
) -> None:
    """Validate that all robots referenced by the input/output specs exist.

    This checks that every robot ID appearing in either the input or output
    robot data specifications is present in the dataset.

    Args:
        dataset: Dataset metadata object.
        dataset_name: Human-readable dataset name (used for error messages).
        input_robot_data_spec: Input robot data specification keyed by robot ID.
        output_robot_data_spec: Output robot data specification keyed by robot ID.

    Raises:
        ValueError: If any robot ID referenced in the specs is not present in the
            dataset.
    """
    robot_ids = input_robot_data_spec.keys() | output_robot_data_spec.keys()
    for robot_id in robot_ids:
        if robot_id not in dataset.robot_ids:
            raise ValueError(
                f"Robot ID {robot_id} not found in dataset {dataset_name}. "
                "Please check the dataset contents."
            )


def validate_algorithm_exists(algorithm_id: str | None, algorithm_name: str) -> None:
    """Validate that the requested algorithm exists.

    Args:
        algorithm_id: Resolved algorithm ID, or None if not found.
        algorithm_name: Algorithm name requested by the user.

    Raises:
        ValueError: If the algorithm ID is None.
    """
    if algorithm_id is None:
        raise ValueError(f"Algorithm {algorithm_name} not found.")


def validate_data_specs(
    dataset: Dataset,
    dataset_name: str,
    algorithm_name: str,
    robot_data_spec: RobotDataSpec,
    supported_data_types: set[DataType],
    spec_kind: str,
) -> None:
    """Validate that a robot data spec is compatible with the dataset and algorithm.

    Validation checks:
      1) Each requested data type must be supported by the algorithm.
      2) Each requested data type must be present in the dataset.

    Args:
        dataset: Dataset metadata object.
        dataset_name: Human-readable dataset name (used for error messages).
        algorithm_name: Algorithm name (used for error messages).
        robot_data_spec: Robot data specification keyed by robot ID.
        supported_data_types: Data types supported by the algorithm for this spec kind.
        spec_kind: Label used in error messages (typically "input" or "output").

    Raises:
        ValueError: If any requested data type is unsupported by the algorithm or
            missing from the dataset.
    """
    for robot_data_types in robot_data_spec.values():
        for data_type in robot_data_types.keys():
            if data_type not in supported_data_types:
                raise ValueError(
                    f"{spec_kind} data type {data_type} is not supported by algorithm "
                    f"{algorithm_name}. Please check the training job requirements."
                )
            if data_type not in dataset.data_types:
                raise ValueError(
                    f"{spec_kind} data type {data_type} is not present in dataset "
                    f"{dataset_name}. Please check the dataset contents."
                )


def get_data_types_for_algorithms(
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
    input_robot_data_spec: RobotDataSpec,
    output_robot_data_spec: RobotDataSpec,
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
        input_robot_data_spec: Input robot data specification keyed by robot ID.
        output_robot_data_spec: Output robot data specification keyed by robot ID.
        algorithm_jsons: List of algorithm metadata dictionaries.

    Raises:
        ValueError: If any validation check fails.
    """
    algorithm_id = get_algorithm_id(algorithm_name, algorithm_jsons)
    validate_algorithm_exists(algorithm_id, algorithm_name)

    supported_inputs, supported_outputs = get_data_types_for_algorithms(
        algorithm_name,
        algorithm_jsons,
    )

    validate_robot_existence(
        dataset=dataset,
        dataset_name=dataset_name,
        input_robot_data_spec=input_robot_data_spec,
        output_robot_data_spec=output_robot_data_spec,
    )

    validate_data_specs(
        dataset=dataset,
        dataset_name=dataset_name,
        algorithm_name=algorithm_name,
        robot_data_spec=input_robot_data_spec,
        supported_data_types=supported_inputs,
        spec_kind="input",
    )

    validate_data_specs(
        dataset=dataset,
        dataset_name=dataset_name,
        algorithm_name=algorithm_name,
        robot_data_spec=output_robot_data_spec,
        supported_data_types=supported_outputs,
        spec_kind="output",
    )
