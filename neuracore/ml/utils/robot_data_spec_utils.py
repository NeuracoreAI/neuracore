"""Utility functions for robot data specifications."""

from neuracore_types import DataType, RobotDataSpec


def convert_str_to_robot_data_spec(
    robot_id_to_data_types: dict[str, dict[str, list[str]]],
) -> RobotDataSpec:
    """Converts string representations of data types to DataType enums.

    Takes a dictionary mapping robot IDs to dictionaries of
    data type strings and their associated item names,
    and converts the data type strings to DataType enums.

    Args:
        robot_id_to_data_types: A dictionary where keys are robot IDs and
            values are dictionaries mapping data type strings to lists of item names.

    Returns:
        A dictionary where keys are robot IDs and values are dictionaries
            mapping DataType enums to lists of item names.
    """
    return {
        robot_id: {DataType(dt): data_list for dt, data_list in dt_dict.items()}
        for robot_id, dt_dict in robot_id_to_data_types.items()
    }


def merge_robot_data_spec(
    data_spec_1: RobotDataSpec,
    data_spec_2: RobotDataSpec,
) -> RobotDataSpec:
    """Merge two robot ID to data types dictionaries.

    Args:
        data_spec_1: First dictionary to merge.
        data_spec_2: Second dictionary to merge.

    Returns:
        Merged dictionary.
    """
    merged_dict: RobotDataSpec = {}

    all_robot_ids = set(data_spec_1.keys()).union(set(data_spec_2.keys()))

    for robot_id in all_robot_ids:
        merged_dict[robot_id] = {}
        data_types1 = data_spec_1.get(robot_id, {})
        data_types2 = data_spec_2.get(robot_id, {})

        all_data_types = set(data_types1.keys()).union(set(data_types2.keys()))

        for data_type in all_data_types:
            items1 = data_types1.get(data_type, [])
            items2 = data_types2.get(data_type, [])
            merged_items = list(set(items1).union(set(items2)))
            merged_dict[robot_id][data_type] = merged_items

    return merged_dict


def extract_data_types(robot_id_to_data_types: RobotDataSpec) -> set[DataType]:
    """Extract unique data types from robot ID to data types dictionary.

    Args:
        robot_id_to_data_types: A dictionary where keys are robot IDs and
            values are dictionaries mapping DataType enums to lists of item names.
    """
    unique_data_types = set()
    for data_types in robot_id_to_data_types.values():
        unique_data_types.update(data_types.keys())
    return unique_data_types
