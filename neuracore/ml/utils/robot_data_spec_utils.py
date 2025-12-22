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
            mapping DataType enums to lists of item names, preserving insertion order.
    """
    result: dict[str, dict[DataType, list[str]]] = {}
    for robot_id, data_type_dict in robot_id_to_data_types.items():
        result[robot_id] = {
            DataType(data_type): list(data_list)
            for data_type, data_list in data_type_dict.items()
        }
    return result


def merge_robot_data_spec(
    data_spec_1: RobotDataSpec,
    data_spec_2: RobotDataSpec,
) -> RobotDataSpec:
    """Merge two robot ID to data types dictionaries.

    Order is preserved: data_spec_1's order takes priority, then data_spec_2's
    items are appended in their original order.

    Args:
        data_spec_1: First dictionary to merge (order takes priority).
        data_spec_2: Second dictionary to merge.

    Returns:
        Merged dictionary with preserved order.
    """
    merged_dict: RobotDataSpec = {}

    # dict.fromkeys() preserves order and removes duplicates
    all_robot_ids = list(dict.fromkeys(list(data_spec_1) + list(data_spec_2)))

    for robot_id in all_robot_ids:
        dt_dict1 = data_spec_1.get(robot_id, {})
        dt_dict2 = data_spec_2.get(robot_id, {})
        all_data_types = list(dict.fromkeys(list(dt_dict1) + list(dt_dict2)))

        merged_dict[robot_id] = {}
        for data_type in all_data_types:
            items = list(dt_dict1.get(data_type, [])) + list(
                dt_dict2.get(data_type, [])
            )
            merged_dict[robot_id][data_type] = list(dict.fromkeys(items))

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
