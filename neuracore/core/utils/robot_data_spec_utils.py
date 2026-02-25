"""Utility functions for robot data specifications.

TODO: Consider moving these functions to neuracore_types/utils/ to avoid
duplication with neuracore_backend/utils.py which has its own copy of
merge_robot_data_spec. Both packages depend on neuracore_types, so it
would be the natural home for these shared utilities.
"""

import re

from neuracore_types import DataType, RobotDataSpec
from ordered_set import OrderedSet

from neuracore.core.robot import get_robot_id_from_name

ID_REGEX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def is_robot_id(string: str) -> bool:
    """Check if a robot identifier is a UUID-style ID.

    Returns:
        True if the identifier matches UUID format, False otherwise.
    """
    return bool(ID_REGEX.match(string))


def convert_str_to_robot_data_spec(
    robot_name_to_data_types: dict[str, dict[str, list[str]]],
) -> RobotDataSpec:
    """Converts string representations of data types to DataType enums.

    Takes a dictionary mapping robot names to dictionaries of
    data type strings and their associated item names,
    and converts the data type strings to DataType enums.

    Args:
        robot_name_to_data_types: A dictionary where keys are robot names and
            values are dictionaries mapping data type strings to lists of item names.

    Returns:
        A dictionary where keys are robot names and values are dictionaries
            mapping DataType enums to lists of item names, preserving insertion order.
    """
    result: dict[str, dict[DataType, list[str]]] = {}
    for robot_name, data_type_dict in robot_name_to_data_types.items():
        result[robot_name] = {
            DataType(data_type): list(data_list)
            for data_type, data_list in data_type_dict.items()
        }
    return result


def convert_robot_data_spec_names_to_ids(
    robot_data_spec: RobotDataSpec,
) -> RobotDataSpec:
    """Convert RobotDataSpec from robot names/IDs to robot IDs.

    Resolves both private and shared robots.

    If name collision occurs between a private and shared robot,
    the private robot will take priority.

    Args:
        robot_data_spec: Robot data spec keyed by robot names or IDs.

    Returns:
        Robot data spec keyed by robot IDs.

    Raises:
        DatasetError: If a robot identifier is ambiguous.
    """
    robot_data_spec_with_ids: RobotDataSpec = {}
    seen_ids = []

    for robot_name_or_id, data_spec in robot_data_spec.items():

        if not is_robot_id(robot_name_or_id):
            robot_name = robot_name_or_id
            # Assume it's a name and try to resolve to an ID
            robot_id = get_robot_id_from_name(robot_name)
            robot_data_spec_with_ids[robot_id] = data_spec
        else:
            robot_id = robot_name_or_id
            robot_data_spec_with_ids[robot_id] = data_spec

        seen_ids.append(robot_id)

    # Check for duplicates and raise an error if found
    if len(seen_ids) != len(set(seen_ids)):
        raise Exception(
            "Duplicate robot identifiers found after conversion. "
            "Please ensure all robot names and IDs are unique."
        )
    return robot_data_spec_with_ids


def merge_robot_data_spec(
    data_spec_1: RobotDataSpec,
    data_spec_2: RobotDataSpec,
) -> RobotDataSpec:
    """Merge two robot name to data types dictionaries.

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
        data_type_dict1 = data_spec_1.get(robot_id, {})
        data_type_dict2 = data_spec_2.get(robot_id, {})
        all_data_types = list(
            dict.fromkeys(list(data_type_dict1) + list(data_type_dict2))
        )

        merged_dict[robot_id] = {}
        for data_type in all_data_types:
            items = list(data_type_dict1.get(data_type, [])) + list(
                data_type_dict2.get(data_type, [])
            )
            merged_dict[robot_id][data_type] = list(dict.fromkeys(items))

    return merged_dict


def extract_data_types(robot_id_to_data_types: RobotDataSpec) -> OrderedSet[DataType]:
    """Extract unique data types from robot name to data types dictionary.

    Args:
        robot_id_to_data_types: A dictionary where keys are robot names and
            values are dictionaries mapping DataType enums to lists of item names.

    Returns:
        OrderedSet of unique data types.
    """
    unique_data_types = OrderedSet()
    for data_types in robot_id_to_data_types.values():
        unique_data_types.update(data_types.keys())
    return unique_data_types
