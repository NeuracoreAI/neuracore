"""Utility functions for robot data specifications.

TODO: Consider moving these functions to neuracore_types/utils/ to avoid
duplication with neuracore_backend/utils.py which has its own copy of
merge_cross_embodiment_description. Both packages depend on neuracore_types, so it
would be the natural home for these shared utilities.
"""

import re

from neuracore_types import CrossEmbodimentDescription, DataType
from omegaconf import DictConfig
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


def convert_omegaconf_to_cross_embodiment_description(
    cross_embodiment_cfg: DictConfig,
) -> CrossEmbodimentDescription:
    """Converts string representations of data types to DataType enums.

    Takes a dictionary mapping robot names to dictionaries of
    data type strings and their associated item names,
    and converts the data type strings to DataType enums.

    Args:
        cross_embodiment_cfg: A dictionary where keys are robot names and
            values are dictionaries mapping data type strings to lists of item names.

    Returns:
        A dictionary where keys are robot names and values are dictionaries
            mapping DataType enums to lists of item names, preserving insertion order.
    """
    result: CrossEmbodimentDescription = {}
    for embodiment, embodiment_values in cross_embodiment_cfg.items():
        result[embodiment] = {}
        for data_type_str, item_names in embodiment_values.items():
            try:
                data_type_enum = DataType(data_type_str)
            except ValueError:
                raise ValueError(
                    f"Invalid data type '{data_type_str}' for robot '{embodiment}'. "
                    f"Expected one of {[dt.value for dt in DataType]}."
                )
            result[embodiment][data_type_enum] = item_names
    return result


def convert_cross_embodiment_description_names_to_ids(
    cross_embodiment_description: CrossEmbodimentDescription,
) -> CrossEmbodimentDescription:
    """Convert CrossEmbodimentDescription from robot names/IDs to robot IDs.

    Resolves both private and shared robots.

    If name collision occurs between a private and shared robot,
    the private robot will take priority.

    Args:
        cross_embodiment_description: Robot data spec keyed by robot names or IDs.

    Returns:
        Robot data spec keyed by robot IDs.

    Raises:
        DatasetError: If a robot identifier is ambiguous.
    """
    robot_data_spec_with_ids: CrossEmbodimentDescription = {}
    seen_ids = []

    for (
        robot_name_or_id,
        embodiment_description,
    ) in cross_embodiment_description.items():

        if not is_robot_id(robot_name_or_id):
            robot_name = robot_name_or_id
            # Assume it's a name and try to resolve to an ID
            robot_id = get_robot_id_from_name(robot_name)
            robot_data_spec_with_ids[robot_id] = embodiment_description
        else:
            robot_id = robot_name_or_id
            robot_data_spec_with_ids[robot_id] = embodiment_description

        seen_ids.append(robot_id)

    # Check for duplicates and raise an error if found
    if len(seen_ids) != len(set(seen_ids)):
        raise Exception(
            "Duplicate robot identifiers found after conversion. "
            "Please ensure all robot names and IDs are unique."
        )
    return robot_data_spec_with_ids


def merge_cross_embodiment_description(
    data_spec_1: CrossEmbodimentDescription,
    data_spec_2: CrossEmbodimentDescription,
) -> dict[str, dict[DataType, list[str]]]:
    """Merge two robot name to data types dictionaries.

    Order is preserved: data_spec_1's order takes priority, then data_spec_2's
    items are appended in their original order.


    Args:
        data_spec_1: First dictionary to merge (order takes priority).
        data_spec_2: Second dictionary to merge.

    Returns:
        Merged dictionary with preserved order.
    """
    cross_embodiment_description: CrossEmbodimentDescription = {}

    # dict.fromkeys() preserves order and removes duplicates
    all_robot_ids = list(dict.fromkeys(list(data_spec_1) + list(data_spec_2)))

    for robot_id in all_robot_ids:
        embodiment_desc_1 = data_spec_1.get(robot_id, {})
        embodiment_desc_2 = data_spec_2.get(robot_id, {})
        all_data_types = list(
            dict.fromkeys(list(embodiment_desc_1) + list(embodiment_desc_2))
        )

        cross_embodiment_description[robot_id] = {}
        for data_type in all_data_types:
            values_1 = embodiment_desc_1.get(data_type, {})
            values_2 = embodiment_desc_2.get(data_type, {})
            items1 = (
                list(values_1.values())
                if isinstance(values_1, dict)
                else list(values_1)
            )
            items2 = (
                list(values_2.values())
                if isinstance(values_2, dict)
                else list(values_2)
            )

            cross_embodiment_description[robot_id][data_type] = list(
                dict.fromkeys(items1 + items2)
            )
    return cross_embodiment_description


def extract_data_types(
    robot_id_to_data_types: CrossEmbodimentDescription,
) -> OrderedSet[DataType]:
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
