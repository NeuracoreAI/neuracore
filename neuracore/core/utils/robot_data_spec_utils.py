"""Utility functions for robot data specifications.

TODO: Consider moving these functions to neuracore_types/utils/ to avoid
duplication with neuracore_backend/utils.py which has its own copy of
merge_cross_embodiment_description. Both packages depend on neuracore_types, so it
would be the natural home for these shared utilities.
"""

import re
from collections.abc import Mapping
from pathlib import Path

import requests
from neuracore_types import (
    CrossEmbodimentDescription,
    CrossEmbodimentUnion,
    DataType,
    EmbodimentDescription,
)
from omegaconf import DictConfig
from ordered_set import OrderedSet

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.robot import get_robot_id_from_name
from neuracore.ml.utils.nc_archive import (
    load_cross_embodiment_descriptions_from_nc_archive,
)

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
        A dictionary where keys are robot ids/name and values are dictionaries
            mapping DataType
        to a dict of index to item name.
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
                    f"Expected one of {[data_type.value for data_type in DataType]}."
                )
            result[embodiment][data_type_enum] = dict(item_names)
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
) -> CrossEmbodimentUnion:
    """Merge two cross-embodiment descriptions into ordered name lists.

    Order is preserved: data_spec_1's order takes priority, then data_spec_2's
    items are appended in their original order. Dict-backed item specs are
    merged by their values, producing the list[str] order spec expected by
    synchronization and ordering code.

    Args:
        data_spec_1: First dictionary to merge (order takes priority).
        data_spec_2: Second dictionary to merge.

    Returns:
        Merged dictionary mapping each robot and data type to an ordered list of
        unique item names.
    """

    def _normalize_item_names(
        values: dict[int, str] | list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        if values is None:
            return []
        if isinstance(values, Mapping):
            return list(values.values())
        return list(values)

    cross_embodiment_description: CrossEmbodimentUnion = {}

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
            items1 = _normalize_item_names(embodiment_desc_1.get(data_type))
            items2 = _normalize_item_names(embodiment_desc_2.get(data_type))

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


def normalize_embodiment_description(
    embodiment_description: dict[DataType | str, object],
) -> EmbodimentDescription:
    """Normalize embodiment description keys to DataType enum values."""
    return {
        data_type if isinstance(data_type, DataType) else DataType(data_type): names
        for data_type, names in embodiment_description.items()
    }


def resolve_embodiment_descriptions(
    input_cross_embodiment_description: CrossEmbodimentDescription,
    output_cross_embodiment_description: CrossEmbodimentDescription,
    robot_id: str,
) -> tuple[EmbodimentDescription, EmbodimentDescription]:
    """Resolve concrete input/output embodiments for a specific robot."""
    if robot_id not in input_cross_embodiment_description:
        raise ValueError(
            f"Robot ID '{robot_id}' not found in input cross-embodiment description."
        )
    if robot_id not in output_cross_embodiment_description:
        raise ValueError(
            f"Robot ID '{robot_id}' not found in output cross-embodiment description."
        )
    return (
        normalize_embodiment_description(input_cross_embodiment_description[robot_id]),
        normalize_embodiment_description(output_cross_embodiment_description[robot_id]),
    )


def resolve_embodiment_descriptions_with_override(
    input_embodiment_description: EmbodimentDescription | None = None,
    output_embodiment_description: EmbodimentDescription | None = None,
    robot_id: str | None = None,
    job_id: str | None = None,
    model_file: Path | None = None,
    input_cross_embodiment_description: CrossEmbodimentDescription | None = None,
    output_cross_embodiment_description: CrossEmbodimentDescription | None = None,
) -> tuple[EmbodimentDescription, EmbodimentDescription]:
    """Resolve embodiments from archive/cross-specs and apply explicit overrides.

    1. Prioritize explicit overrides of embodiment descriptions.
    2. If no explicit overrides, resolve using robot ID from
       cross-embodiment descriptions.
    3. If no cross-embodiment descriptions, resolve from training metadata
       or model archive.
    """
    resolved_input_embodiment_description = None
    resolved_output_embodiment_description = None

    if (
        input_embodiment_description is not None
        and output_embodiment_description is not None
    ):
        return input_embodiment_description, output_embodiment_description

    # Retrieve cross-embodiment descriptions and resolve embodiments from them
    if robot_id is not None:
        if (
            not input_cross_embodiment_description
            or not output_cross_embodiment_description
        ):
            if job_id is not None:
                auth = get_auth()
                org_id = get_current_org()
                response = requests.get(
                    f"{API_URL}/org/{org_id}/training/jobs/{job_id}",
                    headers=auth.get_headers(),
                    timeout=30,
                )
                if response.status_code == 200:
                    training_job = response.json()
                    input_cross_embodiment_description = training_job.get(
                        "input_cross_embodiment_description"
                    )
                    output_cross_embodiment_description = training_job.get(
                        "output_cross_embodiment_description"
                    )

            elif model_file is not None:
                (
                    archive_input_cross_embodiment_description,
                    archive_output_cross_embodiment_description,
                ) = load_cross_embodiment_descriptions_from_nc_archive(model_file)
                if input_cross_embodiment_description is None:
                    input_cross_embodiment_description = (
                        archive_input_cross_embodiment_description
                    )
                if output_cross_embodiment_description is None:
                    output_cross_embodiment_description = (
                        archive_output_cross_embodiment_description
                    )

        if (
            input_cross_embodiment_description is None
            or output_cross_embodiment_description is None
        ):
            raise ValueError(
                "Must provide both input_cross_embodiment_description and "
                "output_cross_embodiment_description, or provide robot_id "
                "with job_id or model_file to load them from training "
                "metadata or the model archive."
            )

        (
            resolved_input_embodiment_description,
            resolved_output_embodiment_description,
        ) = resolve_embodiment_descriptions(
            input_cross_embodiment_description=input_cross_embodiment_description,
            output_cross_embodiment_description=output_cross_embodiment_description,
            robot_id=robot_id,
        )

    if input_embodiment_description is not None:
        resolved_input_embodiment_description = input_embodiment_description
    if output_embodiment_description is not None:
        resolved_output_embodiment_description = output_embodiment_description

    if (
        resolved_input_embodiment_description is None
        or resolved_output_embodiment_description is None
    ):
        raise ValueError(
            "Must provide both input_embodiment_description and "
            "output_embodiment_description, or provide robot_id with job_id/model_file "
            "to load them from training metadata or the model archive."
        )
    return resolved_input_embodiment_description, resolved_output_embodiment_description
