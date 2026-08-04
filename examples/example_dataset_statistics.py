"""This example demonstrates how you can retrieve the statistics of a dataset
from the Neuracore platform and print them."""

import numpy as np
from neuracore_types import (
    CrossEmbodimentDescription,
    CrossEmbodimentUnion,
    DataItemStats,
    DataType,
    NCDataStats,
)

import neuracore as nc

# Statistics are reported for the data a policy reads and the data it predicts,
# so each role gets its own set of data types.
INPUT_DATA_TYPES = [DataType.JOINT_POSITIONS, DataType.RGB_IMAGES]
OUTPUT_DATA_TYPES = [DataType.JOINT_POSITIONS]


def format_array(values: np.ndarray) -> str:
    """Format a statistics array compactly, however many dimensions it has."""
    if values.size == 0:
        return "-"
    return np.array2string(
        values,
        precision=4,
        suppress_small=True,
        threshold=8,
        edgeitems=3,
        separator=", ",
    )


def print_item_statistics(stats: NCDataStats, indent: str) -> None:
    """Print every statistic a data item carries.

    Each data type reports its own set of attributes: a joint has one value, a
    camera has its frame plus intrinsics and extrinsics, and so on.

    Args:
        stats: The statistics for one data item.
        indent: Leading whitespace for the printed lines.
    """
    for attribute_name in vars(stats):
        attribute = getattr(stats, attribute_name)
        if not isinstance(attribute, DataItemStats):
            continue

        print(f"{indent}{attribute_name}:")
        print(f"{indent}  observations: {format_array(attribute.count)}")
        print(f"{indent}  mean:         {format_array(attribute.mean)}")
        print(f"{indent}  std:          {format_array(attribute.std)}")
        print(f"{indent}  min:          {format_array(attribute.min)}")
        print(f"{indent}  max:          {format_array(attribute.max)}")
        print(f"{indent}  q01:          {format_array(attribute.q01)}")
        print(f"{indent}  q99:          {format_array(attribute.q99)}")


def item_names_by_index(
    cross_embodiment_description: CrossEmbodimentDescription,
    data_type: DataType,
) -> dict[int, str]:
    """Collect the item name at each canonical index across every robot.

    Statistics are aggregated across robots into one list per data type, indexed
    by position rather than by name, so this recovers what each position holds.

    Args:
        cross_embodiment_description: The description the statistics were
            calculated for.
        data_type: The data type to collect names for.

    Returns:
        A mapping of canonical index to the name (or names) found there.
    """
    names_by_index: dict[int, set[str]] = {}
    for data_types in cross_embodiment_description.values():
        for index, name in data_types.get(data_type, {}).items():
            names_by_index.setdefault(index, set()).add(name)
    return {index: " / ".join(sorted(names)) for index, names in names_by_index.items()}


def print_statistics(
    role: str,
    statistics: dict[DataType, list[NCDataStats]],
    cross_embodiment_description: CrossEmbodimentDescription,
) -> None:
    """Print one role's statistics, data type by data type.

    Args:
        role: Either "input" or "output".
        statistics: The statistics for that role, dense by canonical index.
        cross_embodiment_description: The description they were calculated for.
    """
    print(f"\n{'=' * 70}\n{role.upper()} STATISTICS\n{'=' * 70}")
    if not statistics:
        print("No statistics for this role.")
        return

    for data_type, stats_by_index in statistics.items():
        names = item_names_by_index(cross_embodiment_description, data_type)
        print(f"\n{data_type.value} ({len(stats_by_index)} items)")
        for index, stats in enumerate(stats_by_index):
            print(f"  [{index}] {names.get(index, '<unnamed>')}")
            print_item_statistics(stats, indent="      ")


def main():
    """Print the statistics of a synchronized dataset."""
    nc.login()

    # Freiburg Franka Play is one of the many public datasets
    dataset = nc.get_dataset("NYU ROT")

    # Statistics are calculated over a synchronized dataset, so the data types
    # being described have to have been synchronized first.
    data_types = sorted(set(INPUT_DATA_TYPES) | set(OUTPUT_DATA_TYPES))
    cross_embodiment_union: CrossEmbodimentUnion = {}
    input_description: CrossEmbodimentDescription = {}
    output_description: CrossEmbodimentDescription = {}
    for robot_id in dataset.robot_ids:
        data_type_to_names = dataset.get_full_embodiment_description(robot_id)
        cross_embodiment_union[robot_id] = {
            data_type: list(data_type_to_names[data_type].values())
            for data_type in data_types
        }
        # A description keeps the index of each item, because statistics are
        # aggregated across robots by position.
        input_description[robot_id] = {
            data_type: data_type_to_names[data_type] for data_type in INPUT_DATA_TYPES
        }
        output_description[robot_id] = {
            data_type: data_type_to_names[data_type] for data_type in OUTPUT_DATA_TYPES
        }

    synced_dataset = dataset.synchronize(
        frequency=20,
        cross_embodiment_union=cross_embodiment_union,
    )
    print(f"Number of episodes: {len(synced_dataset)}")

    # Statistics are calculated on the platform, so this reports progress while
    # it waits. Asking again for the same recordings and descriptions reuses the
    # result rather than recalculating it.
    statistics = synced_dataset.calculate_statistics(
        input_cross_embodiment_description=input_description,
        output_cross_embodiment_description=output_description,
    )

    print_statistics("input", statistics.dataset_statistics["input"], input_description)
    print_statistics(
        "output", statistics.dataset_statistics["output"], output_description
    )


if __name__ == "__main__":
    main()
