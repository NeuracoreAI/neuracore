"""Map embodiment name specs onto model tensor slots.

Deploy and training both need the same index-to-name mapping: a list is
positional, a dict keeps its absolute indices, and a mask of 0 drops a slot.
"""

from collections import defaultdict
from collections.abc import Mapping
from typing import cast

import torch
from neuracore_types import (
    BatchedNCData,
    DataType,
    EmbodimentDescription,
    EmbodimentUnion,
)


def list_indexed_names(
    output_names: list[str] | dict[int, str] | dict[str, str],
) -> list[tuple[int, str]]:
    """Normalize output names to explicit tensor index/name pairs.

    Args:
        output_names: Positional names, or a sparse index-to-name map.

    Returns:
        ``(tensor index, name)`` pairs. Dict specs are sorted by index.
    """
    if isinstance(output_names, list):
        return list(enumerate(output_names))
    indexed_output_names = cast(dict[int | str, str], output_names)
    return sorted((int(index), name) for index, name in indexed_output_names.items())


def normalize_data_names(data_names: list[str] | dict[int, str]) -> dict[int, str]:
    """Normalize list/dict specs to the indexed format used by real datasets.

    Args:
        data_names: Positional names, or an index-to-name map.

    Returns:
        Index-to-name map. List order becomes the index.
    """
    if isinstance(data_names, dict):
        return dict(data_names)
    return {index: name for index, name in enumerate(data_names)}


def order_embodiment_items(
    description: EmbodimentDescription,
) -> dict[DataType, list[tuple[int, str]]]:
    """Flatten an embodiment description into index-ordered (index, name) pairs.

    Args:
        description: Per-type index-to-name maps for one robot.

    Returns:
        The same names as sorted ``(index, name)`` pairs per data type.
    """
    return {
        data_type: list_indexed_names(indexed_names)
        for data_type, indexed_names in description.items()
    }


def convert_to_embodiment_description(
    value: EmbodimentUnion | None,
) -> EmbodimentDescription:
    """Normalize list-based sensor specs into indexed embodiment mappings.

    Converts:
        {
            DataType.JOINT_POSITIONS: ["joint1", "joint2"]
        }

    Into:
        {
            DataType.JOINT_POSITIONS: {
                0: "joint1",
                1: "joint2"
            }
        }

    Guarantees:
    - Order is preserved → index defines semantic position
    - Deterministic mapping
    - No mutation of input

    Args:
        value: Per-type lists of sensor names, or None.

    Returns:
        Indexed embodiment description. None becomes an empty description.

    Raises:
        TypeError: If a data type's spec is not a list.
        ValueError: If a list entry is not a string.
    """
    if value is None:
        return {}

    embodiment_description: EmbodimentDescription = {}

    for data_type, items in value.items():
        if not isinstance(items, list):
            raise TypeError(
                f"Expected list for {data_type}, got {type(items).__name__}"
            )

        if any(not isinstance(name, str) for name in items):
            raise ValueError(f"All entries for {data_type} must be strings")

        embodiment_description[data_type] = normalize_data_names(items)

    return embodiment_description


def _is_slot_masked(
    masks: Mapping[DataType, torch.Tensor] | None,
    data_type: DataType,
    index: int,
) -> bool:
    """Return whether this embodiment slot is padding for the current robot.

    Raises:
        IndexError: If ``index`` is outside the mask for ``data_type``.
    """
    if masks is None or data_type not in masks:
        return False
    mask = masks[data_type]
    if mask.ndim >= 2:
        mask = mask[0]
    if index < 0 or index >= mask.shape[0]:
        raise IndexError(
            f"Slot index {index} is outside the {data_type.value} mask of length "
            f"{mask.shape[0]}."
        )
    return float(mask[index]) == 0.0


def assign_names_to_batches(
    batch: dict[DataType, list[BatchedNCData]],
    embodiment_description: EmbodimentDescription,
    masks: Mapping[DataType, torch.Tensor] | None = None,
) -> dict[DataType, dict[str, BatchedNCData]]:
    """Map tensor slots to the names in an embodiment description.

    Sparse embodiment specs keep their absolute tensor indices. Slots with no
    name in the description are left unnamed so padded cross-embodiment
    tensors are not treated as joints. A mask of 0 drops that named slot too.
    A named index outside the mask is an error.

    Args:
        batch: One tensor list per data type.
        embodiment_description: Per-robot names for those tensors.
        masks: Optional per-type mask aligned with those tensor indices.
            Deploy omits this. Training uses ``outputs_mask`` / ``inputs_mask``.

    Returns:
        The named batch. Image and other non-joint types are included when
        they are in the description; callers decide what to persist.

    Raises:
        ValueError: If a data type is missing from the description, or the
            tensor list is shorter than the highest named index.
        IndexError: If a named index is outside that data type's mask.
    """
    named: dict[DataType, dict[str, BatchedNCData]] = defaultdict(dict)

    for data_type, list_of_batched_ncdata in batch.items():
        names = embodiment_description.get(data_type)

        if names is None:
            raise ValueError(f"DataType {data_type} not in embodiment description.")
        indexed_names = list_indexed_names(names)
        required_tensor_count = (
            max(index for index, _ in indexed_names) + 1 if indexed_names else 0
        )
        if len(list_of_batched_ncdata) < required_tensor_count:
            raise ValueError(
                f"Not enough tensors for DataType {data_type}. "
                "Expected at least "
                f"{required_tensor_count}, "
                f"but got {len(list_of_batched_ncdata)}."
            )

        for tensor_idx, name_of_tensor in indexed_names:
            if _is_slot_masked(masks, data_type, tensor_idx):
                continue
            batched_nc_data = list_of_batched_ncdata[tensor_idx]
            named[data_type][name_of_tensor] = batched_nc_data

    return named
