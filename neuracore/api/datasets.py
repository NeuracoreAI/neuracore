"""Dataset management utilities.

This module provides functions for creating and retrieving datasets
for robot demonstrations.
"""

from typing import Optional

from neuracore.api.globals import GlobalSingleton
from neuracore.core.data.dataset import Dataset


def get_dataset(name: Optional[str] = None, id: Optional[str] = None) -> Dataset:
    """Get a dataset by name or ID.

    Args:
        name: Dataset name
        id: Dataset ID
    Raises:
        ValueError: If neither name nor ID is provided, or if the dataset is not found
    s
    Returns:
        Dataset: The requested dataset instance
    """
    if name is None and id is None:
        raise ValueError("Either name or id must be provided to get_dataset")
    if name is not None and id is not None:
        raise ValueError("Only one of name or id should be provided to get_dataset")
    _active_dataset = None
    if id is not None:
        _active_dataset = Dataset.get_by_id(id)
    elif name is not None:
        _active_dataset = Dataset.get_by_name(name)
    if _active_dataset is None:
        raise ValueError("No active dataset: _active_dataset is None")
    GlobalSingleton()._active_dataset_id = _active_dataset.id
    return _active_dataset


def create_dataset(
    name: str,
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
    shared: bool = False,
) -> Dataset:
    """Create a new dataset for robot demonstrations.

    Args:
        name: Dataset name
        description: Optional description
        tags: Optional list of tags
        shared: Whether the dataset should be shared/open-source.
            Note that setting shared=True is only available to specific
            members allocated by the Neuracore team.

    Returns:
        Dataset: The newly created dataset instance

    Raises:
        DatasetError: If dataset creation fails
    """
    _active_dataset = Dataset.create(name, description, tags, shared)
    GlobalSingleton()._active_dataset_id = _active_dataset.id
    return _active_dataset
