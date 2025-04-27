from typing import Optional

from ..core.dataset import Dataset
from .globals import GlobalSingleton


def get_dataset(name: str) -> Dataset:
    """Get a dataset by name.

    Args:
        name: Dataset name

    """
    _active_dataset = Dataset.get(name)
    GlobalSingleton()._active_dataset_id = _active_dataset.id
    return _active_dataset


def create_dataset(
    name: str,
    description: Optional[str] = None,
    tags: Optional[list[str]] = None,
    shared: bool = False,
) -> Dataset:
    """
    Create a new dataset for robot demonstrations.

    Args:
        name: Dataset name
        description: Optional description
        tags: Optional list of tags

    Raises:
        DatasetError: If dataset creation fails
    """
    _active_dataset = Dataset.create(name, description, tags, shared)
    GlobalSingleton()._active_dataset_id = _active_dataset.id
    return _active_dataset
