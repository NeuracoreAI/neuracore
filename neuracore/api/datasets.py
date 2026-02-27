"""Dataset management utilities.

This module provides functions for creating and retrieving datasets
for robot demonstrations.
"""

import requests
from neuracore_types import Dataset as DatasetModel

from neuracore.api.globals import GlobalSingleton
from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.data.dataset import Dataset
from neuracore.core.exceptions import DatasetError


def get_dataset(name: str | None = None, id: str | None = None) -> Dataset:
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
        raise ValueError(f"No Dataset found with the given name: {name} or ID: {id}")
    GlobalSingleton()._active_dataset_id = _active_dataset.id
    return _active_dataset


def merge_datasets(name: str, dataset_names: list[str]) -> Dataset:
    """Merge multiple datasets into a new combined dataset.

    Args:
        name: Name for the new merged dataset
        dataset_names: List of dataset names to merge

    Returns:
        Dataset: The newly created merged dataset

    Raises:
        DatasetError: If any source dataset is not found or merge fails
        requests.exceptions.HTTPError: If the API request fails
    """
    auth = get_auth()
    org_id = get_current_org()

    source_ids = []
    for dataset_name in dataset_names:
        ds: Dataset = Dataset.get_by_name(dataset_name)  # type: ignore
        source_ids.append(ds.id)

    response = requests.post(
        f"{API_URL}/org/{org_id}/datasets/merge",
        headers=auth.get_headers(),
        json={"name": name, "sourceDatasetIds": source_ids},
    )
    if not response.ok:
        raise DatasetError(
            f"Failed to merge datasets: {response.status_code} {response.text}"
        )
    dataset_model = DatasetModel.model_validate(response.json())
    merged = Dataset(
        id=dataset_model.id,
        org_id=org_id,
        name=dataset_model.name,
        size_bytes=dataset_model.size_bytes,
        tags=dataset_model.tags,
        is_shared=dataset_model.is_shared,
        data_types=list(dataset_model.all_data_types.keys()),
    )
    GlobalSingleton()._active_dataset_id = merged.id
    return merged


def create_dataset(
    name: str,
    description: str | None = None,
    tags: list[str] | None = None,
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
