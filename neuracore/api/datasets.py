"""Dataset management utilities.

This module provides functions for creating and retrieving datasets
for robot demonstrations.
"""

from neuracore_types import Dataset as DatasetModel

from neuracore.api.globals import GlobalSingleton
from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.data.dataset import Dataset
from neuracore.core.exceptions import DatasetError
from neuracore.core.utils.http_errors import extract_error_detail
from neuracore.core.utils.http_session import thread_local_session


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
        ds = Dataset.get_by_name(dataset_name, non_exist_ok=True)
        if ds is None:
            raise DatasetError(f"Dataset '{dataset_name}' not found.")
        source_ids.append(ds.id)

    session = thread_local_session()
    response = session.post(
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


def clone_dataset(
    new_dataset_name: str,
    *,
    source_dataset: Dataset | None = None,
    dataset_name: str | None = None,
    dataset_id: str | None = None,
) -> Dataset:
    """Clone a dataset and all of its recordings.

    Cloned dataset is set to the active dataset.

    Args:
        source_dataset: Source dataset object to clone
        new_dataset_name: Name for the cloned dataset.
        dataset_name: Explicit source dataset name.
        dataset_id: Explicit source dataset ID.

    Returns:
        Dataset: The newly cloned dataset.

    Raises:
        ValueError: If the source or new dataset name is missing, or if multiple
            source arguments are provided.
        DatasetError: If the source dataset cannot be found or cloning fails.
    """
    # Checks only one of dataset, dataset_name, or dataset_id is provided
    assert (
        sum(arg is not None for arg in [source_dataset, dataset_name, dataset_id]) == 1
    ), "Exactly one of dataset, dataset_name, or dataset_id must be provided"

    # If dataset_name is provided, resolve it to a dataset object
    if dataset_name is not None:
        source_dataset = Dataset.get_by_name(dataset_name)

    # Avoid Dataset truthiness here because it delegates to __len__ and may
    # fetch recordings from the API.
    if source_dataset is not None:
        dataset_id = source_dataset.id

    auth = get_auth()
    org_id = get_current_org()
    session = thread_local_session()
    response = session.post(
        f"{API_URL}/org/{org_id}/datasets/clone",
        headers=auth.get_headers(),
        json={"name": new_dataset_name, "sourceDatasetId": dataset_id},
    )
    if not response.ok:
        detail = extract_error_detail(response)
        error_message = detail or f"{response.status_code} {response.reason}"
        raise DatasetError(f"Failed to clone dataset: {error_message}")

    dataset_model = DatasetModel.model_validate(response.json())
    cloned = Dataset(
        id=dataset_model.id,
        org_id=org_id,
        name=dataset_model.name,
        size_bytes=dataset_model.size_bytes,
        tags=dataset_model.tags,
        is_shared=dataset_model.is_shared,
        description=dataset_model.description,
        data_types=list(dataset_model.all_data_types.keys()),
    )

    GlobalSingleton()._active_dataset_id = cloned.id
    return cloned


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
