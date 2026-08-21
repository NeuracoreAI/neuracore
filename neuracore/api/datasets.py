"""Dataset management utilities.

This module provides functions for creating and retrieving datasets
for robot demonstrations.
"""

import logging
import time

from neuracore_types import Dataset as DatasetModel
from tqdm import tqdm

from neuracore.api.globals import GlobalSingleton
from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.data.dataset import Dataset
from neuracore.core.exceptions import DatasetError
from neuracore.core.utils.http_errors import extract_error_detail
from neuracore.core.utils.http_session import thread_local_session

CLONE_PROGRESS_POLL_INTERVAL_S = 1.0

logger = logging.getLogger(__name__)


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
    GlobalSingleton()._active_dataset = _active_dataset
    return _active_dataset


def merge_datasets(
    name: str, dataset_names: list[str], *, wait: bool = True
) -> Dataset:
    """Merge multiple datasets into a new combined dataset.

    Args:
        name: Name for the new merged dataset
        dataset_names: List of dataset names to merge
        wait: Whether to wait for the merging operation to complete before returning.

    Returns:
        Dataset: The newly created merged dataset

    Raises:
        DatasetError: If any source dataset is not found or merge fails
        requests.exceptions.HTTPError: If the API request fails
    """
    auth = get_auth()
    org_id = get_current_org()

    source_datasets = []
    for dataset_name in dataset_names:
        ds = Dataset.get_by_name(dataset_name, non_exist_ok=True)
        if ds is None:
            raise DatasetError(f"Dataset '{dataset_name}' not found.")
        source_datasets.append(ds)

    session = thread_local_session()
    response = session.post(
        f"{API_URL}/org/{org_id}/datasets/merge",
        headers=auth.get_headers(),
        json={"name": name, "sourceDatasetIds": [ds.id for ds in source_datasets]},
    )
    if not response.ok:
        detail = extract_error_detail(response)
        raise DatasetError(detail or f"{response.status_code} {response.reason}")
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
    if not wait:
        logger.warning(
            "Dataset merging is running in the background; recordings may not be "
            "available immediately."
        )
        GlobalSingleton()._active_dataset_id = merged.id
        GlobalSingleton()._active_dataset = merged
        return merged

    source_recording_ids = {
        recording.id for dataset in source_datasets for recording in dataset
    }
    _wait_for_dataset_recordings(
        merged, len(source_recording_ids), desc="Merging datasets"
    )

    GlobalSingleton()._active_dataset_id = merged.id
    GlobalSingleton()._active_dataset = merged
    return merged


def _wait_for_dataset_recordings(
    dataset: Dataset, total_recordings: int, *, desc: str
) -> None:
    """Wait for a background dataset operation and display its progress."""
    completed_recordings = min(len(dataset), total_recordings)
    if completed_recordings >= total_recordings:
        return

    pbar = tqdm(total=total_recordings, desc=desc, unit="recording")
    pbar.n = completed_recordings
    pbar.refresh()
    try:
        while completed_recordings < total_recordings:
            time.sleep(CLONE_PROGRESS_POLL_INTERVAL_S)
            dataset._num_recordings = None
            new_completed_recordings = min(len(dataset), total_recordings)
            if new_completed_recordings > completed_recordings:
                pbar.update(new_completed_recordings - completed_recordings)
                completed_recordings = new_completed_recordings
    finally:
        pbar.close()


def clone_dataset(
    new_dataset_name: str,
    *,
    source_dataset: Dataset | None = None,
    dataset_name: str | None = None,
    dataset_id: str | None = None,
    wait: bool = True,
) -> Dataset:
    """Clone a dataset and all of its recordings.

    Cloned dataset is set to the active dataset.

    Args:
        source_dataset: Source dataset object to clone
        new_dataset_name: Name for the cloned dataset.
        dataset_name: Explicit source dataset name.
        dataset_id: Explicit source dataset ID.
        wait: Whether to wait for the cloning operation to complete before returning.

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
    elif dataset_id is not None:
        source_dataset = Dataset.get_by_id(dataset_id)

    # Avoid Dataset truthiness here because it delegates to __len__ and may
    # fetch recordings from the API.
    if source_dataset is not None:
        dataset_id = source_dataset.id

    description = getattr(source_dataset, "description", None)
    payload = {
        "name": new_dataset_name,
        "sourceDatasetId": dataset_id,
    }
    if description is not None:
        payload["description"] = description

    auth = get_auth()
    org_id = get_current_org()
    session = thread_local_session()
    response = session.post(
        f"{API_URL}/org/{org_id}/datasets/clone",
        headers=auth.get_headers(),
        json=payload,
    )
    if not response.ok:
        detail = extract_error_detail(response)
        raise DatasetError(detail or f"{response.status_code} {response.reason}")

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

    if not wait:
        logger.warning(
            "Dataset cloning is running in the background; recordings may not be "
            "available immediately."
        )
        GlobalSingleton()._active_dataset_id = cloned.id
        GlobalSingleton()._active_dataset = cloned
        return cloned

    if source_dataset is None:
        raise DatasetError("Source dataset could not be found.")

    total_recordings = len(source_dataset)
    _wait_for_dataset_recordings(cloned, total_recordings, desc="Cloning dataset")

    GlobalSingleton()._active_dataset_id = cloned.id
    GlobalSingleton()._active_dataset = cloned
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
    GlobalSingleton()._active_dataset = _active_dataset
    return _active_dataset
