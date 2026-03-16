"""Training job management utilities.

This module provides functions for starting and monitoring training jobs,
including algorithm discovery, dataset resolution, and job status tracking.
"""

import concurrent
import sys
from typing import Any, cast

import requests
from neuracore_types import (
    GPUType,
    RobotDataSpec,
    SynchronizationDetails,
    TrainingJobRequest,
)

from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.utils.robot_data_spec_utils import (
    convert_robot_data_spec_names_to_ids,
    merge_robot_data_spec,
)
from neuracore.core.utils.training_input_args_validation import (
    get_algorithm_id,
    validate_training_params,
)

from ..core.auth import get_auth
from ..core.const import API_URL
from ..core.data.dataset import Dataset


def _resolve_next_name(base_name: str, existing_names: set[str]) -> str:
    """Return the next available name, optionally with _1, _2, ... suffix.

    If base_name is not in use, return it. Otherwise return base_name_N
    for the smallest N >= 1 such that base_name_N is not in existing_names.
    "In use" means exact match or names of the form base_name_<integer>.

    Args:
        base_name: Desired base name.
        existing_names: Set of names already in use.

    Returns:
        base_name or base_name_N for the next free N.
    """
    taken = {
        name
        for name in existing_names
        if name == base_name
        or (name.startswith(base_name + "_") and name[len(base_name) + 1 :].isdigit())
    }
    if base_name not in taken:
        return base_name
    suffix = 1
    while f"{base_name}_{suffix}" in taken:
        suffix += 1
    return f"{base_name}_{suffix}"


def _get_algorithms() -> list[dict]:
    """Retrieve all available algorithms from the API.

    Fetches both organization-specific and shared algorithms concurrently.

    Returns:
        list[dict]: List of algorithm dictionaries containing algorithm metadata

    Raises:
        requests.exceptions.HTTPError: If the API request fails
        requests.exceptions.RequestException: If there is a network problem
        ConfigError: If there is an error trying to get the current org
    """
    auth = get_auth()
    org_id = get_current_org()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        org_alg_req = executor.submit(
            requests.get,
            f"{API_URL}/org/{org_id}/algorithms",
            headers=auth.get_headers(),
            params={"shared": False},
        )
        shared_alg_req = executor.submit(
            requests.get,
            f"{API_URL}/org/{org_id}/algorithms",
            headers=auth.get_headers(),
            params={"shared": True},
        )
        org_alg, shared_alg = org_alg_req.result(), shared_alg_req.result()
    org_alg.raise_for_status()
    shared_alg.raise_for_status()
    return org_alg.json() + shared_alg.json()


def start_training_run(
    name: str,
    dataset_name: str,
    algorithm_name: str,
    algorithm_config: dict[str, Any],
    gpu_type: str,
    num_gpus: int,
    frequency: int,
    input_robot_data_spec: RobotDataSpec,
    output_robot_data_spec: RobotDataSpec,
    max_delay_s: float = sys.float_info.max,
    allow_duplicates: bool = True,
    name_auto_increment: bool = False,
) -> dict:
    """Start a new training run.

    Args:
        name: Name of the training run
        dataset_name: Name of the dataset to use for training
        algorithm_name: Name of the algorithm to use for training
        algorithm_config: Configuration parameters for the algorithm
        gpu_type: Type of GPU to use for training (e.g., "A100", "V100")
        num_gpus: Number of GPUs to use for training
        frequency: Frequency to sync training data to (in Hz)
        input_robot_data_spec: Input robot data specification.
        output_robot_data_spec: Output robot data specification.
        max_delay_s: Maximum allowable delay for data synchronization (in seconds)
        allow_duplicates: Whether to allow duplicate data during synchronization
        name_auto_increment: If True and a job with this name already exists, use
            name_1, name_2, ... instead of failing or duplicating the name.

    Returns:
        dict: Training job data including job ID and status

    Raises:
        ValueError: If dataset or algorithm is not found
        requests.exceptions.HTTPError: If the API request fails
        requests.exceptions.RequestException: If there is a network problem
                ConfigError: If there is an error trying to get the current org
    """
    if name_auto_increment:
        jobs = get_training_jobs()
        existing_names = {j["name"] for j in jobs if isinstance(j.get("name"), str)}
        name = _resolve_next_name(name, existing_names)

    dataset = cast(Dataset, Dataset.get_by_name(dataset_name))
    dataset_id = dataset.id
    input_robot_data_spec_with_ids = convert_robot_data_spec_names_to_ids(
        input_robot_data_spec,
    )
    output_robot_data_spec_with_ids = convert_robot_data_spec_names_to_ids(
        output_robot_data_spec,
    )

    # Get algorithm id
    algorithm_jsons = _get_algorithms()
    algorithm_id = get_algorithm_id(algorithm_name, algorithm_jsons)

    validate_training_params(
        dataset,
        dataset_name,
        algorithm_name,
        input_robot_data_spec_with_ids,
        output_robot_data_spec_with_ids,
        algorithm_jsons,
    )

    data = TrainingJobRequest(
        dataset_id=dataset_id,
        name=name,
        algorithm_id=algorithm_id,
        algorithm_config=algorithm_config,
        gpu_type=GPUType(gpu_type),
        num_gpus=num_gpus,
        synchronization_details=SynchronizationDetails(
            frequency=frequency,
            max_delay_s=max_delay_s,
            allow_duplicates=allow_duplicates,
            robot_data_spec=merge_robot_data_spec(
                input_robot_data_spec_with_ids, output_robot_data_spec_with_ids
            ),
        ),
        input_robot_data_spec=input_robot_data_spec_with_ids,
        output_robot_data_spec=output_robot_data_spec_with_ids,
    )

    auth = get_auth()
    org_id = get_current_org()
    response = requests.post(
        f"{API_URL}/org/{org_id}/training/jobs",
        headers=auth.get_headers(),
        json=data.model_dump(mode="json"),
    )

    response.raise_for_status()

    job_data = response.json()
    return job_data


def resume_training_run(job_id: str, additional_epochs: int) -> dict:
    """Resume a training run.

    Args:
        job_id: The ID of the training job
        additional_epochs: The number of additional epochs to run

    Returns:
        dict: Training job data including job ID and status

    Raises:
        ValueError: If the job is not found or there is an error accessing the job
        requests.exceptions.HTTPError: If the API request returns an error code
        requests.exceptions.RequestException: If there is a problem with the request
    """
    auth = get_auth()
    org_id = get_current_org()
    try:
        response = requests.post(
            f"{API_URL}/org/{org_id}/training/jobs/{job_id}/resume/{additional_epochs}",
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise ValueError(f"Error resuming job {job_id}: {e}")


def get_training_jobs() -> list[dict]:
    """List all training jobs for the current organization.

    Returns:
        List of training job dicts (id, name, status, etc.).

    Raises:
        requests.exceptions.HTTPError: If the API request fails.
        ConfigError: If there is an error trying to get the current org.
    """
    auth = get_auth()
    org_id = get_current_org()
    response = requests.get(
        f"{API_URL}/org/{org_id}/training/jobs",
        headers=auth.get_headers(),
    )
    response.raise_for_status()
    return response.json()


def get_training_job_data(job_id: str) -> dict:
    """Retrieve complete data for a training job.

    Args:
        job_id: The ID of the training job

    Returns:
        dict: Complete job data including status, configuration, and metadata

    Raises:
        ValueError: If the job is not found or there is an error accessing the job
        requests.exceptions.HTTPError: If the API request returns an error code
        requests.exceptions.RequestException: If there is a problem with the request
        ConfigError: If there is an error trying to get the current org
    """
    try:
        jobs = get_training_jobs()
        for job_data in jobs:
            if job_data.get("id") == job_id:
                return job_data
        raise ValueError("Job not found")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error accessing job: {e}")


def get_training_job_status(job_id: str) -> str:
    """Get the current status of a training job.

    Args:
        job_id: The ID of the training job

    Returns:
        str: Current status of the training job (e.g., "running", "completed", "failed")

    Raises:
        ValueError: If the job is not found or there is an error accessing the job
        requests.exceptions.HTTPError: If the API request returns an error code
        requests.exceptions.RequestException: If there is a problem with the request
    """
    try:
        job_data = get_training_job_data(job_id)
        return job_data["status"]
    except Exception as e:
        raise ValueError(f"Error accessing job: {e}")


def get_training_job_logs(
    job_id: str, max_entries: int = 100, severity_filter: str | None = None
) -> dict:
    """Retrieve logs for a training job from neuracore backend.

    Args:
        job_id: The ID of the training job.
        max_entries: Maximum number of log entries to return.
        severity_filter: Optional log severity filter (for example: "ERROR").

    Returns:
        dict: Cloud compute logs payload.

    Raises:
        ValueError: If logs cannot be retrieved.
        requests.exceptions.HTTPError: If the API request returns an error code.
        requests.exceptions.RequestException: If there is a problem with the request.
        ConfigError: If there is an error trying to get the current org.
    """
    auth = get_auth()
    org_id = get_current_org()
    params: dict[str, str | int] = {"max_entries": max_entries}
    if severity_filter is not None:
        params["severity_filter"] = severity_filter

    try:
        response = requests.get(
            f"{API_URL}/org/{org_id}/training/jobs/{job_id}/logs",
            headers=auth.get_headers(),
            params=params,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise ValueError(f"Error getting training job logs: {e}")


def delete_training_job(job_id: str) -> None:
    """Delete a training job and free its resources.

    Args:
        job_id: The ID of the training job to delete

    Raises:
        ValueError: If there is an error deleting the job
        requests.exceptions.HTTPError: If the API request returns an error code
        requests.exceptions.RequestException: If there is a problem with the request
        ConfigError: If there is an error trying to get the current org
    """
    auth = get_auth()
    org_id = get_current_org()
    try:
        response = requests.delete(
            f"{API_URL}/org/{org_id}/training/jobs/{job_id}",
            headers=auth.get_headers(),
        )
        response.raise_for_status()
    except Exception as e:
        raise ValueError(f"Error deleting training job: {e}")
