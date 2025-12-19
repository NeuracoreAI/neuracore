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
from neuracore.core.utils.training_input_args_validation import (
    get_algorithm_id,
    validate_training_params,
)
from neuracore.ml.utils.robot_data_spec_utils import merge_robot_data_spec

from ..core.auth import get_auth
from ..core.const import API_URL
from ..core.data.dataset import Dataset


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


    Returns:
        dict: Training job data including job ID and status

    Raises:
        ValueError: If dataset or algorithm is not found
        requests.exceptions.HTTPError: If the API request fails
        requests.exceptions.RequestException: If there is a network problem
                ConfigError: If there is an error trying to get the current org
    """
    dataset = cast(Dataset, Dataset.get_by_name(dataset_name))
    dataset_id = dataset.id

    # Get algorithm id
    algorithm_jsons = _get_algorithms()
    algorithm_id = get_algorithm_id(algorithm_name, algorithm_jsons)

    validate_training_params(
        dataset,
        dataset_name,
        algorithm_name,
        input_robot_data_spec,
        output_robot_data_spec,
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
                input_robot_data_spec, output_robot_data_spec
            ),
        ),
        input_robot_data_spec=input_robot_data_spec,
        output_robot_data_spec=output_robot_data_spec,
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
    auth = get_auth()
    org_id = get_current_org()
    try:
        response = requests.get(
            f"{API_URL}/org/{org_id}/training/jobs", headers=auth.get_headers()
        )
        response.raise_for_status()

        job = response.json()
        my_job = None
        for job_data in job:
            if job_data["id"] == job_id:
                my_job = job_data
                break
        if my_job is None:
            raise ValueError("Job not found")
        return my_job
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
