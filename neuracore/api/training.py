"""Training job management utilities.

This module provides functions for starting and monitoring training jobs,
including algorithm discovery, dataset resolution, and job status tracking.
"""

import concurrent
import json
from typing import Any, Optional, cast

import requests
from neuracore_types import DataType

from neuracore.core.config.get_current_org import get_current_org

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
    input_data_types: Optional[list[DataType]] = None,
    output_data_types: Optional[list[DataType]] = None,
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
        input_data_types: Optional list of input data types. If not provided,
            uses algorithm's supported input data types
        output_data_types: Optional list of output data types. If not provided,
            uses algorithm's supported output data types

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
    algorithm_id = None
    for algorithm_json in algorithm_jsons:
        if algorithm_json["name"] == algorithm_name:
            algorithm_id = algorithm_json["id"]
            if input_data_types is None:
                input_data_types = [
                    DataType(supported_input_data_type)
                    for supported_input_data_type in algorithm_json[
                        "supported_input_data_types"
                    ]
                ]
            if output_data_types is None:
                output_data_types = [
                    DataType(supported_output_data_type)
                    for supported_output_data_type in algorithm_json[
                        "supported_output_data_types"
                    ]
                ]
            break

    if algorithm_id is None:
        raise ValueError(f"Algorithm {algorithm_name} not found")

    data = {
        "name": name,
        "dataset_id": dataset_id,
        "algorithm_id": algorithm_id,
        "algorithm_config": algorithm_config,
        "gpu_type": gpu_type,
        "num_gpus": num_gpus,
        "frequency": str(frequency),
        "input_data_types": input_data_types,
        "output_data_types": output_data_types,
    }

    auth = get_auth()
    org_id = get_current_org()
    response = requests.post(
        f"{API_URL}/org/{org_id}/training/jobs",
        headers=auth.get_headers(),
        data=json.dumps(data),
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
