import concurrent
import json

import requests

from ..core.auth import get_auth
from ..core.const import API_URL
from ..core.dataset import Dataset
from ..core.nc_types import DataType


def _get_algorithms() -> list[dict]:
    auth = get_auth()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        org_alg_req = executor.submit(
            requests.get,
            f"{API_URL}/algorithms",
            headers=auth.get_headers(),
            params={"shared": False},
        )
        shared_alg_req = executor.submit(
            requests.get,
            f"{API_URL}/algorithms",
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
    algorithm_config: dict[str, any],
    gpu_type: str,
    num_gpus: int,
    frequency: int,
    input_data_types: list[DataType] = None,
    output_data_types: list[DataType] = None,
) -> dict:
    """
    Start a new training run.

    Args:
        name: Name of the training run
        dataset_name: Name of the dataset to use for training
        algorithm_name: Name of the algorithm to use for training
        algorithm_config: Configuration for the algorithm
        gpu_type: Type of GPU to use for training
        num_gpus: Number of GPUs to use for training
        frequency: Frequency of to synced training data to
    """
    # Get dataset id
    dataset_jsons = Dataset._get_datasets()
    dataset_id = None
    for dataset_json in dataset_jsons:
        if dataset_json["name"] == dataset_name:
            dataset_id = dataset_json["id"]
            break

    if dataset_id is None:
        raise ValueError(f"Dataset {dataset_name} not found")

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
    response = requests.post(
        f"{API_URL}/training/jobs", headers=auth.get_headers(), data=json.dumps(data)
    )
    response.raise_for_status()

    job_data = response.json()
    return job_data


def get_training_job_data(job_id: str) -> dict:
    """
    Check if a training job exists and return its status.

    Args:
        job_id: The ID of the training job.
    Raises:
        requests.exceptions.HTTPError: If the api request returns an error code
        requests.exceptions.RequestException: If there is a problem with the request
    """
    auth = get_auth()
    try:
        response = requests.get(f"{API_URL}/training/jobs", headers=auth.get_headers())
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
    """
    Check if a training job exists and return its status.

    Args:
        job_id: The ID of the training job.
    Raises:
        requests.exceptions.HTTPError: If the api request returns an error code
        requests.exceptions.RequestException: If there is a problem with the request
    """
    try:
        job_data = get_training_job_data(job_id)
        return job_data["status"]
    except Exception as e:
        raise ValueError(f"Error accessing job: {e}")
