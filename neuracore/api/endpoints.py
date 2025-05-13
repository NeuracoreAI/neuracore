import json
from typing import Optional

import requests

from ..core.auth import get_auth
from ..core.const import API_URL
from ..core.endpoint import EndpointPolicy
from ..core.endpoint import connect_endpoint as _connect_endpoint
from ..core.endpoint import connect_local_endpoint as _connect_local_endpoint


def connect_endpoint(
    endpoint_name: str, robot_name: Optional[str] = None, instance: Optional[int] = 0
) -> EndpointPolicy:
    """
    Connect to a deployed model endpoint.

    Args:
        endpoint_name: Name of the deployed endpoint
        robot_name: robot name that the data is being logged for. If not
            provided, uses the last initialized robot.
        instance: instance number of the robot. Defaults to 0.

    Returns:
        EndpointPolicy: Policy object that can be used for predictions

    Raises:
        EndpointError: If endpoint connection fails
    """
    return _connect_endpoint(
        endpoint_name=endpoint_name, robot_name=robot_name, instance=instance
    )


def connect_local_endpoint(
    path_to_model: Optional[str] = None,
    train_run_name: Optional[str] = None,
    port: int = 8080,
    robot_name: Optional[str] = None,
    instance: Optional[int] = 0,
) -> EndpointPolicy:
    """
    Connect to a local model endpoint.

    Can supply either path_to_model or train_run_name, but not both.

    Args:
        path_to_model: Path to the local .mar model
        train_run_name: Optional train run name
        port: Port to connect to the local endpoint
        robot_name: robot name that the data is being logged for. If not
            provided, uses the last initialized robot.
        instance: instance number of the robot. Defaults to 0.

    Returns:
        EndpointPolicy: Policy object that can be used for predictions

    Raises:
        EndpointError: If endpoint connection fails
    """
    return _connect_local_endpoint(
        robot_name=robot_name,
        instance=instance,
        path_to_model=path_to_model,
        train_run_name=train_run_name,
        port=port,
    )


def deploy_model(job_id: str, name: str) -> dict:
    """
    Deploy a trained model to an endpoint.

    Args:
        job_id: The ID of the training job.
        name: The name of the endpoint.
    Raises:
        requests.exceptions.HTTPError: If the api request returns an error code
        requests.exceptions.RequestException: If there is a problem with the request
    """
    auth = get_auth()
    try:
        response = requests.post(
            f"{API_URL}/models/deploy",
            headers=auth.get_headers(),
            data=json.dumps({"training_id": job_id, "name": name}),
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise ValueError(f"Error deploying model: {e}")


def get_endpoint_status(endpoint_id: str) -> dict:
    """
    Get the status of an endpoint.
    Args:
        endpoint_id: The ID of the endpoint.
    Raises:
        requests.exceptions.HTTPError: If the api request returns an error code
        requests.exceptions.RequestException: If there is a problem with the request
    """
    auth = get_auth()
    try:
        response = requests.get(
            f"{API_URL}/models/endpoints/{endpoint_id}", headers=auth.get_headers()
        )
        response.raise_for_status()
        return response.json()["status"]
    except Exception as e:
        raise ValueError(f"Error getting endpoint status: {e}")


def delete_endpoint(endpoint_id: str) -> None:
    """
    Delete an endpoint.
    Args:
        endpoint_id: The ID of the endpoint.

    Raises:
        requests.exceptions.HTTPError: If the api request returns an error code
        requests.exceptions.RequestException: If there is a problem with the request
    """
    auth = get_auth()
    try:
        response = requests.delete(
            f"{API_URL}/models/endpoints/{endpoint_id}", headers=auth.get_headers()
        )
        response.raise_for_status()
    except Exception as e:
        raise ValueError(f"Error deleting endpoint: {e}")
