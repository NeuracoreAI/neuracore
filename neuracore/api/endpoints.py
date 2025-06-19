"""Model endpoint management and connection API for Neuracore.

This module provides functionality for connecting to deployed model endpoints,
managing local model endpoints, and handling the lifecycle of model deployments
including deployment, status monitoring, and deletion operations.
"""

import json
from typing import Optional

import requests

from neuracore.core.config.get_current_org import get_current_org

from ..core.auth import get_auth
from ..core.const import API_URL
from ..core.endpoint import EndpointPolicy
from ..core.endpoint import connect_endpoint as _connect_endpoint
from ..core.endpoint import connect_local_endpoint as _connect_local_endpoint


def connect_endpoint(
    endpoint_name: str, robot_name: Optional[str] = None, instance: int = 0
) -> EndpointPolicy:
    """Connect to a deployed model endpoint for inference.

    Establishes a connection to a model endpoint that has been deployed on the
    Neuracore platform. The endpoint can be used to make predictions with the
    deployed model, and data logging is associated with the specified robot.

    Args:
        endpoint_name: Name of the deployed endpoint to connect to.
        robot_name: Robot name that predictions and data will be associated with.
            If not provided, uses the last initialized robot from global state.
        instance: Instance number of the robot for multi-instance deployments.

    Returns:
        Policy object that provides an interface for making predictions
        with the deployed model.

    Raises:
        EndpointError: If the endpoint connection fails due to invalid endpoint
            name, authentication issues, or network problems.
        ConfigError: If there is an error trying to get the current org
    """
    return _connect_endpoint(
        endpoint_name=endpoint_name, robot_name=robot_name, instance=instance
    )


def connect_local_endpoint(
    path_to_model: Optional[str] = None,
    train_run_name: Optional[str] = None,
    port: int = 8080,
    robot_name: Optional[str] = None,
    instance: int = 0,
) -> EndpointPolicy:
    """Connect to a local model endpoint (run locally on your hardware).

    Establishes a connection to a locally hosted model endpoint. The model can
    be specified either by providing a direct path to a .mar model file or by
    referencing a training run name. Only one of these options should be provided.

    Args:
        path_to_model: Direct file path to a local .mar (Model ARchive) model file.
            Mutually exclusive with train_run_name.
        train_run_name: Name of a training run to load the model from. The system
            will locate and load the model from the specified training run.
            Mutually exclusive with path_to_model.
        port: TCP port number where the local endpoint is running.
        robot_name: Robot name that predictions and data will be associated with.
            If not provided, uses the last initialized robot from global state.
        instance: Instance number of the robot for multi-instance deployments.

    Returns:
        Policy object that provides an interface for making predictions
        with the local model.

    Raises:
        EndpointError: If the endpoint connection fails due to invalid model path,
            inaccessible port, or conflicting parameters.
        ValueError: If both path_to_model and train_run_name are provided, or if
            neither is provided.
        FileNotFoundError: If the specified model file doesn't exist.
        ConfigError: If there is an error trying to get the current org
    """
    return _connect_local_endpoint(
        robot_name=robot_name,
        instance=instance,
        path_to_model=path_to_model,
        train_run_name=train_run_name,
        port=port,
    )


def deploy_model(job_id: str, name: str) -> dict:
    """Deploy a trained model to a managed endpoint.

    Takes a completed training job and deploys the resulting model to a managed
    endpoint on the Neuracore platform. The endpoint will be accessible for
    inference once deployment is complete.

    Args:
        job_id: Unique identifier of the completed training job containing
            the model to deploy.
        name: Human-readable name for the endpoint that will be created.

    Returns:
        Deployment response containing endpoint details and deployment status.

    Raises:
        requests.exceptions.HTTPError: If the API request returns an error code
            due to invalid job_id, name conflicts, or server issues.
        requests.exceptions.RequestException: If there are network connectivity
            or request formatting problems.
        ValueError: If the deployment fails due to invalid parameters or
            server-side errors.
        ConfigError: If there is an error trying to get the current org
    """
    auth = get_auth()
    org_id = get_current_org()
    try:
        response = requests.post(
            f"{API_URL}/org/{org_id}/models/deploy",
            headers=auth.get_headers(),
            data=json.dumps({"training_id": job_id, "name": name}),
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise ValueError(f"Error deploying model: {e}")


def get_endpoint_status(endpoint_id: str) -> str:
    """Get the current status of a deployed endpoint.

    Retrieves the operational status of an endpoint, including deployment state,
    health information, and availability for inference requests.

    Args:
        endpoint_id: Unique identifier of the endpoint to check.

    Returns:
        Status information dictionary containing the current state and
        health details of the endpoint.

    Raises:
        requests.exceptions.HTTPError: If the API request returns an error code
            due to invalid endpoint_id or access permissions.
        requests.exceptions.RequestException: If there are network connectivity
            or request formatting problems.
        ValueError: If the status check fails due to server-side errors.
        ConfigError: If there is an error trying to get the current org
    """
    auth = get_auth()
    org_id = get_current_org()
    try:
        response = requests.get(
            f"{API_URL}/org/{org_id}/models/endpoints/{endpoint_id}",
            headers=auth.get_headers(),
        )
        response.raise_for_status()
        return response.json()["status"]
    except Exception as e:
        raise ValueError(f"Error getting endpoint status: {e}")


def delete_endpoint(endpoint_id: str) -> None:
    """Delete a deployed endpoint and free its resources.

    Permanently removes an endpoint from the Neuracore platform, stopping
    all inference capabilities and releasing associated computing resources.
    This operation cannot be undone.

    Args:
        endpoint_id: Unique identifier of the endpoint to delete.

    Raises:
        requests.exceptions.HTTPError: If the API request returns an error code
            due to invalid endpoint_id, insufficient permissions, or if the
            endpoint is currently in use.
        requests.exceptions.RequestException: If there are network connectivity
            or request formatting problems.
        ValueError: If the deletion fails due to server-side errors or
            endpoint dependencies.
        ConfigError: If there is an error trying to get the current org
    """
    auth = get_auth()
    org_id = get_current_org()
    try:
        response = requests.delete(
            f"{API_URL}/org/{org_id}/models/endpoints/{endpoint_id}",
            headers=auth.get_headers(),
        )
        response.raise_for_status()
    except Exception as e:
        raise ValueError(f"Error deleting endpoint: {e}")
