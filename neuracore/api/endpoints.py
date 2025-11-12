"""Model endpoint management and connection API for Neuracore.

This module provides functionality for connecting to deployed model endpoints,
managing local model endpoints, and handling the lifecycle of model deployments
including deployment, status monitoring, and deletion operations.
"""

import json
from typing import Optional

import requests
from neuracore_types import DataType, SyncPoint

from neuracore.api.core import _get_robot
from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.endpoint import DirectPolicy, LocalServerPolicy, RemoteServerPolicy
from neuracore.core.endpoint import policy as _policy
from neuracore.core.endpoint import policy_local_server as _policy_local_server
from neuracore.core.endpoint import policy_remote_server as _policy_remote_server
from neuracore.core.get_latest_sync_point import (
    check_remote_nodes_connected as _check_remote_nodes_connected,
)
from neuracore.core.get_latest_sync_point import (
    get_latest_sync_point as _get_latest_sync_point,
)


def policy(
    train_run_name: Optional[str] = None,
    model_file: Optional[str] = None,
    output_mapping: Optional[dict[DataType, list[str]]] = None,
    device: Optional[str] = None,
) -> DirectPolicy:
    """Launch a direct policy that runs the model in-process without any server.

    This is the fastest option with lowest latency since there's no network overhead.
    The model runs directly in your Python process.

    Args:
        train_run_name: Name of the training run to load the model from.
        model_file: Path to the model file to load. If provided, overrides
        output_mapping: Optional mapping of output data types to response fields.
        device: Torch device to run the model on (CPU or GPU, or MPS)

    Returns:
        DirectPolicy object that provides direct in-process model inference.

    Raises:
        EndpointError: If the model download or initialization fails.
        ConfigError: If there is an error trying to get the current org.
    """
    return _policy(
        train_run_name, model_file, output_mapping=output_mapping, device=device
    )


def policy_local_server(
    train_run_name: Optional[str] = None,
    model_file: Optional[str] = None,
    device: Optional[str] = None,
    port: int = 8080,
    host: str = "127.0.0.1",
) -> LocalServerPolicy:
    """Launch and connect to a local server policy.

    This option provides server-like architecture while maintaining local control.

    Args:
        train_run_name: Name of the training run to load the model from.
        model_file: Path to the model file to load.
        device: Torch device to run the model on (CPU or GPU, or MPS)
        port: TCP port number where the local server will run.
        host: Host address to bind the server to. Defaults to localhost.

    Returns:
        LocalServerPolicy object that manages a local FastAPI server.

    Raises:
        EndpointError: If the server startup or model initialization fails.
        ConfigError: If there is an error trying to get the current org.
    """
    return _policy_local_server(train_run_name, model_file, device, port, host)


def policy_remote_server(endpoint_name: str) -> RemoteServerPolicy:
    """Connects to a policy that is remotely running on neuracore.

    Connects to a model endpoint deployed on the Neuracore cloud platform.
    The endpoint must be active and accessible.

    Args:
        endpoint_name: Name of the deployed endpoint to connect to.

    Returns:
        RemoteServerPolicy object for making predictions with the remote endpoint.

    Raises:
        EndpointError: If the endpoint connection fails due to invalid endpoint
            name, authentication issues, or network problems.
        ConfigError: If there is an error trying to get the current org.
    """
    return _policy_remote_server(endpoint_name)


# Deployment management functions
def deploy_model(job_id: str, name: str, ttl: Optional[int] = None) -> dict:
    """Deploy a trained model to a managed endpoint.

    Takes a completed training job and deploys the resulting model to a managed
    endpoint on the Neuracore platform. The endpoint will be accessible for
    inference once deployment is complete.

    Args:
        job_id: Unique identifier of the completed training job containing
            the model to deploy.
        name: Human-readable name for the endpoint that will be created.
        ttl: Optional time-to-live in seconds for the endpoint. If provided,
            the endpoint will be automatically deleted after this duration.

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
            data=json.dumps({"training_id": job_id, "name": name, "ttl": ttl}),
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


def get_latest_sync_point(
    robot_name: Optional[str] = None, instance: int = 0, include_remote: bool = True
) -> SyncPoint:
    """Creates a sync point from gathering data logged by a robot.

    Note after instantiation it can take time before data is available from
    all remote nodes as it sets up all the necessary connections. to get this
    started sooner and to check the progress call `check_remote_nodes_connected`

    Args:
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot
        include_remote: wether to connect to remote nodes to gather their
            data. This is ignored if NEURACORE_CONSUME_LIVE_DATA is disabled.

    Returns:
        The SyncPoint consisting of the latest data recorded for each of the
            sensors

    Raises:
        RobotError: If the robot is not initialized.
    """
    return _get_latest_sync_point(
        robot=_get_robot(robot_name, instance), include_remote=include_remote
    )


def check_remote_nodes_connected(
    num_remote_nodes: int = 0, robot_name: Optional[str] = None, instance: int = 0
) -> bool:
    """Checks if the required remote nodes are connected to the robot.

    Args:
        num_remote_nodes: The number of remote nodes that are expected to connect
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        instance: Optional instance number of the robot

    Returns:
        True if all remote nodes are connected, False otherwise

    Raises:
        RobotError: If the robot is not initialized.
    """
    return _check_remote_nodes_connected(
        robot=_get_robot(robot_name, instance), num_remote_nodes=num_remote_nodes
    )
