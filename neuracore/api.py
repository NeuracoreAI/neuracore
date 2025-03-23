import concurrent
import json
from threading import Thread
from typing import Optional

import numpy as np
import requests

from .core.auth import get_auth
from .core.auth import login as _login
from .core.auth import logout as _logout
from .core.const import API_URL
from .core.dataset import Dataset
from .core.endpoint import EndpointPolicy
from .core.endpoint import connect_endpoint as _connect_endpoint
from .core.endpoint import connect_local_endpoint as _connect_local_endpoint
from .core.exceptions import RobotError
from .core.robot import Robot, get_robot
from .core.robot import init as _init_robot
from .core.streaming.data_stream import (
    ActionDataStream,
    DataStream,
    DepthDataStream,
    JointDataStream,
    RGBDataStream,
)
from .core.utils.depth_utils import MAX_DEPTH

# Global active robot ID - allows us to avoid passing robot_name to every call
_active_robot: Optional[Robot] = None
_active_dataset_id: Optional[str] = None
_active_recording_ids: dict[str, str] = {}
_data_streams: dict[str, DataStream] = {}


def _stop_recording_wait_for_threads(
    robot: Robot, recording_id: str, threads: list[Thread]
) -> None:
    for thread in threads:
        thread.join()
    robot.stop_recording(recording_id)


def login(api_key: Optional[str] = None) -> None:
    """
    Authenticate with NeuraCore server.

    Args:
        api_key: Optional API key. If not provided, will look for NEURACORE_API_KEY
                environment variable or previously saved configuration.

    Raises:
        AuthenticationError: If authentication fails
    """
    _login(api_key)


def logout() -> None:
    """Clear authentication state."""
    _logout()


def connect_robot(
    robot_name: str,
    urdf_path: Optional[str] = None,
    mjcf_path: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Initialize a robot connection.

    Args:
        robot_name: Unique identifier for the robot
        urdf_path: Optional path to robot's URDF file
        mjcf_path: Optional path to robot's MJCF file
        overwrite: Whether to overwrite an existing robot with the same name
    """
    global _active_robot
    _active_robot = _init_robot(robot_name, urdf_path, mjcf_path, overwrite)


def _get_robot(robot_name: str) -> Robot:
    """Get a robot by name."""
    robot: Robot = _active_robot
    if robot_name is None:
        if _active_robot is None:
            raise RobotError(
                "No active robot. Call init() first or provide robot_name."
            )
    else:
        robot = get_robot(robot_name)
    return robot


def log_joints(
    positions: dict[str, float],
    robot_name: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> None:
    """
    Log joint positions for a robot.

    Args:
        positions: Dictionary mapping joint names to positions (in radians)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
    """
    if not isinstance(positions, dict):
        raise ValueError("Joint positions must be a dictionary of floats")
    for key, value in positions.items():
        if not isinstance(value, float):
            raise ValueError(f"Joint positions must be floats. {key} is not a float.")
    robot = _get_robot(robot_name)
    str_id = f"{robot.name}_joints"
    stream = _data_streams.get(str_id)
    if stream is None:
        stream = JointDataStream()
        _data_streams[str_id] = stream
        if robot.name in _active_recording_ids:
            stream.start_recording(_active_recording_ids[robot.name])
    stream.log(positions, timestamp)


def log_action(
    action: dict[str, float],
    robot_name: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> None:
    """
    Log action for a robot.

    Args:
        action: Dictionary mapping joint names to positions (in radians)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
    """
    if not isinstance(action, dict):
        raise ValueError("Actions must be a dictionary of floats")
    for key, value in action.items():
        if not isinstance(value, float):
            raise ValueError(f"Actions must be floats. {key} is not a float.")
    robot = _get_robot(robot_name)
    str_id = f"{robot.name}_action"
    stream = _data_streams.get(str_id)
    if stream is None:
        stream = ActionDataStream()
        _data_streams[str_id] = stream
        if robot.name in _active_recording_ids:
            stream.start_recording(_active_recording_ids[robot.name])
    stream.log(action, timestamp)


def log_rgb(
    camera_id: str,
    image: np.ndarray,
    robot_name: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> None:
    """
    Log RGB image from a camera.

    Args:
        camera_id: Unique identifier for the camera
        image: RGB image as numpy array (HxWx3, dtype=uint8 or float32)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If image format is invalid
    """
    # Validate image is numpy array of type uint8
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    if image.dtype != np.uint8:
        raise ValueError("Image must be uint8 wth range 0-255")
    robot = _get_robot(robot_name)
    camera_id = f"rgb_{camera_id}"
    str_id = f"{robot.name}_{camera_id}"
    stream = _data_streams.get(str_id)
    if stream is None:
        stream = RGBDataStream(camera_id, image.shape[1], image.shape[0])
        _data_streams[str_id] = stream
        if robot.name in _active_recording_ids:
            stream.start_recording(_active_recording_ids[robot.name])
    if stream.width != image.shape[1] or stream.height != image.shape[0]:
        raise ValueError(
            f"RGB image dimensions {image.shape[1]}x{image.shape[0]} do not match "
            f"stream dimensions {stream.width}x{stream.height}"
        )
    stream.log(image, timestamp)


def log_depth(
    camera_id: str,
    depth: np.ndarray,
    robot_name: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> None:
    """
    Log depth image from a camera.

    Args:
        camera_id: Unique identifier for the camera
        depth: Depth image as numpy array (HxW, dtype=float32, in meters)
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        timestamp: Optional timestamp

    Raises:
        RobotError: If no robot is active and no robot_name provided
        ValueError: If depth format is invalid
    """
    if not isinstance(depth, np.ndarray):
        raise ValueError("Depth image must be a numpy array")
    if depth.dtype not in (np.float16, np.float32):
        raise ValueError(
            f"Depth image must be float16 or float32, but got {depth.dtype}"
        )
    if depth.max() > MAX_DEPTH:
        raise ValueError(
            "Depth image should be in meters. "
            f"You are attempting to log depth values > {MAX_DEPTH}. "
            "The values you are passing in are likely in millimeters."
        )
    robot = _get_robot(robot_name)
    camera_id = f"depth_{camera_id}"
    str_id = f"{robot.name}_{camera_id}"
    stream = _data_streams.get(str_id)
    if stream is None:
        stream = DepthDataStream(camera_id, depth.shape[1], depth.shape[0])
        _data_streams[str_id] = stream
        if robot.name in _active_recording_ids:
            stream.start_recording(_active_recording_ids[robot.name])
    if stream.width != depth.shape[1] or stream.height != depth.shape[0]:
        raise ValueError(
            f"Depth image dimensions {depth.shape[1]}x{depth.shape[0]} do not match "
            f"stream dimensions {stream.width}x{stream.height}"
        )
    stream.log(depth, timestamp)


def start_recording(robot_name: Optional[str] = None) -> None:
    """
    Start recording data for a specific robot.

    Args:
        robot_name: Optional robot ID. If not provided, uses the last initialized robot

    Raises:
        RobotError: If no robot is active and no robot_name provided
    """
    global _active_recording_ids
    robot = _get_robot(robot_name)
    if robot.name in _active_recording_ids:
        raise RobotError("Recording already in progress. Call stop_recording() first.")
    if _active_dataset_id is None:
        raise RobotError("No active dataset. Call create_dataset() first.")
    new_active_recording_id = robot.start_recording(_active_dataset_id)
    for sname, stream in _data_streams.items():
        if sname.startswith(robot.name):
            stream.start_recording(new_active_recording_id)
    _active_recording_ids[robot.name] = new_active_recording_id


def stop_recording(robot_name: Optional[str] = None, wait: bool = False) -> None:
    """
    Stop recording data for a specific robot.

    Args:
        robot_name: Optional robot ID. If not provided, uses the last initialized robot
        wait: Whether to wait for the recording to finish

    Raises:
        RobotError: If no robot is active and no robot_name provided
    """
    global _active_recording_ids
    robot = _get_robot(robot_name)
    if robot.name not in _active_recording_ids:
        raise RobotError("No active recording. Call start_recording() first.")
    threads: Thread = []
    for sname, stream in _data_streams.items():
        if sname.startswith(robot.name):
            threads.append(stream.stop_recording())
    stop_recording_thread = Thread(
        target=_stop_recording_wait_for_threads,
        args=(robot, _active_recording_ids[robot.name], threads),
        daemon=False,
    )
    stop_recording_thread.start()
    _active_recording_ids.pop(robot.name)
    if wait:
        stop_recording_thread.join()


def get_dataset(name: str) -> Dataset:
    """Get a dataset by name.

    Args:
        name: Dataset name

    """
    global _active_dataset_id
    _active_dataset = Dataset.get(name)
    _active_dataset_id = _active_dataset.id
    return _active_dataset


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


def get_training_job_status(job_id: str) -> dict:
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


def create_dataset(
    name: str, description: Optional[str] = None, tags: Optional[list[str]] = None
) -> Dataset:
    """
    Create a new dataset for robot demonstrations.

    Args:
        name: Dataset name
        description: Optional description
        tags: Optional list of tags

    Raises:
        DatasetError: If dataset creation fails
    """
    global _active_dataset_id
    _active_dataset = Dataset.create(name, description, tags)
    _active_dataset_id = _active_dataset.id
    return _active_dataset


def connect_endpoint(name: str) -> EndpointPolicy:
    """
    Connect to a deployed model endpoint.

    Args:
        name: Name of the deployed endpoint

    Returns:
        EndpointPolicy: Policy object that can be used for predictions

    Raises:
        EndpointError: If endpoint connection fails
    """
    return _connect_endpoint(name)


def connect_local_endpoint(
    path_to_model: Optional[str] = None, train_run_name: Optional[str] = None
) -> EndpointPolicy:
    """
    Connect to a local model endpoint.

    Can supply either path_to_model or train_run_name, but not both.

    Args:
        path_to_model: Path to the local .mar model
        train_run_name: Optional train run name

    Returns:
        EndpointPolicy: Policy object that can be used for predictions

    Raises:
        EndpointError: If endpoint connection fails
    """
    return _connect_local_endpoint(path_to_model, train_run_name)
