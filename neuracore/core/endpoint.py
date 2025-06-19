"""Model endpoint management for robot control and inference.

This module provides classes and functions for connecting to and interacting
with machine learning model endpoints, both local and remote. It handles
model prediction requests, data synchronization from robot sensors, and
manages TorchServe instances for local model deployment.
"""

import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path
from subprocess import Popen
from typing import Optional

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from neuracore.api.core import _get_robot
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.robot import Robot
from neuracore.core.utils.depth_utils import depth_to_rgb

from .auth import get_auth
from .const import API_URL
from .exceptions import EndpointError
from .nc_types import CameraData, DataType, JointData, ModelPrediction, SyncPoint

logger = logging.getLogger(__name__)


class EndpointPolicy:
    """Interface to a deployed model endpoint for robot control.

    This class provides methods for sending robot sensor data to a model
    endpoint and receiving action predictions. It handles data encoding,
    request management, and response processing for both local and remote
    endpoints.
    """

    def __init__(
        self,
        robot: Optional[Robot],
        predict_url: str,
        headers: Optional[dict[str, str]] = None,
    ):
        """Initialize the endpoint policy with connection details.

        Args:
            robot: Robot instance for accessing sensor streams.
            predict_url: URL of the model prediction endpoint.
            headers: Optional HTTP headers for authentication.
        """
        self._predict_url = predict_url
        self._headers = headers or {}
        self._process: Optional[Popen] = None
        self._is_local = "localhost" in predict_url
        self.robot = robot

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode numpy image array to base64 string for transmission.

        Converts numpy arrays to PNG format and encodes as base64. For remote
        endpoints, automatically resizes large images to 224x224 to meet
        payload size limits.

        Args:
            image: Numpy array representing an RGB image.

        Returns:
            Base64 encoded string of the PNG image.
        """
        pil_image = Image.fromarray(image)
        if not self._is_local:
            if pil_image.size > (224, 224):
                # There is a limit on the image size for non-local endpoints
                # This is OK as almost all algorithms scale to 224x224
                pil_image = pil_image.resize((224, 224))
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _decode_image(self, encoded_image: str) -> np.ndarray:
        """Decode base64 image string back to numpy array.

        Args:
            encoded_image: Base64 encoded image string.

        Returns:
            Numpy array representing the decoded image.
        """
        img_bytes = base64.b64decode(encoded_image)
        buffer = BytesIO(img_bytes)
        pil_image = Image.open(buffer)
        return np.array(pil_image)

    def _maybe_add_exisiting_data(
        self, existing: Optional[JointData], to_add: JointData
    ) -> JointData:
        """Merge joint data from multiple streams into a single data structure.

        Combines joint data while preserving existing values and updating
        timestamps. Used to aggregate data from multiple joint streams.

        Args:
            existing: Existing joint data or None.
            to_add: New joint data to merge.

        Returns:
            Combined JointData with merged values.
        """
        # Check if the joint data already exists
        if existing is None:
            return to_add
        existing.timestamp = to_add.timestamp
        existing.values.update(to_add.values)
        if existing.additional_values and to_add.additional_values:
            existing.additional_values.update(to_add.additional_values)
        return existing

    def _create_sync_point(self) -> SyncPoint:
        """Create a synchronized data point from current robot sensor streams.

        Collects the latest data from all active robot streams including
        cameras, joint sensors, and language inputs. Organizes the data
        into a synchronized structure with consistent timestamps.

        Returns:
            SyncPoint containing all current sensor data.

        Raises:
            NotImplementedError: If an unsupported stream type is encountered.
        """
        if self.robot is None:
            raise AttributeError("No robot instance")
        sync_point = SyncPoint(timestamp=time.time())
        for stream_name, stream in self.robot.list_all_streams().items():
            if "rgb" in stream_name:
                stream_data = stream.get_latest_data()
                if sync_point.rgb_images is None:
                    sync_point.rgb_images = {}
                sync_point.rgb_images[stream_name] = CameraData(
                    timestamp=time.time(), frame=self._encode_image(stream_data)
                )
            elif "depth" in stream_name:
                stream_data = stream.get_latest_data()
                if sync_point.depth_images is None:
                    sync_point.depth_images = {}
                sync_point.depth_images[stream_name] = CameraData(
                    timestamp=time.time(),
                    frame=self._encode_image(depth_to_rgb(stream_data)),
                )
            elif "joint_positions" in stream_name:
                stream_data = stream.get_latest_data()
                sync_point.joint_positions = self._maybe_add_exisiting_data(
                    sync_point.joint_positions, stream_data
                )
            elif "joint_velocities" in stream_name:
                stream_data = stream.get_latest_data()
                sync_point.joint_velocities = self._maybe_add_exisiting_data(
                    sync_point.joint_velocities, stream_data
                )
            elif "language" in stream_name:
                stream_data = stream.get_latest_data()
                sync_point.language_data = stream_data
            else:
                raise NotImplementedError(
                    f"Support for stream {stream_name} is not implemented yet"
                )
        return sync_point

    def predict(self, sync_point: SyncPoint | None = None) -> ModelPrediction:
        """Get action predictions from the model endpoint.

        Sends robot sensor data to the model and receives action predictions.
        Automatically creates a sync point from current robot data if none
        is provided. Handles image encoding and payload size validation.

        Args:
            sync_point: Synchronized sensor data to send to the model. If None,
                creates a new sync point from the robot's current sensor data.

        Returns:
            Model predictions including actions and any generated outputs.

        Raises:
            EndpointError: If prediction request fails or response is invalid.
            ValueError: If payload size exceeds limits for remote endpoints.
        """
        if sync_point is None:
            sync_point = self._create_sync_point()
        else:
            if sync_point.rgb_images:
                for key in sync_point.rgb_images:
                    if isinstance(sync_point.rgb_images[key].frame, np.ndarray):
                        sync_point.rgb_images[key].frame = self._encode_image(
                            sync_point.rgb_images[key].frame
                        )
            if sync_point.depth_images:
                for key in sync_point.depth_images:
                    if isinstance(sync_point.depth_images[key].frame, np.ndarray):
                        sync_point.depth_images[key].frame = self._encode_image(
                            sync_point.depth_images[key].frame
                        )
        request_data = sync_point.model_dump()
        if not self._is_local:
            payload_size = sys.getsizeof(json.dumps(request_data)) / (
                1024 * 1024
            )  # Size in MB
            if payload_size > 1.5:
                raise ValueError(
                    f"Payload size ({payload_size:.2f}MB) "
                    "exceeds server endpoint limit (1.5MB). "
                    "Please use a local endpoint."
                )

        try:
            # Make prediction request
            response = requests.post(
                self._predict_url,
                headers=self._headers,
                json=request_data,
                timeout=10,
            )
            response.raise_for_status()

            if response.status_code != 200:
                raise EndpointError(
                    f"Failed to get prediction from endpoint: {response.text}"
                )

            # Parse response
            result = response.json()

            if isinstance(result, dict) and "predictions" in result:
                result = result["predictions"]

            model_pred = ModelPrediction.model_validate(result)
            if DataType.RGB_IMAGE in model_pred.outputs:
                rgb_batch = model_pred.outputs[DataType.RGB_IMAGE]
                # Will be [B, T, CAMs, H, W, C]
                for b_idx in range(len(rgb_batch)):
                    for t_idx in range(len(rgb_batch[b_idx])):
                        for cam_idx in range(len(rgb_batch[b_idx][t_idx])):
                            rgb_batch[b_idx][t_idx][cam_idx] = self._decode_image(
                                rgb_batch[b_idx][t_idx][cam_idx]
                            )
                model_pred.outputs[DataType.RGB_IMAGE] = np.array(rgb_batch)
            for key, value in model_pred.outputs.items():
                if isinstance(value, list):
                    model_pred.outputs[key] = np.array(value)
                # Remove batch dimension
                model_pred.outputs[key] = model_pred.outputs[key][0]
            return model_pred

        except requests.exceptions.RequestException as e:
            raise EndpointError(f"Failed to get prediction from endpoint: {str(e)}")
        except Exception as e:
            raise EndpointError(f"Error processing endpoint response: {str(e)}")

    def disconnect(self) -> None:
        """Disconnect from the endpoint and clean up resources.

        For local endpoints, stops the TorchServe process and releases
        associated resources. Should be called when done using the endpoint.
        """
        if self._process:
            subprocess.run(["torchserve", "--stop"], capture_output=True)
            self._process.terminate()
            self._process.wait()
            self._process = None


def connect_endpoint(
    endpoint_name: str,
    robot_name: Optional[str] = None,
    instance: int = 0,
) -> EndpointPolicy:
    """Connect to a remote model endpoint deployed on the Neuracore platform.

    Locates an endpoint by name, verifies it's active, and creates a policy
    interface for making predictions. The endpoint must be deployed and
    running to establish a connection.

    Args:
        endpoint_name: Name or ID of the endpoint to connect to.
        robot_name: Robot identifier. If not provided, uses the currently
            active robot from global state.
        instance: Instance number of the robot for multi-instance deployments.

    Returns:
        EndpointPolicy interface for making predictions with the endpoint.

    Raises:
        EndpointError: If the endpoint is not found, not active, or connection fails.
        ConfigError: If there is an error trying to get the current org
    """
    auth = get_auth()
    org_id = get_current_org()
    robot = _get_robot(robot_name, instance)
    try:
        # If not found by ID, get all endpoints and search by name
        response = requests.get(
            f"{API_URL}/org/{org_id}/models/endpoints", headers=auth.get_headers()
        )
        response.raise_for_status()

        endpoints = response.json()
        endpoint = next((e for e in endpoints if e["name"] == endpoint_name), None)
        if not endpoint:
            raise EndpointError(f"No endpoint found with name or ID: {endpoint_name}")

        # Verify endpoint is active
        if endpoint["status"] != "active":
            raise EndpointError(
                f"Endpoint {endpoint_name} is not active (status: {endpoint['status']})"
            )

        return EndpointPolicy(
            robot=robot,
            predict_url=f"{API_URL}/org/{org_id}/models/endpoints/{endpoint['id']}/predict",
            headers=auth.get_headers(),
        )

    except requests.exceptions.RequestException as e:
        raise EndpointError(f"Failed to connect to endpoint: {str(e)}")


def connect_local_endpoint(
    robot_name: Optional[str] = None,
    instance: int = 0,
    path_to_model: Optional[str] = None,
    train_run_name: Optional[str] = None,
    port: int = 8080,
) -> EndpointPolicy:
    """Connect to a local model endpoint using TorchServe.

    Sets up a local TorchServe instance with the specified model and creates
    a policy interface. The model can be provided as a local file path or
    downloaded from a training run. Only one of path_to_model or train_run_name
    should be specified.

    Args:
        robot_name: Robot identifier. If not provided, uses the currently
            active robot if live data is enabled.
        instance: Instance number of the robot for multi-instance deployments.
        path_to_model: Local file path to a .mar model archive. Mutually
            exclusive with train_run_name.
        train_run_name: Name of a training run to download the model from.
            Mutually exclusive with path_to_model.
        port: TCP port for the local TorchServe instance.

    Returns:
        EndpointPolicy interface for making predictions with the local endpoint.

    Raises:
        ValueError: If both or neither of path_to_model and train_run_name are provided.
        EndpointError: If model download, TorchServe setup, or connection fails.
        FileNotFoundError: If the specified model file doesn't exist.
        ConfigError: If there is an error trying to get the current org
    """
    if path_to_model is None and train_run_name is None:
        raise ValueError("Must provide either path_to_model or train_run_name")
    if path_to_model and train_run_name:
        raise ValueError("Cannot provide both path_to_model and train_run_name")
    robot = None
    if os.getenv("NEURACORE_LIVE_DATA_ENABLED", "True").lower() == "true":
        robot = _get_robot(robot_name, instance)

    if train_run_name:
        auth = get_auth()
        org_id = get_current_org()
        # Get all training runs and search for the job id
        response = requests.get(
            f"{API_URL}/org/{org_id}/training/jobs", headers=auth.get_headers()
        )
        response.raise_for_status()
        jobs = response.json()
        job_id = None
        for job in jobs:
            if job["name"] == train_run_name:
                job_id = job["id"]
                break
        if job_id is None:
            raise EndpointError(f"Training run not found: {train_run_name}")

        print(f"Downloading model '{train_run_name}' from training run...")
        response = requests.get(
            f"{API_URL}/org/{org_id}/training/jobs/{job_id}/model_url",
            headers=auth.get_headers(),
            timeout=30,
        )
        response.raise_for_status()

        model_url_response = response.json()
        model_url = model_url_response["url"]
        response = requests.get(
            model_url,
            timeout=120,
            stream=True,
        )
        response.raise_for_status()

        # Get total file size
        total_size = int(response.headers.get("Content-Length", 0))

        # Create a temporary directory and file path
        tempdir = tempfile.mkdtemp()
        model_path_object: Path = Path(tempdir) / "model.mar"

        # Create progress bar based on file size
        progress_bar = tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading model {train_run_name}",
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )

        # Write the file with progress updates
        with open(model_path_object, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        # Close the progress bar
        progress_bar.close()
        print(f"Model download complete. Saved to {model_path_object}")
    else:
        # path_to_model cannot be none here
        assert path_to_model is not None, "Path to model not found"
        model_path_object = Path(path_to_model)

    try:
        process = _setup_torchserve(str(model_path_object), port=port)
        attemps = 5
        health_check = None
        status_code = 500
        while attemps > 0:
            try:
                # Check if the server is running
                health_check = requests.get(f"http://localhost:{port}/ping", timeout=10)
                status_code = health_check.status_code
                if status_code == 200:
                    logging.info("TorchServe is running...")
                    break
            except requests.exceptions.RequestException:
                health_check = None
                status_code = 500
                pass
            attemps -= 1
            time.sleep(5)
        if status_code != 200:
            raise EndpointError("TorchServe is not running")
        endpoint = EndpointPolicy(
            robot=robot, predict_url=f"http://localhost:{port}/predictions/robot_model"
        )
        endpoint._process = process
        return endpoint

    except requests.exceptions.RequestException as e:
        raise EndpointError(f"Failed to connect to local endpoint: {str(e)}")
    except Exception as e:
        raise EndpointError(f"Error processing local endpoint response: {str(e)}")


def _setup_torchserve(path_to_model: str, port: int = 8080) -> subprocess.Popen:
    """Setup and start a TorchServe instance with the specified model.

    Creates a TorchServe configuration, starts the service with the provided
    model, and returns the process handle for lifecycle management.

    Args:
        path_to_model: File path to the .mar model archive.
        port: Base port for TorchServe (inference, management, and metrics
            ports will be allocated sequentially).

    Returns:
        Subprocess.Popen object representing the TorchServe process.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    model_path = Path(path_to_model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Create config file
    config = {
        "default_workers_per_model": 1,
        "default_response_timeout": 120,
        "inference_address": f"http://localhost:{port}",
        "management_address": f"http://localhost:{port+1}",
        "metrics_address": f"http://localhost:{port+2}",
    }
    config_path = Path(tempfile.gettempdir()) / "config.properties"
    with config_path.open("w") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")

    # Ensure torchserve is not already running
    subprocess.run(["torchserve", "--stop"], capture_output=True)

    # Start TorchServe
    cmd = [
        "torchserve",
        "--start",
        "--model-store",
        str(model_path.resolve().parent),
        "--models",
        f"robot_model={str(model_path.name)}",
        "--ts-config",
        str(config_path.resolve()),
        "--ncs",  # Disable cleanup
        "--disable-token-auth",  # Disable authentication
    ]

    logger.info(f"Starting TorchServe with command:{' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)  # Give time for server to start
    return process
