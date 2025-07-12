"""Model endpoint management for robot control and inference.

This module provides classes and functions for connecting to and interacting
with machine learning model endpoints, both local and remote. It handles
model prediction requests, data synchronization from robot sensors, and
manages FastAPI instance for local model deployment.
"""

import atexit
import base64
import logging
import os
import signal
import socket
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

from neuracore.api.core import _get_robot
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.robot import Robot
from neuracore.core.utils.depth_utils import depth_to_rgb
from neuracore.core.utils.download import download_with_progress
from neuracore.core.utils.server import (
    PING_ENDPOINT,
    PREDICT_ENDPOINT,
    SET_CHECKPOINT_ENDPOINT,
)
from neuracore.ml.utils.policy_inference import PolicyInference

from .auth import get_auth
from .const import API_URL
from .exceptions import EndpointError
from .nc_types import CameraData, DataType, JointData, ModelPrediction, SyncPoint

logger = logging.getLogger(__name__)


class Policy:
    """Base class for all policies."""

    def __init__(self, robot: Optional[Robot]):
        """Initialize the policy with an optional robot instance."""
        self.robot = robot

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
                    timestamp=time.time(), frame=stream_data
                )
            elif "depth" in stream_name:
                stream_data = stream.get_latest_data()
                if sync_point.depth_images is None:
                    sync_point.depth_images = {}
                sync_point.depth_images[stream_name] = CameraData(
                    timestamp=time.time(),
                    frame=depth_to_rgb(stream_data),
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

    def set_checkpoint(
        self, epoch: Optional[int] = None, checkpoint_file: Optional[str] = None
    ) -> None:
        """Set the model checkpoint to use for inference.

        Args:
            epoch: The epoch number of the checkpoint to load.
            checkpoint_file: Optional path to a specific checkpoint file.
                If provided, overrides the epoch setting.
        """
        if epoch is not None and checkpoint_file is not None:
            raise ValueError("Specify either epoch or checkpoint_file, not both.")
        if epoch is None and checkpoint_file is None:
            raise ValueError("Must specify either epoch or checkpoint_file.")

    def predict(self, sync_point: Optional[SyncPoint] = None) -> ModelPrediction:
        """Make a prediction using the policy."""
        raise NotImplementedError("Subclasses must implement this method.")

    def disconnect(self) -> None:
        """Disconnect from the policy and clean up resources."""
        pass


class DirectPolicy(Policy):
    """Direct model inference without any server infrastructure.

    This policy loads the model directly in the current process and runs
    inference without any network overhead. Ideal for low-latency applications.
    """

    def __init__(
        self,
        robot: Optional[Robot],
        model_path: Path,
        org_id: str,
        job_id: Optional[str] = None,
    ):
        """Initialize the direct policy with a robot instance."""
        super().__init__(robot)
        self._policy = PolicyInference(
            org_id=org_id, job_id=job_id, model_file=model_path
        )

    def set_checkpoint(
        self, epoch: Optional[int] = None, checkpoint_file: Optional[str] = None
    ) -> None:
        """Set the model checkpoint to use for inference.

        Args:
            epoch: The epoch number of the checkpoint to load.
            checkpoint_file: Optional path to a specific checkpoint file.
                If provided, overrides the epoch setting.
        """
        super().set_checkpoint(epoch, checkpoint_file)
        self._policy.set_checkpoint(epoch, checkpoint_file)

    def predict(self, sync_point: Optional[SyncPoint] = None) -> ModelPrediction:
        """Run direct model inference.

        Args:
            sync_point: Optional sync point. If None, creates from robot sensors.

        Returns:
            Model predictions.
        """
        if sync_point is None:
            sync_point = self._create_sync_point()
        model_prediction = self._policy(sync_point)
        model_prediction.outputs = {
            key: value[0] if isinstance(value, np.ndarray) and value.ndim > 0 else value
            for key, value in model_prediction.outputs.items()
        }
        return model_prediction


class ServerPolicy(Policy):
    """Base class for server-based policies that communicate via HTTP.

    This class provides common functionality for policies that send requests
    to HTTP endpoints, whether local or remote.
    """

    def __init__(
        self,
        robot: Optional[Robot],
        base_url: str,
        headers: Optional[dict[str, str]] = None,
    ):
        """Initialize the server policy with connection details.

        Args:
            robot: Robot instance for accessing sensor streams.
            base_url: Base URL of the server.
            headers: Optional HTTP headers for authentication.
        """
        super().__init__(robot)
        self._base_url = base_url
        self._headers = headers or {}
        self._is_local = "localhost" in base_url or "127.0.0.1" in base_url
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

    def set_checkpoint(
        self, epoch: Optional[int] = None, checkpoint_file: Optional[str] = None
    ) -> None:
        """Set the model checkpoint via HTTP request.

        Args:
            epoch: The epoch number of the checkpoint to load.
            checkpoint_file: Optional path to a specific checkpoint file.
                If provided, overrides the epoch setting.
        """
        if checkpoint_file is not None:
            raise ValueError(
                "Setting checkpoint by file is not supported in server policies."
            )
        if epoch is None:
            raise ValueError("Must specify epoch to set checkpoint.")
        if epoch < -1:
            raise ValueError("Epoch must be -1 (last) or a non-negative integer.")
        try:
            response = requests.post(
                f"{self._base_url}{SET_CHECKPOINT_ENDPOINT}",
                headers=self._headers,
                json={"epoch": epoch},
                timeout=30,
            )
            if response.status_code != 200:
                raise EndpointError(
                    "Failed to set checkpoint: "
                    f"{response.status_code} - {response.text}"
                )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise EndpointError(f"Failed to set checkpoint: {str(e)}")

    def predict(self, sync_point: Optional[SyncPoint] = None) -> ModelPrediction:
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

        # Encode images if they are numpy arrays
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
        try:
            # Make prediction request
            response = requests.post(
                f"{self._base_url}{PREDICT_ENDPOINT}",
                headers=self._headers,
                json=sync_point.model_dump(),
                timeout=int(os.getenv("NEURACORE_ENDPOINT_TIMEOUT", 10)),
            )
            response.raise_for_status()

            # Parse response
            result = response.json()
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


class LocalServerPolicy(ServerPolicy):
    """Policy that manages a local FastAPI server instance.

    This policy starts and manages a local FastAPI server for model inference,
    providing the flexibility of a server architecture with local control.
    """

    def __init__(
        self,
        robot: Optional[Robot],
        org_id: str,
        model_path: Path,
        job_id: Optional[str] = None,
        port: int = 8080,
        host: str = "127.0.0.1",
    ):
        """Initialize the local server policy.

        Args:
            robot: Robot instance for accessing sensor streams.
            org_id: Organization ID
            model_path: Path to the .nc.zip model file
            job_id: Optional job ID to associate with the server
            port: Port to run the server on
            host: Host to bind to
        """
        super().__init__(robot, f"http://{host}:{port}")
        self.org_id = org_id
        self.job_id = job_id
        self.model_path = model_path
        self.port = port
        self.host = host
        self.server_process: Optional[Popen] = None
        atexit.register(self.disconnect)
        self._start_server()

    def _start_server(self) -> None:
        """Start the FastAPI server in a subprocess using module execution."""
        # Start the server process using module execution
        cmd = [
            sys.executable,
            "-m",
            "neuracore.core.utils.server",
            "--model_file",
            str(self.model_path),
            "--org-id",
            self.org_id,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--log-level",
            "info",
        ]
        if self.job_id:
            cmd.extend(["--job-id", self.job_id])

        if self._is_port_in_use(self.host, self.port):
            raise EndpointError(
                f"Port {self.port} is already in use. "
                "Kill the process using it or choose a different port."
            )

        logger.info(f"Starting FastAPI server with command: {' '.join(cmd)}")

        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Ensure clean process termination
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        # Wait for server to start
        self._wait_for_server()

    def _is_port_in_use(self, host: str, port: int) -> bool:
        """Check if a port is in use on the specified host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex((host, port)) == 0

    def _wait_for_server(self, max_attempts: int = 60) -> None:
        """Wait for the server to become available."""
        for attempt in range(max_attempts):
            try:
                response = requests.get(
                    f"http://{self.host}:{self.port}{PING_ENDPOINT}", timeout=1
                )
                if response.status_code == 200:
                    logger.info(
                        f"Local server started successfully on {self.host}:{self.port}"
                    )
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        raise EndpointError(
            f"Local server failed to start after {max_attempts} attempts"
        )

    def set_checkpoint(
        self, epoch: Optional[int] = None, checkpoint_file: Optional[str] = None
    ) -> None:
        """Set the model checkpoint via HTTP request to the local server.

        Args:
            epoch: The epoch number of the checkpoint to load.
            checkpoint_file: Optional path to a specific checkpoint file.
                If provided, overrides the epoch setting.
        """
        if self.job_id is None:
            raise ValueError("Cannot set a checkpoint when loading from .nc.zip file")
        return super().set_checkpoint(epoch, checkpoint_file)

    def disconnect(self) -> None:
        """Stop the local server and clean up resources."""
        if not self.server_process:
            return
        try:
            # Try graceful termination first
            if hasattr(os, "killpg"):
                # Unix-like systems: kill the process group
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
            else:
                # Windows: terminate the process
                self.server_process.terminate()

            # Wait for graceful shutdown
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                else:
                    self.server_process.kill()
                self.server_process.wait()

        except (ProcessLookupError, OSError):
            # Process already terminated
            pass
        finally:
            self.server_process = None
            logger.info("Local server stopped")


class RemoteServerPolicy(ServerPolicy):
    """Policy for connecting to remote endpoints on the Neuracore platform."""

    def __init__(self, robot: Optional[Robot], base_url: str, headers: dict[str, str]):
        """Initialize the remote server policy.

        Args:
            robot: Robot instance for accessing sensor streams.
            base_url: Base URL of the remote server.
            headers: HTTP headers for authentication.
        """
        super().__init__(robot, base_url, headers)


# Main connection functions
def policy(
    train_run_name: Optional[str] = None,
    model_file: Optional[str] = None,
    robot_name: Optional[str] = None,
    instance: int = 0,
) -> DirectPolicy:
    """Launch a direct policy that runs the model in-process.

    Args:
        train_run_name: Name of the training run to load the model from.
        robot_name: Robot identifier.
        instance: Instance number of the robot.

    Returns:
        DirectPolicy instance for direct model inference.
    """
    robot = None
    if os.getenv("NEURACORE_LIVE_DATA_ENABLED", "True").lower() == "true":
        robot = _get_robot(robot_name, instance)

    org_id = get_current_org()
    job_id = None
    if train_run_name is not None:
        job_id = _get_job_id(train_run_name, org_id)
        model_path = _download_model(job_id, org_id)
    elif model_file is not None:
        model_path = Path(model_file)
    else:
        raise ValueError("Must specify either train_run_name or model_file")

    return DirectPolicy(
        robot=robot, org_id=org_id, job_id=job_id, model_path=model_path
    )


def policy_local_server(
    train_run_name: Optional[str] = None,
    model_file: Optional[str] = None,
    port: int = 8080,
    robot_name: Optional[str] = None,
    instance: int = 0,
    host: str = "127.0.0.1",
    job_id: Optional[str] = None,
) -> LocalServerPolicy:
    """Launch a local server policy with a FastAPI server.

    Args:
        train_run_name: Name of the training run to load the model from.
        port: Port to run the server on.
        robot_name: Robot identifier.
        instance: Instance number of the robot.
        host: Host to bind to.
        job_id: Optional job ID to associate with the server.

    Returns:
        LocalServerPolicy instance managing a local FastAPI server.
    """
    if train_run_name is None and model_file is None:
        raise ValueError("Must specify either train_run_name or model_file")
    if train_run_name and model_file:
        raise ValueError("Cannot specify both train_run_name and model_file")

    robot = None
    if os.getenv("NEURACORE_LIVE_DATA_ENABLED", "True").lower() == "true":
        robot = _get_robot(robot_name, instance)

    org_id = get_current_org()

    # Download model
    if train_run_name is not None:
        if job_id is None:
            job_id = _get_job_id(train_run_name, org_id)
        model_path = _download_model(job_id, org_id)
    elif model_file is not None:
        model_path = Path(model_file)
    else:
        raise ValueError("Must specify either train_run_name or model_file")

    return LocalServerPolicy(
        robot=robot,
        org_id=org_id,
        model_path=model_path,
        job_id=job_id,
        port=port,
        host=host,
    )


def policy_remote_server(
    endpoint_name: str,
    robot_name: Optional[str] = None,
    instance: int = 0,
) -> RemoteServerPolicy:
    """Launch a remote server policy connected to a deployed endpoint.

    Args:
        endpoint_name: Name of the deployed endpoint.
        robot_name: Robot identifier.
        instance: Instance number of the robot.

    Returns:
        RemoteServerPolicy instance for remote inference.
    """
    auth = get_auth()
    org_id = get_current_org()
    robot = _get_robot(robot_name, instance)

    try:
        # Find endpoint by name
        response = requests.get(
            f"{API_URL}/org/{org_id}/models/endpoints", headers=auth.get_headers()
        )
        response.raise_for_status()

        endpoints = response.json()
        endpoint = next((e for e in endpoints if e["name"] == endpoint_name), None)
        if not endpoint:
            raise EndpointError(f"No endpoint found with name: {endpoint_name}")

        # Verify endpoint is active
        if endpoint["status"] != "active":
            raise EndpointError(
                f"Endpoint {endpoint_name} is not active (status: {endpoint['status']})"
            )

        return RemoteServerPolicy(
            robot=robot,
            base_url=f"{API_URL}/org/{org_id}/models/endpoints/{endpoint['id']}",
            headers=auth.get_headers(),
        )

    except requests.exceptions.RequestException as e:
        raise EndpointError(f"Failed to connect to endpoint: {str(e)}")


# Helper functions
def _download_model(job_id: str, org_id: str) -> Path:
    """Download model from training run."""
    auth = get_auth()
    destination = Path(tempfile.gettempdir()) / job_id / "model.nc.zip"
    if destination.exists():
        print(f"Model already downloaded at {destination}. Skipping download.")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading model from training run...")
    response = requests.get(
        f"{API_URL}/org/{org_id}/training/jobs/{job_id}/model_url",
        headers=auth.get_headers(),
        timeout=30,
    )
    response.raise_for_status()

    model_url_response = response.json()
    model_path = download_with_progress(
        model_url_response["url"],
        "Downloading model...",
        destination=destination,
    )
    print(f"Model download complete. Saved to {model_path}")
    return model_path


def _get_job_id(train_run_name: str, org_id: str) -> str:
    """Get job ID from training run name."""
    auth = get_auth()
    response = requests.get(
        f"{API_URL}/org/{org_id}/training/jobs", headers=auth.get_headers()
    )
    response.raise_for_status()
    jobs = response.json()

    for job in jobs:
        if job["name"] == train_run_name:
            return job["id"]

    raise EndpointError(f"Training run not found: {train_run_name}")
