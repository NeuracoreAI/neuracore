"""Model endpoint management for robot control and inference.

This module provides classes and functions for connecting to and interacting
with machine learning model endpoints, both local and remote. It handles
model prediction requests, data synchronization from robot sensors, and
manages FastAPI instance for local model deployment.
"""

import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from subprocess import Popen
from typing import Optional

import numpy as np
import requests
from neuracore_types import DataType, SyncPoint

from neuracore.api.globals import GlobalSingleton
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.exceptions import InsufficientSyncPointError
from neuracore.core.get_latest_sync_point import get_latest_sync_point
from neuracore.core.utils.download import download_with_progress
from neuracore.core.utils.image_string_encoder import ImageStringEncoder
from neuracore.core.utils.server import (
    PING_ENDPOINT,
    PREDICT_ENDPOINT,
    SET_CHECKPOINT_ENDPOINT,
)

from .auth import get_auth
from .const import API_URL
from .exceptions import EndpointError

logger = logging.getLogger(__name__)

PREDICTION_WAIT_TIME = 0.1


class Policy:
    """Base class for all policies."""

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

    def predict(
        self, sync_point: Optional[SyncPoint] = None, timeout: float = 5
    ) -> list[SyncPoint]:
        """Get action predictions from the model.

        Sends robot sensor data to the model and receives action predictions.
        Automatically creates a sync point from current robot data if none
        is provided.

        Args:
            sync_point: Synchronized sensor data to send to the model. If None,
                creates a new sync point from the robot's current sensor data.
            timeout: Maximum time to wait (in seconds) to accumulate asynchronous
                sensor data. Raises error if timeout is reached without sufficient data.

        Returns:
            Model predictions as a list of SyncPoint objects.

        Raises:
            InsufficientSyncPointError: If the sync point doesn't contain required data.
            EndpointError: If prediction request fails or response is invalid.
        """
        if timeout <= 0:
            raise ValueError("Timeout must be a positive number.")
        t = time.time()
        sync_points = None
        while sync_points is None:
            try:
                sync_points = self._predict(sync_point)
            except InsufficientSyncPointError as e:
                if time.time() - t > timeout:
                    raise e
                time.sleep(PREDICTION_WAIT_TIME)
        return sync_points

    def disconnect(self) -> None:
        """Disconnect from the policy and clean up resources."""
        pass

    def _predict(self, sync_point: Optional[SyncPoint] = None) -> list[SyncPoint]:
        """Internal get action predictions from the model.

        Sends robot sensor data to the model and receives action predictions.
        Automatically creates a sync point from current robot data if none
        is provided.

        Args:
            sync_point: Synchronized sensor data to send to the model. If None,
                creates a new sync point from the robot's current sensor data.

        Returns:
            Model predictions as a list of SyncPoint objects.
        """
        raise NotImplementedError(
            "Subclasses must implement the _predict method to run model inference."
        )


class DirectPolicy(Policy):
    """Direct model inference without any server infrastructure.

    This policy loads the model directly in the current process and runs
    inference without any network overhead. Ideal for low-latency applications.
    """

    def __init__(
        self,
        model_path: Path,
        org_id: str,
        job_id: Optional[str] = None,
        output_mapping: Optional[dict[DataType, list[str]]] = None,
        device: Optional[str] = None,
    ):
        """Initialize the direct policy with a robot instance."""
        super().__init__()
        # Import here to avoid the need for pytorch unless the user uses this policy
        from neuracore.ml.utils.policy_inference import PolicyInference

        self._policy = PolicyInference(
            org_id=org_id,
            job_id=job_id,
            model_file=model_path,
            output_mapping=output_mapping,
            device=device,
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

    def _predict(self, sync_point: Optional[SyncPoint] = None) -> list[SyncPoint]:
        """Run direct model inference.

        Args:
            sync_point: Optional sync point. If None, creates from robot sensors.

        Returns:
            Model predictions as a list of SyncPoint objects.

        Raises:
            InsufficientSyncPointError: If the sync point doesn't contain required data.
        """
        if sync_point is None:
            sync_point = get_latest_sync_point()
        return self._policy(sync_point)


class ServerPolicy(Policy):
    """Base class for server-based policies that communicate via HTTP.

    This class provides common functionality for policies that send requests
    to HTTP endpoints, whether local or remote.
    """

    def __init__(
        self,
        base_url: str,
        headers: Optional[dict[str, str]] = None,
    ):
        """Initialize the server policy with connection details.

        Args:
            robot: Robot instance for accessing sensor streams.
            base_url: Base URL of the server.
            headers: Optional HTTP headers for authentication.
        """
        super().__init__()
        self._base_url = base_url
        self._headers = headers or {}
        self._is_local = "localhost" in base_url or "127.0.0.1" in base_url

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
        except requests.exceptions.ConnectionError:
            raise EndpointError((
                "Failed to connect to endpoint, "
                "please check your internet connection and try again."
            ))
        except requests.exceptions.RequestException as e:
            raise EndpointError(f"Failed to set checkpoint: {str(e)}")

    def _predict(self, sync_point: Optional[SyncPoint] = None) -> list[SyncPoint]:
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
            InsufficientSyncPointError: If the sync point doesn't contain required data.
            ValueError: If payload size exceeds limits for remote endpoints.
        """
        if sync_point is None:
            sync_point = get_latest_sync_point()

        if sync_point.robot_id is None:
            robot = GlobalSingleton()._active_robot
            if robot is None:
                raise ValueError(
                    "No active robot found. Please connect a robot before predicting."
                )
            sync_point.robot_id = robot.id

        # Encode images if they are numpy arrays
        if sync_point.rgb_images:
            for key in sync_point.rgb_images:
                if isinstance(sync_point.rgb_images[key].frame, np.ndarray):
                    sync_point.rgb_images[key].frame = ImageStringEncoder.encode_image(
                        sync_point.rgb_images[key].frame
                    )
        if sync_point.depth_images:
            for key in sync_point.depth_images:
                if isinstance(sync_point.depth_images[key].frame, np.ndarray):
                    sync_point.depth_images[key].frame = (
                        ImageStringEncoder.encode_image(
                            sync_point.depth_images[key].frame
                        )
                    )
        response = None
        try:
            # Make prediction request
            response = requests.post(
                f"{self._base_url}{PREDICT_ENDPOINT}",
                headers=self._headers,
                json=sync_point.model_dump(mode="json"),
                timeout=int(os.getenv("NEURACORE_ENDPOINT_TIMEOUT", 10)),
            )
            response.raise_for_status()

            # Parse response
            result = response.json()
            sync_point_preds = [SyncPoint.model_validate(res) for res in result]
            for sync_point_pred in sync_point_preds:
                if sync_point_pred.rgb_images:
                    # Decode images back to numpy arrays
                    rgb = sync_point_pred.rgb_images
                    for cam_name, cam_data in rgb.items():
                        if isinstance(cam_data.frame, str):
                            cam_data.frame = ImageStringEncoder.decode_image(
                                cam_data.frame
                            )
                if sync_point_pred.depth_images:
                    # Decode depth images back to numpy arrays
                    depth = sync_point_pred.depth_images
                    for cam_name, cam_data in depth.items():
                        if isinstance(cam_data.frame, str):
                            cam_data.frame = ImageStringEncoder.decode_image(
                                cam_data.frame
                            )
            return sync_point_preds
        except requests.exceptions.ConnectionError:
            raise EndpointError((
                "Failed to connect to endpoint, "
                "please check your internet connection and try again."
            ))
        except requests.exceptions.RequestException as e:
            if response is not None:
                if response.status_code == 422:
                    raise InsufficientSyncPointError(
                        "Insufficient sync point data for inference."
                    )
                raise EndpointError(
                    "Failed to get prediction from endpoint: "
                    f"{response.json().get('detail', 'Unknown error')}"
                )
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
        org_id: str,
        model_path: Path,
        device: Optional[str] = None,
        job_id: Optional[str] = None,
        port: int = 8080,
        host: str = "127.0.0.1",
    ):
        """Initialize the local server policy.

        Args:
            robot: Robot instance for accessing sensor streams.
            org_id: Organization ID
            model_path: Path to the .nc.zip model file
            device: Device model to be loaded on
            job_id: Optional job ID to associate with the server
            port: Port to run the server on
            host: Host to bind to
        """
        super().__init__(f"http://{host}:{port}")
        self.org_id = org_id
        self.job_id = job_id
        self.model_path = model_path
        self.device = device
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
        if self.device:
            cmd.extend(["--device", self.device])
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

    def __init__(self, base_url: str, headers: dict[str, str]):
        """Initialize the remote server policy.

        Args:
            base_url: Base URL of the remote server.
            headers: HTTP headers for authentication.
        """
        super().__init__(base_url, headers)


# Main connection functions
def policy(
    train_run_name: Optional[str] = None,
    model_file: Optional[str] = None,
    output_mapping: Optional[dict[DataType, list[str]]] = None,
    device: Optional[str] = None,
) -> DirectPolicy:
    """Launch a direct policy that runs the model in-process.

    Args:
        train_run_name: Name of the training run to load the model from.
        robot_name: Robot identifier.
        instance: Instance number of the robot.
        output_mapping: Optional mapping of data types to output keys.
        device: Torch device to run the model on (CPU or GPU, or MPS).

    Returns:
        DirectPolicy instance for direct model inference.
    """
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
        org_id=org_id,
        job_id=job_id,
        model_path=model_path,
        output_mapping=output_mapping,
        device=device,
    )


def policy_local_server(
    train_run_name: Optional[str] = None,
    model_file: Optional[str] = None,
    device: Optional[str] = None,
    port: int = 8080,
    host: str = "127.0.0.1",
    job_id: Optional[str] = None,
) -> LocalServerPolicy:
    """Launch a local server policy with a FastAPI server.

    Args:
        train_run_name: Name of the training run to load the model from.
        port: Port to run the server on.
        device: Device model to be loaded on
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
        org_id=org_id,
        model_path=model_path,
        device=device,
        job_id=job_id,
        port=port,
        host=host,
    )


def policy_remote_server(
    endpoint_name: str,
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
            base_url=f"{API_URL}/org/{org_id}/models/endpoints/{endpoint['id']}",
            headers=auth.get_headers(),
        )
    except requests.exceptions.ConnectionError:
        raise EndpointError((
            "Failed to connect to endpoint: Connection Error. "
            "Please check your internet connection and try again."
        ))
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
