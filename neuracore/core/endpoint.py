import base64
import json
import logging
import subprocess
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from neuracore.api.core import _get_robot
from neuracore.core.robot import Robot
from neuracore.core.utils.depth_utils import depth_to_rgb

from .auth import get_auth
from .const import API_URL
from .exceptions import EndpointError
from .nc_types import (
    CameraData,
    DataType,
    JointData,
    LanguageData,
    ModelPrediction,
    SyncPoint,
)

logger = logging.getLogger(__name__)


class EndpointPolicy:
    """Interface to a deployed model endpoint."""

    def __init__(self, robot: Robot, predict_url: str, headers: dict[str, str] = None):
        self._predict_url = predict_url
        self._headers = headers or {}
        self._process = None
        self._is_local = "localhost" in predict_url
        self.robot = robot

    def _encode_image(self, image: np.ndarray) -> str:
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
        """Decode base64 image string to numpy array."""
        img_bytes = base64.b64decode(encoded_image)
        buffer = BytesIO(img_bytes)
        pil_image = Image.open(buffer)
        return np.array(pil_image)

    def _maybe_add_exisiting_data(
        self, existing: JointData, to_add: JointData
    ) -> JointData:
        # Check if the joint data already exists
        if existing is None:
            return to_add
        existing.timestamp = to_add.timestamp
        existing.values.update(to_add.values)
        if existing.additional_values:
            existing.additional_values.update(to_add.additional_values)
        return existing

    def _create_sync_point(self) -> SyncPoint:
        sync_point = SyncPoint(timestamp=time.time())
        for stream_name, stream in self.robot.list_all_streams().items():
            if "rgb" in stream_name:
                stream_data: np.ndarray = stream.get_latest_data()
                if sync_point.rgb_images is None:
                    sync_point.rgb_images = {}
                sync_point.rgb_images[stream_name] = CameraData(
                    timestamp=time.time(), frame=self._encode_image(stream_data)
                )
            elif "depth" in stream_name:
                stream_data: np.ndarray = stream.get_latest_data()
                if sync_point.depth_images is None:
                    sync_point.depth_images = {}
                sync_point.depth_images[stream_name] = CameraData(
                    timestamp=time.time(),
                    frame=self._encode_image(depth_to_rgb(stream_data)),
                )
            elif "joint_positions" in stream_name:
                stream_data: JointData = stream.get_latest_data()
                sync_point.joint_positions = self._maybe_add_exisiting_data(
                    sync_point.joint_positions, stream_data
                )
            elif "joint_velocities" in stream_name:
                stream_data: JointData = stream.get_latest_data()
                sync_point.joint_velocities = self._maybe_add_exisiting_data(
                    sync_point.joint_velocities, stream_data
                )
            elif "language" in stream_name:
                stream_data: LanguageData = stream.get_latest_data()
                sync_point.language_data = stream_data
            else:
                raise NotImplementedError(
                    f"Support for stream {stream_name} is not implemented yet"
                )
        return sync_point

    def predict(self, sync_point: SyncPoint | None = None) -> ModelPrediction:
        """
        Get action predictions from the model.

        Args:
            sync_point: SyncPoint object containing the data to be sent to the model.
                If None, a new SyncPoint will be created with the latest data.

        Returns:
            numpy.ndarray: Predicted action/joint velocities

        Raises:
            EndpointError: If prediction fails
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
        """Disconnect from the endpoint."""
        if self._process:
            subprocess.run(["torchserve", "--stop"], capture_output=True)
            self._process.terminate()
            self._process.wait()
            self._process = None


def connect_endpoint(
    endpoint_name: str,
    robot_name: Optional[str] = None,
    instance: Optional[int] = 0,
) -> EndpointPolicy:
    """Connect to a remote model endpoint.

    Args:
        endpoint_name (str): the name or ID of the endpoint to connect to.
        robot_name: (Optional[str], optional) robot ID. If not provided, uses
            the last initialized robot.
        instance: (Optional[int], optional) instance number of the robot.
            Defaults to 0.
    Raises:
        EndpointError: Endpoint Not Found
        EndpointError: Endpoint Not Active
        EndpointError: Failed to connect to endpoint

    Returns:
        EndpointPolicy: The policy by which actions are selected.
    """
    auth = get_auth()
    robot = _get_robot(robot_name, instance)
    try:
        # If not found by ID, get all endpoints and search by name
        response = requests.get(
            f"{API_URL}/models/endpoints", headers=auth.get_headers()
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
            predict_url=f"{API_URL}/models/endpoints/{endpoint['id']}/predict",
            headers=auth.get_headers(),
        )

    except requests.exceptions.RequestException as e:
        raise EndpointError(f"Failed to connect to endpoint: {str(e)}")


def connect_local_endpoint(
    robot_name: Optional[str] = None,
    instance: Optional[int] = 0,
    path_to_model: Optional[str] = None,
    train_run_name: Optional[str] = None,
    port: int = 8080,
) -> EndpointPolicy:
    """Connect to a local model endpoint.

    Args:
        robot_name: (Optional[str], optional) robot ID. If not provided, uses
            the last initialized robot.
        instance: (Optional[int], optional) instance number of the robot.
            Defaults to 0.
        path_to_model:  Path to the model file
        train_run_name: Optional train run name
        port: Port to run the local endpoint on
    """
    if path_to_model is None and train_run_name is None:
        raise ValueError("Must provide either path_to_model or train_run_name")
    if path_to_model and train_run_name:
        raise ValueError("Cannot provide both path_to_model and train_run_name")
    robot = _get_robot(robot_name, instance)
    auth = get_auth()
    if train_run_name:
        # Get all training runs and search for the job id
        response = requests.get(f"{API_URL}/training/jobs", headers=auth.get_headers())
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
            f"{API_URL}/training/jobs/{job_id}/model",
            headers=auth.get_headers(),
            timeout=120,
            stream=True,
        )
        response.raise_for_status()

        # Get total file size
        total_size = int(response.headers.get("Content-Length", 0))

        # Create a temporary directory and file path
        tempdir = tempfile.mkdtemp()
        path_to_model = Path(tempdir) / "model.mar"

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
        with open(path_to_model, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        # Close the progress bar
        progress_bar.close()
        print(f"Model download complete. Saved to {path_to_model}")

    try:
        process = _setup_torchserve(path_to_model, port=port)
        attemps = 5
        while attemps > 0:
            try:
                # Check if the server is running
                health_check = requests.get(f"http://localhost:{port}/ping", timeout=10)
                if health_check.status_code == 200:
                    logging.info("TorchServe is running...")
                    break
            except requests.exceptions.RequestException:
                pass
            attemps -= 1
            time.sleep(5)
        if health_check.status_code != 200:
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
    """Setup and start TorchServe with our model."""
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
