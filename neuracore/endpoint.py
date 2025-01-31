import base64
import json
import logging
import subprocess
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from neuracore.const import API_URL

from .auth import get_auth
from .exceptions import EndpointError

logger = logging.getLogger(__name__)


class EndpointPolicy:
    """Interface to a deployed model endpoint."""

    def __init__(self, predict_url: str, headers: dict[str, str] = None):
        self._predict_url = predict_url
        self._headers = headers or {}
        self._process = None
        self._is_local = "localhost" in predict_url

    def predict(
        self, joint_positions: list[float], images: dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Get action predictions from the model.

        Args:
            joint_positions: List of joint positions in radians
            images: Dictionary mapping camera IDs to RGB images (HxWx3 numpy arrays)

        Returns:
            numpy.ndarray: Predicted action/joint velocities

        Raises:
            EndpointError: If prediction fails
        """
        # Encode images as base64
        encoded_images = {}
        for camera_id, image in images.items():
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Image for camera {camera_id} must be a numpy array")

            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            # Ensure RGB format
            if len(image.shape) == 2:  # Grayscale
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]

            pil_image = Image.fromarray(image)
            if not self._is_local:
                if pil_image.size > (224, 224):
                    # There is a limit on the image size for non-local endpoints
                    # This is OK as almost all algorithms scale to 224x224
                    pil_image = pil_image.resize((224, 224))

            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            encoded_images[camera_id] = base64.b64encode(buffer.getvalue()).decode(
                "utf-8"
            )

        # Prepare request data
        request_data = {"joint_positions": joint_positions, "images": encoded_images}

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
            else:
                result = result[0]  # One item in batch

            return np.array(result)

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


def connect_endpoint(name: str) -> EndpointPolicy:
    """Implementation of connect_endpoint."""
    auth = get_auth()

    try:
        # If not found by ID, get all endpoints and search by name
        response = requests.get(
            f"{API_URL}/models/endpoints", headers=auth.get_headers()
        )
        response.raise_for_status()

        endpoints = response.json()
        endpoint = next((e for e in endpoints if e["name"] == name), None)
        if not endpoint:
            raise EndpointError(f"No endpoint found with name or ID: {name}")

        # Verify endpoint is active
        if endpoint["status"] != "active":
            raise EndpointError(
                f"Endpoint {name} is not active (status: {endpoint['status']})"
            )

        return EndpointPolicy(
            f"{API_URL}/models/endpoints/{endpoint['id']}/predict",
            auth.get_headers(),
        )

    except requests.exceptions.RequestException as e:
        raise EndpointError(f"Failed to connect to endpoint: {str(e)}")


def connect_local_endpoint(
    path_to_model: str | None = None, train_run_name: str | None = None
) -> EndpointPolicy:
    """Connect to a local model.

    Args:
        path_to_model:  Path to the model file
        train_run_name: Optional train run name
    """
    if path_to_model is None and train_run_name is None:
        raise ValueError("Must provide either path_to_model or train_run_name")
    if path_to_model and train_run_name:
        raise ValueError("Cannot provide both path_to_model and train_run_name")
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
        print("Downloading model from training run. This may take a while...")
        response = requests.get(
            f"{API_URL}/training/jobs/{job_id}/model",
            headers=auth.get_headers(),
            timeout=120,
            stream=True,
        )
        response.raise_for_status()
        tempdir = tempfile.mkdtemp()
        path_to_model = Path(tempdir) / f"{train_run_name}.mar"
        with open(path_to_model, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    try:
        process = _setup_torchserve(path_to_model)
        health_check = requests.get("http://localhost:8080/ping")
        if health_check.status_code == 200:
            logging.info("TorchServe is running...")
        else:
            raise EndpointError("TorchServe is not running")

        endpoint = EndpointPolicy("http://localhost:8080/predictions/robot_model")
        endpoint._process = process
        return endpoint

    except requests.exceptions.RequestException as e:
        raise EndpointError(f"Failed to connect to local endpoint: {str(e)}")
    except Exception as e:
        raise EndpointError(f"Error processing local endpoint response: {str(e)}")


def _setup_torchserve(path_to_model: str):
    """Setup and start TorchServe with our model."""
    model_path = Path(path_to_model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Create config file
    config = {"default_workers_per_model": 1, "default_response_timeout": 120}
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
