"""Model endpoint management for robot control and inference.

This module provides classes and functions for connecting to and interacting
with machine learning model endpoints, both local and remote. It handles
model prediction requests, data synchronization from robot sensors, and
manages FastAPI instance for local model deployment.
"""

import atexit
import json
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
from typing import TYPE_CHECKING, Any

import requests
from neuracore_types import DataType, EmbodimentDescription, SynchronizedPoint

from neuracore.core.utils.http_session import thread_local_session

if TYPE_CHECKING:
    from neuracore_types import BatchedNCData

    from neuracore.ml.utils.real_time_chunking import RTCConfig
    from neuracore.ml.utils.rtc_controller import ChunkingController, ChunkingStats
    from neuracore.ml.utils.temporal_ensemble import TemporalEnsembleConfig

from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.exceptions import InsufficientSynchronizedPointError
from neuracore.core.get_latest_sync_point import get_latest_sync_point
from neuracore.core.utils.download import download_to_cache
from neuracore.ml.logging.endpoint_log_streamer import EndpointLogStreamer
from neuracore.ml.preprocessing.base import PreprocessingConfiguration
from neuracore.ml.utils.endpoint_storage_handler import EndpointStorageHandler

from .auth import get_auth
from .const import API_URL, PING_ENDPOINT, PREDICT_ENDPOINT, SET_CHECKPOINT_ENDPOINT
from .exceptions import EndpointError

logger = logging.getLogger(__name__)

PREDICTION_WAIT_TIME = 0.1


def _parse_embodiment_description(raw_description: dict) -> EmbodimentDescription:
    """Parse an API embodiment description, restoring typed keys."""
    return {
        DataType(data_type): {int(index): name for index, name in indexed_names.items()}
        for data_type, indexed_names in raw_description.items()
    }


class Policy:
    """Base class for all policies."""

    def set_checkpoint(
        self, epoch: int | None = None, checkpoint_file: str | None = None
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
        self,
        sync_point: SynchronizedPoint | None = None,
        timeout: float = 5,
    ) -> dict[DataType, dict[str, "BatchedNCData"]]:
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
            Model predictions as dict[DataType, dict[str, "BatchedNCData"]].

        Raises:
            InsufficientSynchronizedPointError:
                If the sync point doesn't contain required data.
            EndpointError: If prediction request fails or response is invalid.
        """
        if timeout <= 0:
            raise ValueError("Timeout must be a positive number.")
        t = time.time()
        prediction = None
        while prediction is None:
            try:
                prediction = self._predict(sync_point)
            except InsufficientSynchronizedPointError as e:
                if time.time() - t > timeout:
                    raise e
                time.sleep(PREDICTION_WAIT_TIME)
        return prediction

    def disconnect(self) -> None:
        """Disconnect from the policy and clean up resources."""
        pass

    def _predict(
        self,
        sync_point: SynchronizedPoint | None = None,
    ) -> dict[DataType, dict[str, "BatchedNCData"]]:
        """Internal get action predictions from the model.

        Sends robot sensor data to the model and receives action predictions.
        Automatically creates a sync point from current robot data if none
        is provided.

        Args:
            sync_point: Synchronized sensor data to send to the model. If None,
                creates a new sync point from the robot's current sensor data.

        Returns:
            Model predictions as dict[DataType, dict[str, "BatchedNCData"]].
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
        input_embodiment_description: EmbodimentDescription | None = None,
        output_embodiment_description: EmbodimentDescription | None = None,
        input_preprocessing_config: PreprocessingConfiguration | None = None,
        job_id: str | None = None,
        device: str | None = None,
        robot_id: str | None = None,
    ):
        """Initialize the direct policy with a robot instance."""
        super().__init__()
        # Import here to avoid the need for pytorch unless the user uses this policy
        from neuracore.ml.utils.policy_inference import PolicyInference

        self._policy = PolicyInference(
            input_embodiment_description=input_embodiment_description,
            output_embodiment_description=output_embodiment_description,
            input_preprocessing_config=input_preprocessing_config,
            org_id=org_id,
            job_id=job_id,
            model_file=model_path,
            device=device,
            robot_id=robot_id,
        )

    def set_checkpoint(
        self, epoch: int | None = None, checkpoint_file: str | None = None
    ) -> None:
        """Set the model checkpoint to use for inference.

        Args:
            epoch: The epoch number of the checkpoint to load.
            checkpoint_file: Optional path to a specific checkpoint file.
                If provided, overrides the epoch setting.
        """
        super().set_checkpoint(epoch, checkpoint_file)
        self._policy.set_checkpoint(epoch, checkpoint_file)

    def _predict(
        self,
        sync_point: SynchronizedPoint | None = None,
    ) -> dict[DataType, dict[str, "BatchedNCData"]]:
        """Run direct model inference.

        Args:
            sync_point: Optional sync point. If None, creates from robot sensors.

        Returns:
            Model predictions as dict[DataType, dict[str, "BatchedNCData"]].

        Raises:
            InsufficientSynchronizedPointError:
                If the sync point doesn't contain required data.
        """
        if sync_point is None:
            sync_point = get_latest_sync_point()

        # Filter sync point to only include data types the model expects as input
        filtered_data = {
            data_type: sync_point.data[data_type]
            for data_type in self._policy.input_embodiment_description.keys()
            if data_type in sync_point.data
        }
        sync_point.data = filtered_data

        return self._policy(sync_point)


class RealTimePolicy(DirectPolicy):
    """In-process policy with an async overlapping-chunk controller.

    Supports two mutually exclusive modes bound at construction:

    * ``"rtc"`` — guided denoising (diffusion/flow only; arXiv:2506.07339)
    * ``"temporal_ensemble"`` — unguided predict + exponential merge

    Drive execution with :meth:`start` / :meth:`get_action` / :meth:`stop`.
    """

    def __init__(
        self,
        *args: Any,
        mode: str,
        config: "RTCConfig | TemporalEnsembleConfig",
        control_hz: float,
        adapt_inference_delay: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the realtime policy and validate mode/config.

        Args:
            *args: Forwarded to :class:`DirectPolicy`.
            mode: ``"rtc"`` or ``"temporal_ensemble"``.
            config: Mode-specific configuration.
            control_hz: Rate at which :meth:`get_action` will be called.
            adapt_inference_delay: For RTC, track measured latency with ``d``.
                Ignored for temporal ensemble.
            **kwargs: Forwarded to :class:`DirectPolicy`.

        Raises:
            EndpointError: If mode/config are inconsistent or RTC is requested
                for a model without the required hooks.
            ValueError: If ``mode`` is unknown.
        """
        super().__init__(*args, **kwargs)
        if mode not in ("rtc", "temporal_ensemble"):
            raise ValueError(
                f"Unknown realtime mode {mode!r}; expected 'rtc' or "
                "'temporal_ensemble'."
            )
        self._mode = mode
        self._realtime_config = config
        self._control_hz = control_hz
        self._adapt_inference_delay = adapt_inference_delay
        self._controller: ChunkingController | None = None

        if mode == "rtc":
            from neuracore.ml.utils.real_time_chunking import RTCConfig

            if not isinstance(config, RTCConfig):
                raise EndpointError(
                    "mode='rtc' requires an RTCConfig; " f"got {type(config).__name__}."
                )
            if not self._policy.supports_real_time_chunking:
                raise EndpointError(
                    "Real-time chunking requires a diffusion policy; the loaded "
                    f"model is a {type(self._policy.model).__name__}."
                )
        else:
            from neuracore.ml.utils.temporal_ensemble import TemporalEnsembleConfig

            if not isinstance(config, TemporalEnsembleConfig):
                raise EndpointError(
                    "mode='temporal_ensemble' requires a TemporalEnsembleConfig; "
                    f"got {type(config).__name__}."
                )

    @property
    def mode(self) -> str:
        """Realtime execution mode bound at construction."""
        return self._mode

    @property
    def prediction_horizon(self) -> int:
        """Number of actions in a chunk, as the model was trained."""
        return self._policy.prediction_horizon

    @property
    def action_names(self) -> list[tuple[DataType, str | None]]:
        """Column layout of a raw action chunk."""
        return self._policy.output_action_names()

    def replace_config(
        self,
        config: "RTCConfig | TemporalEnsembleConfig",
        *,
        control_hz: float | None = None,
        adapt_inference_delay: bool | None = None,
    ) -> None:
        """Replace session config (e.g. after latency sizing) and drop the controller.

        Args:
            config: New mode-specific configuration (must match ``mode``).
            control_hz: Optional updated control rate.
            adapt_inference_delay: Optional RTC adaptation flag.
        """
        if self._mode == "rtc":
            from neuracore.ml.utils.real_time_chunking import RTCConfig

            if not isinstance(config, RTCConfig):
                raise EndpointError(
                    "mode='rtc' requires an RTCConfig; " f"got {type(config).__name__}."
                )
        else:
            from neuracore.ml.utils.temporal_ensemble import TemporalEnsembleConfig

            if not isinstance(config, TemporalEnsembleConfig):
                raise EndpointError(
                    "mode='temporal_ensemble' requires a TemporalEnsembleConfig; "
                    f"got {type(config).__name__}."
                )
        if self._controller is not None:
            self._controller.request_stop()
        self._realtime_config = config
        if control_hz is not None:
            self._control_hz = control_hz
        if adapt_inference_delay is not None:
            self._adapt_inference_delay = adapt_inference_delay
        self._controller = None

    def make_chunker(self) -> "ChunkingController":
        """Build the async chunk controller for this policy.

        Session settings come from construction; observations are pushed via
        :meth:`ChunkingController.get_action`.

        Returns:
            ChunkingController: Not yet started; call ``start()`` on it (or on
            this policy).
        """
        from neuracore.ml.utils.rtc_controller import (
            ChunkingController,
            RTCReplanner,
            TemporalEnsembleReplanner,
        )

        if self._mode == "rtc":
            from neuracore.ml.utils.real_time_chunking import RTCConfig

            assert isinstance(self._realtime_config, RTCConfig)
            return ChunkingController(
                self._policy,
                RTCReplanner(self._policy, self._realtime_config),
                execution_horizon=self._realtime_config.execution_horizon,
                inference_delay=self._realtime_config.inference_delay,
                control_hz=self._control_hz,
                adapt_inference_delay=self._adapt_inference_delay,
                enforce_rtc_invariant=True,
            )

        from neuracore.ml.utils.temporal_ensemble import TemporalEnsembleConfig

        assert isinstance(self._realtime_config, TemporalEnsembleConfig)
        return ChunkingController(
            self._policy,
            TemporalEnsembleReplanner(self._policy, self._realtime_config),
            execution_horizon=self._realtime_config.execution_horizon,
            inference_delay=0,
            control_hz=self._control_hz,
            adapt_inference_delay=False,
            enforce_rtc_invariant=False,
        )

    def start(self) -> None:
        """Start the async chunk controller."""
        if self._controller is None:
            self._controller = self.make_chunker()
        self._controller.start()

    def request_stop(self) -> None:
        """Ask the controller to stop without joining the inference thread."""
        if self._controller is not None:
            self._controller.request_stop()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the controller and wait for the inference thread.

        Args:
            timeout: Seconds to wait for the thread to join.
        """
        if self._controller is not None:
            self._controller.stop(timeout=timeout)

    def wait_for_first_chunk(self, timeout: float = 30.0) -> bool:
        """Block until the first action chunk is ready.

        Args:
            timeout: Seconds to wait.

        Returns:
            bool: True if a chunk is ready.

        Raises:
            RuntimeError: If :meth:`start` has not been called.
        """
        if self._controller is None:
            raise RuntimeError("Call start() before wait_for_first_chunk().")
        return self._controller.wait_for_first_chunk(timeout=timeout)

    def get_action(self, observation: SynchronizedPoint | None = None) -> Any:
        """Return the next action and advance the controller cursor.

        Prefer passing ``observation``: omitting it pulls a fresh sync point on
        every tick, which walks every active robot stream and may merge remote
        node data. See :meth:`ChunkingController.get_action`.

        Args:
            observation: Explicit sync point. When ``None``, falls back to
                :func:`~neuracore.core.get_latest_sync_point.get_latest_sync_point`
                on every call.

        Returns:
            The action for this tick (``np.ndarray``), or ``None`` before the
            first chunk.

        Raises:
            RuntimeError: If :meth:`start` has not been called.
        """
        if self._controller is None:
            raise RuntimeError("Call start() before get_action().")
        return self._controller.get_action(observation)

    def stats(self) -> "ChunkingStats":
        """Return chunking health counters.

        Raises:
            RuntimeError: If :meth:`start` has not been called.
        """
        if self._controller is None:
            raise RuntimeError("Call start() before stats().")
        return self._controller.stats()

    def benchmark(
        self,
        sync_point: SynchronizedPoint,
        iterations: int = 10,
    ) -> list[float]:
        """Time one replan so a caller can size its execution horizon.

        The first iteration is discarded as warm-up, since it pays for lazy CUDA
        initialisation and any cuDNN autotuning.

        Args:
            sync_point: A representative observation.
            iterations: Timed iterations to run, after the warm-up.

        Returns:
            list[float]: Per-iteration durations in seconds, ascending.
        """
        import time

        import numpy as np
        import torch

        from neuracore.ml.utils.real_time_chunking import RTCConfig

        horizon = self.prediction_horizon
        action_dim = len(self.action_names)
        prev_chunk = np.zeros((horizon, action_dim), dtype=np.float32)

        durations = []
        for index in range(iterations + 1):
            started = time.monotonic()
            if isinstance(self._realtime_config, RTCConfig):
                self._policy.predict_action_chunk(
                    sync_point, prev_chunk, self._realtime_config
                )
            else:
                self._policy.predict_action_chunk(sync_point)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if index > 0:
                durations.append(time.monotonic() - started)
        return sorted(durations)


class ServerPolicy(Policy):
    """Base class for server-based policies that communicate via HTTP.

    This class provides common functionality for policies that send requests
    to HTTP endpoints, whether local or remote.
    """

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        input_embodiment_description: EmbodimentDescription | None = None,
    ):
        """Initialize the server policy with connection details.

        Args:
            robot: Robot instance for accessing sensor streams.
            base_url: Base URL of the server.
            headers: Optional HTTP headers for authentication.
            input_embodiment_description: Optional input spec used to project
                local sync points before sending them to the server.
        """
        super().__init__()
        self._base_url = base_url
        self._headers = headers or {}
        self._input_embodiment_description = input_embodiment_description
        self._is_local = "localhost" in base_url or "127.0.0.1" in base_url

    def set_checkpoint(
        self, epoch: int | None = None, checkpoint_file: str | None = None
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
            session = thread_local_session()
            response = session.post(
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
            raise EndpointError(
                "Failed to connect to endpoint, "
                "please check your internet connection and try again."
            )
        except requests.exceptions.RequestException as e:
            raise EndpointError(f"Failed to set checkpoint: {str(e)}")

    def _predict(
        self,
        sync_point: SynchronizedPoint | None = None,
    ) -> dict[DataType, dict[str, "BatchedNCData"]]:
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
            InsufficientSynchronizedPointError:
                If the sync point doesn't contain required data.
            ValueError: If payload size exceeds limits for remote endpoints.
        """
        # Lazy import to avoid torch dependency at module load time
        from neuracore_types import DATA_TYPE_TO_BATCHED_NC_DATA_CLASS

        if sync_point is None:
            sync_point = get_latest_sync_point()
        if self._input_embodiment_description is not None:
            filtered_data = {
                data_type: sync_point.data[data_type]
                for data_type in self._input_embodiment_description.keys()
                if data_type in sync_point.data
            }
            sync_point.data = filtered_data
        response = None
        try:
            session = thread_local_session()
            response = session.post(
                f"{self._base_url}{PREDICT_ENDPOINT}",
                headers=self._headers,
                json=sync_point.model_dump(mode="json"),
                timeout=int(os.getenv("NEURACORE_ENDPOINT_TIMEOUT", 10)),
            )
            response.raise_for_status()
            result = response.json()
            sync_point_preds = {
                DataType(data_type): {
                    key: DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[data_type].model_validate(
                        value
                    )
                    for key, value in data_type_dict.items()
                }
                for data_type, data_type_dict in result.items()
            }
            return sync_point_preds
        except requests.exceptions.ConnectionError:
            raise EndpointError(
                "Failed to connect to endpoint, "
                "please check your internet connection and try again."
            )
        except requests.exceptions.RequestException as e:
            if response is not None:
                if response.status_code == 422:
                    raise InsufficientSynchronizedPointError(
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
        input_embodiment_description: EmbodimentDescription | None = None,
        output_embodiment_description: EmbodimentDescription | None = None,
        input_preprocessing_config: PreprocessingConfiguration | None = None,
        device: str | None = None,
        job_id: str | None = None,
        port: int = 8080,
        host: str = "127.0.0.1",
        endpoint_id: str | None = None,
        robot_id: str | None = None,
    ):
        """Initialize the local server policy.

        Args:
            input_embodiment_description: Specification of the order that will
                be fed into the model
            output_embodiment_description: Specification of the order that will
                be fed into the model outputs
            org_id: Organization ID
            model_path: Path to the .nc.zip model file
            device: Device model to be loaded on
            job_id: Optional job ID to associate with the server
            port: Port to run the server on
            host: Host to bind to
            endpoint_id: Optional deployed endpoint ID used for cloud log uploads.
            robot_id: Optional robot ID used to resolve embodiments from model
                metadata when explicit embodiments are omitted.
            input_preprocessing_config: Preprocessing configuration for the input
                data.
        """
        super().__init__(
            f"http://{host}:{port}",
            input_embodiment_description=input_embodiment_description,
        )
        self.input_embodiment_description = input_embodiment_description
        self.output_embodiment_description = output_embodiment_description
        self.input_preprocessing_config = input_preprocessing_config
        self.robot_id = robot_id
        self.org_id = org_id
        self.job_id = job_id
        self.endpoint_id = endpoint_id
        self.model_path = model_path
        self.device = device
        self.port = port
        self.host = host
        self.server_process: Popen | None = None
        self._log_streamer: EndpointLogStreamer | None = None
        self._endpoint_log_path: Path | None = None
        self._startup_status_path: Path | None = None
        if endpoint_id is not None:
            output_dir = (
                Path(tempfile.gettempdir()) / "neuracore" / "endpoints" / endpoint_id
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            self._endpoint_log_path = output_dir / "endpoint.log"
            self._startup_status_path = output_dir / "startup-status.json"
            storage_handler = EndpointStorageHandler(endpoint_id=endpoint_id)
            self._log_streamer = EndpointLogStreamer(
                storage_handler=storage_handler,
                output_dir=output_dir,
            )
            self._log_streamer.start()
        else:
            status_dir = Path(tempfile.gettempdir()) / "neuracore" / "endpoints"
            status_dir.mkdir(parents=True, exist_ok=True)
            self._startup_status_path = (
                status_dir / f"startup-status-{self.host}-{self.port}.json"
            )
        atexit.register(self.disconnect)
        self._start_server()

    def _start_server(self) -> None:
        """Start the FastAPI server in a subprocess using module execution."""
        # Start the server process using module execution
        cmd = [
            sys.executable,
            "-m",
            "neuracore.core.utils.server",
            "--model-file",
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
        if self.input_embodiment_description is not None:
            input_embodiment_description_str = json.dumps(
                {k.value: v for k, v in self.input_embodiment_description.items()}
            )
            cmd.extend([
                "--input-embodiment-description",
                f"{input_embodiment_description_str}",
            ])
        if self.output_embodiment_description is not None:
            output_embodiment_description_str = json.dumps(
                {k.value: v for k, v in self.output_embodiment_description.items()}
            )
            cmd.extend([
                "--output-embodiment-description",
                f"{output_embodiment_description_str}",
            ])
        if self.input_preprocessing_config is not None:
            input_preprocessing_config_serialized = {
                data_type.value: [m.to_dict() for m in methods]
                for data_type, methods in self.input_preprocessing_config.items()
            }
            cmd.extend([
                "--input-preprocessing-config",
                json.dumps(input_preprocessing_config_serialized),
            ])
        if self.robot_id is not None:
            cmd.extend(["--robot-id", self.robot_id])
        if self.device:
            cmd.extend(["--device", self.device])
        if self.job_id:
            cmd.extend(["--job-id", self.job_id])
        if self._endpoint_log_path is not None:
            cmd.extend(["--log-file-path", str(self._endpoint_log_path)])
        if self._startup_status_path is not None:
            if self._startup_status_path.exists():
                self._startup_status_path.unlink()
            cmd.extend(["--startup-status-file-path", str(self._startup_status_path)])

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

    def _wait_for_server(self, max_attempts: int = 900) -> None:
        """Wait for the server to become available."""
        for attempt in range(max_attempts):
            if (
                self._startup_status_path is not None
                and self._startup_status_path.exists()
            ):
                try:
                    status_payload = json.loads(
                        self._startup_status_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    status_payload = {}
                if status_payload.get("status") == "error":
                    error = status_payload.get(
                        "error", "Unknown server initialization error."
                    )
                    raise EndpointError(f"Local server failed to initialize: {error}")
            # Check if the process has terminated unexpectedly
            if self.server_process and self.server_process.poll() is not None:
                raise EndpointError("Local server process terminated unexpectedly.")
            try:
                session = thread_local_session()
                response = session.get(
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
        self, epoch: int | None = None, checkpoint_file: str | None = None
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
            if self._log_streamer is not None:
                self._log_streamer.close()
                self._log_streamer = None
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
            if self._log_streamer is not None:
                self._log_streamer.close()
                self._log_streamer = None
            logger.info("Local server stopped")


class RemoteServerPolicy(ServerPolicy):
    """Policy for connecting to remote endpoints on the Neuracore platform."""

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str],
        input_embodiment_description: EmbodimentDescription | None = None,
    ):
        """Initialize the remote server policy.

        Args:
            base_url: Base URL of the remote server.
            headers: HTTP headers for authentication.
            input_embodiment_description: Optional input spec used to project
                local sync points before sending them to the server.
        """
        super().__init__(
            base_url,
            headers,
            input_embodiment_description=input_embodiment_description,
        )


def policy(
    input_embodiment_description: EmbodimentDescription | None = None,
    output_embodiment_description: EmbodimentDescription | None = None,
    input_preprocessing_config: PreprocessingConfiguration | None = None,
    train_run_name: str | None = None,
    model_file: str | None = None,
    device: str | None = None,
    robot_id: str | None = None,
) -> DirectPolicy:
    """Launch a direct policy that runs the model in-process.

    Args:
        input_embodiment_description: Specification of the order that will
            be fed into the model
        output_embodiment_description: Specification of the order that will
            be output from the model
        input_preprocessing_config: Preprocessing configuration for the input data.
        train_run_name: Name of the training run to load the model from.
        model_file: Path to the model file to load.
        device: Torch device to run the model on (CPU or GPU, or MPS).
        robot_id: Robot ID used to select embodiments from the model archive
            when embodiment descriptions are not explicitly provided.

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
        input_embodiment_description=input_embodiment_description,
        output_embodiment_description=output_embodiment_description,
        input_preprocessing_config=input_preprocessing_config,
        org_id=org_id,
        job_id=job_id,
        model_path=model_path,
        device=device,
        robot_id=robot_id,
    )


def policy_realtime(
    input_embodiment_description: EmbodimentDescription | None = None,
    output_embodiment_description: EmbodimentDescription | None = None,
    input_preprocessing_config: PreprocessingConfiguration | None = None,
    train_run_name: str | None = None,
    model_file: str | None = None,
    device: str | None = None,
    robot_id: str | None = None,
    *,
    mode: str,
    config: "RTCConfig | TemporalEnsembleConfig",
    control_hz: float,
    adapt_inference_delay: bool = True,
) -> RealTimePolicy:
    """Launch an in-process policy with async overlapping-chunk execution.

    Args:
        input_embodiment_description: Specification of the order that will
            be fed into the model
        output_embodiment_description: Specification of the order that will
            be output from the model
        input_preprocessing_config: Preprocessing configuration for the input data.
        train_run_name: Name of the training run to load the model from.
        model_file: Path to the model file to load.
        device: Torch device to run the model on (CPU or GPU, or MPS).
        robot_id: Robot ID used to select embodiments from the model archive
            when embodiment descriptions are not explicitly provided.
        mode: ``"rtc"`` or ``"temporal_ensemble"`` (required; no default).
        config: Mode-specific configuration (:class:`RTCConfig` or
            :class:`TemporalEnsembleConfig`).
        control_hz: Rate at which :meth:`RealTimePolicy.get_action` will be called.
        adapt_inference_delay: For RTC, track measured latency with ``d``.

    Returns:
        RealTimePolicy ready for :meth:`~RealTimePolicy.start`.

    Raises:
        ValueError: If neither train_run_name nor model_file is provided, or
            ``mode`` is invalid.
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

    return RealTimePolicy(
        input_embodiment_description=input_embodiment_description,
        output_embodiment_description=output_embodiment_description,
        input_preprocessing_config=input_preprocessing_config,
        org_id=org_id,
        job_id=job_id,
        model_path=model_path,
        device=device,
        robot_id=robot_id,
        mode=mode,
        config=config,
        control_hz=control_hz,
        adapt_inference_delay=adapt_inference_delay,
    )


def policy_local_server(
    input_embodiment_description: EmbodimentDescription | None = None,
    output_embodiment_description: EmbodimentDescription | None = None,
    input_preprocessing_config: PreprocessingConfiguration | None = None,
    train_run_name: str | None = None,
    model_file: str | None = None,
    device: str | None = None,
    port: int = 8080,
    host: str = "127.0.0.1",
    job_id: str | None = None,
    endpoint_id: str | None = None,
    robot_id: str | None = None,
) -> LocalServerPolicy:
    """Launch a local server policy with a FastAPI server.

    Args:
        input_embodiment_description: Specification of the order that
            will be fed into the model
        output_embodiment_description: Specification of the order that
            will be output from the model
        input_preprocessing_config: Preprocessing configuration for the input data.
        train_run_name: Name of the training run to load the model from.
        model_file: Path to the model file to load.
        device: Device model to be loaded on.
        port: Port to run the server on.
        host: Host to bind to.
        job_id: Optional job ID to associate with the server.
        endpoint_id: Optional endpoint ID used for endpoint log streaming.
        robot_id: Robot ID used to select embodiments from the model archive
            when embodiment descriptions are not explicitly provided.

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
        input_embodiment_description=input_embodiment_description,
        output_embodiment_description=output_embodiment_description,
        input_preprocessing_config=input_preprocessing_config,
        device=device,
        job_id=job_id,
        port=port,
        host=host,
        endpoint_id=endpoint_id,
        robot_id=robot_id,
    )


def policy_remote_server(
    endpoint_name: str,
) -> RemoteServerPolicy:
    """Launch a remote server policy connected to a deployed endpoint.

    Args:
        endpoint_name: Name of the deployed endpoint.

    Returns:
        RemoteServerPolicy instance for remote inference.
    """
    auth = get_auth()
    org_id = get_current_org()

    try:
        # Find endpoint by name
        session = thread_local_session()
        response = session.get(
            f"{API_URL}/org/{org_id}/models/endpoints", headers=auth.get_headers()
        )
        response.raise_for_status()

        endpoints = response.json()
        matching_endpoints = [e for e in endpoints if e["name"] == endpoint_name]
        if not matching_endpoints:
            raise EndpointError(f"No endpoint found with name: {endpoint_name}")

        active_endpoints = [e for e in matching_endpoints if e["status"] == "active"]
        if not active_endpoints:
            raise EndpointError(f"Endpoint {endpoint_name} is not active")
        if len(active_endpoints) > 1:
            raise EndpointError(
                f"Multiple active endpoints found with name {endpoint_name} "
            )
        endpoint = active_endpoints[0]
        input_embodiment_description = None
        if endpoint.get("input_embodiment_description"):
            input_embodiment_description = _parse_embodiment_description(
                endpoint["input_embodiment_description"]
            )

        return RemoteServerPolicy(
            base_url=f"{API_URL}/org/{org_id}/models/endpoints/{endpoint['id']}",
            headers=auth.get_headers(),
            input_embodiment_description=input_embodiment_description,
        )
    except requests.exceptions.ConnectionError:
        raise EndpointError(
            "Failed to connect to endpoint: Connection Error. "
            "Please check your internet connection and try again."
        )
    except requests.exceptions.RequestException as e:
        raise EndpointError(f"Failed to connect to endpoint: {str(e)}")


# Helper functions
def _download_model(job_id: str, org_id: str) -> Path:
    """Download model from training run."""
    auth = get_auth()
    session = thread_local_session()
    response = session.get(
        f"{API_URL}/org/{org_id}/training/jobs/{job_id}/model_url",
        headers=auth.get_headers(),
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    train_run_name = data["train_run_name"]
    destination = Path(tempfile.gettempdir()) / job_id / f"{train_run_name}.nc.zip"

    model_path = download_to_cache(
        data["url"],
        destination,
        "Downloading model...",
    )
    print(f"Model available at {model_path}")
    return model_path


def _get_job_id(train_run_name: str, org_id: str) -> str:
    """Get job ID from training run name."""
    auth = get_auth()
    session = thread_local_session()
    response = session.get(
        f"{API_URL}/org/{org_id}/training/jobs", headers=auth.get_headers()
    )
    response.raise_for_status()
    jobs = response.json()

    for job in jobs:
        if job["name"] == train_run_name:
            return job["id"]

    raise EndpointError(f"Training run not found: {train_run_name}")
