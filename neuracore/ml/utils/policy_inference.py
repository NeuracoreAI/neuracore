"""Policy Inference Module."""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional, cast

import numpy as np
import requests
import torch
from neuracore_types import DataItemStats, DataType, ModelPrediction, SynchronizedPoint

from neuracore.api.globals import GlobalSingleton
from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.utils.download import download_with_progress
from neuracore.ml import BatchedInferenceInputs, MaskableData
from neuracore.ml.utils.device_utils import get_default_device
from neuracore.ml.utils.nc_archive import load_model_from_nc_archive

logger = logging.getLogger(__name__)


class PolicyInference:
    """PolicyInference class for handling model inference.

    This class is responsible for loading a model from a Neuracore archive,
    processing incoming data from SynchronizedPoints, and running inference to
    generate predictions.
    """

    def __init__(
        self,
        model_file: Path,
        org_id: str,
        job_id: Optional[str] = None,
        device: Optional[str] = None,
        robot_to_output_mapping: Optional[dict[str, dict[DataType, list[str]]]] = None,
    ) -> None:
        """Initialize the policy inference.

        Args:
            model_file: Path to the model file to load.
            org_id: ID of the organization for loading checkpoints.
            job_id: ID of the training job for loading checkpoints.
            device: Torch device to run the model inference on.
            robot_to_output_mapping: Output mapping per supported robot type.
            Each output mapping is a dictionary of data types to list of output names.
        """
        self.org_id = org_id
        self.job_id = job_id
        self.model = load_model_from_nc_archive(model_file, device=device)
        self.dataset_statistics = self.model.model_init_description.dataset_statistics
        self.device = torch.device(device) if device else get_default_device()
        self.robot_to_output_mapping = (
            robot_to_output_mapping or self.model.robot_to_output_mapping
        )

    def _validate_robot_to_ncdata_keys(
        self, robot_id: str, data_item_stats: DataItemStats, data_name: str
    ) -> list[str]:
        keys = data_item_stats.robot_to_ncdata_keys.get(robot_id, [])
        if not keys:
            raise ValueError(
                f"No {data_name} found for robot {robot_id} in dataset description."
            )
        return keys

    def _preprocess(self, sync_point: SynchronizedPoint) -> BatchedInferenceInputs:
        """Preprocess incoming sync point into model-compatible format.

        Converts a single SynchronizedPoint data into batched tensors suitable
        for model inference.
        Handles multiple data modalities including joint states,
        images, and language instructions.

        Args:
            sync_point: SynchronizedPoint containing data from a single time step.

        Returns:
            BatchedInferenceSamples object ready for model inference.
        """
        inputs: Dict[DataType, Dict[str, MaskableData]] = {}
        for data_type in sync_point.data.keys():
            inputs[data_type] = {}
            for name, nc_data in sync_point.data[data_type].items():
                tensor = torch.from_numpy(nc_data.numpy())
                num_existing = tensor.shape[0]
                stats = self.dataset_statistics.data[data_type][name]
                extra_states = stats.max_len - num_existing
                if extra_states > 0:
                    tensor = torch.cat(
                        [tensor, torch.zeros(extra_states, dtype=torch.float32)], dim=0
                    )
                tensor_mask = torch.tensor(
                    [1.0] * num_existing + [0.0] * extra_states, dtype=torch.float32
                )
                inputs[data_type][name] = MaskableData(
                    data=tensor,
                    mask=tensor_mask,
                )
        return BatchedInferenceInputs(inputs=inputs).to(self.device)

    def set_checkpoint(
        self, epoch: Optional[int] = None, checkpoint_file: Optional[str] = None
    ) -> None:
        """Set the model checkpoint to use for inference.

        Args:
            epoch: The epoch number of the checkpoint to load.
                -1 to load the latest checkpoint.
            checkpoint_file: Optional path to a specific checkpoint file.
                If provided, overrides the epoch setting.
        """
        if epoch is not None:
            if epoch < -1:
                raise ValueError("Epoch must be -1 (latest) or a non-negative integer.")
            if self.org_id is None or self.job_id is None:
                raise ValueError(
                    "Organization ID and Job ID must be set to load checkpoints."
                )
            checkpoint_name = f"checkpoint_{epoch if epoch != -1 else 'latest'}.pt"
            checkpoint_path = (
                Path(tempfile.gettempdir()) / self.job_id / checkpoint_name
            )
            if not checkpoint_path.exists():
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                response = requests.get(
                    f"{API_URL}/org/{self.org_id}/training/jobs/{self.job_id}/checkpoint_url/{checkpoint_name}",
                    headers=get_auth().get_headers(),
                    timeout=30,
                )
                if response.status_code == 404:
                    raise ValueError(f"Checkpoint {checkpoint_name} does not exist.")
                checkpoint_path = download_with_progress(
                    response.json()["url"],
                    f"Downloading checkpoint {checkpoint_name}",
                    destination=checkpoint_path,
                )
        elif checkpoint_file is not None:
            checkpoint_path = Path(checkpoint_file)
        else:
            raise ValueError("Must specify either epoch or checkpoint_file.")

        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True),
            strict=False,
        )

    def _model_prediction_to_sync_points(
        self,
        batch_output: ModelPrediction,
        robot_id: Optional[str] = None,
    ) -> list[SynchronizedPoint]:
        """Convert model prediction output to SynchronizedPoint format.

        Args:
            batch_output: ModelPrediction containing the model's outputs.

        Returns:
            SynchronizedPoint with processed outputs.
        """
        horizon = list(batch_output.outputs.values())[0].shape[1]
        sync_points: list[SynchronizedPoint] = [
            SynchronizedPoint(robot_id=robot_id) for _ in range(horizon)
        ]

        # Map outputs to SynchronizedPoint fields based on output_mapping
        for data_type, output in batch_output.outputs.items():
            # Remove batch dimension if present
            if isinstance(output, np.ndarray) and output.ndim > 0:
                output = output[0]
            output = cast(np.ndarray, output)
            for t in range(horizon):
                nc_data = data_type.from_numpy(output[t])
                # TODO: Fix naming for multiple data items
                sync_points[t].data[data_type] = {"0": nc_data}

        return sync_points

    def _validate_input_sync_point(self, sync_point: SynchronizedPoint) -> None:
        """Validate the sync point with what the model had as input.

        Ensures that the sync point contains all required data types
        as specified in the model's input data types.

        Args:
            sync_point: SynchronizedPoint containing data from a single time step.

        Raises:
            ValueError: If the sync point does not contain required data types.
        """
        input_data_types = self.model.model_init_description.input_data_types
        missing_data_types = []
        for data_type in input_data_types:
            if data_type not in sync_point.data:
                missing_data_types.append(f"{data_type.name}")
        if missing_data_types:
            raise ValueError(
                "SynchronizedPoint is missing required data types: "
                f"{', '.join(missing_data_types)}"
            )

    def __call__(
        self, sync_point: SynchronizedPoint, robot_name: Optional[str] = None
    ) -> list[SynchronizedPoint]:
        """Process a single sync point and run inference.

        Args:
            sync_point: SynchronizedPoint containing data from a single time step.
            robot_name: Name of the robot to predict on. If None, uses the active robot.

        Returns:
            SynchronizedPoint with model predictions filled in for each robot.
        """
        sync_point = sync_point.order()
        active_robot = GlobalSingleton()._active_robot
        available_robots = set(self.robot_to_output_mapping.keys())

        if robot_name is None and active_robot is None:
            raise ValueError(
                "Robot name must be provided if no active robot is set. "
                f"Available robots: {', '.join(available_robots)}"
            )

        # Fallback to active robot if robot name is not provided
        if robot_name is None and active_robot is not None:
            robot_name = active_robot.name
            if sync_point.robot_id is None:
                sync_point.robot_id = active_robot.id

        if robot_name not in available_robots:
            raise ValueError(
                f"Robot name {robot_name} is not in the list of "
                f"available robots: {', '.join(available_robots)}"
            )
        self.robot_to_output_mapping[robot_name]

        self._validate_input_sync_point(sync_point)
        batch = self._preprocess(sync_point)
        with torch.no_grad():
            batch_output: ModelPrediction = self.model(batch)
            return self._model_prediction_to_sync_points(
                batch_output, robot_id=sync_point.robot_id
            )
