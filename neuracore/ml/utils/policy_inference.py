"""Policy Inference Module."""

import logging
import tempfile
from collections import defaultdict
from pathlib import Path

import requests
import torch
from neuracore_types import (
    DATA_TYPE_TO_BATCHED_NC_DATA_CLASS,
    BatchedNCData,
    DataType,
    SynchronizedPoint,
)

from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.utils.download import download_with_progress
from neuracore.ml import BatchedInferenceInputs
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
        model_input_order: dict[DataType, list[str]],
        model_output_order: dict[DataType, list[str]],
        model_file: Path,
        org_id: str,
        job_id: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the policy inference.

        Args:
            model_input_order: Input mapping per supported robot type.
            model_output_order: Output mapping per supported robot type.
            model_file: Path to the model file to load.
            org_id: ID of the organization for loading checkpoints.
            job_id: ID of the training job for loading checkpoints.
            device: Torch device to run the model inference on.
            robot_to_output_mapping: Output mapping per supported robot type.
        """
        self.org_id = org_id
        self.job_id = job_id
        self.model = load_model_from_nc_archive(model_file, device=device)
        self.dataset_statistics = self.model.model_init_description.dataset_statistics
        self.device = torch.device(device) if device else get_default_device()
        self.model_input_order = model_input_order
        self.model_output_order = model_output_order
        self.prediction_horizon = (
            self.model.model_init_description.output_prediction_horizon
        )

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
        inputs: dict[DataType, list[BatchedNCData]] = {}
        inputs_mask: dict[DataType, torch.Tensor] = {}  # Dict[DataType, (B, MAX_LEN)]
        # We need to go from sync_point (single time step) to BatchedNCData
        for data_type in sync_point.data.keys():
            inputs[data_type] = []
            max_items_for_this_data_type = len(sync_point.data[data_type])
            max_items_trained_on = len(self.dataset_statistics[data_type])
            if max_items_for_this_data_type > max_items_trained_on:
                raise ValueError(
                    f"Received {max_items_for_this_data_type} items for data type "
                    f"{data_type.name}, but model was trained on maximum of "
                    f"{max_items_trained_on} items."
                )
            inputs_mask[data_type] = torch.tensor(
                [1.0] * max_items_for_this_data_type
                + [0.0] * (max_items_trained_on - max_items_for_this_data_type),
                dtype=torch.float32,
            )
            inputs_mask[data_type].unsqueeze_(0)  # Add batch dimension
            for name, nc_data in sync_point.data[data_type].items():
                tensor = DATA_TYPE_TO_BATCHED_NC_DATA_CLASS[data_type].from_nc_data(
                    nc_data
                )
                inputs[data_type].append(tensor)
        return BatchedInferenceInputs(
            inputs=inputs,
            inputs_mask=inputs_mask,
            batch_size=1,
        ).to(self.device)

    def set_checkpoint(
        self, epoch: int | None = None, checkpoint_file: str | None = None
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

    def _assign_names_to_model_outputs(
        self,
        batch_output: dict[DataType, list[BatchedNCData]],
    ) -> dict[DataType, dict[str, BatchedNCData]]:
        """Convert model prediction output to SynchronizedPoint format.

        Args:
            batch_output: ModelPrediction containing the model's outputs.

        Returns:
            SynchronizedPoint with processed outputs.
        """
        outputs: dict[DataType, dict[str, BatchedNCData]] = defaultdict(dict)

        # Map outputs to SynchronizedPoint fields based on output_mapping
        for data_type, list_of_batched_ncdata in batch_output.items():

            # Check that there are enough output names for the data type
            if data_type not in self.model_output_order:
                raise ValueError(f"DataType {data_type} not in output configuration.")
            # There can be more tensors than names due to
            # output padding (multi robot training)
            if len(list_of_batched_ncdata) < len(self.model_output_order[data_type]):
                raise ValueError(
                    f"Not enough output names for DataType {data_type}. "
                    f"Expected at least {len(self.model_output_order[data_type])}, "
                    f"but got {len(list_of_batched_ncdata)}."
                )

            for tensor_idx, batched_nc_data in enumerate(list_of_batched_ncdata):
                name_of_tensor = self.model_output_order[data_type][tensor_idx]
                outputs[data_type][name_of_tensor] = batched_nc_data

        return outputs

    def _validate_input_sync_point(self, sync_point: SynchronizedPoint) -> None:
        """Validate the sync point with what the model had as input.

        Ensures that the sync point contains all required data types
        as specified in the model's input data types.

        Args:
            sync_point: SynchronizedPoint containing data from a single time step.

        Raises:
            ValueError: If the sync point does not contain required data types.
        """
        input_robot_data_spec = self.model.model_init_description.input_data_types
        missing_data_types = []
        for data_type in input_robot_data_spec:
            if data_type not in sync_point.data:
                missing_data_types.append(f"{data_type.name}")
        if missing_data_types:
            raise ValueError(
                "SynchronizedPoint is missing required data types: "
                f"{', '.join(missing_data_types)}"
            )

    def __call__(
        self,
        sync_point: SynchronizedPoint,
    ) -> dict[DataType, dict[str, BatchedNCData]]:
        """Process a single sync point and run inference.

        Args:
            sync_point: SynchronizedPoint containing data from a single time step.

        Returns:
            SynchronizedPoint with model predictions filled in for each robot.
        """
        sync_point = sync_point.order(self.model_input_order)
        self._validate_input_sync_point(sync_point)
        batch = self._preprocess(sync_point)
        with torch.no_grad():
            batch_output: dict[DataType, list[BatchedNCData]] = self.model(batch)
            return self._assign_names_to_model_outputs(batch_output)
