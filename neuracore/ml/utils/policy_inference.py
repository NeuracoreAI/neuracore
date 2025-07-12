"""Policy Inference Module."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image

from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.nc_types import (
    CameraData,
    CustomData,
    EndEffectorData,
    JointData,
    LanguageData,
    ModelPrediction,
    PointCloudData,
    PoseData,
    SyncPoint,
)
from neuracore.core.utils.download import download_with_progress
from neuracore.ml import BatchedInferenceSamples, MaskableData
from neuracore.ml.utils.nc_archive import load_model_from_nc_archive

logger = logging.getLogger(__name__)


class PolicyInference:
    """PolicyInference class for handling model inference.

    This class is responsible for loading a model from a Neuracore archive,
    processing incoming data from SyncPoints, and running inference to
    generate predictions.
    """

    def __init__(
        self,
        model_file: Path,
        org_id: str,
        job_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the policy inference."""
        self.org_id = org_id
        self.job_id = job_id
        self.model = load_model_from_nc_archive(model_file)
        self.dataset_description = self.model.model_init_description.dataset_description
        self.device = (
            torch.device(device)
            if device
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )

    def _process_joint_data(self, joint_data: JointData, max_len: int) -> MaskableData:
        """Process joint state data into batched tensor format.

        Converts joint data from a single sample into a batched tensor with
        appropriate padding and masking for variable-length joint configurations.

        Args:
            joint_data: JointData object from the sample.
            max_len: Maximum joint dimension for padding.

        Returns:
            MaskableData containing batched joint values and attention masks.
        """
        values = np.zeros((1, max_len))
        mask = np.zeros((1, max_len))
        v = list(joint_data.values.values())
        values[0, : len(v)] = v
        mask[0, : len(v)] = 1.0
        return MaskableData(
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )

    def _process_image_data(
        self, image_data: dict[str, CameraData], max_len: int, is_depth: bool
    ) -> MaskableData:
        """Process camera image data into batched tensor format.

        Decodes base64 images, applies standard preprocessing transforms,
        and creates batched tensors with masking for variable numbers of cameras.

        Args:
            image_data: Dictionary mapping camera names to CameraData.
            max_len: Maximum number of cameras to support with padding.
            is_depth: Whether the images are depth images (single channel).

        Returns:
            MaskableData containing batched image tensors and attention masks.
        """
        channels = 1 if is_depth else 3
        values = np.zeros((1, max_len, channels, 224, 224))
        mask = np.zeros((1, max_len))

        for j, (camera_name, camera_data) in enumerate(image_data.items()):
            if j >= max_len:
                break

            image = camera_data.frame
            assert isinstance(
                image, np.ndarray
            ), f"Expected numpy array for image, got {type(image)}"

            # Handle different image formats
            if is_depth:
                if len(image.shape) == 3:
                    image = np.mean(image, axis=2)  # Convert to grayscale
                image = np.expand_dims(image, axis=0)  # Add channel dimension
            else:
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=2)  # Convert grayscale to RGB
                image = np.transpose(image, (2, 0, 1))  # HWC to CHW

            # Resize and normalize
            image = Image.fromarray(
                image.transpose(1, 2, 0) if not is_depth else image[0]
            )
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])
            values[0, j] = transform(image).numpy()
            mask[0, j] = 1.0

        return MaskableData(
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )

    def _process_end_effector_data(
        self, end_effector_data: EndEffectorData, max_len: int
    ) -> MaskableData:
        """Process end-effector data into batched tensor format."""
        values = np.zeros((1, max_len))
        mask = np.zeros((1, max_len))

        ee_values = list(end_effector_data.open_amounts.values())
        values[0, : len(ee_values)] = ee_values
        mask[0, : len(ee_values)] = 1.0

        return MaskableData(
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )

    def _process_pose_data(
        self, pose_data: dict[str, PoseData], max_len: int
    ) -> MaskableData:
        """Process pose data into batched tensor format."""
        values = np.zeros((1, max_len))
        mask = np.zeros((1, max_len))

        all_poses = []
        for pose_name, pose_data_item in pose_data.items():
            all_poses.extend(pose_data_item.pose[pose_name])  # 6DOF pose

        values[0, : len(all_poses)] = all_poses
        mask[0, : len(all_poses)] = 1.0

        return MaskableData(
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )

    def _process_point_cloud_data(
        self, point_cloud_data: dict[str, PointCloudData], max_clouds: int
    ) -> MaskableData:
        """Process point cloud data into batched tensor format."""
        target_num_points = 1024  # Standard point cloud size
        values = np.zeros((1, max_clouds, target_num_points, 3))
        mask = np.zeros((1, max_clouds))

        for j, (cloud_name, cloud_data) in enumerate(point_cloud_data.items()):
            if j >= max_clouds:
                break

            points = np.array(cloud_data.points)  # [num_points, 3]
            current_num_points = points.shape[0]

            if current_num_points < target_num_points:
                # Pad with zeros
                padding = np.zeros((target_num_points - current_num_points, 3))
                points = np.concatenate([points, padding], axis=0)
            elif current_num_points > target_num_points:
                # Subsample
                indices = np.random.choice(
                    current_num_points, target_num_points, replace=False
                )
                points = points[indices]

            values[0, j] = points
            mask[0, j] = 1.0

        return MaskableData(
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )

    def _process_custom_data(
        self, custom_data: dict[str, CustomData]
    ) -> dict[str, MaskableData]:
        """Process custom data into batched tensor format."""
        result = {}

        for key, custom_data_item in custom_data.items():
            data = custom_data_item.data
            if isinstance(data, (list, np.ndarray)):
                batch_data = np.array(data, dtype=np.float32)
            else:
                # Convert other types to float
                batch_data = np.array([float(hash(str(data)) % 1000)], dtype=np.float32)

            # Add batch dimension
            batch_data = np.expand_dims(batch_data, axis=0)
            mask = np.ones((1, batch_data.shape[-1]))

            result[key] = MaskableData(
                torch.tensor(batch_data, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32),
            )

        return result

    def _process_language_data(self, language_data: LanguageData) -> MaskableData:
        """Process natural language instruction data using model tokenizer.

        Tokenizes text instructions into input IDs and attention masks using
        the model's built-in tokenization functionality.

        Args:
            language_data: LanguageData object containing text instruction.

        Returns:
            MaskableData containing tokenized text and attention masks.
        """
        # Tokenize the text (create batch of size 1)
        texts = [language_data.text]
        input_ids, attention_mask = self.model.tokenize_text(texts)
        return MaskableData(
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.float32),
        )

    def _preprocess(self, sync_point: SyncPoint) -> BatchedInferenceSamples:
        """Preprocess incoming sync point into model-compatible format.

        Converts a single SyncPoint data into batched tensors suitable
        for model inference.
        Handles multiple data modalities including joint states,
        images, and language instructions.

        Args:
            sync_point: SyncPoint containing data from a single time step.

        Returns:
            BatchedInferenceSamples object ready for model inference.
        """
        batch = BatchedInferenceSamples()

        # Process joint data
        if sync_point.joint_positions:
            batch.joint_positions = self._process_joint_data(
                sync_point.joint_positions,
                self.dataset_description.joint_positions.max_len,
            )
        if sync_point.joint_velocities:
            batch.joint_velocities = self._process_joint_data(
                sync_point.joint_velocities,
                self.dataset_description.joint_velocities.max_len,
            )
        if sync_point.joint_torques:
            batch.joint_torques = self._process_joint_data(
                sync_point.joint_torques,
                self.dataset_description.joint_torques.max_len,
            )
        if sync_point.joint_target_positions:
            batch.joint_target_positions = self._process_joint_data(
                sync_point.joint_target_positions,
                self.dataset_description.joint_target_positions.max_len,
            )

        # Process visual data
        if sync_point.rgb_images:
            batch.rgb_images = self._process_image_data(
                sync_point.rgb_images,
                self.dataset_description.max_num_rgb_images,
                is_depth=False,
            )
        if sync_point.depth_images:
            batch.depth_images = self._process_image_data(
                sync_point.depth_images,
                self.dataset_description.max_num_depth_images,
                is_depth=True,
            )

        # Process end-effector data
        if sync_point.end_effectors:
            batch.end_effectors = self._process_end_effector_data(
                sync_point.end_effectors,
                self.dataset_description.end_effector_states.max_len,
            )

        # Process pose data
        if sync_point.poses:
            batch.poses = self._process_pose_data(
                sync_point.poses,
                self.dataset_description.poses.max_len,
            )

        # Process point cloud data
        if sync_point.point_clouds:
            batch.point_clouds = self._process_point_cloud_data(
                sync_point.point_clouds,
                self.dataset_description.max_num_point_clouds,
            )

        # Process language data
        if sync_point.language_data:
            batch.language_tokens = self._process_language_data(
                sync_point.language_data,
            )

        # Process custom data
        if sync_point.custom_data:
            batch.custom_data = self._process_custom_data(
                sync_point.custom_data,
            )

        return batch.to(self.device)

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

    def __call__(self, sync_point: SyncPoint) -> ModelPrediction:
        """Process a single sync point and run inference.

        Args:
            sync_point: SyncPoint containing data from a single time step.

        Returns:
            ModelPrediction containing the model's output predictions.
        """
        batch = self._preprocess(sync_point)
        with torch.no_grad():
            batch_output: ModelPrediction = self.model(batch)
            return batch_output
