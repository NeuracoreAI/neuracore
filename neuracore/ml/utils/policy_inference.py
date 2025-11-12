"""Policy Inference Module."""

import logging
import tempfile
from pathlib import Path
from typing import Optional, cast

import numpy as np
import requests
import torch
import torchvision.transforms as T
from neuracore_types import (
    CameraData,
    CustomData,
    DataItemStats,
    DataType,
    EndEffectorData,
    EndEffectorPoseData,
    JointData,
    LanguageData,
    ModelPrediction,
    ParallelGripperOpenAmountData,
    PointCloudData,
    PoseData,
    SyncPoint,
)
from PIL import Image

from neuracore.api.globals import GlobalSingleton
from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.utils.download import download_with_progress
from neuracore.ml import BatchedInferenceSamples, MaskableData
from neuracore.ml.utils.device_utils import get_default_device
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
        output_mapping: Optional[dict[DataType, list[str]]] = None,
    ) -> None:
        """Initialize the policy inference."""
        self.org_id = org_id
        self.job_id = job_id
        self.model = load_model_from_nc_archive(model_file, device=device)
        self.dataset_description = self.model.model_init_description.dataset_description
        self.device = torch.device(device) if device else get_default_device()
        self.output_mapping = output_mapping
        self.robot_ids_to_output_mapping: dict[str, dict[DataType, list[str]]] = {}

    def _validate_robot_to_ncdata_keys(
        self, robot_id: str, data_item_stats: DataItemStats, data_name: str
    ) -> list[str]:
        keys = data_item_stats.robot_to_ncdata_keys.get(robot_id, [])
        if not keys:
            raise ValueError(
                f"No {data_name} found for robot {robot_id} in dataset description."
            )
        return keys

    def _get_output_mapping(self, robot_id: Optional[str]) -> dict[DataType, list[str]]:
        if self.output_mapping is not None:
            return self.output_mapping
        if robot_id is None:
            raise ValueError(
                "You must either set an active robot or provide an output mapping."
            )
        if robot_id in self.robot_ids_to_output_mapping:
            return self.robot_ids_to_output_mapping[robot_id]

        output_data_types = self.model.model_init_description.output_data_types
        output_mapping: dict[DataType, list[str]] = {}
        if DataType.JOINT_TARGET_POSITIONS in output_data_types:
            keys = self._validate_robot_to_ncdata_keys(
                robot_id,
                self.dataset_description.joint_target_positions,
                "joint target positions",
            )
            output_mapping[DataType.JOINT_TARGET_POSITIONS] = keys
        if DataType.JOINT_POSITIONS in output_data_types:
            keys = self._validate_robot_to_ncdata_keys(
                robot_id,
                self.dataset_description.joint_positions,
                "joint positions",
            )
            output_mapping[DataType.JOINT_POSITIONS] = keys
        if DataType.JOINT_VELOCITIES in output_data_types:
            keys = self._validate_robot_to_ncdata_keys(
                robot_id,
                self.dataset_description.joint_velocities,
                "joint velocities",
            )
            output_mapping[DataType.JOINT_VELOCITIES] = keys
        if DataType.JOINT_TORQUES in output_data_types:
            keys = self._validate_robot_to_ncdata_keys(
                robot_id,
                self.dataset_description.joint_torques,
                "joint torques",
            )
            output_mapping[DataType.JOINT_TORQUES] = keys
        if DataType.END_EFFECTORS in output_data_types:
            keys = self._validate_robot_to_ncdata_keys(
                robot_id,
                self.dataset_description.end_effector_states,
                "end effector states",
            )
            output_mapping[DataType.END_EFFECTORS] = keys
        if DataType.POSES in output_data_types:
            keys = self._validate_robot_to_ncdata_keys(
                robot_id,
                self.dataset_description.poses,
                "poses",
            )
            output_mapping[DataType.POSES] = keys
        if DataType.END_EFFECTOR_POSES in output_data_types:
            keys = self._validate_robot_to_ncdata_keys(
                robot_id,
                self.dataset_description.end_effector_poses,
                "end effector poses",
            )
            output_mapping[DataType.END_EFFECTOR_POSES] = keys
        if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in output_data_types:
            keys = self._validate_robot_to_ncdata_keys(
                robot_id,
                self.dataset_description.parallel_gripper_open_amounts,
                "parallel gripper open amounts",
            )
            output_mapping[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS] = keys
        if DataType.RGB_IMAGE in output_data_types:
            keys = self._validate_robot_to_ncdata_keys(
                robot_id,
                self.dataset_description.rgb_images,
                "RGB images",
            )
            output_mapping[DataType.RGB_IMAGE] = keys
        if DataType.DEPTH_IMAGE in output_data_types:
            keys = self._validate_robot_to_ncdata_keys(
                robot_id,
                self.dataset_description.depth_images,
                "depth images",
            )
            output_mapping[DataType.DEPTH_IMAGE] = keys
        if DataType.POINT_CLOUD in output_data_types:
            keys = self._validate_robot_to_ncdata_keys(
                robot_id,
                self.dataset_description.point_clouds,
                "point clouds",
            )
            output_mapping[DataType.POINT_CLOUD] = keys
        if DataType.LANGUAGE in output_data_types:
            pass  # Language data typically does not require robot-specific keys
        if DataType.CUSTOM in output_data_types:
            all_custom_keys = []
            for (
                custom_data_name,
                data_item_stats,
            ) in self.dataset_description.custom_data.items():
                keys = self._validate_robot_to_ncdata_keys(
                    robot_id, data_item_stats, "custom data"
                )
                all_custom_keys.extend(keys)
            output_mapping[DataType.CUSTOM] = all_custom_keys

        self.robot_ids_to_output_mapping[robot_id] = output_mapping
        return output_mapping

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

    def _process_end_effector_pose_data(
        self, end_effector_pose_data: EndEffectorPoseData, max_len: int
    ) -> MaskableData:
        """Process end-effector pose data into batched tensor format."""
        values = np.zeros((1, max_len))
        mask = np.zeros((1, max_len))

        all_poses = list(end_effector_pose_data.poses.values())
        values[0, : len(all_poses)] = all_poses
        mask[0, : len(all_poses)] = 1.0
        return MaskableData(
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )

    def _process_parallel_gripper_open_amount_data(
        self,
        parallel_gripper_open_amount_data: ParallelGripperOpenAmountData,
        max_len: int,
    ) -> MaskableData:
        """Process parallel gripper open amount data into batched tensor format."""
        values = np.zeros((1, max_len))
        mask = np.zeros((1, max_len))

        all_open_amounts = list(parallel_gripper_open_amount_data.open_amounts.values())
        values[0, : len(all_open_amounts)] = all_open_amounts
        mask[0, : len(all_open_amounts)] = 1.0
        return MaskableData(
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )

    def _process_pose_data(self, pose_data: PoseData, max_len: int) -> MaskableData:
        """Process pose data into batched tensor format."""
        values = np.zeros((1, max_len))
        mask = np.zeros((1, max_len))

        all_poses = []
        for pose_name, pose_data_item in pose_data.pose.items():
            all_poses.extend(pose_data_item)  # 6DOF pose

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
                self.dataset_description.rgb_images.max_len,
                is_depth=False,
            )
        if sync_point.depth_images:
            batch.depth_images = self._process_image_data(
                sync_point.depth_images,
                self.dataset_description.depth_images.max_len,
                is_depth=True,
            )

        # Process end-effector data
        if sync_point.end_effectors:
            batch.end_effectors = self._process_end_effector_data(
                sync_point.end_effectors,
                self.dataset_description.end_effector_states.max_len,
            )

        # Process end-effector pose data
        if sync_point.end_effector_poses:
            batch.end_effector_poses = self._process_end_effector_pose_data(
                sync_point.end_effector_poses,
                self.dataset_description.end_effector_poses.max_len,
            )
        # Process parallel gripper data
        if sync_point.parallel_gripper_open_amounts:
            batch.parallel_gripper_open_amounts = (
                self._process_parallel_gripper_open_amount_data(
                    sync_point.parallel_gripper_open_amounts,
                    self.dataset_description.parallel_gripper_open_amounts.max_len,
                )
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
                self.dataset_description.point_clouds.max_len,
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

    def _model_prediction_to_sync_points(
        self,
        batch_output: ModelPrediction,
        output_mapping: dict[DataType, list[str]],
        robot_id: Optional[str] = None,
    ) -> list[SyncPoint]:
        """Convert model prediction output to SyncPoint format.

        Args:
            batch_output: ModelPrediction containing the model's outputs.

        Returns:
            SyncPoint with processed outputs.
        """
        horizon = list(batch_output.outputs.values())[0].shape[1]
        sync_points: list[SyncPoint] = [
            SyncPoint(robot_id=robot_id) for _ in range(horizon)
        ]

        # Map outputs to SyncPoint fields based on output_mapping
        for data_type, output in batch_output.outputs.items():
            # Remove batch dimension if present
            if isinstance(output, np.ndarray) and output.ndim > 0:
                output = output[0]
            output = cast(np.ndarray, output)
            keys = output_mapping[data_type]

            if data_type == DataType.JOINT_POSITIONS:
                for t in range(horizon):
                    sync_points[t].joint_positions = JointData(
                        values=dict(zip(keys, output[t].tolist()))
                    )
            elif data_type == DataType.JOINT_VELOCITIES:
                for t in range(horizon):
                    sync_points[t].joint_velocities = JointData(
                        values=dict(zip(keys, output[t].tolist()))
                    )
            elif data_type == DataType.JOINT_TORQUES:
                for t in range(horizon):
                    sync_points[t].joint_torques = JointData(
                        values=dict(zip(keys, output[t].tolist()))
                    )
            elif data_type == DataType.JOINT_TARGET_POSITIONS:
                for t in range(horizon):
                    sync_points[t].joint_target_positions = JointData(
                        values=dict(zip(keys, output[t].tolist()))
                    )
            elif data_type == DataType.END_EFFECTOR_POSES:
                for t in range(horizon):
                    sync_points[t].end_effector_poses = EndEffectorPoseData(
                        poses=dict(zip(keys, output[t].tolist()))
                    )
            elif data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                for t in range(horizon):
                    sync_points[t].parallel_gripper_open_amounts = (
                        ParallelGripperOpenAmountData(
                            open_amounts=dict(zip(keys, output[t].tolist()))
                        )
                    )
            elif data_type == DataType.POINT_CLOUD:
                # [T, CLOUDs, N, 3]
                for i in range(horizon):
                    sync_points[i].point_clouds = {
                        cloud_name: PointCloudData(points=output[i, j])
                        for j, cloud_name in enumerate(keys)
                    }
            elif data_type == DataType.RGB_IMAGE:
                # [T, CAMs, H, W, C]
                camera_names = keys
                for t in range(horizon):
                    sync_points[t].rgb_images = {
                        cam_name: CameraData(frame=output[t, i])
                        for i, cam_name in enumerate(camera_names)
                    }
            elif data_type == DataType.DEPTH_IMAGE:
                # [T, CAMs, H, W]
                camera_names = keys
                for t in range(horizon):
                    sync_points[t].depth_images = {
                        cam_name: CameraData(frame=output[t, i])
                        for i, cam_name in enumerate(camera_names)
                    }
            elif data_type == DataType.LANGUAGE:
                raise NotImplementedError(
                    "Language data processing is not implemented yet."
                )
            elif data_type == DataType.CUSTOM:
                # Assuming output is a dictionary with custom data
                for t in range(horizon):
                    if len(output[t]) != len(keys):
                        raise ValueError(
                            f"Output length {len(output[t])} does not match expected "
                            f"keys length {len(keys)} for custom data."
                        )
                    sync_points[t].custom_data = {
                        key: CustomData(
                            data=(
                                value.tolist()
                                if isinstance(value, np.ndarray)
                                else value
                            )
                        )
                        for key, value in zip(keys, output)
                    }
                    if len(output[t]) != len(keys):
                        raise ValueError(
                            f"Output length {len(output[t])} does not match expected "
                            f"keys length {len(keys)} for custom data."
                        )

        return sync_points

    def _validate_input_sync_point(self, sync_point: SyncPoint) -> None:
        """Validate the sync point with what the model had as input.

        Ensures that the sync point contains all required data types
        as specified in the model's input data types.

        Args:
            sync_point: SyncPoint containing data from a single time step.

        Raises:
            ValueError: If the sync point does not contain required data types.
        """
        input_data_types = self.model.model_init_description.input_data_types
        missing_data_types = []
        for data_type in input_data_types:
            if data_type == DataType.JOINT_POSITIONS and not sync_point.joint_positions:
                missing_data_types.append("joint positions")
            elif (
                data_type == DataType.JOINT_VELOCITIES
                and not sync_point.joint_velocities
            ):
                missing_data_types.append("joint velocities")
            elif data_type == DataType.JOINT_TORQUES and not sync_point.joint_torques:
                missing_data_types.append("joint torques")
            elif (
                data_type == DataType.JOINT_TARGET_POSITIONS
                and not sync_point.joint_target_positions
            ):
                missing_data_types.append("joint target positions")
            elif data_type == DataType.END_EFFECTORS and not sync_point.end_effectors:
                missing_data_types.append("end effector states")
            elif (
                data_type == DataType.END_EFFECTOR_POSES
                and not sync_point.end_effector_poses
            ):
                missing_data_types.append("end effector poses")
            elif (
                data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS
                and not sync_point.parallel_gripper_open_amounts
            ):
                missing_data_types.append("parallel gripper open amounts")
            elif data_type == DataType.POSES and not sync_point.poses:
                missing_data_types.append("poses")
            elif data_type == DataType.RGB_IMAGE and not sync_point.rgb_images:
                missing_data_types.append("RGB images")
            elif data_type == DataType.DEPTH_IMAGE and not sync_point.depth_images:
                missing_data_types.append("depth images")
            elif data_type == DataType.POINT_CLOUD and not sync_point.point_clouds:
                missing_data_types.append("point clouds")
            elif data_type == DataType.LANGUAGE and not sync_point.language_data:
                missing_data_types.append("language data")
            elif data_type == DataType.CUSTOM and not sync_point.custom_data:
                missing_data_types.append("custom data")
        if missing_data_types:
            raise ValueError(
                "SyncPoint is missing required data types: "
                f"{', '.join(missing_data_types)}"
            )

    def __call__(self, sync_point: SyncPoint) -> list[SyncPoint]:
        """Process a single sync point and run inference.

        Args:
            sync_point: SyncPoint containing data from a single time step.

        Returns:
            SyncPoint with model predictions filled in.
        """
        sync_point = sync_point.order()
        if sync_point.robot_id is None:
            active_robot = GlobalSingleton()._active_robot
            if active_robot is None:
                raise ValueError("No active robot set. Please set an active robot.")
            sync_point.robot_id = active_robot.id
        self._validate_input_sync_point(sync_point)
        batch = self._preprocess(sync_point)
        with torch.no_grad():
            batch_output: ModelPrediction = self.model(batch)
            output_mapping = self._get_output_mapping(sync_point.robot_id)
            return self._model_prediction_to_sync_points(
                batch_output, output_mapping, robot_id=sync_point.robot_id
            )
