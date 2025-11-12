"""PyTorch dataset for loading synchronized robot data with filesystem caching."""

import logging
from typing import Callable, Optional, Set, cast

import numpy as np
import torch
from neuracore_types import (
    CustomData,
    DataType,
    EndEffectorData,
    EndEffectorPoseData,
    JointData,
    ParallelGripperOpenAmountData,
    PointCloudData,
    PoseData,
    SyncPoint,
)
from PIL import Image

import neuracore as nc
from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.ml import BatchedTrainingSamples, MaskableData
from neuracore.ml.datasets.pytorch_neuracore_dataset import PytorchNeuracoreDataset
from neuracore.ml.utils.memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples
CHECK_MEMORY_INTERVAL = 100


class PytorchSynchronizedDataset(PytorchNeuracoreDataset):
    """Dataset for loading episodic robot data from GCS with filesystem caching.

    Enhanced to support all data types including depth images, point clouds,
    poses, end-effectors, and custom sensor data.
    """

    def __init__(
        self,
        synchronized_dataset: SynchronizedDataset,
        input_data_types: list[DataType],
        output_data_types: list[DataType],
        output_prediction_horizon: int,
        tokenize_text: Optional[
            Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ):
        """Initialize the dataset.

        Args:
            synchronized_dataset: The synchronized dataset to load data from.
            input_data_types: List of input data types to include in the dataset.
            output_data_types: List of output data types to include in the dataset.
            output_prediction_horizon: Number of future timesteps to predict.
            tokenize_text: Optional function to tokenize text data.
        """
        if not isinstance(synchronized_dataset, SynchronizedDataset):
            raise TypeError(
                "synchronized_dataset must be an instance of SynchronizedDataset"
            )
        super().__init__(
            input_data_types=input_data_types,
            output_data_types=output_data_types,
            output_prediction_horizon=output_prediction_horizon,
            tokenize_text=tokenize_text,
            num_recordings=len(synchronized_dataset),
        )
        self.synchronized_dataset = synchronized_dataset
        self.dataset_description = self.synchronized_dataset.dataset_description

        self._max_error_count = 100
        self._error_count = 0
        self._memory_monitor = MemoryMonitor(
            max_ram_utilization=0.8, max_gpu_utilization=1.0, gpu_id=None
        )
        self._mem_check_counter = 0
        self._num_samples = self.synchronized_dataset.num_transitions
        self._logged_in = False

    @staticmethod
    def _get_timestep(episode_length: int) -> int:
        max_start = max(0, episode_length)
        return np.random.randint(0, max_start - 1)

    def load_sample(
        self, episode_idx: int, timestep: Optional[int] = None
    ) -> BatchedTrainingSamples:
        """Load sample from cache or GCS with full data type support."""
        if not self._logged_in:
            nc.login()
            self._logged_in = True

        if self._mem_check_counter % CHECK_MEMORY_INTERVAL == 0:
            self._memory_monitor.check_memory()
            self._mem_check_counter = 0
        self._mem_check_counter += 1

        try:
            synced_recording = self.synchronized_dataset[episode_idx]
            synced_recording = cast(SynchronizedRecording, synced_recording)
            episode_length = len(synced_recording)
            if timestep is None:
                timestep = self._get_timestep(episode_length)

            sample = TrainingSample(
                output_prediction_mask=torch.ones(
                    (self.output_prediction_horizon,), dtype=torch.float32
                ),
            )
            sync_point = cast(SyncPoint, synced_recording[timestep])
            future_sync_points = cast(
                list[SyncPoint],
                synced_recording[
                    timestep + 1 : timestep + 1 + self.output_prediction_horizon
                ],
            )
            # Padding for future sync points
            for _ in range(self.output_prediction_horizon - len(future_sync_points)):
                future_sync_points.append(future_sync_points[-1])

            # Process RGB images
            if sync_point.rgb_images:
                if DataType.RGB_IMAGE in self.input_data_types:
                    rgbs_for_each_camera: list[Image.Image] = list(
                        [sp.frame for sp in sync_point.rgb_images.values()]
                    )
                    sample.inputs.rgb_images = self._create_camera_maskable_input_data(
                        rgbs_for_each_camera,
                        self.dataset_description.rgb_images.max_len,
                    )
                if DataType.RGB_IMAGE in self.output_data_types:
                    future_frames = [
                        [cam_data.frame for cam_data in sp.rgb_images.values()]
                        for sp in future_sync_points
                        if sp.rgb_images is not None
                    ]
                    sample.outputs.rgb_images = (
                        self._create_camera_maskable_output_data(
                            future_frames,
                            self.dataset_description.rgb_images.max_len,
                        )
                    )

            # Process depth images
            if sync_point.depth_images:
                if DataType.DEPTH_IMAGE in self.input_data_types:
                    depth_for_each_camera: list[Image.Image] = list(
                        [sp.frame for sp in sync_point.depth_images.values()]
                    )
                    sample.inputs.depth_images = (
                        self._create_camera_maskable_input_data(
                            depth_for_each_camera,
                            self.dataset_description.depth_images.max_len,
                        )
                    )
                if DataType.DEPTH_IMAGE in self.output_data_types:
                    future_frames = [
                        [cam_data.frame for cam_data in sp.depth_images.values()]
                        for sp in future_sync_points
                        if sp.depth_images is not None
                    ]
                    sample.outputs.depth_images = (
                        self._create_camera_maskable_output_data(
                            future_frames,
                            self.dataset_description.depth_images.max_len,
                        )
                    )

            # Process point clouds
            if sync_point.point_clouds:
                if DataType.POINT_CLOUD in self.input_data_types:
                    sample.inputs.point_clouds = (
                        self._create_point_cloud_maskable_input_data(
                            sync_point.point_clouds
                        )
                    )
                if DataType.POINT_CLOUD in self.output_data_types:
                    future_point_clouds = [
                        sp.point_clouds
                        for sp in future_sync_points
                        if sp.point_clouds is not None
                    ]
                    sample.outputs.point_clouds = (
                        self._create_point_cloud_maskable_output_data(
                            future_point_clouds
                        )
                    )

            # Process joint data
            if sync_point.joint_positions:
                if DataType.JOINT_POSITIONS in self.input_data_types:
                    sample.inputs.joint_positions = (
                        self._create_joint_maskable_input_data(
                            sync_point.joint_positions,
                            self.dataset_description.joint_positions.max_len,
                        )
                    )
                if DataType.JOINT_POSITIONS in self.output_data_types:
                    sample.outputs.joint_positions = (
                        self._create_joint_maskable_output_data(
                            [
                                sp.joint_positions
                                for sp in future_sync_points
                                if sp.joint_positions is not None
                            ],
                            self.dataset_description.joint_positions.max_len,
                        )
                    )

            if sync_point.joint_velocities:
                if DataType.JOINT_VELOCITIES in self.input_data_types:
                    sample.inputs.joint_velocities = (
                        self._create_joint_maskable_input_data(
                            sync_point.joint_velocities,
                            self.dataset_description.joint_velocities.max_len,
                        )
                    )
                if DataType.JOINT_VELOCITIES in self.output_data_types:
                    sample.outputs.joint_velocities = (
                        self._create_joint_maskable_output_data(
                            [
                                sp.joint_velocities
                                for sp in future_sync_points
                                if sp.joint_velocities is not None
                            ],
                            self.dataset_description.joint_velocities.max_len,
                        )
                    )

            if sync_point.joint_torques:
                if DataType.JOINT_TORQUES in self.input_data_types:
                    sample.inputs.joint_torques = (
                        self._create_joint_maskable_input_data(
                            sync_point.joint_torques,
                            self.dataset_description.joint_torques.max_len,
                        )
                    )
                if DataType.JOINT_TORQUES in self.output_data_types:
                    sample.outputs.joint_torques = (
                        self._create_joint_maskable_output_data(
                            [
                                sp.joint_torques
                                for sp in future_sync_points
                                if sp.joint_torques is not None
                            ],
                            self.dataset_description.joint_torques.max_len,
                        )
                    )

            if sync_point.joint_target_positions:
                if DataType.JOINT_TARGET_POSITIONS in self.input_data_types:
                    sample.inputs.joint_target_positions = (
                        self._create_joint_maskable_input_data(
                            sync_point.joint_target_positions,
                            self.dataset_description.joint_target_positions.max_len,
                        )
                    )
                if DataType.JOINT_TARGET_POSITIONS in self.output_data_types:
                    # We dont need to shift the sync_point by 1, since we are
                    # using the target joint positions as the action
                    jtp_points = [sync_point] + future_sync_points
                    jtp_points = jtp_points[: self.output_prediction_horizon]

                    sample.outputs.joint_target_positions = (
                        self._create_joint_maskable_output_data(
                            [
                                sp.joint_target_positions
                                for sp in jtp_points
                                if sp.joint_target_positions is not None
                            ],
                            self.dataset_description.joint_target_positions.max_len,
                        )
                    )

            # Process end-effector data
            if sync_point.end_effectors:
                if DataType.END_EFFECTORS in self.input_data_types:
                    sample.inputs.end_effectors = (
                        self._create_end_effector_maskable_input_data(
                            sync_point.end_effectors
                        )
                    )
                if DataType.END_EFFECTORS in self.output_data_types:
                    future_end_effectors = [
                        sp.end_effectors
                        for sp in future_sync_points
                        if sp.end_effectors is not None
                    ]
                    sample.outputs.end_effectors = (
                        self._create_end_effector_maskable_output_data(
                            future_end_effectors
                        )
                    )

            if sync_point.end_effector_poses:
                if DataType.END_EFFECTOR_POSES in self.input_data_types:
                    sample.inputs.end_effector_poses = (
                        self._create_end_effector_pose_maskable_input_data(
                            sync_point.end_effector_poses
                        )
                    )
                if DataType.END_EFFECTOR_POSES in self.output_data_types:
                    future_ee_poses = [
                        sp.end_effector_poses
                        for sp in future_sync_points
                        if sp.end_effector_poses is not None
                    ]
                    sample.outputs.end_effector_poses = (
                        self._create_end_effector_pose_maskable_output_data(
                            future_ee_poses
                        )
                    )
            if sync_point.parallel_gripper_open_amounts:
                if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in self.input_data_types:
                    sample.inputs.parallel_gripper_open_amounts = (
                        self._create_parallel_gripper_open_amounts_maskable_input_data(
                            sync_point.parallel_gripper_open_amounts
                        )
                    )
                if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in self.output_data_types:
                    future_parallel_gripper_open_amounts = [
                        sp.parallel_gripper_open_amounts
                        for sp in future_sync_points
                        if sp.parallel_gripper_open_amounts is not None
                    ]
                    sample.outputs.parallel_gripper_open_amounts = (
                        self._create_parallel_gripper_open_amounts_maskable_output_data(
                            future_parallel_gripper_open_amounts
                        )
                    )

            # Process pose data
            if sync_point.poses:
                if DataType.POSES in self.input_data_types:
                    sample.inputs.poses = self._create_pose_maskable_input_data(
                        sync_point.poses
                    )
                if DataType.POSES in self.output_data_types:
                    future_poses = [
                        sp.poses for sp in future_sync_points if sp.poses is not None
                    ]
                    sample.outputs.poses = self._create_pose_maskable_output_data(
                        future_poses
                    )

            # Process language data
            if sync_point.language_data and (
                DataType.LANGUAGE in self.input_data_types
                or DataType.LANGUAGE in self.output_data_types
            ):
                if self.tokenize_text is None:
                    raise ValueError(
                        "Failed to initialize tokenize_text for DataType.LANGUAGE"
                    )
                input_ids, attention_mask = self.tokenize_text(
                    [sync_point.language_data.text]
                )

                language_tokens = MaskableData(input_ids, attention_mask)
                if DataType.LANGUAGE in self.input_data_types:
                    sample.inputs.language_tokens = language_tokens
                if DataType.LANGUAGE in self.output_data_types:
                    sample.outputs.language_tokens = language_tokens

            # Process custom data
            if sync_point.custom_data:
                if DataType.CUSTOM in self.input_data_types:
                    sample.inputs.custom_data = self._create_custom_maskable_input_data(
                        sync_point.custom_data
                    )
                if DataType.CUSTOM in self.output_data_types:
                    future_custom_data = [
                        sp.custom_data
                        for sp in future_sync_points
                        if sp.custom_data is not None
                    ]
                    sample.outputs.custom_data = (
                        self._create_custom_maskable_output_data(future_custom_data)
                    )

            sample.output_prediction_mask = self._create_output_prediction_mask(
                episode_length,
                timestep,
                self.output_prediction_horizon,
            )

            return sample

        except Exception:
            logger.error(
                f"Error loading frame {timestep} from episode {episode_idx}.",
                exc_info=True,
            )
            raise

    def _create_joint_maskable_input_data(
        self, joint_data: JointData, max_len: int
    ) -> MaskableData:
        """Create MaskableData for joint input."""
        jdata = torch.tensor(list(joint_data.values.values()), dtype=torch.float32)
        num_existing_states = jdata.shape[0]
        extra_states = max_len - num_existing_states
        if extra_states > 0:
            jdata = torch.cat(
                [jdata, torch.zeros(extra_states, dtype=torch.float32)], dim=0
            )
        jdata_mask = torch.tensor(
            [1.0] * num_existing_states + [0.0] * extra_states, dtype=torch.float32
        )
        return MaskableData(jdata, jdata_mask)

    def _create_joint_maskable_output_data(
        self, joint_data: list[JointData], max_len: int
    ) -> MaskableData:
        """Create MaskableData for joint output."""
        maskable_data_for_each_t = [
            self._create_joint_maskable_input_data(jd, max_len) for jd in joint_data
        ]
        stacked_maskable_data = torch.stack(
            [maskable_data.data for maskable_data in maskable_data_for_each_t]
        )
        stacked_maskable_mask = torch.stack(
            [maskable_data.mask for maskable_data in maskable_data_for_each_t]
        )
        return MaskableData(stacked_maskable_data, stacked_maskable_mask)

    def _create_end_effector_maskable_input_data(
        self, end_effector_data: EndEffectorData
    ) -> MaskableData:
        """Create MaskableData for end-effector input."""
        ee_values = list(end_effector_data.open_amounts.values())
        ee_tensor = torch.tensor(ee_values, dtype=torch.float32)

        max_len = self.dataset_description.end_effector_states.max_len
        num_existing = ee_tensor.shape[0]
        extra = max_len - num_existing

        if extra > 0:
            ee_tensor = torch.cat(
                [ee_tensor, torch.zeros(extra, dtype=torch.float32)], dim=0
            )

        ee_mask = torch.tensor(
            [1.0] * num_existing + [0.0] * extra, dtype=torch.float32
        )
        return MaskableData(ee_tensor, ee_mask)

    def _create_end_effector_maskable_output_data(
        self, end_effector_data: list[EndEffectorData]
    ) -> MaskableData:
        """Create MaskableData for end-effector output."""
        maskable_data_for_each_t = [
            self._create_end_effector_maskable_input_data(eed)
            for eed in end_effector_data
        ]
        stacked_data = torch.stack(
            [maskable_data.data for maskable_data in maskable_data_for_each_t]
        )
        stacked_mask = torch.stack(
            [maskable_data.mask for maskable_data in maskable_data_for_each_t]
        )
        return MaskableData(stacked_data, stacked_mask)

    def _create_end_effector_pose_maskable_output_data(
        self, end_effector_data: list[EndEffectorPoseData]
    ) -> MaskableData:
        """Create MaskableData for end-effector pose output."""
        maskable_data_for_each_t = [
            self._create_end_effector_pose_maskable_input_data(eed)
            for eed in end_effector_data
        ]
        stacked_data = torch.stack(
            [maskable_data.data for maskable_data in maskable_data_for_each_t]
        )
        stacked_mask = torch.stack(
            [maskable_data.mask for maskable_data in maskable_data_for_each_t]
        )
        return MaskableData(stacked_data, stacked_mask)

    def _create_end_effector_pose_maskable_input_data(
        self, end_effector_pose_data: EndEffectorPoseData
    ) -> MaskableData:
        """Create MaskableData for end-effector pose input."""
        ee_poses = []
        for ee_name, ee_pose in end_effector_pose_data.poses.items():
            ee_poses.extend(ee_pose)  # 6DOF pose

        ee_tensor = torch.tensor(ee_poses, dtype=torch.float32)
        max_len = self.dataset_description.end_effector_poses.max_len
        num_existing = ee_tensor.shape[0]
        extra = max_len - num_existing

        if extra > 0:
            ee_tensor = torch.cat(
                [ee_tensor, torch.zeros(extra, dtype=torch.float32)], dim=0
            )

        ee_mask = torch.tensor(
            [1.0] * num_existing + [0.0] * extra, dtype=torch.float32
        )
        return MaskableData(ee_tensor, ee_mask)

    def _create_parallel_gripper_open_amounts_maskable_input_data(
        self, parallel_gripper_open_amounts: ParallelGripperOpenAmountData
    ) -> MaskableData:
        """Create MaskableData for parallel gripper open amounts input."""
        parallel_gripper_open_amounts_tensor = torch.tensor(
            list(parallel_gripper_open_amounts.values()), dtype=torch.float32
        )
        max_len = self.dataset_description.parallel_gripper_open_amounts.max_len
        num_existing = parallel_gripper_open_amounts_tensor.shape[0]
        extra = max_len - num_existing
        if extra > 0:
            parallel_gripper_open_amounts_tensor = torch.cat(
                [
                    parallel_gripper_open_amounts_tensor,
                    torch.zeros(extra, dtype=torch.float32),
                ],
                dim=0,
            )
        parallel_gripper_open_amounts_mask = torch.tensor(
            [1.0] * num_existing + [0.0] * extra, dtype=torch.float32
        )
        return MaskableData(
            parallel_gripper_open_amounts_tensor, parallel_gripper_open_amounts_mask
        )

    def _create_parallel_gripper_open_amounts_maskable_output_data(
        self, parallel_gripper_open_amounts_list: list[ParallelGripperOpenAmountData]
    ) -> MaskableData:
        """Create MaskableData for parallel gripper open amounts output."""
        maskable_data_for_each_t = [
            self._create_parallel_gripper_open_amounts_maskable_input_data(pgoa)
            for pgoa in parallel_gripper_open_amounts_list
        ]

        stacked_data = torch.stack(
            [maskable_data.data for maskable_data in maskable_data_for_each_t]
        )
        stacked_mask = torch.stack(
            [maskable_data.mask for maskable_data in maskable_data_for_each_t]
        )
        return MaskableData(stacked_data, stacked_mask)

    def _create_pose_maskable_input_data(self, poses: PoseData) -> MaskableData:
        """Create MaskableData for pose input."""
        all_poses = []
        for pose_name, pose_data in poses.pose.items():
            all_poses.extend(pose_data)  # 6DOF pose

        pose_tensor = torch.tensor(all_poses, dtype=torch.float32)
        max_len = self.dataset_description.poses.max_len
        num_existing = pose_tensor.shape[0]
        extra = max_len - num_existing

        if extra > 0:
            pose_tensor = torch.cat(
                [pose_tensor, torch.zeros(extra, dtype=torch.float32)], dim=0
            )

        pose_mask = torch.tensor(
            [1.0] * num_existing + [0.0] * extra, dtype=torch.float32
        )
        return MaskableData(pose_tensor, pose_mask)

    def _create_pose_maskable_output_data(
        self, poses_list: list[PoseData]
    ) -> MaskableData:
        """Create MaskableData for pose output."""
        maskable_data_for_each_t = [
            self._create_pose_maskable_input_data(poses) for poses in poses_list
        ]
        stacked_data = torch.stack(
            [maskable_data.data for maskable_data in maskable_data_for_each_t]
        )
        stacked_mask = torch.stack(
            [maskable_data.mask for maskable_data in maskable_data_for_each_t]
        )
        return MaskableData(stacked_data, stacked_mask)

    def _create_point_cloud_maskable_input_data(
        self, point_clouds: dict[str, PointCloudData]
    ) -> MaskableData:
        """Create MaskableData for point cloud input."""
        # Stack point clouds from all sensors
        all_clouds = []
        for pc_name, pc_data in point_clouds.items():
            # Convert points to tensor: [num_points, 3]
            points = torch.tensor(pc_data.points, dtype=torch.float32)
            all_clouds.append(points)

        # For now, we'll use the first point cloud and pad to standard size
        if all_clouds:
            points = all_clouds[0]  # [num_points, 3]
            target_num_points = 1024  # Standard size
            current_num_points = points.shape[0]

            if current_num_points < target_num_points:
                # Pad with zeros
                padding = torch.zeros(target_num_points - current_num_points, 3)
                points = torch.cat([points, padding], dim=0)
            elif current_num_points > target_num_points:
                # Subsample
                indices = torch.randperm(current_num_points)[:target_num_points]
                points = points[indices]

            # Create mask for valid points
            mask = torch.tensor(
                [1.0] * min(current_num_points, target_num_points)
                + [0.0] * max(0, target_num_points - current_num_points)
            )

            # Reshape for batching: [1, num_points, 3] for single point cloud
            points = points.unsqueeze(0)
            mask = mask.unsqueeze(0)  # [1, num_points]
        else:
            # Empty point cloud
            points = torch.zeros(1, 1024, 3)
            mask = torch.zeros(1, 1024)

        return MaskableData(points, mask)

    def _create_point_cloud_maskable_output_data(
        self, point_clouds_list: list[dict[str, PointCloudData]]
    ) -> MaskableData:
        """Create MaskableData for point cloud output."""
        maskable_data_for_each_t = [
            self._create_point_cloud_maskable_input_data(pcs)
            for pcs in point_clouds_list
        ]
        stacked_data = torch.stack(
            [maskable_data.data for maskable_data in maskable_data_for_each_t]
        )
        stacked_mask = torch.stack(
            [maskable_data.mask for maskable_data in maskable_data_for_each_t]
        )
        return MaskableData(stacked_data, stacked_mask)

    def _create_custom_maskable_input_data(
        self, custom_data: dict[str, CustomData]
    ) -> dict[str, MaskableData]:
        """Create MaskableData for custom input data."""
        result = {}
        for key, data in custom_data.items():
            # Convert custom data to tensor
            if isinstance(data.data, (list, np.ndarray)):
                tensor_data = torch.tensor(data.data, dtype=torch.float32)
                if tensor_data.dim() == 0:  # Scalar
                    tensor_data = tensor_data.unsqueeze(0)

                # Create simple mask (all valid)
                mask = torch.ones(tensor_data.shape[0], dtype=torch.float32)
                result[key] = MaskableData(tensor_data, mask)
            else:
                # For other data types, create a simple representation
                tensor_data = torch.tensor(
                    [float(hash(str(data.data)) % 1000)], dtype=torch.float32
                )
                mask = torch.ones(1, dtype=torch.float32)
                result[key] = MaskableData(tensor_data, mask)

        return result

    def _create_custom_maskable_output_data(
        self, custom_data_list: list[dict[str, CustomData]]
    ) -> dict[str, MaskableData]:
        """Create MaskableData for custom output data."""
        result = {}

        # Get all keys from all timesteps
        all_keys: Set[str] = set()
        for custom_dict in custom_data_list:
            all_keys.update(custom_dict.keys())

        for key in all_keys:
            maskable_data_for_each_t = []
            for custom_dict in custom_data_list:
                if key in custom_dict:
                    single_data = self._create_custom_maskable_input_data(
                        {key: custom_dict[key]}
                    )
                    maskable_data_for_each_t.append(single_data[key])
                else:
                    # Create dummy data for missing timesteps
                    tensor_data = torch.zeros(1, dtype=torch.float32)
                    mask = torch.zeros(1, dtype=torch.float32)
                    maskable_data_for_each_t.append(MaskableData(tensor_data, mask))

            stacked_data = torch.stack(
                [maskable_data.data for maskable_data in maskable_data_for_each_t]
            )
            stacked_mask = torch.stack(
                [maskable_data.mask for maskable_data in maskable_data_for_each_t]
            )
            result[key] = MaskableData(stacked_data, stacked_mask)

        return result

    def _create_output_prediction_mask(
        self, episode_length: int, timestep: int, output_prediction_horizon: int
    ) -> torch.FloatTensor:
        """Create mask for output predictions."""
        output_prediction_mask = torch.zeros(
            output_prediction_horizon, dtype=torch.float32
        )
        for i in range(output_prediction_horizon):
            if timestep + i >= episode_length:
                break
            else:
                output_prediction_mask[i] = 1.0
        return output_prediction_mask

    def _create_camera_maskable_input_data(
        self, camera_data: list[Image.Image], max_cameras: int
    ) -> MaskableData:
        """Create MaskableData for camera input.

        Returns:
            MaskableData containing camera images of shape [num_cameras, H, W, C]
            and a mask indicating which cameras are present.
        """
        cam_image_tensors = torch.stack(
            [self.camera_transform(cam_data) for cam_data in camera_data]
        )
        num_cameras = cam_image_tensors.shape[0]
        extra_cameras = max_cameras - num_cameras
        if extra_cameras > 0:
            empty_image = torch.zeros_like(cam_image_tensors[0])
            cam_image_tensors = torch.cat(
                [cam_image_tensors, empty_image.repeat(extra_cameras, 1, 1, 1)],
                dim=0,
            )
        camera_images_mask = torch.tensor(
            [1.0] * num_cameras + [0.0] * extra_cameras,
            dtype=torch.float32,
        )
        return MaskableData(cam_image_tensors, camera_images_mask)

    def _create_camera_maskable_output_data(
        self, temporal_camera_data: list[list[Image.Image]], max_cameras: int
    ) -> MaskableData:
        """Create maskable data for multiple cameras.

        Args:
            temporal_camera_data: A list of lists of shape [T, CAMS, ...].

        Returns:
            MaskableData: A MaskableData object containing the stacked camera images
                and their masks of shape [T, CAMS, C, H, W].
        """
        maskable_data_for_each_t = [
            self._create_camera_maskable_input_data(camera_data, max_cameras)
            for camera_data in temporal_camera_data
        ]
        stacked_maskable_data = torch.stack(
            [maskable_data.data for maskable_data in maskable_data_for_each_t]
        )
        stacked_maskable_mask = torch.stack(
            [maskable_data.mask for maskable_data in maskable_data_for_each_t]
        )
        return MaskableData(stacked_maskable_data, stacked_maskable_mask)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self._num_samples
