"""Dummy dataset for algorithm validation and testing without real data.

This module provides a synthetic dataset that generates random data matching
the structure of real Neuracore datasets. It's used for algorithm development,
testing, and validation without requiring actual robot demonstration data.
"""

import logging
from typing import Callable, Optional

import numpy as np
import torch
from neuracore_types import DataItemStats, DatasetDescription, DataType

from neuracore.core.robot import Robot
from neuracore.ml import BatchedTrainingSamples, MaskableData
from neuracore.ml.datasets.pytorch_neuracore_dataset import PytorchNeuracoreDataset

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples


class PytorchDummyDataset(PytorchNeuracoreDataset):
    """Synthetic dataset for algorithm validation and testing.

    This dataset generates random data with the same structure and dimensions
    as real Neuracore datasets, allowing for algorithm development and testing
    without requiring actual robot demonstration data. It supports all standard
    data types including images, joint data, depth images, point clouds,
    poses, end-effectors, and language instructions.
    """

    def __init__(
        self,
        input_data_types: list[DataType],
        output_data_types: list[DataType],
        num_samples: int = 50,
        num_episodes: int = 10,
        output_prediction_horizon: int = 5,
        tokenize_text: Optional[
            Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ):
        """Initialize the dummy dataset with specified data types and dimensions.

        Args:
            input_data_types: List of data types to include as model inputs.
            output_data_types: List of data types to include as model outputs.
            num_samples: Total number of training samples to generate.
            num_episodes: Number of distinct episodes in the dataset.
            output_prediction_horizon: Length of output action sequences.
            tokenize_text: Function to convert text strings to token tensors.
                Should return (input_ids, attention_mask) tuple.
        """
        super().__init__(
            num_recordings=num_episodes,
            input_data_types=input_data_types,
            output_data_types=output_data_types,
            output_prediction_horizon=output_prediction_horizon,
            tokenize_text=tokenize_text,
        )
        self.num_samples = num_samples
        self.robot = Robot("dummy_robot", 0)
        self.robot.id = "dummy_robot_id"

        self.image_size = (224, 224)

        # Sample instructions for dummy data
        self.sample_instructions = [
            "Pick up the red block",
            "Move the cup to the left",
            "Open the drawer",
            "Place the object on the table",
            "Push the button",
            "Grasp the handle",
            "Move the arm up",
            "Turn the knob clockwise",
            "Close the gripper",
            "Slide the object forward",
        ]

        self.dataset_description = DatasetDescription()

        # Joint data
        if DataType.JOINT_POSITIONS in self.data_types:
            self.dataset_description.joint_positions = DataItemStats(
                mean=np.zeros(6),
                std=np.ones(6),
                min=-np.ones(6),
                max=np.ones(6),
                max_len=6,
                robot_to_ncdata_keys={self.robot.id: [f"jps_{i}" for i in range(6)]},
            )
        if DataType.JOINT_VELOCITIES in self.data_types:
            self.dataset_description.joint_velocities = DataItemStats(
                mean=np.zeros(6),
                std=np.ones(6),
                min=-np.ones(6),
                max=np.ones(6),
                max_len=6,
                robot_to_ncdata_keys={self.robot.id: [f"jvs_{i}" for i in range(6)]},
            )
        if DataType.JOINT_TORQUES in self.data_types:
            self.dataset_description.joint_torques = DataItemStats(
                mean=np.zeros(6),
                std=np.ones(6),
                min=-np.ones(6),
                max=np.ones(6),
                max_len=6,
                robot_to_ncdata_keys={self.robot.id: [f"jts_{i}" for i in range(6)]},
            )
        if DataType.JOINT_TARGET_POSITIONS in self.data_types:
            self.dataset_description.joint_target_positions = DataItemStats(
                mean=np.zeros(7),
                std=np.ones(7),
                min=-np.ones(7),
                max=np.ones(7),
                max_len=7,
                robot_to_ncdata_keys={self.robot.id: [f"jtps_{i}" for i in range(7)]},
            )

        # End-effector data
        # TODO: Remove later.
        if DataType.END_EFFECTORS in self.data_types:
            self.dataset_description.end_effector_states = DataItemStats(
                mean=np.zeros(2),
                std=np.ones(2),
                max_len=2,  # e.g., gripper open amounts
                robot_to_ncdata_keys={
                    self.robot.id: [f"end_effector_{i}" for i in range(2)]
                },
            )

        # End-effector pose data
        if DataType.END_EFFECTOR_POSES in self.data_types:
            self.dataset_description.end_effector_poses = DataItemStats(
                mean=np.zeros(7),
                std=np.ones(7),
                max_len=7,  # 1 gripper x 7DOF pose
                robot_to_ncdata_keys={
                    self.robot.id: [f"end_effector_pose_{i}" for i in range(7)]
                },
            )
        # Parallel gripper open amounts
        if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in self.data_types:
            self.dataset_description.parallel_gripper_open_amounts = DataItemStats(
                mean=np.zeros(1),
                std=np.ones(1),
                max_len=1,  # 1 parallel gripper x 1 open amount float value
                robot_to_ncdata_keys={self.robot.id: ["parallel_gripper_open_amount"]},
            )

        # Pose data
        # TODO: Remove or change later.
        if DataType.POSES in self.data_types:
            self.dataset_description.poses = DataItemStats(
                mean=np.zeros(12),
                std=np.ones(12),
                max_len=12,  # 2 poses x 6DOF each
                robot_to_ncdata_keys={self.robot.id: [f"pose_{i}" for i in range(12)]},
            )

        # Visual data
        if DataType.RGB_IMAGE in self.data_types:
            self.dataset_description.rgb_images = DataItemStats(
                max_len=2,  # e.g., two RGB images per sample
                robot_to_ncdata_keys={self.robot.id: [f"rgb_{i}" for i in range(2)]},
            )
        if DataType.DEPTH_IMAGE in self.data_types:
            self.dataset_description.depth_images = DataItemStats(
                max_len=2,  # e.g., two depth images per sample
                robot_to_ncdata_keys={self.robot.id: [f"depth_{i}" for i in range(2)]},
            )
        if DataType.POINT_CLOUD in self.data_types:
            self.dataset_description.point_clouds = DataItemStats(
                max_len=1,  # e.g., one point cloud per sample
                robot_to_ncdata_keys={
                    self.robot.id: [f"point_cloud_{i}" for i in range(1)]
                },
            )

        # Language data
        if DataType.LANGUAGE in self.data_types:
            self.dataset_description.language = DataItemStats(
                max_len=max(
                    len(instruction) for instruction in self.sample_instructions
                )
            )

        # Custom data
        if DataType.CUSTOM in self.data_types:
            self.dataset_description.custom_data = {
                "sensor_1": DataItemStats(
                    mean=np.zeros(10),
                    std=np.ones(10),
                    min=-np.ones(10),
                    max=np.ones(10),
                    max_len=10,
                    robot_to_ncdata_keys={self.robot.id: ["sensor_1"]},
                ),
                "sensor_2": DataItemStats(
                    mean=np.zeros(5),
                    std=np.ones(5),
                    min=-np.ones(5),
                    max=np.ones(5),
                    max_len=5,
                    robot_to_ncdata_keys={self.robot.id: ["sensor_2"]},
                ),
            }

        self._error_count = 0
        self._max_error_count = 1

    def load_sample(
        self, episode_idx: int, timestep: Optional[int] = None
    ) -> TrainingSample:
        """Generate a random training sample with realistic data structure.

        Creates synthetic data that matches the format and dimensions of real
        robot demonstration data, including appropriate masking and tensor shapes.

        Args:
            episode_idx: Index of the episode (used for reproducible randomness).
            timestep: Optional timestep within the episode (currently unused).

        Returns:
            A TrainingSample containing randomly generated input and output data
            matching the specified data types and dimensions.

        Raises:
            Exception: If there's an error generating the sample data.
        """
        try:
            sample = TrainingSample(
                output_prediction_mask=torch.ones(
                    (self.output_prediction_horizon,), dtype=torch.float32
                ),
            )

            # Visual data
            if DataType.RGB_IMAGE in self.data_types:
                max_rgb_len = self.dataset_description.rgb_images.max_len
                rgb_images = MaskableData(
                    torch.zeros(
                        (max_rgb_len, 3, *self.image_size), dtype=torch.float32
                    ),
                    torch.ones((max_rgb_len,), dtype=torch.float32),
                )
                if DataType.RGB_IMAGE in self.input_data_types:
                    sample.inputs.rgb_images = rgb_images
                if DataType.RGB_IMAGE in self.output_data_types:
                    sample.outputs.rgb_images = rgb_images

            if DataType.DEPTH_IMAGE in self.data_types:
                max_depth_len = self.dataset_description.depth_images.max_len
                depth_images = MaskableData(
                    torch.zeros(
                        (max_depth_len, 1, *self.image_size), dtype=torch.float32
                    ),
                    torch.ones((max_depth_len,), dtype=torch.float32),
                )
                if DataType.DEPTH_IMAGE in self.input_data_types:
                    sample.inputs.depth_images = depth_images
                if DataType.DEPTH_IMAGE in self.output_data_types:
                    sample.outputs.depth_images = depth_images

            if DataType.POINT_CLOUD in self.data_types:
                max_pc_len = self.dataset_description.point_clouds.max_len
                # Point clouds: [num_clouds, num_points, 3 (x,y,z)]
                num_points = 1024  # Standard point cloud size
                point_clouds = MaskableData(
                    torch.randn((max_pc_len, num_points, 3), dtype=torch.float32),
                    torch.ones((max_pc_len,), dtype=torch.float32),
                )
                if DataType.POINT_CLOUD in self.input_data_types:
                    sample.inputs.point_clouds = point_clouds
                if DataType.POINT_CLOUD in self.output_data_types:
                    sample.outputs.point_clouds = point_clouds

            # Joint data
            if DataType.JOINT_POSITIONS in self.data_types:
                max_jp_len = self.dataset_description.joint_positions.max_len
                joint_positions = MaskableData(
                    torch.zeros((max_jp_len,), dtype=torch.float32),
                    torch.ones((max_jp_len,), dtype=torch.float32),
                )
                if DataType.JOINT_POSITIONS in self.input_data_types:
                    sample.inputs.joint_positions = joint_positions
                if DataType.JOINT_POSITIONS in self.output_data_types:
                    sample.outputs.joint_positions = joint_positions

            if DataType.JOINT_VELOCITIES in self.data_types:
                max_jv_len = self.dataset_description.joint_velocities.max_len
                joint_velocities = MaskableData(
                    torch.zeros((max_jv_len,), dtype=torch.float32),
                    torch.ones((max_jv_len,), dtype=torch.float32),
                )
                if DataType.JOINT_VELOCITIES in self.input_data_types:
                    sample.inputs.joint_velocities = joint_velocities
                if DataType.JOINT_VELOCITIES in self.output_data_types:
                    sample.outputs.joint_velocities = joint_velocities

            if DataType.JOINT_TORQUES in self.data_types:
                max_jt_len = self.dataset_description.joint_torques.max_len
                joint_torques = MaskableData(
                    torch.zeros((max_jt_len,), dtype=torch.float32),
                    torch.ones((max_jt_len,), dtype=torch.float32),
                )
                if DataType.JOINT_TORQUES in self.input_data_types:
                    sample.inputs.joint_torques = joint_torques
                if DataType.JOINT_TORQUES in self.output_data_types:
                    sample.outputs.joint_torques = joint_torques

            if DataType.JOINT_TARGET_POSITIONS in self.data_types:
                max_jtp_len = self.dataset_description.joint_target_positions.max_len
                joint_target_positions = MaskableData(
                    torch.zeros((max_jtp_len,), dtype=torch.float32),
                    torch.ones((max_jtp_len,), dtype=torch.float32),
                )
                if DataType.JOINT_TARGET_POSITIONS in self.input_data_types:
                    sample.inputs.joint_target_positions = joint_target_positions
                if DataType.JOINT_TARGET_POSITIONS in self.output_data_types:
                    sample.outputs.joint_target_positions = joint_target_positions

            # End-effector data
            if DataType.END_EFFECTORS in self.data_types:
                max_ee_len = self.dataset_description.end_effector_states.max_len
                end_effectors = MaskableData(
                    torch.zeros((max_ee_len,), dtype=torch.float32),
                    torch.ones((max_ee_len,), dtype=torch.float32),
                )
                if DataType.END_EFFECTORS in self.input_data_types:
                    sample.inputs.end_effectors = end_effectors
                if DataType.END_EFFECTORS in self.output_data_types:
                    sample.outputs.end_effectors = end_effectors

            # End-effector pose data
            if DataType.END_EFFECTOR_POSES in self.data_types:
                max_ee_pose_len = self.dataset_description.end_effector_poses.max_len
                end_effector_poses = MaskableData(
                    torch.zeros((max_ee_pose_len,), dtype=torch.float32),
                    torch.ones((max_ee_pose_len,), dtype=torch.float32),
                )
                if DataType.END_EFFECTOR_POSES in self.input_data_types:
                    sample.inputs.end_effector_poses = end_effector_poses
                if DataType.END_EFFECTOR_POSES in self.output_data_types:
                    sample.outputs.end_effector_poses = end_effector_poses

            # Parallel gripper open amounts
            if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in self.data_types:
                max_pg_len = (
                    self.dataset_description.parallel_gripper_open_amounts.max_len
                )
                parallel_gripper_open_amounts = MaskableData(
                    torch.zeros((max_pg_len,), dtype=torch.float32),
                    torch.ones((max_pg_len,), dtype=torch.float32),
                )
                if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in self.input_data_types:
                    sample.inputs.parallel_gripper_open_amounts = (
                        parallel_gripper_open_amounts
                    )
                if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in self.output_data_types:
                    sample.outputs.parallel_gripper_open_amounts = (
                        parallel_gripper_open_amounts
                    )

            # Pose data
            if DataType.POSES in self.data_types:
                max_pose_len = self.dataset_description.poses.max_len
                poses = MaskableData(
                    torch.zeros((max_pose_len,), dtype=torch.float32),
                    torch.ones((max_pose_len,), dtype=torch.float32),
                )
                if DataType.POSES in self.input_data_types:
                    sample.inputs.poses = poses
                if DataType.POSES in self.output_data_types:
                    sample.outputs.poses = poses

            # Language data
            if DataType.LANGUAGE in self.data_types:
                if self.tokenize_text is None:
                    raise ValueError(
                        "Failed to initialize tokenize_text for DataType.LANGUAGE"
                    )

                # Randomly select an instruction
                instruction = np.random.choice(self.sample_instructions)
                # Tokenize the instruction
                input_ids, attention_mask = self.tokenize_text([instruction])
                language_tokens = MaskableData(input_ids, attention_mask)

                if DataType.LANGUAGE in self.input_data_types:
                    sample.inputs.language_tokens = language_tokens
                if DataType.LANGUAGE in self.output_data_types:
                    sample.outputs.language_tokens = language_tokens

            # Custom data
            if DataType.CUSTOM in self.data_types:
                # Generate some example custom data
                custom_data = {
                    "sensor_1": MaskableData(
                        torch.randn((10,), dtype=torch.float32),
                        torch.ones((10,), dtype=torch.float32),
                    ),
                    "sensor_2": MaskableData(
                        torch.randn((5,), dtype=torch.float32),
                        torch.ones((5,), dtype=torch.float32),
                    ),
                }
                if DataType.CUSTOM in self.input_data_types:
                    sample.inputs.custom_data = custom_data
                if DataType.CUSTOM in self.output_data_types:
                    sample.outputs.custom_data = custom_data

            return sample

        except Exception:
            logger.error("Error generating random sample", exc_info=True)
            raise

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            The number of training samples available in this dataset.
        """
        return self.num_samples

    def collate_fn(self, samples: list[TrainingSample]) -> BatchedTrainingSamples:
        """Collate individual samples into a batched training sample.

        Combines multiple training samples into a single batch with proper
        tensor stacking and masking. Handles the expansion of output data
        across the prediction horizon for sequence generation tasks.

        Args:
            samples: List of individual TrainingSample instances to batch together.

        Returns:
            A BatchedTrainingSamples instance containing the batched inputs,
            outputs, and prediction masks ready for model training.
        """
        bd = self._collate_fn([s.outputs for s in samples], self.output_data_types)
        for key in bd.__dict__.keys():
            if bd.__dict__[key] is not None:
                if isinstance(bd.__dict__[key], MaskableData):
                    # Skip language tokens for expansion
                    if key == "language_tokens":
                        continue
                    data = bd.__dict__[key].data.unsqueeze(1)
                    data = data.expand(
                        -1, self.output_prediction_horizon, *data.shape[2:]
                    )
                    mask = bd.__dict__[key].mask.unsqueeze(1)
                    mask = mask.expand(
                        -1, self.output_prediction_horizon, *mask.shape[2:]
                    )
                    bd.__dict__[key].data = data
                    bd.__dict__[key].mask = mask
                elif isinstance(bd.__dict__[key], dict):
                    # Handle custom_data dictionary
                    for custom_key, custom_value in bd.__dict__[key].items():
                        if isinstance(custom_value, MaskableData):
                            data = custom_value.data.unsqueeze(1)
                            data = data.expand(
                                -1, self.output_prediction_horizon, *data.shape[2:]
                            )
                            mask = custom_value.mask.unsqueeze(1)
                            mask = mask.expand(
                                -1, self.output_prediction_horizon, *mask.shape[2:]
                            )
                            bd.__dict__[key][custom_key] = MaskableData(data, mask)
        return BatchedTrainingSamples(
            inputs=self._collate_fn([s.inputs for s in samples], self.input_data_types),
            outputs=bd,
            output_prediction_mask=torch.stack(
                [sample.output_prediction_mask for sample in samples]
            ),
        )
