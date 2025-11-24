"""Abstract base class for Neuracore datasets with multi-modal data support.

This module provides the foundation for creating datasets that handle robot
demonstration data including images, joint states, depth images, point clouds,
poses, end-effectors, and language instructions.
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, Set, cast

import torch
import torchvision.transforms as T
from neuracore_types import DataType
from torch.utils.data import Dataset

from neuracore.ml import BatchedTrainingSamples, MaskableData
from neuracore.ml.core.ml_types import BatchedData

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples


class PytorchNeuracoreDataset(Dataset, ABC):
    """Abstract base class for Neuracore multi-modal robot datasets.

    This class provides a standardized interface for datasets containing robot
    demonstration data. It handles data type validation, preprocessing setup,
    batch collation, and error management for training machine learning models
    on robot data including images, joint states, depth images, point clouds,
    poses, end-effectors, and language instructions.
    """

    def __init__(
        self,
        num_recordings: int,
        input_data_types: list[DataType],
        output_data_types: list[DataType],
        output_prediction_horizon: int = 5,
        tokenize_text: Optional[
            Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ):
        """Initialize the dataset with data type specifications and preprocessing.

        Args:
            input_data_types: List of data types to include as model inputs
                (e.g., RGB images, joint positions).
            output_data_types: List of data types to include as model outputs
                (e.g., joint target positions, actions).
            output_prediction_horizon: Number of future timesteps to predict
                for sequential output tasks.
            tokenize_text: Function to convert text strings to tokenized tensors.
                Required if DataType.LANGUAGE is in the data types. Should return
                (input_ids, attention_mask) tuple.

        Raises:
            ValueError: If language data is requested but no tokenizer is provided.
        """
        if len(input_data_types) == 0 and len(output_data_types) == 0:
            raise ValueError(
                "Must supply both input and output data types for the dataset"
            )
        self.num_recordings = num_recordings
        self.input_data_types = input_data_types
        self.output_data_types = output_data_types
        self.output_prediction_horizon = output_prediction_horizon

        self.data_types = set(input_data_types + output_data_types)

        # Setup camera transform to match EpisodicDataset
        self.camera_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

        # Create tokenizer if language data is used
        self.tokenize_text = tokenize_text
        self._error_count = 0
        self._max_error_count = 1

    @abstractmethod
    def load_sample(
        self, episode_idx: int, timestep: Optional[int] = None
    ) -> TrainingSample:
        """Load a single training sample from the dataset.

        This method must be implemented by concrete subclasses to define how
        individual samples are loaded and formatted.

        Args:
            episode_idx: Index of the episode to load data from.
            timestep: Optional specific timestep within the episode.
                If None, may load entire episode or use class-specific logic.

        Returns:
            A TrainingSample containing input and output data formatted
            for model training.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            The number of training samples available.
        """
        pass

    def __getitem__(self, idx: int) -> TrainingSample:
        """Get a training sample by index with error handling.

        Implements the PyTorch Dataset interface with robust error handling
        to manage data loading failures gracefully during training.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A TrainingSample containing the requested data.

        Raises:
            Exception: If sample loading fails after exhausting retry attempts.
        """
        if idx < 0:
            # Handle negative indices by wrapping around
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self)}"
            )
        while self._error_count < self._max_error_count:
            try:
                episode_idx = idx % self.num_recordings
                return self.load_sample(episode_idx)
            except Exception:
                self._error_count += 1
                logger.error(f"Error loading item {idx}.", exc_info=True)
                if self._error_count >= self._max_error_count:
                    raise
        raise Exception(
            f"Maximum error count ({self._max_error_count}) already reached"
        )

    def _collate_fn(
        self, samples: list[BatchedData], data_types: list[DataType]
    ) -> BatchedData:
        """Collate individual data samples into a batched format.

        Combines multiple samples into batched tensors with appropriate stacking
        for different data modalities. Handles masking for variable-length data.

        Args:
            samples: List of BatchedData objects to combine.
            data_types: List of data types to include in the batch.

        Returns:
            A single BatchedData object containing the stacked samples.
        """
        bd = BatchedData()

        # Joint state data
        if DataType.JOINT_POSITIONS in data_types:
            if any(s.joint_positions is None for s in samples):
                raise ValueError(
                    "All samples must have joint_positions when "
                    "JOINT_POSITIONS data type is requested"
                )
            bd.joint_positions = MaskableData(
                torch.stack(
                    [cast(MaskableData, s.joint_positions).data for s in samples]
                ),
                torch.stack(
                    [cast(MaskableData, s.joint_positions).mask for s in samples]
                ),
            )

        if DataType.JOINT_VELOCITIES in data_types:
            if any(s.joint_velocities is None for s in samples):
                raise ValueError(
                    "All samples must have joint_velocities when "
                    "JOINT_VELOCITIES data type is requested"
                )
            bd.joint_velocities = MaskableData(
                torch.stack(
                    [cast(MaskableData, s.joint_velocities).data for s in samples]
                ),
                torch.stack(
                    [cast(MaskableData, s.joint_velocities).mask for s in samples]
                ),
            )

        if DataType.JOINT_TORQUES in data_types:
            if any(s.joint_torques is None for s in samples):
                raise ValueError(
                    "All samples must have joint_torques when "
                    "JOINT_TORQUES data type is requested"
                )
            bd.joint_torques = MaskableData(
                torch.stack(
                    [cast(MaskableData, s.joint_torques).data for s in samples]
                ),
                torch.stack(
                    [cast(MaskableData, s.joint_torques).mask for s in samples]
                ),
            )

        if DataType.JOINT_TARGET_POSITIONS in data_types:
            if any(s.joint_target_positions is None for s in samples):
                raise ValueError(
                    "All samples must have joint_target_positions when "
                    "JOINT_TARGET_POSITIONS data type is requested"
                )
            bd.joint_target_positions = MaskableData(
                torch.stack(
                    [cast(MaskableData, s.joint_target_positions).data for s in samples]
                ),
                torch.stack(
                    [cast(MaskableData, s.joint_target_positions).mask for s in samples]
                ),
            )

        # End-effector data
        if DataType.END_EFFECTORS in data_types:
            if any(s.end_effectors is None for s in samples):
                raise ValueError(
                    "All samples must have end_effectors when "
                    "END_EFFECTORS data type is requested"
                )
            bd.end_effectors = MaskableData(
                torch.stack(
                    [cast(MaskableData, s.end_effectors).data for s in samples]
                ),
                torch.stack(
                    [cast(MaskableData, s.end_effectors).mask for s in samples]
                ),
            )

        # End Effector Poses
        if DataType.END_EFFECTOR_POSES in data_types:
            if any(s.end_effector_poses is None for s in samples):
                raise ValueError(
                    "All samples must have end_effector_poses when "
                    "END_EFFECTOR_POSES data type is requested"
                )
            bd.end_effector_poses = MaskableData(
                torch.stack(
                    [cast(MaskableData, s.end_effector_poses).data for s in samples]
                ),
                torch.stack(
                    [cast(MaskableData, s.end_effector_poses).mask for s in samples]
                ),
            )

        # Parallel Gripper Open Amount Data
        if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in data_types:
            if any(s.parallel_gripper_open_amounts is None for s in samples):
                raise ValueError(
                    "All samples must have parallel_gripper_open_amounts when "
                    "GRIPPER_OPEN_AMOUNTS data type is requested"
                )
            bd.parallel_gripper_open_amounts = MaskableData(
                torch.stack([
                    cast(MaskableData, s.parallel_gripper_open_amounts).data
                    for s in samples
                ]),
                torch.stack([
                    cast(MaskableData, s.parallel_gripper_open_amounts).mask
                    for s in samples
                ]),
            )

        # Pose data
        if DataType.POSES in data_types:
            if any(s.poses is None for s in samples):
                raise ValueError(
                    "All samples must have poses when " "POSES data type is requested"
                )
            bd.poses = MaskableData(
                torch.stack([cast(MaskableData, s.poses).data for s in samples]),
                torch.stack([cast(MaskableData, s.poses).mask for s in samples]),
            )

        # Visual data
        if DataType.RGB_IMAGE in data_types:
            if any(s.rgb_images is None for s in samples):
                raise ValueError(
                    "All samples must have rgb_images when "
                    "RGB_IMAGE data type is requested"
                )
            bd.rgb_images = MaskableData(
                torch.stack([cast(MaskableData, s.rgb_images).data for s in samples]),
                torch.stack([cast(MaskableData, s.rgb_images).mask for s in samples]),
            )

        if DataType.DEPTH_IMAGE in data_types:
            if any(s.depth_images is None for s in samples):
                raise ValueError(
                    "All samples must have depth_images when "
                    "DEPTH_IMAGE data type is requested"
                )
            bd.depth_images = MaskableData(
                torch.stack([cast(MaskableData, s.depth_images).data for s in samples]),
                torch.stack([cast(MaskableData, s.depth_images).mask for s in samples]),
            )

        if DataType.POINT_CLOUD in data_types:
            if any(s.point_clouds is None for s in samples):
                raise ValueError(
                    "All samples must have point_clouds when "
                    "POINT_CLOUD data type is requested"
                )
            bd.point_clouds = MaskableData(
                torch.stack([cast(MaskableData, s.point_clouds).data for s in samples]),
                torch.stack([cast(MaskableData, s.point_clouds).mask for s in samples]),
            )

        # Language data
        if DataType.LANGUAGE in data_types:
            if any(s.language_tokens is None for s in samples):
                raise ValueError(
                    "All samples must have language_tokens when "
                    "LANGUAGE data type is requested"
                )
            bd.language_tokens = MaskableData(
                torch.cat(
                    [cast(MaskableData, s.language_tokens).data for s in samples]
                ),
                torch.cat(
                    [cast(MaskableData, s.language_tokens).mask for s in samples]
                ),
            )

        # Custom data
        if DataType.CUSTOM in data_types:
            # Collect all custom data keys from all samples
            all_custom_keys: Set[str] = set()
            for sample in samples:
                if sample.custom_data:
                    all_custom_keys.update(sample.custom_data.keys())

            bd.custom_data = {}
            for key in all_custom_keys:
                # Check if all samples have this custom data key
                custom_data_list = []
                custom_mask_list = []
                for sample in samples:
                    if sample.custom_data and key in sample.custom_data:
                        custom_data_list.append(sample.custom_data[key].data)
                        custom_mask_list.append(sample.custom_data[key].mask)
                    else:
                        # Create zero tensors for missing data
                        if custom_data_list:
                            # Use the shape of the first sample for consistency
                            zero_data = torch.zeros_like(custom_data_list[0])
                            zero_mask = torch.zeros_like(custom_mask_list[0])
                        else:
                            # If this is the first sample and it's missing, skip
                            continue
                        custom_data_list.append(zero_data)
                        custom_mask_list.append(zero_mask)

                if custom_data_list:
                    bd.custom_data[key] = MaskableData(
                        torch.stack(custom_data_list),
                        torch.stack(custom_mask_list),
                    )

        return bd

    def collate_fn(self, samples: list[TrainingSample]) -> BatchedTrainingSamples:
        """Collate training samples into a complete batch for model training.

        Combines individual training samples into batched inputs, outputs, and
        prediction masks suitable for model training. This function is typically
        used with PyTorch DataLoader.

        Args:
            samples: List of TrainingSample objects to batch together.

        Returns:
            A BatchedTrainingSamples object containing batched inputs, outputs,
            and prediction masks ready for model training.
        """
        return BatchedTrainingSamples(
            inputs=self._collate_fn([s.inputs for s in samples], self.input_data_types),
            outputs=self._collate_fn(
                [s.outputs for s in samples], self.output_data_types
            ),
            output_prediction_mask=torch.stack(
                [sample.output_prediction_mask for sample in samples]
            ),
        )
