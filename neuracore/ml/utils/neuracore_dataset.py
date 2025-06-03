"""Abstract base class for Neuracore datasets with multi-modal data support.

This module provides the foundation for creating datasets that handle robot
demonstration data including images, joint states, and language instructions.
It includes standardized preprocessing, batching, and error handling for
machine learning training workflows.
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from neuracore.core.nc_types import DataType
from neuracore.ml import BatchedTrainingSamples, MaskableData
from neuracore.ml.ml_types import BatchedData

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples


class NeuracoreDataset(Dataset, ABC):
    """Abstract base class for Neuracore multi-modal robot datasets.

    This class provides a standardized interface for datasets containing robot
    demonstration data. It handles data type validation, preprocessing setup,
    batch collation, and error management for training machine learning models
    on robot data including images, joint states, and language instructions.
    """

    def __init__(
        self,
        input_data_types: list[DataType],
        output_data_types: list[DataType],
        output_prediction_horizon: int = 5,
        tokenize_text: Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]] = None,
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
        if DataType.LANGUAGE in self.data_types and tokenize_text is None:
            raise ValueError(
                "Tokenizer not provided but language data requested. "
                "Please provide a tokenizer function."
            )

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
        while self._error_count < self._max_error_count:
            try:
                episode_idx = idx % self.num_episodes
                return self.load_sample(episode_idx)
            except Exception as e:
                self._error_count += 1
                logger.error(f"Error loading item {idx}: {str(e)}")
                if self._error_count >= self._max_error_count:
                    raise e

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
        if DataType.JOINT_POSITIONS in data_types:
            bd.joint_positions = MaskableData(
                torch.stack([s.joint_positions.data for s in samples]),
                torch.stack([s.joint_positions.mask for s in samples]),
            )
        if DataType.JOINT_VELOCITIES in data_types:
            bd.joint_velocities = MaskableData(
                torch.stack([s.joint_velocities.data for s in samples]),
                torch.stack([s.joint_velocities.mask for s in samples]),
            )
        if DataType.JOINT_TORQUES in data_types:
            bd.joint_torques = MaskableData(
                torch.stack([s.joint_torques.data for s in samples]),
                torch.stack([s.joint_torques.mask for s in samples]),
            )
        if DataType.JOINT_TARGET_POSITIONS in data_types:
            bd.joint_target_positions = MaskableData(
                torch.stack([s.joint_target_positions.data for s in samples]),
                torch.stack([s.joint_target_positions.mask for s in samples]),
            )
        if DataType.RGB_IMAGE in data_types:
            bd.rgb_images = MaskableData(
                torch.stack([s.rgb_images.data for s in samples]),
                torch.stack([s.rgb_images.mask for s in samples]),
            )
        if DataType.LANGUAGE in data_types:
            bd.language_tokens = MaskableData(
                torch.cat([s.language_tokens.data for s in samples]),
                torch.cat([s.language_tokens.mask for s in samples]),
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
            output_predicition_mask=torch.stack(
                [sample.output_predicition_mask for sample in samples]
            ),
        )
