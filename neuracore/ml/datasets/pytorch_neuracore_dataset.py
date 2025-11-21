"""Abstract base class for Neuracore datasets with multi-modal data support.

This module provides the foundation for creating datasets that handle robot
demonstration data including images, joint states, depth images, point clouds,
poses, end-effectors, and language instructions.
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import torch
from neuracore_types import DataType
from torch.utils.data import Dataset

from neuracore.ml import BatchedTrainingSamples, MaskableData

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
        self, samples: List[Dict[DataType, Dict[str, MaskableData]]]
    ) -> Dict[DataType, Dict[str, MaskableData]]:
        """Collate individual data samples into a batched format.

        Combines multiple samples into batched tensors with appropriate stacking
        for different data modalities. Handles masking for variable-length data.

        Args:
            samples: List of BatchedData objects to combine.
            data_types: List of data types to include in the batch.

        Returns:
            A single BatchedData object containing the stacked samples.
        """
        batched_data: Dict[DataType, Dict[str, MaskableData]] = {}
        for data_type in samples[0].keys():
            batched_data[data_type] = {}
            group_names = samples[0][data_type].keys()
            for group_name in group_names:
                batched_data[data_type][group_name] = MaskableData(
                    data=torch.stack(
                        [sample[data_type][group_name].data for sample in samples]
                    ),
                    mask=torch.stack(
                        [sample[data_type][group_name].mask for sample in samples]
                    ),
                )
        return batched_data

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
            inputs=self._collate_fn([s.inputs for s in samples]),
            outputs=self._collate_fn([s.outputs for s in samples]),
            output_prediction_mask=torch.stack(
                [sample.output_prediction_mask for sample in samples]
            ),
        )
