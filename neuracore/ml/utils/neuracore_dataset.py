import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from neuracore.core.nc_types import DataType
from neuracore.ml import BatchedTrainingSamples, MaskableData
from neuracore.ml.ml_types import BatchedData

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples


class NeuracoreDataset(Dataset, ABC):
    """
    A small dummy dataset for validating algorithm plumbing.

    This dataset generates random data with the same structure as the real
    EpisodicDataset, but without requiring any actual data files.
    """

    def __init__(
        self,
        input_data_types: list[DataType],
        output_data_types: list[DataType],
        output_prediction_horizon: int = 5,
        language_model_name: str = "bert-base-uncased",
        language_max_token_length: int = 125,
    ):
        """
        Initialize the dummy dataset.

        Args:
            input_data_types: List of supported input data types
            output_data_types: List of supported output data types
            output_prediction_horizon: Length of the action sequence
            language_model_name: Name of the language model for tokenization
        """
        self.input_data_types = input_data_types
        self.output_data_types = output_data_types
        self.output_prediction_horizon = output_prediction_horizon
        self.language_model_name = language_model_name
        self.language_max_token_length = language_max_token_length

        self.data_types = set(input_data_types + output_data_types)

        # Setup camera transform to match EpisodicDataset
        self.camera_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

        # Create tokenizer if language data is used
        self.tokenizer = None
        if DataType.LANGUAGE in self.data_types:
            self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)

        self._error_count = 0
        self._max_error_count = 1

    def _tokenize_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text using the specified tokenizer."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized but language data requested")

        # Tokenize the text
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.language_max_token_length,
            truncation=True,
            return_tensors="pt",
        )

        # Extract token ids and attention mask
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        return input_ids, attention_mask

    @abstractmethod
    def load_sample(
        self, episode_idx: int, timestep: Optional[int] = None
    ) -> TrainingSample:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> TrainingSample:
        """Get a sample from the dataset."""
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
        """Collate a list of samples into a single batch."""
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
        """Collate a list of samples into a single batch."""
        return BatchedTrainingSamples(
            inputs=self._collate_fn([s.inputs for s in samples], self.input_data_types),
            outputs=self._collate_fn(
                [s.outputs for s in samples], self.output_data_types
            ),
            output_predicition_mask=torch.stack(
                [sample.output_predicition_mask for sample in samples]
            ),
        )
