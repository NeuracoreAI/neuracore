import logging
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from neuracore.core.nc_types import DataItemStats, DatasetDescription, DataType
from neuracore.ml import BatchedTrainingSamples, MaskableData
from neuracore.ml.ml_types import BatchedData

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples


class DummyDataset(Dataset):
    """
    A small dummy dataset for validating algorithm plumbing.

    This dataset generates random data with the same structure as the real
    EpisodicDataset, but without requiring any actual data files.
    """

    def __init__(
        self,
        input_data_types: list[DataType],
        output_data_types: list[DataType],
        num_samples: int = 50,
        num_episodes: int = 10,
        output_prediction_horizon: int = 5,
        language_model_name: str = "bert-base-uncased",
    ):
        """
        Initialize the dummy dataset.

        Args:
            input_data_types: List of supported input data types
            output_data_types: List of supported output data types
            num_samples: Number of samples in the dataset
            num_episodes: Number of episodes in the dataset
            output_prediction_horizon: Length of the action sequence
            language_model_name: Name of the language model for tokenization
        """
        self.input_data_types = input_data_types
        self.output_data_types = output_data_types
        self.num_samples = num_samples
        self.num_episodes = num_episodes
        self.output_prediction_horizon = output_prediction_horizon
        self.language_model_name = language_model_name

        self.data_types = set(input_data_types + output_data_types)

        # Setup camera transform to match EpisodicDataset
        self.camera_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

        self.image_size = (224, 224)

        # Create tokenizer if language data is used
        self.tokenizer = None
        if DataType.LANGUAGE_DATA in self.data_types:
            self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
            self.max_token_length = 32  # Max token length for dummy data

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
        if DataType.RGB_IMAGE in self.data_types:
            self.dataset_description.max_num_rgb_images = 2
        if DataType.JOINT_POSITIONS in self.data_types:
            self.dataset_description.joint_positions = DataItemStats(
                mean=np.zeros(6), std=np.ones(6), max_len=6
            )
        if DataType.JOINT_VELOCITIES in self.data_types:
            self.dataset_description.joint_velocities = DataItemStats(
                mean=np.zeros(6), std=np.ones(6), max_len=6
            )
        if DataType.JOINT_TORQUES in self.data_types:
            self.dataset_description.joint_torques = DataItemStats(
                mean=np.zeros(6), std=np.ones(6), max_len=6
            )
        if DataType.JOINT_TARGET_POSITIONS in self.data_types:
            self.dataset_description.joint_target_positions = DataItemStats(
                mean=np.zeros(7), std=np.ones(7), max_len=7
            )
        if DataType.LANGUAGE_DATA in self.data_types:
            # Add language data stats to dataset description
            self.dataset_description.language_data = DataItemStats(
                mean=np.zeros(1), std=np.ones(1), max_len=self.max_token_length
            )

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
            max_length=self.max_token_length,
            truncation=True,
            return_tensors="pt",
        )

        # Extract token ids and attention mask
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        return input_ids, attention_mask

    def _load_sample(
        self, episode_idx: int, timestep: Optional[int] = None
    ) -> TrainingSample:
        """Generate a random sample in the format of EpisodicDataset."""
        try:
            sample = TrainingSample(
                output_predicition_mask=torch.ones(
                    (self.output_prediction_horizon,), dtype=torch.float32
                ),
            )

            if DataType.RGB_IMAGE in self.data_types:
                max_rgb_len = self.dataset_description.max_num_rgb_images
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

            if DataType.LANGUAGE_DATA in self.data_types:
                # Randomly select an instruction
                instruction = np.random.choice(self.sample_instructions)
                # Tokenize the instruction
                input_ids, attention_mask = self._tokenize_text(instruction)

                language_tokens = MaskableData(
                    input_ids.unsqueeze(0),  # Add batch dimension
                    attention_mask.unsqueeze(0),  # Add batch dimension
                )

                if DataType.LANGUAGE_DATA in self.input_data_types:
                    sample.inputs.language_tokens = language_tokens
                if DataType.LANGUAGE_DATA in self.output_data_types:
                    sample.outputs.language_tokens = language_tokens

            return sample

        except Exception as e:
            logger.error(f"Error generating random sample: {str(e)}")
            raise

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> TrainingSample:
        """Get a sample from the dataset."""
        while self._error_count < self._max_error_count:
            try:
                episode_idx = idx % self.num_episodes
                return self._load_sample(episode_idx)
            except Exception as e:
                self._error_count += 1
                logger.error(f"Error loading item {idx}: {str(e)}")
                if self._error_count >= self._max_error_count:
                    raise e

    def _collate_fn(
        self, samples: list[BatchedData], data_types: list[DataType], expand_data: bool
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
        if DataType.LANGUAGE_DATA in data_types:
            bd.language_tokens = MaskableData(
                torch.cat([s.language_tokens.data for s in samples]),
                torch.cat([s.language_tokens.mask for s in samples]),
            )

        if expand_data:
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
        return bd

    def collate_fn(self, samples: list[TrainingSample]) -> BatchedTrainingSamples:
        """Collate a list of samples into a single batch."""
        return BatchedTrainingSamples(
            inputs=self._collate_fn(
                [s.inputs for s in samples], self.input_data_types, False
            ),
            outputs=self._collate_fn(
                [s.outputs for s in samples], self.output_data_types, True
            ),
            output_predicition_mask=torch.stack(
                [sample.output_predicition_mask for sample in samples]
            ),
        )
