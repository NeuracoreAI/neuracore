import logging
from typing import Callable, Optional

import numpy as np
import torch

from neuracore.core.nc_types import DataItemStats, DatasetDescription, DataType
from neuracore.ml import BatchedTrainingSamples, MaskableData
from neuracore.ml.utils.neuracore_dataset import NeuracoreDataset

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples


class DummyDataset(NeuracoreDataset):
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
        tokenize_text: Optional[
            Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ):
        """
        Initialize the dummy dataset.

        Args:
            input_data_types: List of supported input data types
            output_data_types: List of supported output data types
            num_samples: Number of samples in the dataset
            num_episodes: Number of episodes in the dataset
            output_prediction_horizon: Length of the action sequence
            tokenize_text: Function to tokenize text data
        """
        super().__init__(
            input_data_types=input_data_types,
            output_data_types=output_data_types,
            output_prediction_horizon=output_prediction_horizon,
            tokenize_text=tokenize_text,
        )
        self.num_samples = num_samples
        self.num_episodes = num_episodes

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
        max_language_length = max(
            len(instruction) for instruction in self.sample_instructions
        )

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
        if DataType.LANGUAGE in self.data_types:
            # Add language data stats to dataset description
            self.dataset_description.max_language_length = max_language_length

        self._error_count = 0
        self._max_error_count = 1

    def load_sample(
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

            if DataType.LANGUAGE in self.data_types:
                # Randomly select an instruction
                instruction = np.random.choice(self.sample_instructions)
                # Tokenize the instruction
                input_ids, attention_mask = self.tokenize_text([instruction])

                language_tokens = MaskableData(input_ids, attention_mask)

                if DataType.LANGUAGE in self.input_data_types:
                    sample.inputs.language_tokens = language_tokens
                if DataType.LANGUAGE in self.output_data_types:
                    sample.outputs.language_tokens = language_tokens

            return sample

        except Exception as e:
            logger.error(f"Error generating random sample: {str(e)}")
            raise

    def __len__(self) -> int:
        return self.num_samples

    def collate_fn(self, samples: list[TrainingSample]) -> BatchedTrainingSamples:
        """Collate a list of samples into a single batch."""
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
        return BatchedTrainingSamples(
            inputs=self._collate_fn([s.inputs for s in samples], self.input_data_types),
            outputs=bd,
            output_predicition_mask=torch.stack(
                [sample.output_predicition_mask for sample in samples]
            ),
        )
