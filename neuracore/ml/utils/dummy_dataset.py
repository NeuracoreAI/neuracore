import logging
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from neuracore.core.nc_types import DataItemStats, DatasetDescription, DataType
from neuracore.ml import ActionMaskableData, BatchedTrainingSamples, MaskableData

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
        data_types: list[DataType],
        num_samples: int = 50,
        num_episodes: int = 10,
        action_sequence_length: int = 5,
    ):
        """
        Initialize the dummy dataset.

        Args:
            data_types: List of data types to include in the dataset
            num_samples: Number of samples in the dataset
            num_episodes: Number of episodes in the dataset
            action_sequence_length: Length of the action sequence
        """
        self.data_types = data_types
        self.num_samples = num_samples
        self.num_episodes = num_episodes
        self.action_sequence_length = action_sequence_length

        # Setup camera transform to match EpisodicDataset
        self.camera_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

        self.image_size = (224, 224)

        self.dataset_description = DatasetDescription()
        if DataType.RGB_IMAGE in data_types:
            self.dataset_description.max_num_rgb_images = 2
        if DataType.JOINT_POSITIONS in data_types:
            self.dataset_description.joint_positions = DataItemStats(
                mean=np.zeros(6), std=np.ones(6)
            )
        if DataType.ACTIONS in data_types:
            self.dataset_description.actions = DataItemStats(
                mean=np.zeros(7), std=np.ones(7)
            )

        self._error_count = 0
        self._max_error_count = 1

    def _load_sample(
        self, episode_idx: int, timestep: Optional[int] = None
    ) -> BatchedTrainingSamples:
        """Generate a random sample in the format of EpisodicDataset."""
        try:

            sample = BatchedTrainingSamples()

            if DataType.RGB_IMAGE in self.data_types:
                max_rgb_len = self.dataset_description.max_num_rgb_images
                sample.rgb_images = MaskableData(
                    torch.zeros(
                        (max_rgb_len, 3, *self.image_size), dtype=torch.float32
                    ),
                    torch.ones((max_rgb_len,), dtype=torch.float32),
                )

            if DataType.JOINT_POSITIONS in self.data_types:
                max_jp_len = self.dataset_description.joint_positions.max_len
                sample.joint_positions = MaskableData(
                    torch.zeros((max_jp_len,), dtype=torch.float32),
                    torch.ones((max_jp_len,), dtype=torch.float32),
                )

            if DataType.ACTIONS in self.data_types:
                max_action_len = self.dataset_description.actions.max_len
                sample.actions = ActionMaskableData(
                    torch.zeros(
                        (self.action_sequence_length, max_action_len),
                        dtype=torch.float32,
                    ),
                    torch.ones((max_action_len,), dtype=torch.float32),
                    torch.ones((self.action_sequence_length,), dtype=torch.float32),
                )

            return sample

        except Exception as e:
            logger.error(f"Error generating random sample: {str(e)}")
            raise

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> BatchedTrainingSamples:
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

    def collate_fn(self, samples: list[TrainingSample]) -> BatchedTrainingSamples:
        """Collate a list of samples into a single batch."""
        joint_positions = torch.stack(
            [sample.joint_positions.data for sample in samples]
        )
        joint_positions_mask = torch.stack(
            [sample.joint_positions.data for sample in samples]
        )
        rgb = torch.stack([sample.rgb_images.data for sample in samples])
        rgb_mask = torch.stack([sample.rgb_images.mask for sample in samples])
        actions = torch.stack([sample.actions.data for sample in samples])
        actions_mask = torch.stack([sample.actions.mask for sample in samples])
        actions_sequence_mask = torch.stack(
            [sample.actions.sequence_mask for sample in samples]
        )
        return BatchedTrainingSamples(
            actions=ActionMaskableData(actions, actions_mask, actions_sequence_mask),
            joint_positions=MaskableData(joint_positions, joint_positions_mask),
            rgb_images=MaskableData(rgb, rgb_mask),
        )
