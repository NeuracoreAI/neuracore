"""Dummy dataset for algorithm validation and testing without real data.

This module provides a synthetic dataset that generates random data matching
the structure of real Neuracore datasets. It's used for algorithm development,
testing, and validation without requiring actual robot demonstration data.
"""

import logging
from typing import Callable, Dict, Optional

import numpy as np
import torch
from neuracore_types import DataItemStats, DatasetStatistics, DataType

from neuracore.core.robot import Robot
from neuracore.ml import BatchedTrainingSamples, MaskableData
from neuracore.ml.datasets.pytorch_neuracore_dataset import PytorchNeuracoreDataset

logger = logging.getLogger(__name__)

TrainingSample = BatchedTrainingSamples


DATA_TYPE_TO_DUMMY_SHAPE_AND_MAX_LEN = {
    DataType.JOINT_POSITIONS: ((7,), 50),
    DataType.JOINT_VELOCITIES: ((7,), 50),
    DataType.JOINT_TORQUES: ((7,), 50),
    DataType.JOINT_TARGET_POSITIONS: ((7,), 50),
    DataType.END_EFFECTOR_POSES: ((7,), 50),
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: ((1,), 50),
    DataType.RGB_IMAGES: ((3, 224, 224), 10),
    DataType.DEPTH_IMAGES: ((1, 224, 224), 10),
    DataType.POINT_CLOUDS: ((1024, 3), 10),
    DataType.POSES: ((7,), 50),
    DataType.LANGUAGE: ((32,), 1),  # Assuming max 32 tokens
}


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

        self.dataset_statistics = DatasetStatistics()
        for data_type in self.data_types:
            max_len = DATA_TYPE_TO_DUMMY_SHAPE_AND_MAX_LEN[data_type][1]
            self.dataset_statistics.data[data_type] = {
                "dummy_key": DataItemStats(
                    mean=np.zeros(max_len).tolist(),
                    std=np.ones(max_len).tolist(),
                    min=-np.ones(max_len).tolist(),
                    max=np.ones(max_len).tolist(),
                    max_len=max_len,
                    robot_to_ncdata_keys={
                        self.robot.id: [f"keys_{i}" for i in range(max_len)]
                    },
                )
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
            inputs_and_outputs: Dict[DataType, Dict[str, MaskableData]] = {}
            for data_type in set(self.input_data_types + self.output_data_types):
                inputs_and_outputs[data_type] = {}
                for group_name, data_item_stats in self.dataset_statistics.data[
                    data_type
                ].items():
                    max_rgb_len = data_item_stats.max_len
                    data_shape = DATA_TYPE_TO_DUMMY_SHAPE_AND_MAX_LEN[data_type][0]
                    data = MaskableData(
                        data=torch.zeros(
                            (data_item_stats.max_len, *data_shape), dtype=torch.float32
                        ),
                        mask=torch.ones((max_rgb_len,), dtype=torch.float32),
                    )
                    inputs_and_outputs[data_type][group_name] = data

            inputs: Dict[DataType, Dict[str, MaskableData]] = {}
            for data_type in self.input_data_types:
                inputs[data_type] = {}
                for group_name, data_item_stats in self.dataset_statistics.data[
                    data_type
                ].items():
                    inputs[data_type][group_name] = inputs_and_outputs[data_type][
                        group_name
                    ]

            outputs: Dict[DataType, Dict[str, MaskableData]] = {}
            for data_type in self.output_data_types:
                outputs[data_type] = {}
                for group_name, data_item_stats in self.dataset_statistics.data[
                    data_type
                ].items():
                    outputs[data_type][group_name] = inputs_and_outputs[data_type][
                        group_name
                    ]

            return TrainingSample(
                inputs=inputs,
                outputs=outputs,
                output_prediction_mask=torch.ones(
                    (self.output_prediction_horizon,), dtype=torch.float32
                ),
            )

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
        bd = self._collate_fn([s.outputs for s in samples])
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
            inputs=self._collate_fn([s.inputs for s in samples]),
            outputs=bd,
            output_prediction_mask=torch.stack(
                [sample.output_prediction_mask for sample in samples]
            ),
        )
