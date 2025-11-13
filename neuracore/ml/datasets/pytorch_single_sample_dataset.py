"""Dataset that returns the same sample from a real dataset for quick testing."""

from typing import Optional

from neuracore_types import DatasetDescription, DataType

from neuracore.ml import BatchedTrainingSamples
from neuracore.ml.datasets.pytorch_neuracore_dataset import PytorchNeuracoreDataset


class SingleSampleDataset(PytorchNeuracoreDataset):
    """Fast dataset wrapper that loads and saves the first sample from a real dataset.

    It saves this sample to avoid costly loading of the samples
    every time __getitem__ or load_sample is called.
    """

    def __init__(
        self,
        sample: BatchedTrainingSamples,
        input_data_types: list[DataType],
        output_data_types: list[DataType],
        output_prediction_horizon: int,
        dataset_description: DatasetDescription,
        num_recordings: int,
    ):
        """Initialize the decoy dataset."""
        super().__init__(
            num_recordings=num_recordings,
            input_data_types=input_data_types,
            output_data_types=output_data_types,
            output_prediction_horizon=output_prediction_horizon,
        )

        # Create a template sample from the first sample of the dataset
        self._sample = sample
        self._num_recordings = num_recordings
        self._dataset_description = dataset_description

    def __len__(self) -> int:
        """Return the number of samples in the dataset this dataset is mimicking."""
        return self._num_recordings

    def load_sample(
        self, episode_idx: int, timestep: Optional[int] = None
    ) -> BatchedTrainingSamples:
        """Load the same sample from the dataset.

        Passed arguments are ignored.
        """
        return self._sample

    @property
    def dataset_description(self) -> DatasetDescription:
        """Return the dataset description."""
        return self._dataset_description
