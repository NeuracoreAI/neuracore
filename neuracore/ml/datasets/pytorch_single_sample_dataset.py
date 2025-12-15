"""Dataset that returns the same sample from a real dataset for quick testing."""

from neuracore_types import DataType, NCDataStats, RobotDataSpec

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
        input_robot_data_spec: RobotDataSpec,
        output_robot_data_spec: RobotDataSpec,
        output_prediction_horizon: int,
        num_recordings: int,
        dataset_statistics: dict[DataType, list[NCDataStats]],
    ):
        """Initialize the decoy dataset."""
        super().__init__(
            num_recordings=num_recordings,
            input_robot_data_spec=input_robot_data_spec,
            output_robot_data_spec=output_robot_data_spec,
            output_prediction_horizon=output_prediction_horizon,
        )

        # Create a template sample from the first sample of the dataset
        self._sample = sample
        self._num_recordings = num_recordings
        self._dataset_statistics = dataset_statistics

    def __len__(self) -> int:
        """Return the number of samples in the dataset this dataset is mimicking."""
        return self._num_recordings

    def __getitem__(self, idx: int) -> BatchedTrainingSamples:
        """Get a training sample."""
        return self.load_sample(idx)

    def load_sample(
        self, episode_idx: int, timestep: int | None = None
    ) -> BatchedTrainingSamples:
        """Load the same sample from the dataset.

        Passed arguments are ignored.
        """
        return self._sample

    @property
    def dataset_statistics(self) -> dict[DataType, list[NCDataStats]]:
        """Return the dataset description."""
        return self._dataset_statistics
