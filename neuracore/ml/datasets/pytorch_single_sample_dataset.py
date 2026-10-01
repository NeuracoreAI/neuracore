"""Dataset that returns the same sample from a real dataset for quick testing."""

from neuracore_types import CrossEmbodimentDescription, DataType, NCDataStats

from neuracore.ml import BatchedTrainingSamples
from neuracore.ml.datasets.pytorch_neuracore_dataset import (
    PytorchNeuracoreDataset,
    SampleIdentity,
)


class SingleSampleDataset(PytorchNeuracoreDataset):
    """Fast dataset wrapper that loads and saves the first sample from a real dataset.

    It saves this sample to avoid costly loading of the samples
    every time __getitem__ or load_sample is called.
    """

    def __init__(
        self,
        sample: BatchedTrainingSamples,
        input_cross_embodiment_description: CrossEmbodimentDescription,
        output_cross_embodiment_description: CrossEmbodimentDescription,
        output_prediction_horizon: int,
        num_recordings: int,
        dataset_statistics: dict[str, dict[DataType, list[NCDataStats]]],
    ):
        """Initialize the decoy dataset."""
        super().__init__(
            num_recordings=num_recordings,
            input_cross_embodiment_description=input_cross_embodiment_description,
            output_cross_embodiment_description=output_cross_embodiment_description,
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

    def get_sample_identity(self, idx: int) -> SampleIdentity:
        """Return a synthetic identity for this decoy dataset.

        Args:
            idx: Flat sample index.

        Returns:
            One recording per index at timestep 0, using a robot from the
            embodiment description.

        Raises:
            IndexError: If ``idx`` is outside the dataset.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Sample index {idx} is outside the dataset of length {len(self)}. "
                "Expected an index in "
                f"[0, {len(self)})."
            )
        robot_ids = list(self.input_cross_embodiment_description) or list(
            self.output_cross_embodiment_description
        )
        return SampleIdentity(
            recording_id=f"single-sample-{idx}",
            timestep=0,
            robot_id=robot_ids[idx % len(robot_ids)],
        )

    def load_sample(
        self, episode_idx: int, timestep: int | None = None
    ) -> BatchedTrainingSamples:
        """Load the same sample from the dataset.

        Passed arguments are ignored.
        """
        return self._sample

    @property
    def dataset_statistics(self) -> dict[str, dict[DataType, list[NCDataStats]]]:
        """Return the dataset description."""
        return self._dataset_statistics
