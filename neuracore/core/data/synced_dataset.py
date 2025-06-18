"""SynchronizedDataset class for managing synchronized datasets."""

import logging
from typing import TYPE_CHECKING, Optional, Union

from neuracore.core.data.synced_recording import SynchronizedRecording

from ..nc_types import DatasetDescription, DataType

if TYPE_CHECKING:
    from neuracore.core.data.dataset import Dataset


logger = logging.getLogger(__name__)


class SynchronizedDataset:
    """Class for managing synchronized datasets."""

    def __init__(
        self,
        dataset: "Dataset",
        frequency: int,
        data_types: Optional[list[DataType]],
        dataset_description: DatasetDescription,
    ):
        """Initialize a dataset from server response data.

        Args:
            dataset: Dataset object containing recordings.
            frequency: Frequency of the dataset in Hz.
            data_types: List of data types to include in the dataset.
            dataset_description: Description of the dataset.
        """
        self.dataset = dataset
        self.frequency = frequency
        self.data_types = data_types or []
        self.dataset_description = dataset_description
        self._recording_idx = 0
        self._synced_recording_cache: dict[int, SynchronizedRecording] = {}

    @property
    def num_transitions(self) -> int:
        """Get the number of transitions in the dataset."""
        return sum(len(synced_recording) for synced_recording in self)

    def __iter__(self) -> "SynchronizedDataset":
        """Initialize iterator over episodes in the dataset.

        Returns:
            Self for iteration over episodes.
        """
        return self

    def __len__(self) -> int:
        """Get the number of episodes in the dataset.

        Returns:
            Number of demonstration episodes in the dataset.
        """
        return len(self.dataset)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union["SynchronizedRecording", "SynchronizedDataset"]:
        """Support for indexing and slicing dataset episodes.

        Args:
            idx: Integer index or slice object for accessing episodes.

        Returns:
            SynchronizedRecording for a single episode or
            SynchronizedDataset for a slice of episodes.

        Raises:
            IndexError: If the index is out of range.
            TypeError: If the index is not an integer or slice.
        """
        if isinstance(idx, slice):
            # Handle slice
            self.dataset.recordings = self.dataset.recordings[
                idx.start : idx.stop : idx.step
            ]
            ds = SynchronizedDataset(
                dataset=self.dataset,
                frequency=self.frequency,
                data_types=self.data_types,
                dataset_description=self.dataset_description,
            )
            return ds
        else:
            # Handle single index
            if isinstance(idx, int):
                if idx < 0:  # Handle negative indices
                    idx += len(self.dataset.recordings)
                if not 0 <= idx < len(self.dataset.recordings):
                    raise IndexError("Dataset index out of range")
                if idx not in self._synced_recording_cache:
                    synced_recording = SynchronizedRecording(
                        recording_id=self.dataset.recordings[idx]["id"],
                        dataset=self.dataset,
                        frequency=self.frequency,
                        data_types=self.data_types,
                    )
                    self._synced_recording_cache[idx] = synced_recording
                return self._synced_recording_cache[idx]
            raise TypeError(
                f"Dataset indices must be integers or slices, not {type(idx)}"
            )

    def __next__(self) -> SynchronizedRecording:
        """Get the next episode in the dataset iteration.

        Returns:
            SynchronizedRecording for the next episode.

        Raises:
            StopIteration: When all episodes have been processed.
        """
        if self._recording_idx >= len(self.dataset.recordings):
            raise StopIteration

        if self._recording_idx in self._synced_recording_cache:
            return self._synced_recording_cache[self._recording_idx]

        recording = self.dataset.recordings[self._recording_idx]
        self._recording_idx += 1  # Increment counter
        return SynchronizedRecording(
            recording_id=recording["id"],
            dataset=self.dataset,
            frequency=self.frequency,
            data_types=self.data_types,
        )
