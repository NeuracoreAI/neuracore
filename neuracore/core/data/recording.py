"""Recording class for managing synchronized data streams in a dataset."""

from typing import TYPE_CHECKING, Optional

from neuracore_types import DataType

from neuracore.core.data.synced_recording import SynchronizedRecording

from ..exceptions import SynchronizationError

if TYPE_CHECKING:
    from neuracore.core.data.dataset import Dataset


class Recording:
    """Class representing a recording episode in a dataset.

    This class provides methods to synchronize the recording with a specified
    frequency and data types, and to iterate over the synchronized data.
    """

    def __init__(
        self,
        dataset: "Dataset",
        recording_id: str,
        size_bytes: int,
        robot_id: str,
        instance: int,
    ):
        """Initialize episode iterator for a specific recording.

        Args:
            dataset: Parent Dataset instance.
            recording_id: Unique identifier for the recording episode.
            size_bytes: Size of the recording episode in bytes.
            robot_id: The robot that created this recording.
            instance: The instance of the robot that created this recording.
        """
        self.dataset = dataset
        self.id = recording_id
        self.size_bytes = size_bytes
        self.robot_id = robot_id
        self.instance = instance

    def synchronize(
        self,
        frequency: int = 0,
        data_types: Optional[list[DataType]] = None,
    ) -> SynchronizedRecording:
        """Synchronize the episode with specified frequency and data types.

        Args:
            frequency: Frequency at which to synchronize the episode.
                Use 0 for aperiodic data.
            data_types: List of DataType to include in synchronization.
                If None, uses the default data types from the recording.

        Raises:
            SynchronizationError: If synchronization fails.
        """
        if frequency < 0:
            raise SynchronizationError("Frequency must be >= 0")

        # check valid data types if provided
        if data_types is not None:
            if not all(isinstance(dt, DataType) for dt in data_types):
                raise ValueError(
                    "Invalid data types provided. "
                    "All items must be DataType enum values."
                )
            if not set(data_types).issubset(set(self.dataset.data_types)):
                raise SynchronizationError(
                    "Invalid data type requested for synchronization"
                )

        return SynchronizedRecording(
            dataset=self.dataset,
            recording_id=self.id,
            robot_id=self.robot_id,
            instance=self.instance,
            frequency=frequency,
            data_types=data_types or [],
        )

    def __iter__(self) -> None:
        """Initialize iterator over synchronized recording data.

        Raises:
            RuntimeError: Always raised to indicate that this method is not
            supported for unsynchronized recordings.
        """
        raise RuntimeError(
            "Only synchronized recordings can be iterated over. "
            "Use the synchronize method to create a synchronized recording."
        )
