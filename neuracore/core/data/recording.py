"""Recording class for managing synchronized data streams in a dataset."""

from typing import TYPE_CHECKING

from neuracore_types import DataSpec, DataType, RobotDataSpec

from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.ml.utils.robot_data_spec_utils import extract_data_types

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
        total_bytes: int,
        robot_id: str,
        instance: int,
        start_time: float,
        end_time: float,
    ):
        """Initialize episode iterator for a specific recording.

        Args:
            dataset: Parent Dataset instance.
            recording_id: Unique identifier for the recording episode.
            total_bytes: Size of the recording episode in bytes.
            robot_id: The robot that created this recording.
            instance: The instance of the robot that created this recording.
            start_time: Unix timestamp when recording started.
            end_time: Unix timestamp when recording ended.
        """
        self.dataset = dataset
        self.id = recording_id
        self.total_bytes = total_bytes
        self.robot_id = robot_id
        self.instance = instance
        self.start_time = start_time
        self.end_time = end_time
        self._raw = {
            "id": recording_id,
            "total_bytes": total_bytes,
            "robot_id": robot_id,
            "instance": instance,
        }

    def __getitem__(self, key: str) -> object:
        """Support old dict-style access dynamically."""
        try:
            return self._raw[key]
        except KeyError:
            raise KeyError(f"Recording has no key '{key}'")

    def synchronize(
        self,
        frequency: int = 0,
        data_spec: DataSpec | None = None,
        robot_data_spec: RobotDataSpec | None = None,
    ) -> SynchronizedRecording:
        """Synchronize the episode with specified frequency and data types.

        Args:
            frequency: Frequency at which to synchronize the episode.
                Use 0 for aperiodic data.
            data_spec: Dict specifying data types and their names to include
                in synchronization. If None, will use all available data
                types from the dataset.
            robot_data_spec: Dict specifying robot id to
                data types and their names to include in synchronization.
                If None, will use all available data types from the dataset.

        Raises:
            SynchronizationError: If synchronization fails.
        """
        if frequency < 0:
            raise SynchronizationError("Frequency must be >= 0")

        if data_spec is not None:
            # Special case for backend.
            # TODO: Find a better way to handle this.
            robot_data_spec = {
                "": data_spec,
            }

        data_types = extract_data_types(robot_data_spec) if robot_data_spec else None
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
            start_time=self.start_time,
            end_time=self.end_time,
            frequency=frequency,
            robot_data_spec=robot_data_spec,
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
