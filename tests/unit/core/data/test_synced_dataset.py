"""Tests for SynchronizedDataset class."""

from unittest.mock import patch

import pytest
from neuracore_types import DatasetDescription, DataType

from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.data.synced_recording import SynchronizedRecording


class TestSynchronizedDataset:
    """Tests for the SynchronizedDataset class."""

    @pytest.fixture
    def dataset_mock(self, dataset_dict, recordings_list, tmp_path):
        """Create a mock dataset object."""
        from neuracore.core.data.dataset import Dataset

        dataset = Dataset(**dataset_dict, recordings=recordings_list)
        dataset.cache_dir = tmp_path / "cache"
        dataset.cache_dir.mkdir(parents=True, exist_ok=True)
        return dataset

    @pytest.fixture
    def dataset_description(self):
        """Create a dataset description fixture."""
        return DatasetDescription()

    @pytest.fixture
    def synced_dataset(self, dataset_mock, dataset_description):
        """Create a SynchronizedDataset instance for testing."""
        return SynchronizedDataset(
            dataset=dataset_mock,
            frequency=30,
            data_types=None,
            dataset_description=dataset_description,
            prefetch_videos=False,
        )

    def test_init(self, synced_dataset, dataset_mock, dataset_description):
        """Test SynchronizedDataset initialization."""
        assert synced_dataset.dataset == dataset_mock
        assert synced_dataset.frequency == 30
        assert synced_dataset.data_types == []
        assert synced_dataset.dataset_description == dataset_description
        assert synced_dataset._recording_idx == 0
        assert synced_dataset._synced_recording_cache == {}

    def test_init_with_data_types(self, dataset_mock, dataset_description):
        """Test initialization with specific data types."""
        data_types = [DataType.RGB_IMAGE, DataType.DEPTH_IMAGE]
        synced = SynchronizedDataset(
            dataset=dataset_mock,
            frequency=30,
            data_types=data_types,
            dataset_description=dataset_description,
        )

        assert synced.data_types == data_types

    def test_len(self, synced_dataset):
        """Test __len__ returns correct number of recordings."""
        assert len(synced_dataset) == 2

    def test_iter_reset(self, synced_dataset):
        """Test that __iter__ resets the iteration index."""
        synced_dataset._recording_idx = 5
        result = iter(synced_dataset)

        assert result is synced_dataset
        assert synced_dataset._recording_idx == 0

    def test_getitem_single_index(self, synced_dataset, mock_auth_requests):
        """Test accessing a single recording by index."""
        recording = synced_dataset[0]

        assert isinstance(recording, SynchronizedRecording)
        assert recording.id == "rec1"
        assert recording.robot_id == "robot1"
        assert recording.frequency == 30

    def test_getitem_second_recording(self, synced_dataset, mock_auth_requests):
        """Test accessing the second recording."""
        recording = synced_dataset[1]

        assert isinstance(recording, SynchronizedRecording)
        assert recording.id == "rec2"
        assert recording.robot_id == "robot2"

    def test_getitem_negative_index(self, synced_dataset, mock_auth_requests):
        """Test accessing recordings with negative indices."""
        recording = synced_dataset[-1]

        assert isinstance(recording, SynchronizedRecording)
        assert recording.id == "rec2"

    def test_getitem_negative_first(self, synced_dataset, mock_auth_requests):
        """Test accessing first recording with negative index."""
        recording = synced_dataset[-2]

        assert isinstance(recording, SynchronizedRecording)
        assert recording.id == "rec1"

    def test_getitem_out_of_range(self, synced_dataset):
        """Test that out of range index raises IndexError."""
        with pytest.raises(IndexError, match="Dataset index out of range"):
            _ = synced_dataset[10]

    def test_getitem_negative_out_of_range(self, synced_dataset):
        """Test that negative out of range index raises IndexError."""
        with pytest.raises(IndexError, match="Dataset index out of range"):
            _ = synced_dataset[-10]

    def test_getitem_invalid_type(self, synced_dataset):
        """Test that non-integer/slice index raises TypeError."""
        with pytest.raises(
            TypeError, match="Dataset indices must be integers or slices"
        ):
            _ = synced_dataset["invalid"]

    def test_getitem_slice(self, synced_dataset, mock_auth_requests):
        """Test slicing synchronized dataset."""
        sliced = synced_dataset[0:1]

        assert isinstance(sliced, SynchronizedDataset)
        assert len(sliced) == 1

    def test_getitem_slice_full(self, synced_dataset, mock_auth_requests):
        """Test slicing entire dataset."""
        sliced = synced_dataset[0:2]

        assert isinstance(sliced, SynchronizedDataset)
        assert len(sliced) == 2

    def test_getitem_slice_with_step(self, synced_dataset, mock_auth_requests):
        """Test slicing with step parameter."""
        sliced = synced_dataset[0:2:1]

        assert isinstance(sliced, SynchronizedDataset)
        assert len(sliced) == 2

    def test_getitem_slice_preserves_properties(
        self, synced_dataset, mock_auth_requests
    ):
        """Test that slicing preserves dataset properties."""
        sliced = synced_dataset[0:1]

        assert sliced.frequency == synced_dataset.frequency
        assert sliced.data_types == synced_dataset.data_types
        assert sliced.dataset_description == synced_dataset.dataset_description

    def test_iteration(self, synced_dataset, mock_auth_requests):
        """Test iterating through synchronized dataset."""
        recordings = list(synced_dataset)

        assert len(recordings) == 2
        assert all(isinstance(r, SynchronizedRecording) for r in recordings)
        assert recordings[0].id == "rec1"
        assert recordings[1].id == "rec2"

    def test_iteration_multiple_times(self, synced_dataset, mock_auth_requests):
        """Test that the dataset can be iterated multiple times."""
        recordings1 = list(synced_dataset)
        recordings2 = list(synced_dataset)

        assert len(recordings1) == len(recordings2)
        assert recordings1[0].id == recordings2[0].id

    def test_next_stop_iteration(self, synced_dataset, mock_auth_requests):
        """Test that __next__ raises StopIteration when exhausted."""
        iter(synced_dataset)

        # Exhaust the iterator
        for _ in range(len(synced_dataset.dataset.recordings)):
            next(synced_dataset)

        with pytest.raises(StopIteration):
            next(synced_dataset)

    def test_caching_on_access(self, synced_dataset, mock_auth_requests):
        """Test that recordings are cached after first access."""
        # First access
        recording1 = synced_dataset[0]

        # Check cache
        assert 0 in synced_dataset._synced_recording_cache
        assert synced_dataset._synced_recording_cache[0] is recording1

        # Second access should return cached instance
        recording2 = synced_dataset[0]
        assert recording2 is recording1

    def test_caching_during_iteration(self, synced_dataset, mock_auth_requests):
        """Test that recordings are cached during iteration."""
        # Iterate through dataset
        recordings = list(synced_dataset)

        # Check that all recordings are cached
        assert 0 in synced_dataset._synced_recording_cache
        assert 1 in synced_dataset._synced_recording_cache

        # Accessing by index should return cached instances
        assert synced_dataset[0] is recordings[0]
        assert synced_dataset[1] is recordings[1]

    def test_prefetch_videos_disabled(self, dataset_mock, dataset_description):
        """Test that prefetch_videos=False doesn't trigger prefetch."""
        with patch.object(
            SynchronizedDataset, "_perform_videos_prefetch"
        ) as mock_prefetch:
            SynchronizedDataset(
                dataset=dataset_mock,
                frequency=30,
                data_types=None,
                dataset_description=dataset_description,
                prefetch_videos=False,
            )

            mock_prefetch.assert_not_called()

    def test_prefetch_videos_enabled_no_cache(
        self, dataset_mock, dataset_description, mock_auth_requests
    ):
        """Test that prefetch_videos=True triggers prefetch when no cache exists."""
        with patch.object(
            SynchronizedDataset, "_perform_videos_prefetch"
        ) as mock_prefetch:
            SynchronizedDataset(
                dataset=dataset_mock,
                frequency=30,
                data_types=None,
                dataset_description=dataset_description,
                prefetch_videos=True,
            )

            mock_prefetch.assert_called_once()

    def test_prefetch_videos_enabled_with_cache(
        self, dataset_mock, dataset_description, mock_auth_requests, tmp_path
    ):
        """Test that prefetch is skipped when cache exists."""
        # Create cache directories for all recordings
        for rec in dataset_mock.recordings:
            cache_path = dataset_mock.cache_dir / f"{rec['id']}" / "30Hz"
            cache_path.mkdir(parents=True, exist_ok=True)

        with patch.object(
            SynchronizedDataset, "_perform_videos_prefetch"
        ) as mock_prefetch:
            SynchronizedDataset(
                dataset=dataset_mock,
                frequency=30,
                data_types=None,
                dataset_description=dataset_description,
                prefetch_videos=True,
            )

            mock_prefetch.assert_not_called()

    def test_prefetch_videos_partial_cache(
        self, dataset_mock, dataset_description, mock_auth_requests, tmp_path
    ):
        """Test that prefetch runs if only some recordings are cached."""
        # Create cache for only first recording
        cache_path = (
            dataset_mock.cache_dir / f"{dataset_mock.recordings[0]['id']}" / "30Hz"
        )
        cache_path.mkdir(parents=True, exist_ok=True)

        with patch.object(
            SynchronizedDataset, "_perform_videos_prefetch"
        ) as mock_prefetch:
            SynchronizedDataset(
                dataset=dataset_mock,
                frequency=30,
                data_types=None,
                dataset_description=dataset_description,
                prefetch_videos=True,
            )

            mock_prefetch.assert_called_once()

    def test_max_workers_parameter(self, dataset_mock, dataset_description):
        """Test that max_workers parameter is used in prefetch."""
        with patch.object(
            SynchronizedDataset, "_perform_videos_prefetch"
        ) as mock_prefetch:
            SynchronizedDataset(
                dataset=dataset_mock,
                frequency=30,
                data_types=None,
                dataset_description=dataset_description,
                prefetch_videos=True,
                max_workers=8,
            )

            mock_prefetch.assert_called_once_with(max_workers=8)

    def test_slice_does_not_prefetch(self, synced_dataset, mock_auth_requests):
        """Test that slicing creates a new dataset without prefetching."""
        sliced = synced_dataset[0:1]

        # The sliced dataset should not have prefetch_videos enabled
        assert sliced._prefetch_videos is False

    def test_getitem_with_instance_info(self, synced_dataset, mock_auth_requests):
        """Test that getitem preserves instance information."""
        recording = synced_dataset[0]

        assert recording.instance == 1
        assert recording.robot_id == "robot1"

    def test_nested_iteration(
        self, synced_dataset, mock_auth_requests, mock_wget_download
    ):
        """Test nested iteration through dataset and recordings."""
        total_frames = 0

        for recording in synced_dataset:
            for frame in recording:
                total_frames += 1

        # Each recording has 2 frames
        assert total_frames == 4

    def test_mixed_access_patterns(self, synced_dataset, mock_auth_requests):
        """Test mixing iteration and indexing."""
        # Access by index
        rec0 = synced_dataset[0]

        # Then iterate
        recordings = list(synced_dataset)

        # First recording should be same cached instance
        assert recordings[0] is rec0

        # Access by negative index
        rec_last = synced_dataset[-1]
        assert recordings[-1] is rec_last

    def test_empty_data_types_list(self, dataset_mock, dataset_description):
        """Test initialization with empty data types list."""
        synced = SynchronizedDataset(
            dataset=dataset_mock,
            frequency=30,
            data_types=[],
            dataset_description=dataset_description,
        )

        assert synced.data_types == []

    def test_multiple_data_types(self, dataset_mock, dataset_description):
        """Test initialization with multiple data types."""
        data_types = [
            DataType.RGB_IMAGE,
            DataType.DEPTH_IMAGE,
            DataType.JOINT_POSITIONS,
        ]
        synced = SynchronizedDataset(
            dataset=dataset_mock,
            frequency=30,
            data_types=data_types,
            dataset_description=dataset_description,
        )

        assert synced.data_types == data_types

    def test_cache_independence_between_instances(
        self, dataset_mock, dataset_description, mock_auth_requests
    ):
        """Test that different SynchronizedDataset instances have independent caches."""
        synced1 = SynchronizedDataset(
            dataset=dataset_mock,
            frequency=30,
            data_types=None,
            dataset_description=dataset_description,
        )

        synced2 = SynchronizedDataset(
            dataset=dataset_mock,
            frequency=30,
            data_types=None,
            dataset_description=dataset_description,
        )

        # Access in first instance
        rec1 = synced1[0]

        # Cache should be independent
        assert 0 in synced1._synced_recording_cache
        assert 0 not in synced2._synced_recording_cache

        # Access in second instance
        rec2 = synced2[0]

        # Should be different objects
        assert rec1 is not rec2
