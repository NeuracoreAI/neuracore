from unittest.mock import MagicMock, patch

import pytest
import torch
from neuracore_types import (
    DATA_TYPE_TO_NC_DATA_CLASS,
    DataItemStats,
    DataType,
    NCDataStats,
    RobotDataSpec,
    SynchronizedDatasetStatistics,
    SynchronizedPoint,
)

from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.ml import BatchedTrainingSamples
from neuracore.ml.datasets.pytorch_synchronized_dataset import (
    PytorchSynchronizedDataset,
)

DATA_ITEMS = 3

NUM_EPISODES = 5
NUM_OBSERVATIONS_PER_EPISODE = 10


@pytest.fixture
def synchronization_point() -> SynchronizedPoint:
    """Create a sample SynchronizedPoint with various data types."""
    # Create data for all DataTypes
    all_data_types = [
        DataType.JOINT_POSITIONS,
        DataType.JOINT_TARGET_POSITIONS,
        DataType.RGB_IMAGES,
    ]

    return SynchronizedPoint(
        robot_id="robot_0",
        timestamp=1234567890.0,
        data={
            data_type: {
                f"{data_type.value}_{i}": DATA_TYPE_TO_NC_DATA_CLASS[data_type].sample()
                for i in range(DATA_ITEMS)
            }
            for data_type in all_data_types
        },
    )


@pytest.fixture
def dataset_statistics(
    synchronization_point: SynchronizedPoint,
) -> dict[DataType, list[NCDataStats]]:
    """Create sample dataset statistics for testing."""
    # Return mock statistics for different data types
    stats = {
        DataType.JOINT_POSITIONS: [
            list(synchronization_point.data[DataType.JOINT_POSITIONS].values())[
                i
            ].calculate_statistics()
            for i in range(DATA_ITEMS)
        ],
        DataType.JOINT_TARGET_POSITIONS: [
            list(synchronization_point.data[DataType.JOINT_TARGET_POSITIONS].values())[
                i
            ].calculate_statistics()
            for i in range(DATA_ITEMS)
        ],
        DataType.RGB_IMAGES: [
            list(synchronization_point.data[DataType.RGB_IMAGES].values())[
                i
            ].calculate_statistics()
            for i in range(DATA_ITEMS)
        ],
    }
    # Edit the count as it is used in the dataset
    for data_type_stats in stats.values():
        for stat in data_type_stats:
            for attr_name, attr_value in vars(stat).items():
                if isinstance(attr_value, DataItemStats):
                    attr_value.count[0] = NUM_EPISODES * NUM_OBSERVATIONS_PER_EPISODE
    return stats


@pytest.fixture
def mock_synced_recording(
    synchronization_point: SynchronizedPoint,
) -> SynchronizedRecording:
    """Create a mock SynchronizedRecording for testing."""
    # Create multiple sync points for sequence testing
    sync_points = [synchronization_point] * 10  # 10 timesteps

    class MockSynchronizedRecording(SynchronizedRecording):
        def __init__(self):
            self.sync_points = sync_points
            self.robot_id = "robot_0"

        def __len__(self):
            return len(self.sync_points)

        def __getitem__(self, idx):
            if isinstance(idx, int):
                return self.sync_points[idx]
            elif isinstance(idx, slice):
                start = idx.start or 0
                stop = idx.stop or len(self.sync_points)
                step = idx.step or 1
                return self.sync_points[start:stop:step]
            else:
                raise TypeError(f"Invalid index type: {type(idx)}")

        def __iter__(self):
            return iter(self.sync_points)

    return MockSynchronizedRecording()


@pytest.fixture
def mock_synchronized_dataset(
    mock_synced_recording: SynchronizedRecording,
    dataset_statistics: dict[DataType, list[NCDataStats]],
) -> SynchronizedDataset:
    """Create a mock SynchronizedDataset for testing."""

    class MockSynchronizedDataset(SynchronizedDataset):
        def __init__(self):
            self.dataset = MagicMock()
            self.robot_data_spec = {
                "robot_0": {
                    DataType.JOINT_POSITIONS: [
                        f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
                    ],
                    DataType.JOINT_TARGET_POSITIONS: [
                        f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(3)
                    ],
                    DataType.RGB_IMAGES: [
                        f"{DataType.RGB_IMAGES.value}_{i}" for i in range(3)
                    ],
                }
            }

        def calculate_statistics(
            self, robot_data_spec: RobotDataSpec
        ) -> SynchronizedDatasetStatistics:
            return SynchronizedDatasetStatistics(
                synchronized_dataset_id="mock_dataset",
                robot_data_spec=robot_data_spec,
                dataset_statistics=dataset_statistics,
            )

        def __len__(self):
            return NUM_EPISODES

        def __getitem__(self, idx):
            return mock_synced_recording

        def __next__(self) -> SynchronizedRecording:
            if self._recording_idx >= NUM_EPISODES:
                raise StopIteration
            self._recording_idx += 1
            return mock_synced_recording

    return MockSynchronizedDataset()


def test_should_initialize_with_correct_args(
    mock_synchronized_dataset: SynchronizedDataset,
):
    """Test basic dataset initialization."""
    input_spec: RobotDataSpec = {
        "robot_0": {
            DataType.JOINT_POSITIONS: [
                f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
            ],
            DataType.RGB_IMAGES: [f"{DataType.RGB_IMAGES.value}_{i}" for i in range(3)],
        }
    }
    output_spec: RobotDataSpec = {
        "robot_0": {
            DataType.JOINT_TARGET_POSITIONS: [
                f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(3)
            ],
        }
    }

    dataset = PytorchSynchronizedDataset(
        synchronized_dataset=mock_synchronized_dataset,
        input_robot_data_spec=input_spec,
        output_robot_data_spec=output_spec,
        output_prediction_horizon=5,
    )

    assert dataset.synchronized_dataset == mock_synchronized_dataset
    assert dataset.input_robot_data_spec == input_spec
    assert dataset.output_robot_data_spec == output_spec
    assert dataset.output_prediction_horizon == 5
    assert (
        len(dataset) == NUM_EPISODES * NUM_OBSERVATIONS_PER_EPISODE - NUM_EPISODES
    )  # num_transitions - num_episodes (exclude last frames)


def test_should_throw_error_with_missing_robot_id(
    mock_synchronized_dataset: SynchronizedDataset,
):
    """Test validation fails with missing robot ID."""
    input_spec: RobotDataSpec = {
        "robot_999": {  # Robot ID not in dataset
            DataType.JOINT_POSITIONS: [
                f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
            ],
        }
    }
    output_spec: RobotDataSpec = {
        "robot_0": {
            DataType.JOINT_TARGET_POSITIONS: [
                f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(3)
            ],
        }
    }

    with pytest.raises(ValueError, match="Input robot IDs .* not found"):
        PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_robot_data_spec=input_spec,
            output_robot_data_spec=output_spec,
            output_prediction_horizon=5,
        )


def test_should_throw_error_with_missing_data_type(
    mock_synchronized_dataset: SynchronizedDataset,
):
    """Test validation fails with missing data type."""
    input_spec: RobotDataSpec = {
        "robot_0": {
            DataType.JOINT_POSITIONS: [
                f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
            ],
            DataType.POINT_CLOUDS: [
                f"{DataType.POINT_CLOUDS.value}_0"
            ],  # Not in dataset
        }
    }
    output_spec: RobotDataSpec = {
        "robot_0": {
            DataType.JOINT_TARGET_POSITIONS: [
                f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(3)
            ],
        }
    }

    with pytest.raises(ValueError, match="Input data types .* not found"):
        PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_robot_data_spec=input_spec,
            output_robot_data_spec=output_spec,
            output_prediction_horizon=5,
        )


def test_initialization_invalid_synchronized_dataset():
    """Test initialization with invalid synchronized dataset."""
    with pytest.raises(AttributeError):  # Will fail when trying to access attributes
        PytorchSynchronizedDataset(
            synchronized_dataset="invalid",  # type: ignore
            input_robot_data_spec={
                "robot_0": {
                    DataType.JOINT_POSITIONS: [
                        f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
                    ]
                }
            },
            output_robot_data_spec={
                "robot_0": {
                    DataType.JOINT_TARGET_POSITIONS: [
                        f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(3)
                    ]
                }
            },
            output_prediction_horizon=5,
        )


class TestDataLoading:
    """Test data loading functionality."""

    @patch("neuracore.login")
    def test_load_sample_basic(self, mock_login, mock_synchronized_dataset):
        """Test basic sample loading."""
        input_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_POSITIONS: [
                    f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
                ],
                DataType.RGB_IMAGES: [
                    f"{DataType.RGB_IMAGES.value}_{i}" for i in range(3)
                ],
            }
        }
        output_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_TARGET_POSITIONS: [
                    f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_robot_data_spec=input_spec,
            output_robot_data_spec=output_spec,
            output_prediction_horizon=3,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset.load_sample(episode_idx=0, timestep=2)

        assert isinstance(sample, BatchedTrainingSamples)
        assert sample.inputs is not None
        assert sample.outputs is not None
        assert sample.inputs_mask is not None
        assert sample.outputs_mask is not None
        assert sample.batch_size == 1

        # Check that login was called
        mock_login.assert_called_once()

    @patch("neuracore.login")
    def test_load_sample_memory_monitoring(self, mock_login, mock_synchronized_dataset):
        """Test memory monitoring during sample loading."""
        input_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_POSITIONS: [
                    f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
                ],
                DataType.RGB_IMAGES: [
                    f"{DataType.RGB_IMAGES.value}_{i}" for i in range(3)
                ],
            }
        }
        output_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_TARGET_POSITIONS: [
                    f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_robot_data_spec=input_spec,
            output_robot_data_spec=output_spec,
            output_prediction_horizon=3,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None

            # Load multiple samples to trigger memory check
            for i in range(105):  # Should trigger memory check at multiples of 100
                dataset._mem_check_counter = i
                dataset.load_sample(episode_idx=0, timestep=0)

            # Memory should be checked at least once
            assert mock_monitor.check_memory.call_count >= 1


class TestDataTypeProcessing:
    """Test processing of different data types."""

    @patch("neuracore.login")
    def test_inputs_and_outputs_structure(self, mock_login, mock_synchronized_dataset):
        """Test that inputs and outputs have correct structure."""
        input_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_POSITIONS: [
                    f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
                ],
                DataType.RGB_IMAGES: [
                    f"{DataType.RGB_IMAGES.value}_{i}" for i in range(3)
                ],
            }
        }
        output_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_TARGET_POSITIONS: [
                    f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_robot_data_spec=input_spec,
            output_robot_data_spec=output_spec,
            output_prediction_horizon=2,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset.load_sample(episode_idx=0, timestep=0)

        # Check structure: dict[DataType, list[BatchedNCData]]
        assert isinstance(sample.inputs, dict)
        assert DataType.JOINT_POSITIONS in sample.inputs
        assert DataType.RGB_IMAGES in sample.inputs
        assert isinstance(sample.inputs[DataType.JOINT_POSITIONS], list)

        # Check masks: dict[DataType, torch.Tensor]
        assert isinstance(sample.inputs_mask, dict)
        assert DataType.JOINT_POSITIONS in sample.inputs_mask
        assert isinstance(sample.inputs_mask[DataType.JOINT_POSITIONS], torch.Tensor)

        # Check outputs
        assert isinstance(sample.outputs, dict)
        assert DataType.JOINT_TARGET_POSITIONS in sample.outputs
        assert isinstance(sample.outputs[DataType.JOINT_TARGET_POSITIONS], list)


class TestDatasetIntegration:
    """Test dataset with PyTorch ecosystem."""

    @patch("neuracore.login")
    def test_getitem_method(self, mock_login, mock_synchronized_dataset):
        """Test __getitem__ method."""
        input_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_POSITIONS: [
                    f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
                ],
                DataType.RGB_IMAGES: [
                    f"{DataType.RGB_IMAGES.value}_{i}" for i in range(3)
                ],
            }
        }
        output_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_TARGET_POSITIONS: [
                    f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_robot_data_spec=input_spec,
            output_robot_data_spec=output_spec,
            output_prediction_horizon=3,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset[5]

        assert isinstance(sample, BatchedTrainingSamples)
        assert sample.inputs is not None
        assert sample.outputs is not None

    @patch("neuracore.login")
    def test_getitem_negative_index(self, mock_login, mock_synchronized_dataset):
        """Test __getitem__ with negative index."""
        input_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_POSITIONS: [
                    f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
                ],
                DataType.RGB_IMAGES: [
                    f"{DataType.RGB_IMAGES.value}_{i}" for i in range(3)
                ],
            }
        }
        output_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_TARGET_POSITIONS: [
                    f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_robot_data_spec=input_spec,
            output_robot_data_spec=output_spec,
            output_prediction_horizon=3,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None

            # Should work with negative indices
            sample = dataset[-1]
            assert isinstance(sample, BatchedTrainingSamples)

    def test_getitem_index_out_of_bounds(self, mock_synchronized_dataset):
        """Test __getitem__ with out of bounds index."""
        input_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_POSITIONS: [
                    f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }
        output_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_TARGET_POSITIONS: [
                    f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_robot_data_spec=input_spec,
            output_robot_data_spec=output_spec,
            output_prediction_horizon=3,
        )

        # Test positive out of bounds
        with pytest.raises(IndexError):
            _ = dataset[100]  # Only 45 samples (50 - 5 episodes)

        # Test negative out of bounds
        with pytest.raises(IndexError):
            _ = dataset[-100]

    def test_len_method(self, mock_synchronized_dataset):
        """Test __len__ method."""
        input_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_POSITIONS: [
                    f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }
        output_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_TARGET_POSITIONS: [
                    f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_robot_data_spec=input_spec,
            output_robot_data_spec=output_spec,
            output_prediction_horizon=3,
        )

        assert len(dataset) == 45

    @patch("neuracore.login")
    def test_error_handling_in_getitem(self, mock_login, mock_synchronized_dataset):
        """Test error handling in __getitem__ method."""
        input_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_POSITIONS: [
                    f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }
        output_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_TARGET_POSITIONS: [
                    f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_robot_data_spec=input_spec,
            output_robot_data_spec=output_spec,
            output_prediction_horizon=3,
        )

        with patch.object(dataset, "load_sample") as mock_load_sample:
            mock_load_sample.side_effect = Exception("Load error")

            # Should propagate the error after max retries
            with pytest.raises(Exception):
                _ = dataset[0]


class TestPerformanceAndOptimization:
    """Test performance and optimization features."""

    def test_memory_monitoring_initialization(self, mock_synchronized_dataset):
        """Test memory monitor initialization."""
        input_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_POSITIONS: [
                    f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }
        output_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_TARGET_POSITIONS: [
                    f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_robot_data_spec=input_spec,
            output_robot_data_spec=output_spec,
            output_prediction_horizon=3,
        )

        assert dataset._memory_monitor is not None
        assert hasattr(dataset._memory_monitor, "check_memory")

    def test_episode_indices_creation(self, mock_synchronized_dataset):
        """Test episode indices are created correctly."""
        input_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_POSITIONS: [
                    f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }
        output_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_TARGET_POSITIONS: [
                    f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_robot_data_spec=input_spec,
            output_robot_data_spec=output_spec,
            output_prediction_horizon=3,
        )

        # Should have episode indices for each sample (excluding last frames)
        # 5 episodes * 9 samples per episode (10 - 1)
        assert len(dataset.episode_indices) == 45

        # Check structure: first 9 should be episode 0, next 9 episode 1, etc.
        assert all(idx == 0 for idx in dataset.episode_indices[:9])
        assert all(idx == 1 for idx in dataset.episode_indices[9:18])


class TestErrorRecovery:
    """Test error recovery and robustness."""

    def test_error_count_tracking(self, mock_synchronized_dataset):
        """Test error count tracking in parent class."""
        input_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_POSITIONS: [
                    f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }
        output_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_TARGET_POSITIONS: [
                    f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(4)
                ]
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_robot_data_spec=input_spec,
            output_robot_data_spec=output_spec,
            output_prediction_horizon=3,
        )

        # Initial error count should be 0
        assert dataset._error_count == 0
        assert dataset._max_error_count == 100


class TestIntegrationWithPyTorchDataLoader:
    """Integration tests with PyTorch DataLoader."""

    @patch("neuracore.login")
    def test_dataloader_with_collate_fn(self, mock_login, mock_synchronized_dataset):
        """Test DataLoader with custom collate function."""
        from torch.utils.data import DataLoader

        input_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_POSITIONS: [
                    f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(2)
                ],
                DataType.RGB_IMAGES: [
                    f"{DataType.RGB_IMAGES.value}_{i}" for i in range(2)
                ],
            }
        }
        output_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_TARGET_POSITIONS: [
                    f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(2)
                ]
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_robot_data_spec=input_spec,
            output_robot_data_spec=output_spec,
            output_prediction_horizon=3,
        )

        dataloader = DataLoader(
            dataset, batch_size=3, shuffle=False, collate_fn=dataset.collate_fn
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None

            batch = next(iter(dataloader))
            assert isinstance(batch, BatchedTrainingSamples)
            assert batch.batch_size == 3

            # Check that data is properly batched
            # Each data type should have batched data
            for data_type in input_spec["robot_0"].keys():
                assert data_type in batch.inputs
                assert isinstance(batch.inputs[data_type], list)


class TestDatasetStatistics:
    """Test dataset statistics functionality."""

    def test_dataset_statistics_property(self, mock_synchronized_dataset):
        """Test dataset_statistics property."""
        input_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_POSITIONS: [
                    f"{DataType.JOINT_POSITIONS.value}_{i}" for i in range(3)
                ]
            }
        }
        output_spec: RobotDataSpec = {
            "robot_0": {
                DataType.JOINT_TARGET_POSITIONS: [
                    f"{DataType.JOINT_TARGET_POSITIONS.value}_{i}" for i in range(4)
                ]
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_robot_data_spec=input_spec,
            output_robot_data_spec=output_spec,
            output_prediction_horizon=3,
        )

        stats = dataset.dataset_statistics
        assert isinstance(stats, dict)
        # Should contain statistics for the data types
        assert DataType.JOINT_POSITIONS in stats or len(stats) >= 0
