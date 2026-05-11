import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from neuracore_types import (
    DATA_TYPE_TO_NC_DATA_CLASS,
    CrossEmbodimentDescription,
    DataItemStats,
    DataType,
    NCDataStats,
    SynchronizedDatasetStatistics,
    SynchronizedPoint,
)
from omegaconf import OmegaConf

from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.data.synced_recording import SynchronizedRecording
from neuracore.core.utils.robot_data_spec_utils import (
    convert_omegaconf_to_cross_embodiment_description,
    merge_cross_embodiment_description,
)
from neuracore.ml import BatchedTrainingSamples
from neuracore.ml.datasets.pytorch_synchronized_dataset import (
    PytorchSynchronizedDataset,
)
from neuracore.ml.preprocessing.methods.resize_pad import ResizePad
from neuracore.ml.utils.preprocessing_utils import PreprocessingConfiguration

DATA_ITEMS = 3

NUM_EPISODES = 5
NUM_OBSERVATIONS_PER_EPISODE = 10
ROBOT_ID = "11111111-1111-1111-1111-111111111111"
MISSING_ROBOT_ID = "22222222-2222-2222-2222-222222222222"


def _indexed_names(data_type: DataType, count: int) -> dict[int, str]:
    return {i: f"{data_type.value}_{i}" for i in range(count)}


def _full_data_spec() -> dict[DataType, dict[int, str]]:
    return {
        DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
        DataType.JOINT_TARGET_POSITIONS: _indexed_names(
            DataType.JOINT_TARGET_POSITIONS, 3
        ),
        DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
    }


def _default_preprocessing_config() -> PreprocessingConfiguration:
    return {
        DataType.RGB_IMAGES: [ResizePad(size=(224, 224))],
        DataType.DEPTH_IMAGES: [ResizePad(size=(224, 224))],
    }


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
        robot_id=ROBOT_ID,
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
) -> dict[str, dict[DataType, list[NCDataStats]]]:
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
    return {
        "input": {
            DataType.JOINT_POSITIONS: stats[DataType.JOINT_POSITIONS],
            DataType.RGB_IMAGES: stats[DataType.RGB_IMAGES],
        },
        "output": {
            DataType.JOINT_TARGET_POSITIONS: stats[DataType.JOINT_TARGET_POSITIONS],
        },
    }


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
            self.robot_id = ROBOT_ID

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
    dataset_statistics: dict[str, dict[DataType, list[NCDataStats]]],
) -> SynchronizedDataset:
    """Create a mock SynchronizedDataset for testing."""

    class MockSynchronizedDataset(SynchronizedDataset):
        def __init__(self):
            self.id = "mock_dataset"
            self.dataset = MagicMock()
            self.dataset.data_types = [
                DataType.JOINT_POSITIONS,
                DataType.JOINT_TARGET_POSITIONS,
                DataType.RGB_IMAGES,
            ]
            self.dataset.get_full_embodiment_description.side_effect = (
                lambda robot_id: (
                    _full_data_spec()
                    if robot_id == ROBOT_ID
                    else (_ for _ in ()).throw(
                        ValueError(f"Input robot IDs [{robot_id}] not found")
                    )
                )
            )
            self.cross_embodiment_description = {
                ROBOT_ID: {
                    DataType.JOINT_POSITIONS: _indexed_names(
                        DataType.JOINT_POSITIONS, 3
                    ),
                    DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                        DataType.JOINT_TARGET_POSITIONS, 3
                    ),
                    DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
                }
            }

        def calculate_statistics(
            self,
            input_cross_embodiment_description: CrossEmbodimentDescription,
            output_cross_embodiment_description: CrossEmbodimentDescription,
        ) -> SynchronizedDatasetStatistics:
            return SynchronizedDatasetStatistics(
                synchronized_dataset_id="mock_dataset",
                input_cross_embodiment_description=input_cross_embodiment_description,
                output_cross_embodiment_description=output_cross_embodiment_description,
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


def _cacheable_cross_embodiment_description(
    cross_embodiment_description: CrossEmbodimentDescription,
):
    if hasattr(cross_embodiment_description, "model_dump"):
        return cross_embodiment_description.model_dump(mode="json")
    return cross_embodiment_description


def _stats_cache_path(cache_root, sync_id, input_spec, output_spec):
    spec_key = json.dumps(
        {
            "input_cross_embodiment_description": (
                _cacheable_cross_embodiment_description(input_spec)
            ),
            "output_cross_embodiment_description": (
                _cacheable_cross_embodiment_description(output_spec)
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    spec_hash = hashlib.sha256(spec_key.encode("utf-8")).hexdigest()[:12]
    return cache_root / "dataset_cache" / f"{sync_id}_statistics_{spec_hash}.json"


@pytest.fixture
def synchronization_point_with_depth() -> SynchronizedPoint:
    """Create a sample SynchronizedPoint including depth data."""
    all_data_types = [
        DataType.JOINT_POSITIONS,
        DataType.JOINT_TARGET_POSITIONS,
        DataType.RGB_IMAGES,
        DataType.DEPTH_IMAGES,
    ]

    return SynchronizedPoint(
        robot_id=ROBOT_ID,
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
def dataset_statistics_with_depth(
    synchronization_point_with_depth: SynchronizedPoint,
) -> dict[str, dict[DataType, list[NCDataStats]]]:
    """Create sample dataset statistics for tests including depth."""
    stats = {
        DataType.JOINT_POSITIONS: [
            list(
                synchronization_point_with_depth.data[DataType.JOINT_POSITIONS].values()
            )[i].calculate_statistics()
            for i in range(DATA_ITEMS)
        ],
        DataType.JOINT_TARGET_POSITIONS: [
            list(
                synchronization_point_with_depth.data[
                    DataType.JOINT_TARGET_POSITIONS
                ].values()
            )[i].calculate_statistics()
            for i in range(DATA_ITEMS)
        ],
        DataType.RGB_IMAGES: [
            list(synchronization_point_with_depth.data[DataType.RGB_IMAGES].values())[
                i
            ].calculate_statistics()
            for i in range(DATA_ITEMS)
        ],
        DataType.DEPTH_IMAGES: [
            list(synchronization_point_with_depth.data[DataType.DEPTH_IMAGES].values())[
                i
            ].calculate_statistics()
            for i in range(DATA_ITEMS)
        ],
    }
    for data_type_stats in stats.values():
        for stat in data_type_stats:
            for attr_name, attr_value in vars(stat).items():
                if isinstance(attr_value, DataItemStats):
                    attr_value.count[0] = NUM_EPISODES * NUM_OBSERVATIONS_PER_EPISODE
    return {
        "input": {
            DataType.JOINT_POSITIONS: stats[DataType.JOINT_POSITIONS],
            DataType.RGB_IMAGES: stats[DataType.RGB_IMAGES],
            DataType.DEPTH_IMAGES: stats[DataType.DEPTH_IMAGES],
        },
        "output": {
            DataType.JOINT_TARGET_POSITIONS: stats[DataType.JOINT_TARGET_POSITIONS],
        },
    }


@pytest.fixture
def mock_synced_recording_with_depth(
    synchronization_point_with_depth: SynchronizedPoint,
) -> SynchronizedRecording:
    """Create a mock recording including depth data."""
    sync_points = [synchronization_point_with_depth] * 10

    class MockSynchronizedRecording(SynchronizedRecording):
        def __init__(self):
            self.sync_points = sync_points
            self.robot_id = ROBOT_ID

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
def mock_synchronized_dataset_with_depth(
    mock_synced_recording_with_depth: SynchronizedRecording,
    dataset_statistics_with_depth: dict[str, dict[DataType, list[NCDataStats]]],
) -> SynchronizedDataset:
    """Create a mock synchronized dataset including depth data."""

    class MockSynchronizedDataset(SynchronizedDataset):
        def __init__(self):
            self.id = "mock_dataset"
            self.dataset = MagicMock()
            self.dataset.data_types = [
                DataType.JOINT_POSITIONS,
                DataType.JOINT_TARGET_POSITIONS,
                DataType.RGB_IMAGES,
                DataType.DEPTH_IMAGES,
            ]
            self.dataset.get_full_embodiment_description.side_effect = (
                lambda robot_id: (
                    {
                        **_full_data_spec(),
                        DataType.DEPTH_IMAGES: _indexed_names(DataType.DEPTH_IMAGES, 3),
                    }
                    if robot_id == ROBOT_ID
                    else (_ for _ in ()).throw(
                        ValueError(f"Input robot IDs [{robot_id}] not found")
                    )
                )
            )
            self.cross_embodiment_description = {
                ROBOT_ID: {
                    DataType.JOINT_POSITIONS: _indexed_names(
                        DataType.JOINT_POSITIONS, 3
                    ),
                    DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                        DataType.JOINT_TARGET_POSITIONS, 3
                    ),
                    DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
                    DataType.DEPTH_IMAGES: _indexed_names(DataType.DEPTH_IMAGES, 3),
                }
            }

        def calculate_statistics(
            self,
            input_cross_embodiment_description: CrossEmbodimentDescription,
            output_cross_embodiment_description: CrossEmbodimentDescription,
        ) -> SynchronizedDatasetStatistics:
            return SynchronizedDatasetStatistics(
                synchronized_dataset_id="mock_dataset",
                input_cross_embodiment_description=input_cross_embodiment_description,
                output_cross_embodiment_description=output_cross_embodiment_description,
                dataset_statistics=dataset_statistics_with_depth,
            )

        def __len__(self):
            return NUM_EPISODES

        def __getitem__(self, idx):
            return mock_synced_recording_with_depth

        def __next__(self) -> SynchronizedRecording:
            if self._recording_idx >= NUM_EPISODES:
                raise StopIteration
            self._recording_idx += 1
            return mock_synced_recording_with_depth

    return MockSynchronizedDataset()


def test_should_initialize_with_correct_args(
    mock_synchronized_dataset: SynchronizedDataset,
):
    """Test basic dataset initialization."""
    input_embodiment_description: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
            DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
        }
    }
    output_embodiment_description: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                DataType.JOINT_TARGET_POSITIONS, 3
            ),
        }
    }

    dataset = PytorchSynchronizedDataset(
        synchronized_dataset=mock_synchronized_dataset,
        input_cross_embodiment_description=input_embodiment_description,
        output_cross_embodiment_description=output_embodiment_description,
        output_prediction_horizon=5,
        input_preprocessing_config=_default_preprocessing_config(),
        output_preprocessing_config=_default_preprocessing_config(),
    )

    assert dataset.synchronized_dataset == mock_synchronized_dataset
    assert dataset.input_cross_embodiment_description == input_embodiment_description
    assert dataset.output_cross_embodiment_description == output_embodiment_description
    assert dataset.output_prediction_horizon == 5
    assert (
        len(dataset) == NUM_EPISODES * NUM_OBSERVATIONS_PER_EPISODE - NUM_EPISODES
    )  # num_transitions - num_episodes (exclude last frames)


def test_should_throw_error_with_missing_robot_id(
    mock_synchronized_dataset: SynchronizedDataset,
):
    """Test validation fails with missing robot ID."""
    input_spec: CrossEmbodimentDescription = {
        MISSING_ROBOT_ID: {  # Robot ID not in dataset
            DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
        }
    }
    output_spec: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                DataType.JOINT_TARGET_POSITIONS, 3
            ),
        }
    }

    with pytest.raises(ValueError, match="Input robot IDs .* not found"):
        PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=5,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )


def test_should_throw_error_with_missing_data_type(
    mock_synchronized_dataset: SynchronizedDataset,
):
    """Test validation fails with missing data type."""
    input_spec: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
            DataType.POINT_CLOUDS: _indexed_names(DataType.POINT_CLOUDS, 1),
        }
    }
    output_spec: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                DataType.JOINT_TARGET_POSITIONS, 3
            ),
        }
    }

    with pytest.raises(ValueError, match="Input data type .* is not present"):
        PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=5,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )


def test_initialization_invalid_synchronized_dataset():
    """Test initialization with invalid synchronized dataset."""
    with pytest.raises(AttributeError):  # Will fail when trying to access attributes
        PytorchSynchronizedDataset(
            synchronized_dataset="invalid",  # type: ignore
            input_cross_embodiment_description={
                ROBOT_ID: {
                    DataType.JOINT_POSITIONS: _indexed_names(
                        DataType.JOINT_POSITIONS, 3
                    )
                }
            },
            output_cross_embodiment_description={
                ROBOT_ID: {
                    DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                        DataType.JOINT_TARGET_POSITIONS, 3
                    )
                }
            },
            output_prediction_horizon=5,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )


def test_merge_cross_embodiment_description_uses_dict_values_in_order():
    input_spec: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: {
                10: "joint_a",
                20: "joint_b",
            }
        }
    }
    output_spec: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: {
                99: "joint_c",
            },
            DataType.JOINT_TARGET_POSITIONS: {
                0: "target_a",
            },
        }
    }

    assert merge_cross_embodiment_description(input_spec, output_spec) == {
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: ["joint_a", "joint_b", "joint_c"],
            DataType.JOINT_TARGET_POSITIONS: ["target_a"],
        }
    }


def test_merge_cross_embodiment_description_deduplicates_by_name_not_index():
    input_spec: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.RGB_IMAGES: {
                1: "front_camera",
                5: "wrist_camera",
            }
        }
    }
    output_spec: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.RGB_IMAGES: {
                3: "wrist_camera",
                9: "side_camera",
            }
        }
    }

    assert merge_cross_embodiment_description(input_spec, output_spec) == {
        ROBOT_ID: {
            DataType.RGB_IMAGES: [
                "front_camera",
                "wrist_camera",
                "side_camera",
            ]
        }
    }


def test_merge_cross_embodiment_description_preserves_robot_order():
    other_robot_id = "33333333-3333-3333-3333-333333333333"
    input_spec: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: {
                0: "joint_0",
            }
        }
    }
    output_spec: CrossEmbodimentDescription = {
        other_robot_id: {
            DataType.JOINT_TARGET_POSITIONS: {
                0: "target_0",
            }
        }
    }

    assert list(merge_cross_embodiment_description(input_spec, output_spec)) == [
        ROBOT_ID,
        other_robot_id,
    ]


def test_merge_cross_embodiment_description_handles_omegaconf_dict_values():
    input_cfg = OmegaConf.create({
        ROBOT_ID: {
            "JOINT_POSITIONS": {
                0: "vx300s_right/wrist_angle",
                1: "vx300s_left/waist",
            },
            "RGB_IMAGES": {
                0: "rgb_angle",
            },
        }
    })
    output_cfg = OmegaConf.create({
        ROBOT_ID: {
            "JOINT_POSITIONS": {
                0: "vx300s_right/wrist_angle",
                1: "vx300s_left/waist",
            }
        }
    })

    input_spec = convert_omegaconf_to_cross_embodiment_description(input_cfg)
    output_spec = convert_omegaconf_to_cross_embodiment_description(output_cfg)

    assert merge_cross_embodiment_description(input_spec, output_spec) == {
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: [
                "vx300s_right/wrist_angle",
                "vx300s_left/waist",
            ],
            DataType.RGB_IMAGES: ["rgb_angle"],
        }
    }


class TestDataLoading:
    """Test data loading functionality."""

    @patch("neuracore.login")
    def test_load_sample_basic(self, mock_login, mock_synchronized_dataset):
        """Test basic sample loading."""
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
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
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None

            # Load multiple samples to trigger memory check
            for i in range(105):  # Should trigger memory check at multiples of 100
                dataset._mem_check_counter = i
                dataset.load_sample(episode_idx=0, timestep=0)

            # Memory should be checked at least once
            assert mock_monitor.check_memory.call_count >= 1

    @patch("neuracore.login")
    def test_load_sample_applies_input_preprocessing(
        self, mock_login, mock_synchronized_dataset_with_depth
    ):
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
                DataType.DEPTH_IMAGES: _indexed_names(DataType.DEPTH_IMAGES, 3),
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }
        RGB_TEST_SHAPE = (123, 456)
        DEPTH_TEST_SHAPE = (789, 101)
        input_preprocessing_config = {
            DataType.RGB_IMAGES: [ResizePad(size=RGB_TEST_SHAPE)],
            DataType.DEPTH_IMAGES: [ResizePad(size=DEPTH_TEST_SHAPE)],
        }
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset_with_depth,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=input_preprocessing_config,
            output_preprocessing_config=_default_preprocessing_config(),
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset.load_sample(episode_idx=0, timestep=0)

        assert sample.inputs[DataType.RGB_IMAGES][0].frame.shape[-2:] == RGB_TEST_SHAPE
        assert (
            sample.inputs[DataType.DEPTH_IMAGES][0].frame.shape[-2:] == DEPTH_TEST_SHAPE
        )

    @patch("neuracore.login")
    def test_load_sample_applies_output_preprocessing(
        self, mock_login, mock_synchronized_dataset_with_depth
    ):
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
                DataType.DEPTH_IMAGES: _indexed_names(DataType.DEPTH_IMAGES, 3),
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
                DataType.DEPTH_IMAGES: _indexed_names(DataType.DEPTH_IMAGES, 3),
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                ),
            }
        }
        RGB_TEST_SHAPE = (160, 200)
        DEPTH_TEST_SHAPE = (180, 220)
        output_preprocessing_config = {
            DataType.RGB_IMAGES: [ResizePad(size=RGB_TEST_SHAPE)],
            DataType.DEPTH_IMAGES: [ResizePad(size=DEPTH_TEST_SHAPE)],
        }
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset_with_depth,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=output_preprocessing_config,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset.load_sample(episode_idx=0, timestep=0)

        assert sample.outputs[DataType.RGB_IMAGES][0].frame.shape[-2:] == RGB_TEST_SHAPE
        assert (
            sample.outputs[DataType.DEPTH_IMAGES][0].frame.shape[-2:]
            == DEPTH_TEST_SHAPE
        )


class TestDataTypeProcessing:
    """Test processing of different data types."""

    @patch("neuracore.login")
    def test_inputs_and_outputs_structure(self, mock_login, mock_synchronized_dataset):
        """Test that inputs and outputs have correct structure."""
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=2,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
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
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
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
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None

            # Should work with negative indices
            sample = dataset[-1]
            assert isinstance(sample, BatchedTrainingSamples)

    def test_getitem_index_out_of_bounds(self, mock_synchronized_dataset):
        """Test __getitem__ with out of bounds index."""
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )

        # Test positive out of bounds
        with pytest.raises(IndexError):
            _ = dataset[100]  # Only 45 samples (50 - 5 episodes)

        # Test negative out of bounds
        with pytest.raises(IndexError):
            _ = dataset[-100]

    def test_len_method(self, mock_synchronized_dataset):
        """Test __len__ method."""
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )

        assert len(dataset) == 45

    @patch("neuracore.login")
    def test_error_handling_in_getitem(self, mock_login, mock_synchronized_dataset):
        """Test error handling in __getitem__ method."""
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
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
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )

        assert dataset._memory_monitor is not None
        assert hasattr(dataset._memory_monitor, "check_memory")

    def test_episode_indices_creation(self, mock_synchronized_dataset):
        """Test episode indices are created correctly."""
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
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
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
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

        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 2),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 2),
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 2
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
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
            for data_type in input_spec[ROBOT_ID].keys():
                assert data_type in batch.inputs
                assert isinstance(batch.inputs[data_type], list)


class TestDatasetStatistics:
    """Test dataset statistics functionality."""

    def test_dataset_statistics_property(self, mock_synchronized_dataset):
        """Test dataset_statistics property."""
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )

        stats = dataset.dataset_statistics
        assert isinstance(stats, dict)
        assert DataType.JOINT_POSITIONS in stats["input"]

    def test_dataset_statistics_cache_hit(
        self, tmp_path, mock_synchronized_dataset, dataset_statistics, monkeypatch
    ):
        """Test cached statistics are loaded without recomputation."""
        import neuracore.ml.datasets.pytorch_synchronized_dataset as psd

        monkeypatch.setattr(psd, "DEFAULT_CACHE_DIR", tmp_path)
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }
        stats = SynchronizedDatasetStatistics(
            synchronized_dataset_id=mock_synchronized_dataset.id,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            dataset_statistics=dataset_statistics,
        )
        cache_path = _stats_cache_path(
            tmp_path, mock_synchronized_dataset.id, input_spec, output_spec
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(stats.model_dump(mode="json"), handle)

        mock_synchronized_dataset.calculate_statistics = MagicMock(
            side_effect=AssertionError("calculate_statistics should not be called")
        )

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )

        assert mock_synchronized_dataset.calculate_statistics.call_count == 0
        assert (
            dataset.synchronized_dataset_statistics.synchronized_dataset_id
            == mock_synchronized_dataset.id
        )

    def test_dataset_statistics_cache_miss_writes_cache(
        self, tmp_path, mock_synchronized_dataset, dataset_statistics, monkeypatch
    ):
        """Test cache miss computes stats and writes cache file."""
        import neuracore.ml.datasets.pytorch_synchronized_dataset as psd

        monkeypatch.setattr(psd, "DEFAULT_CACHE_DIR", tmp_path)
        input_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_spec: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }
        stats = SynchronizedDatasetStatistics(
            synchronized_dataset_id=mock_synchronized_dataset.id,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            dataset_statistics=dataset_statistics,
        )
        mock_synchronized_dataset.calculate_statistics = MagicMock(return_value=stats)

        PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_spec,
            output_cross_embodiment_description=output_spec,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )

        assert mock_synchronized_dataset.calculate_statistics.call_count == 1
        cache_path = _stats_cache_path(
            tmp_path, mock_synchronized_dataset.id, input_spec, output_spec
        )
        assert cache_path.exists()

        assert cache_path.exists()
