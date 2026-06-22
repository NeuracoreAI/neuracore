import hashlib
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from neuracore_types import (
    DATA_TYPE_TO_NC_DATA_CLASS,
    TARGET_OUTPUT_DATA_TYPES,
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
from neuracore.core.utils.embodiment_description_utils import (
    merge_cross_embodiment_description,
)
from neuracore.ml import BatchedTrainingSamples
from neuracore.ml.datasets.pytorch_synchronized_dataset import (
    PytorchSynchronizedDataset,
    _cacheable_cross_embodiment_description,
)
from neuracore.ml.preprocessing.methods.resize_pad import ResizePad
from neuracore.ml.utils.preprocessing_utils import PreprocessingConfiguration

DATA_ITEMS = 3

NUM_EPISODES = 5
NUM_OBSERVATIONS_PER_EPISODE = 10
ROBOT_ID = "11111111-1111-1111-1111-111111111111"
MISSING_ROBOT_ID = "22222222-2222-2222-2222-222222222222"


class ModelDumpCrossEmbodimentDescription:
    def model_dump(self, mode):
        assert mode == "json"
        return {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: OmegaConf.create({0: "joint_positions_0"})
            }
        }


def _indexed_names(data_type: DataType, count: int) -> dict[int, str]:
    return {i: f"{data_type.value}_{i}" for i in range(count)}


def _full_embodiment_description() -> dict[DataType, dict[int, str]]:
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


NON_TARGET_OUTPUT_DATA_TYPES = tuple(
    data_type for data_type in DataType if data_type not in TARGET_OUTPUT_DATA_TYPES
)


def _timestep_marker(data_type: DataType, timestep: int, item_index: int = 0) -> float:
    base = (list(DataType).index(data_type) + 1) * 10.0
    return base + float(timestep) + float(item_index)


def _create_nc_data_at_timestep(
    data_type: DataType, timestep: int, item_index: int = 0
):
    marker = _timestep_marker(data_type, timestep, item_index)
    if data_type in {
        DataType.JOINT_POSITIONS,
        DataType.JOINT_VELOCITIES,
        DataType.JOINT_TORQUES,
        DataType.JOINT_TARGET_POSITIONS,
        DataType.VISUAL_JOINT_POSITIONS,
    }:
        from neuracore_types.nc_data.joint_data import JointData

        return JointData(value=marker)
    if data_type in {
        DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
    }:
        from neuracore_types.nc_data.parallel_gripper_open_amount_data import (
            ParallelGripperOpenAmountData,
        )

        return ParallelGripperOpenAmountData(open_amount=marker)
    if data_type == DataType.END_EFFECTOR_POSES:
        from neuracore_types.nc_data.end_effector_pose_data import EndEffectorPoseData

        return EndEffectorPoseData(
            pose=np.array([marker, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )
    if data_type == DataType.POSES:
        from neuracore_types.nc_data.pose_data import PoseData

        return PoseData(
            pose=np.array([marker, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )
    if data_type == DataType.RGB_IMAGES:
        from neuracore_types.nc_data.camera_data import RGBCameraData

        channel_value = int(marker) % 256
        return RGBCameraData(
            extrinsics=np.eye(4, dtype=np.float16),
            intrinsics=np.eye(3, dtype=np.float16),
            frame=np.full((2, 2, 3), channel_value, dtype=np.uint8),
        )
    if data_type == DataType.DEPTH_IMAGES:
        from neuracore_types.nc_data.camera_data import DepthCameraData

        return DepthCameraData(
            extrinsics=np.eye(4, dtype=np.float32),
            intrinsics=np.eye(3, dtype=np.float32),
            frame=np.full((2, 2), marker, dtype=np.float32),
        )
    if data_type == DataType.POINT_CLOUDS:
        from neuracore_types.nc_data.point_cloud_data import PointCloudData

        return PointCloudData(
            points=np.array([[marker, 0.0, 0.0]], dtype=np.float16),
            rgb_points=np.zeros((1, 3), dtype=np.uint8),
            extrinsics=np.eye(4, dtype=np.float16),
            intrinsics=np.eye(3, dtype=np.float16),
        )
    if data_type == DataType.LANGUAGE:
        from neuracore_types.nc_data.language_data import LanguageData

        return LanguageData(text=str(marker))
    if data_type == DataType.CUSTOM_1D:
        from neuracore_types.nc_data.custom_1d_data import Custom1DData

        return Custom1DData(data=np.array([marker], dtype=np.float32))
    raise ValueError(f"Unhandled data type: {data_type}")


def _timestep_full_embodiment_description() -> dict[DataType, dict[int, str]]:
    return {data_type: _indexed_names(data_type, DATA_ITEMS) for data_type in DataType}


def _dataset_statistics_for_output_type(
    output_data_type: DataType,
) -> dict[str, dict[DataType, list[NCDataStats]]]:
    input_stats = [
        _create_nc_data_at_timestep(
            DataType.JOINT_POSITIONS, 0, i
        ).calculate_statistics()
        for i in range(DATA_ITEMS)
    ]
    output_stats = [
        _create_nc_data_at_timestep(output_data_type, 0, i).calculate_statistics()
        for i in range(DATA_ITEMS)
    ]
    for stats_list in (input_stats, output_stats):
        for stat in stats_list:
            for attr_value in vars(stat).values():
                if isinstance(attr_value, DataItemStats) and attr_value.count.size > 0:
                    attr_value.count[0] = NUM_EPISODES * NUM_OBSERVATIONS_PER_EPISODE
    return {
        "input": {DataType.JOINT_POSITIONS: input_stats},
        "output": {output_data_type: output_stats},
    }


def _extract_output_marker(batched_nc_data, data_type: DataType) -> float:
    if data_type in {
        DataType.JOINT_POSITIONS,
        DataType.JOINT_VELOCITIES,
        DataType.JOINT_TORQUES,
        DataType.JOINT_TARGET_POSITIONS,
        DataType.VISUAL_JOINT_POSITIONS,
    }:
        return float(batched_nc_data.value[0, 0, 0])
    if data_type in {
        DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
    }:
        return float(batched_nc_data.open_amount[0, 0, 0])
    if data_type in {DataType.END_EFFECTOR_POSES, DataType.POSES}:
        return float(batched_nc_data.pose[0, 0, 0])
    if data_type == DataType.RGB_IMAGES:
        return float(batched_nc_data.frame[0, 0, 0, 0, 0])
    if data_type == DataType.DEPTH_IMAGES:
        return float(batched_nc_data.frame[0, 0, 0, 0, 0])
    if data_type == DataType.POINT_CLOUDS:
        return float(batched_nc_data.points[0, 0, 0, 0])
    if data_type == DataType.CUSTOM_1D:
        return float(batched_nc_data.data[0, 0, 0])
    raise ValueError(f"Cannot extract scalar marker for data type: {data_type}")


def _assert_output_matches_timestep(
    sample, output_data_type: DataType, expected_timestep: int
) -> None:
    batched_nc_data = sample.outputs[output_data_type][0]
    if output_data_type == DataType.LANGUAGE:
        from neuracore_types.batched_nc_data.batched_language_data import (
            BatchedLanguageData,
        )

        expected_nc_data = _create_nc_data_at_timestep(
            output_data_type, expected_timestep, 0
        )
        expected_batched = BatchedLanguageData.from_nc_data(expected_nc_data)
        assert torch.equal(batched_nc_data.input_ids, expected_batched.input_ids)
        assert torch.equal(
            batched_nc_data.attention_mask, expected_batched.attention_mask
        )
        return

    marker = _extract_output_marker(batched_nc_data, output_data_type)
    expected_marker = _timestep_marker(output_data_type, expected_timestep, 0)
    assert marker == pytest.approx(expected_marker)


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
                    _full_embodiment_description()
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


def _stats_cache_path(
    cache_root,
    synchronized_dataset,
    input_description,
    output_description,
):
    recording_fingerprint = [
        {
            "id": recording.id,
            "total_bytes": recording.total_bytes,
            "robot_id": recording.robot_id,
            "instance": recording.instance,
            "start_time": recording.start_time,
            "end_time": recording.end_time,
        }
        for recording in synchronized_dataset.dataset
    ]
    spec_key = json.dumps(
        {
            "recordings": recording_fingerprint,
            "input_cross_embodiment_description": (
                _cacheable_cross_embodiment_description(input_description)
            ),
            "output_cross_embodiment_description": (
                _cacheable_cross_embodiment_description(output_description)
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    spec_hash = hashlib.sha256(spec_key.encode("utf-8")).hexdigest()[:12]
    return (
        cache_root
        / "dataset_cache"
        / f"{synchronized_dataset.id}_statistics_{spec_hash}.json"
    )


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
                        **_full_embodiment_description(),
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


def test_cacheable_cross_embodiment_description_handles_nested_omegaconf():
    """Test OmegaConf descriptions can be used in statistics cache keys."""
    spec = OmegaConf.create({
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: {
                0: "joint_positions_0",
                1: "joint_positions_1",
            }
        }
    })

    cacheable_spec = _cacheable_cross_embodiment_description(spec)

    json.dumps(cacheable_spec, sort_keys=True, separators=(",", ":"))
    assert cacheable_spec == {
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: {
                0: "joint_positions_0",
                1: "joint_positions_1",
            }
        }
    }


def test_cacheable_cross_embodiment_description_recurses_after_model_dump():
    cacheable_spec = _cacheable_cross_embodiment_description(
        ModelDumpCrossEmbodimentDescription()
    )

    json.dumps(cacheable_spec, sort_keys=True, separators=(",", ":"))
    assert cacheable_spec == {
        ROBOT_ID: {DataType.JOINT_POSITIONS: {0: "joint_positions_0"}}
    }


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
    input_description: CrossEmbodimentDescription = {
        MISSING_ROBOT_ID: {  # Robot ID not in dataset
            DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
        }
    }
    output_description: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                DataType.JOINT_TARGET_POSITIONS, 3
            ),
        }
    }

    with pytest.raises(ValueError, match="Input robot IDs .* not found"):
        PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
            output_prediction_horizon=5,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )


def test_should_throw_error_with_missing_data_type(
    mock_synchronized_dataset: SynchronizedDataset,
):
    """Test validation fails with missing data type."""
    input_description: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
            DataType.POINT_CLOUDS: _indexed_names(DataType.POINT_CLOUDS, 1),
        }
    }
    output_description: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                DataType.JOINT_TARGET_POSITIONS, 3
            ),
        }
    }

    with pytest.raises(ValueError, match="Input data type .* is not present"):
        PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
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
    input_description: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: {
                10: "joint_a",
                20: "joint_b",
            }
        }
    }
    output_description: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: {
                99: "joint_c",
            },
            DataType.JOINT_TARGET_POSITIONS: {
                0: "target_a",
            },
        }
    }

    assert merge_cross_embodiment_description(
        input_description, output_description
    ) == {
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: ["joint_a", "joint_b", "joint_c"],
            DataType.JOINT_TARGET_POSITIONS: ["target_a"],
        }
    }


def test_merge_cross_embodiment_description_deduplicates_by_name_not_index():
    input_description: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.RGB_IMAGES: {
                1: "front_camera",
                5: "wrist_camera",
            }
        }
    }
    output_description: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.RGB_IMAGES: {
                3: "wrist_camera",
                9: "side_camera",
            }
        }
    }

    assert merge_cross_embodiment_description(
        input_description, output_description
    ) == {
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
    input_description: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: {
                0: "joint_0",
            }
        }
    }
    output_description: CrossEmbodimentDescription = {
        other_robot_id: {
            DataType.JOINT_TARGET_POSITIONS: {
                0: "target_0",
            }
        }
    }

    assert list(
        merge_cross_embodiment_description(input_description, output_description)
    ) == [
        ROBOT_ID,
        other_robot_id,
    ]


def test_merge_cross_embodiment_description_handles_config_dict_values():
    input_description: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: {
                0: "vx300s_right/wrist_angle",
                1: "vx300s_left/waist",
            },
            DataType.RGB_IMAGES: {
                0: "rgb_angle",
            },
        }
    }
    output_description: CrossEmbodimentDescription = {
        ROBOT_ID: {
            DataType.JOINT_POSITIONS: {
                0: "vx300s_right/wrist_angle",
                1: "vx300s_left/waist",
            }
        }
    }

    assert merge_cross_embodiment_description(
        input_description, output_description
    ) == {
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
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
            }
        }
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
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
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
            }
        }
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
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
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
                DataType.DEPTH_IMAGES: _indexed_names(DataType.DEPTH_IMAGES, 3),
            }
        }
        output_description: CrossEmbodimentDescription = {
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
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
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
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
                DataType.DEPTH_IMAGES: _indexed_names(DataType.DEPTH_IMAGES, 3),
            }
        }
        output_description: CrossEmbodimentDescription = {
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
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
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
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
            }
        }
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
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


class TestOutputTimestepAlignment:
    """Test per-output-type input/output timestep alignment."""

    @staticmethod
    def _sync_point_at_timestep(timestep: int) -> SynchronizedPoint:
        return SynchronizedPoint(
            robot_id=ROBOT_ID,
            timestamp=float(timestep),
            data={
                data_type: {
                    f"{data_type.value}_{i}": _create_nc_data_at_timestep(
                        data_type, timestep, i
                    )
                    for i in range(DATA_ITEMS)
                }
                for data_type in DataType
            },
        )

    @staticmethod
    def _recording_with_timestep_values(
        num_timesteps: int,
    ) -> SynchronizedRecording:
        sync_points = [
            TestOutputTimestepAlignment._sync_point_at_timestep(t)
            for t in range(num_timesteps)
        ]

        class TimestepRecording(SynchronizedRecording):
            def __init__(self) -> None:
                self.sync_points = sync_points
                self.robot_id = ROBOT_ID
                self.name = "timestep_recording"

            def __len__(self) -> int:
                return len(self.sync_points)

            def __getitem__(self, idx):
                if isinstance(idx, int):
                    return self.sync_points[idx]
                if isinstance(idx, slice):
                    start = idx.start or 0
                    stop = idx.stop or len(self.sync_points)
                    step = idx.step or 1
                    return self.sync_points[start:stop:step]
                raise TypeError(f"Invalid index type: {type(idx)}")

        return TimestepRecording()

    @staticmethod
    def _dataset_with_recording(
        recording: SynchronizedRecording,
        dataset_statistics: dict[str, dict[DataType, list[NCDataStats]]],
    ) -> SynchronizedDataset:
        class TimestepSynchronizedDataset(SynchronizedDataset):
            def __init__(self) -> None:
                self.id = "timestep_dataset"
                self.dataset = MagicMock()
                self.dataset.data_types = list(DataType)
                self.dataset.get_full_embodiment_description.side_effect = (
                    lambda robot_id: (
                        _timestep_full_embodiment_description()
                        if robot_id == ROBOT_ID
                        else (_ for _ in ()).throw(
                            ValueError(f"Input robot IDs [{robot_id}] not found")
                        )
                    )
                )

            def calculate_statistics(
                self,
                input_cross_embodiment_description: CrossEmbodimentDescription,
                output_cross_embodiment_description: CrossEmbodimentDescription,
            ) -> SynchronizedDatasetStatistics:
                return SynchronizedDatasetStatistics(
                    synchronized_dataset_id=self.id,
                    input_cross_embodiment_description=(
                        input_cross_embodiment_description
                    ),
                    output_cross_embodiment_description=(
                        output_cross_embodiment_description
                    ),
                    dataset_statistics=dataset_statistics,
                )

            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int) -> SynchronizedRecording:
                return recording

        return TimestepSynchronizedDataset()

    def _load_sample_for_output_type(
        self, output_data_type: DataType, timestep: int = 4
    ):
        recording = self._recording_with_timestep_values(NUM_OBSERVATIONS_PER_EPISODE)
        synchronized_dataset = self._dataset_with_recording(
            recording, _dataset_statistics_for_output_type(output_data_type)
        )
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
            }
        }
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {output_data_type: _indexed_names(output_data_type, 3)}
        }
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
            output_prediction_horizon=1,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config={},
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            return dataset.load_sample(episode_idx=0, timestep=timestep)

    @patch("neuracore.login")
    @pytest.mark.parametrize(
        "output_data_type",
        sorted(TARGET_OUTPUT_DATA_TYPES, key=lambda data_type: data_type.value),
    )
    def test_target_output_uses_same_timestep(
        self, mock_login, output_data_type: DataType
    ) -> None:
        """Target output types align with the input timestep."""
        timestep = 4
        sample = self._load_sample_for_output_type(output_data_type, timestep=timestep)
        _assert_output_matches_timestep(sample, output_data_type, timestep)

    @patch("neuracore.login")
    @pytest.mark.parametrize(
        "output_data_type",
        sorted(NON_TARGET_OUTPUT_DATA_TYPES, key=lambda data_type: data_type.value),
    )
    def test_non_target_output_uses_next_timestep(
        self, mock_login, output_data_type: DataType
    ) -> None:
        """Non-target outputs use next-step action-chunking alignment."""
        timestep = 4
        sample = self._load_sample_for_output_type(output_data_type, timestep=timestep)
        _assert_output_matches_timestep(sample, output_data_type, timestep + 1)

    @patch("neuracore.login")
    def test_mixed_outputs_use_per_type_alignment(self, mock_login) -> None:
        """Target and non-target outputs can use different alignments in one sample."""
        recording = self._recording_with_timestep_values(NUM_OBSERVATIONS_PER_EPISODE)
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                ),
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
            }
        }
        dataset_statistics = _dataset_statistics_for_output_type(
            DataType.JOINT_TARGET_POSITIONS
        )
        dataset_statistics["output"][DataType.JOINT_POSITIONS] = dataset_statistics[
            "input"
        ][DataType.JOINT_POSITIONS]
        synchronized_dataset = self._dataset_with_recording(
            recording, dataset_statistics
        )
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
            }
        }
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
            output_prediction_horizon=1,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config={},
        )

        timestep = 4
        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset.load_sample(episode_idx=0, timestep=timestep)

        _assert_output_matches_timestep(
            sample, DataType.JOINT_TARGET_POSITIONS, timestep
        )
        _assert_output_matches_timestep(sample, DataType.JOINT_POSITIONS, timestep + 1)


class TestDatasetIntegration:
    """Test dataset with PyTorch ecosystem."""

    @patch("neuracore.login")
    def test_getitem_method(self, mock_login, mock_synchronized_dataset):
        """Test __getitem__ method."""
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
            }
        }
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
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
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 3),
            }
        }
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
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
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
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
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )

        assert len(dataset) == 45

    @patch("neuracore.login")
    def test_getitem_propagates_load_sample_error(
        self, mock_login, mock_synchronized_dataset
    ):
        """Test __getitem__ propagates load_sample failures."""
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )

        with patch.object(dataset, "load_sample") as mock_load_sample:
            mock_load_sample.side_effect = Exception("Load error")

            with pytest.raises(Exception):
                _ = dataset[0]

            mock_load_sample.assert_called_once_with(0, 0)


class TestPerformanceAndOptimization:
    """Test performance and optimization features."""

    def test_memory_monitoring_initialization(self, mock_synchronized_dataset):
        """Test memory monitor initialization."""
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )

        assert dataset._memory_monitor is not None
        assert hasattr(dataset._memory_monitor, "check_memory")

    def test_episode_indices_creation(self, mock_synchronized_dataset):
        """Test episode indices are created correctly."""
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
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


class TestIntegrationWithPyTorchDataLoader:
    """Integration tests with PyTorch DataLoader."""

    @patch("neuracore.login")
    def test_dataloader_with_collate_fn(self, mock_login, mock_synchronized_dataset):
        """Test DataLoader with custom collate function."""
        from torch.utils.data import DataLoader

        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 2),
                DataType.RGB_IMAGES: _indexed_names(DataType.RGB_IMAGES, 2),
            }
        }
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 2
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
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
            for data_type in input_description[ROBOT_ID].keys():
                assert data_type in batch.inputs
                assert isinstance(batch.inputs[data_type], list)


class TestDatasetStatistics:
    """Test dataset statistics functionality."""

    def test_dataset_statistics_property(self, mock_synchronized_dataset):
        """Test dataset_statistics property."""
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
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
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }
        stats = SynchronizedDatasetStatistics(
            synchronized_dataset_id=mock_synchronized_dataset.id,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
            dataset_statistics=dataset_statistics,
        )
        cache_path = _stats_cache_path(
            tmp_path,
            mock_synchronized_dataset,
            input_description,
            output_description,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(stats.model_dump(mode="json"), handle)

        mock_synchronized_dataset.calculate_statistics = MagicMock(
            side_effect=AssertionError("calculate_statistics should not be called")
        )

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
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
        input_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_POSITIONS: _indexed_names(DataType.JOINT_POSITIONS, 3)
            }
        }
        output_description: CrossEmbodimentDescription = {
            ROBOT_ID: {
                DataType.JOINT_TARGET_POSITIONS: _indexed_names(
                    DataType.JOINT_TARGET_POSITIONS, 3
                )
            }
        }
        stats = SynchronizedDatasetStatistics(
            synchronized_dataset_id=mock_synchronized_dataset.id,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
            dataset_statistics=dataset_statistics,
        )
        mock_synchronized_dataset.calculate_statistics = MagicMock(return_value=stats)

        PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_cross_embodiment_description=input_description,
            output_cross_embodiment_description=output_description,
            output_prediction_horizon=3,
            input_preprocessing_config=_default_preprocessing_config(),
            output_preprocessing_config=_default_preprocessing_config(),
        )

        assert mock_synchronized_dataset.calculate_statistics.call_count == 1
        cache_path = _stats_cache_path(
            tmp_path,
            mock_synchronized_dataset,
            input_description,
            output_description,
        )
        assert cache_path.exists()

        assert cache_path.exists()
