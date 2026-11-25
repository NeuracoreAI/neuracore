"""Tests for LeRobot importer behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from neuracore_types import DataType, JointPositionInputTypeConfig
from neuracore_types.importer.config import LanguageConfig
from neuracore_types.timestamps import MICROSECONDS_PER_SECOND, seconds_to_us

from neuracore.importer.core.base import ImportItem, WorkerError
from neuracore.importer.core.exceptions import ImportError
from neuracore.importer.lerobot_importer import LeRobotDatasetImporter


class _FakeTensor:
    """Simple test tensor-like object exposing a numpy() method."""

    def __init__(self, value):
        self._value = value

    def numpy(self):
        return self._value

    def __getitem__(self, key):
        return _FakeTensor(self._value[key])


class _FakeEpisodeRows:
    """Simple rows object matching the subset of HF Dataset API we use."""

    def __init__(self, rows):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def sort(self, key):
        return _FakeEpisodeRows(sorted(self._rows, key=lambda row: row[key]))

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self._rows]
        return self._rows[key]


class _FakeHFDataset:
    """Simple dataset object supporting filter(...) used by importer."""

    def __init__(self, rows):
        self._rows = rows

    def filter(self, predicate):
        return _FakeEpisodeRows([row for row in self._rows if predicate(row)])


def test_lerobot_getstate_drops_worker_local_handles():
    """Pickle state should omit worker-local dataset handles."""
    importer = object.__new__(LeRobotDatasetImporter)
    importer.dataset_name = "demo"
    importer._dataset = object()
    importer.frequency = 30.0

    state = importer.__getstate__()

    assert "_dataset" not in state
    assert state["dataset_name"] == "demo"
    assert state["frequency"] == 30.0


def test_lerobot_work_items_carry_their_episode_id():
    """Work items should name their episode so any chunk imports the right one."""
    importer = object.__new__(LeRobotDatasetImporter)
    importer.episode_ids = [10, 20, 40]

    items = importer.build_work_items()

    assert [item.index for item in items] == [0, 1, 2]
    assert [importer.item_key(item) for item in items] == ["10", "20", "40"]


def test_lerobot_metadata_episode_ids_are_sorted_and_unique():
    meta = SimpleNamespace(
        total_episodes=4, episodes={"episode_index": [3, 1, 2, 3, 1]}
    )

    assert LeRobotDatasetImporter._metadata_episode_ids(meta) == [1, 2, 3]


def test_lerobot_metadata_episode_ids_fall_back_to_the_episode_count():
    meta = SimpleNamespace(total_episodes=3, episodes=None)

    assert LeRobotDatasetImporter._metadata_episode_ids(meta) == [0, 1, 2]


def test_lerobot_prepare_worker_loads_the_dataset(monkeypatch):
    importer = object.__new__(LeRobotDatasetImporter)
    base_calls = []

    def fake_base_prepare_worker(self, worker_id, chunk=None):
        base_calls.append((worker_id, chunk))

    monkeypatch.setattr(
        "neuracore.importer.lerobot_importer.NeuracoreDatasetImporter.prepare_worker",
        fake_base_prepare_worker,
    )
    fake_dataset = object()
    importer._load_dataset = MagicMock(return_value=fake_dataset)

    chunk = [ImportItem(index=2), ImportItem(index=3)]
    importer.prepare_worker(worker_id=7, chunk=chunk)

    assert base_calls == [(7, chunk)]
    assert importer._dataset is fake_dataset


def _episode_item(index, episode_id):
    return ImportItem(index=index, metadata={"episode_id": episode_id})


def _step_importer(worker_id, skip_on_error):
    importer = object.__new__(LeRobotDatasetImporter)
    importer._dataset = object()
    importer.ik_init_config = None
    importer.frequency = 10.0
    importer._worker_id = worker_id
    importer.num_episodes = 1
    importer.skip_on_error = skip_on_error
    importer.robot_name = "test_robot"
    importer.dry_run = False
    importer.logger = MagicMock()
    importer._error_queue = MagicMock()
    importer._log_worker_error = MagicMock()
    importer._emit_progress = MagicMock()
    importer._start_recording = MagicMock()
    importer._stop_recording = MagicMock()
    return importer


def test_lerobot_import_item_step_mode_skips_failing_steps():
    """LeRobot importer should continue when a step fails in step mode."""
    importer = _step_importer(worker_id=2, skip_on_error="step")
    importer._iter_episode_steps = MagicMock(
        return_value=(iter([{"v": 1}, {"v": 2}]), 2)
    )
    importer._record_step = MagicMock(side_effect=[ValueError("bad step"), None])
    item = _episode_item(0, 123)

    with patch(
        "neuracore.importer.lerobot_importer.now_us",
        return_value=100_000_000,
    ):
        importer.import_item(item)

    importer._iter_episode_steps.assert_called_once_with(importer._dataset, 123)
    step_timestamps = [call.args[1] for call in importer._record_step.call_args_list]
    assert [seconds_to_us(stamp) for stamp in step_timestamps] == [
        100_100_000,
        100_200_000,
    ]
    importer._error_queue.put.assert_called_once()
    queued_error = importer._error_queue.put.call_args.args[0]
    assert isinstance(queued_error, WorkerError)
    assert queued_error.worker_id == 2
    assert queued_error.item_index == 0
    assert queued_error.message == "Step 1: bad step"
    assert queued_error.traceback is not None
    assert "ValueError: bad step" in queued_error.traceback
    importer._log_worker_error.assert_called_once_with(2, 0, "Step 1: bad step")
    importer._emit_progress.assert_has_calls([
        call(0, step=0, total_steps=2, episode_label="123"),
        call(0, step=2, total_steps=2, episode_label="123"),
    ])
    importer._start_recording.assert_called_once_with(
        item, 100_000_000 / MICROSECONDS_PER_SECOND
    )
    importer._stop_recording.assert_called_once_with(
        100_300_000 / MICROSECONDS_PER_SECOND
    )


def test_lerobot_import_item_non_step_mode_re_raises():
    """LeRobot importer should re-raise step errors when not in step mode."""
    importer = _step_importer(worker_id=1, skip_on_error="episode")
    importer._iter_episode_steps = MagicMock(return_value=(iter([{"v": 1}]), 1))
    importer._record_step = MagicMock(side_effect=RuntimeError("explode"))

    with pytest.raises(RuntimeError, match="explode"):
        importer.import_item(_episode_item(0, 555))

    importer._error_queue.put.assert_not_called()
    importer._log_worker_error.assert_not_called()
    importer._stop_recording.assert_not_called()


def test_lerobot_import_item_imports_the_episode_of_a_non_contiguous_item():
    """Resumed chunks skip saved episodes, so items need not be contiguous."""
    importer = _step_importer(worker_id=0, skip_on_error="episode")
    importer._iter_episode_steps = MagicMock(return_value=(iter([{"v": 1}]), 1))
    importer._record_step = MagicMock()

    importer.import_item(_episode_item(5, 42))

    importer._iter_episode_steps.assert_called_once_with(importer._dataset, 42)


def test_lerobot_an_episode_without_frames_fails_before_recording():
    importer = _step_importer(worker_id=0, skip_on_error="episode")
    importer._iter_episode_steps = MagicMock(return_value=(iter([]), 0))

    with pytest.raises(ImportError, match="Episode 7 has no frames"):
        importer.import_item(_episode_item(0, 7))

    importer._start_recording.assert_not_called()


def test_lerobot_import_item_requires_initialized_worker_dataset():
    """Importer should fail if worker-local dataset state is missing."""
    importer = object.__new__(LeRobotDatasetImporter)
    importer._dataset = None
    importer.ik_init_config = None

    with pytest.raises(ImportError, match="Worker dataset was not initialized"):
        importer.import_item(_episode_item(0, 1))


def test_lerobot_import_item_requires_frequency():
    """Importer should require an explicit episode frequency before logging steps."""
    importer = object.__new__(LeRobotDatasetImporter)
    importer._dataset = object()
    importer.ik_init_config = None
    importer.frequency = None

    with pytest.raises(
        ImportError, match="Frequency is required for importing episodes"
    ):
        importer.import_item(_episode_item(0, 9))


def test_lerobot_record_step_supports_empty_source_path_for_language():
    """LeRobot _record_step should support empty source path for language values."""
    importer = object.__new__(LeRobotDatasetImporter)
    mapping_item = SimpleNamespace(
        source_name="instruction",
        index=None,
        index_range=None,
        name="instruction",
    )
    import_format = SimpleNamespace(language_type=LanguageConfig.STRING)
    import_config = SimpleNamespace(
        source="", mapping=[mapping_item], format=import_format
    )
    importer.dataset_config = SimpleNamespace(
        data_import_config={DataType.LANGUAGE: import_config}
    )
    importer.ordered_import_configs = [(DataType.LANGUAGE, import_config)]
    importer._log_data = MagicMock()

    importer._record_step({"instruction": "close gripper"}, timestamp=7.5)

    importer._log_data.assert_called_once_with(
        DataType.LANGUAGE,
        "close gripper",
        mapping_item,
        import_format,
        7.5,
        extrinsics=None,
        intrinsics=None,
    )


def test_lerobot_record_step_reads_dotted_source_key_and_converts_tensor():
    """LeRobot _record_step should read dotted keys and call numpy()."""
    importer = object.__new__(LeRobotDatasetImporter)
    mapping_item = SimpleNamespace(
        source_name="state",
        index=None,
        index_range=None,
        name="joint_positions",
    )
    import_format = SimpleNamespace(
        language_type=LanguageConfig.STRING,
        joint_position_input_type=JointPositionInputTypeConfig.CUSTOM,
        ee_pose_input_type=None,
    )
    import_config = SimpleNamespace(
        source="observation",
        mapping=[mapping_item],
        format=import_format,
    )
    importer.dataset_config = SimpleNamespace(
        data_import_config={DataType.JOINT_POSITIONS: import_config}
    )
    importer.ordered_import_configs = [(DataType.JOINT_POSITIONS, import_config)]
    importer._log_data = MagicMock()

    importer._record_step(
        {"observation.state": _FakeTensor([0.1, -0.2])},
        timestamp=2.0,
    )

    importer._log_data.assert_called_once_with(
        DataType.JOINT_POSITIONS,
        [0.1, -0.2],
        mapping_item,
        import_format,
        2.0,
        extrinsics=None,
        intrinsics=None,
    )


@pytest.mark.parametrize(
    ("import_source_path", "expected"),
    [
        ("observation", [1.0, 2.0, 3.0]),
        ("episode.observation", [4.0, 5.0, 6.0]),
        ("run.episode.observation", [7.0, 8.0, 9.0]),
    ],
)
def test_lerobot_resolve_source_path_supports_dot_delimited_source_depths(
    import_source_path, expected
):
    """Import source path should resolve flattened 1-3 segment keys."""
    importer = object.__new__(LeRobotDatasetImporter)

    resolved = importer._resolve_source_path(
        source={
            "observation": _FakeTensor([1.0, 2.0, 3.0]),
            "episode.observation": _FakeTensor([4.0, 5.0, 6.0]),
            "run.episode.observation": _FakeTensor([7.0, 8.0, 9.0]),
        },
        source_name=import_source_path,
    )

    assert resolved.numpy() == expected


@pytest.mark.parametrize(
    ("item_source_name", "expected"),
    [
        ("joint_positions", [1.0, 2.0, 3.0]),
        ("robot.joint_positions", [4.0, 5.0, 6.0]),
        ("arm.robot.joint_positions", [7.0, 8.0, 9.0]),
    ],
)
def test_lerobot_extract_source_data_supports_dot_delimited_source_name_depths(
    item_source_name, expected
):
    """Mapping source_name should resolve flattened 1-3 segment keys."""
    importer = object.__new__(LeRobotDatasetImporter)
    item = SimpleNamespace(
        source_name=item_source_name,
        index=None,
        index_range=None,
        name="joint_positions",
    )

    resolved = importer._extract_source_data(
        source={
            "joint_positions": _FakeTensor([1.0, 2.0, 3.0]),
            "robot.joint_positions": _FakeTensor([4.0, 5.0, 6.0]),
            "arm.robot.joint_positions": _FakeTensor([7.0, 8.0, 9.0]),
        },
        item=item,
        import_source_path="",
        data_type=DataType.JOINT_POSITIONS,
    )

    assert resolved.numpy() == expected


def test_lerobot_record_step_resolves_mixed_dot_delimited_source_and_source_name():
    """_record_step should combine source and source_name dot paths."""
    importer = object.__new__(LeRobotDatasetImporter)
    mapping_item = SimpleNamespace(
        source_name="robot.joint_positions",
        index=None,
        index_range=None,
        name="joint_positions",
    )
    import_format = SimpleNamespace(
        language_type=LanguageConfig.STRING,
        joint_position_input_type=JointPositionInputTypeConfig.CUSTOM,
        ee_pose_input_type=None,
    )
    import_config = SimpleNamespace(
        source="episode.steps",
        mapping=[mapping_item],
        format=import_format,
    )
    importer.dataset_config = SimpleNamespace(
        data_import_config={DataType.JOINT_POSITIONS: import_config}
    )
    importer.ordered_import_configs = [(DataType.JOINT_POSITIONS, import_config)]
    importer._log_data = MagicMock()

    importer._record_step(
        {"episode.steps.robot.joint_positions": _FakeTensor([0.1, 0.2, 0.3])},
        timestamp=1.5,
    )

    importer._log_data.assert_called_once_with(
        DataType.JOINT_POSITIONS,
        [0.1, 0.2, 0.3],
        mapping_item,
        import_format,
        1.5,
        extrinsics=None,
        intrinsics=None,
    )


def test_lerobot_extract_source_data_uses_split_pose_sources_and_concatenates():
    """Split pose source config should override source_name and concatenate slices."""
    importer = object.__new__(LeRobotDatasetImporter)
    item = SimpleNamespace(
        pose_position_source_name="position",
        pose_orientation_source_name="orientation",
        pose_position_index_range=SimpleNamespace(start=1, end=4),
        pose_orientation_index_range=SimpleNamespace(start=0, end=4),
    )

    extracted = importer._extract_source_data(
        source={
            "observation.combined_pose": _FakeTensor(
                [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
            ),
            "observation.position": _FakeTensor([10.0, 1.0, 2.0, 3.0, 20.0]),
            "observation.orientation": _FakeTensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        },
        item=item,
        import_source_path="observation",
        data_type=DataType.POSES,
    )

    converted = importer._convert_source_data(
        source_data=extracted,
        data_type=DataType.POSES,
        item_name="ee_pose",
    )
    assert converted.tolist() == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4]


def test_lerobot_extract_source_data_split_pose_requires_all_split_fields():
    """Split pose extraction should reject partially specified configs."""
    importer = object.__new__(LeRobotDatasetImporter)
    item = SimpleNamespace(
        source_name="combined_pose",
        index=None,
        index_range=None,
        pose_position_source_name="position",
        pose_orientation_source_name=None,
        pose_position_index_range=SimpleNamespace(start=0, end=3),
        pose_orientation_index_range=SimpleNamespace(start=0, end=4),
    )

    with pytest.raises(ImportError, match="must be provided together"):
        importer._extract_source_data(
            source={
                "observation.combined_pose": _FakeTensor(
                    [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
                ),
                "observation.position": _FakeTensor([1.0, 2.0, 3.0]),
                "observation.orientation": _FakeTensor([0.1, 0.2, 0.3, 0.4]),
            },
            item=item,
            import_source_path="observation",
            data_type=DataType.POSES,
        )


def test_lerobot_init_forwards_ik_args_to_base(monkeypatch):
    """LeRobot importer should forward IK args to NeuracoreDatasetImporter."""
    captured = {}

    def fake_base_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "neuracore.importer.lerobot_importer.NeuracoreDatasetImporter.__init__",
        fake_base_init,
    )
    monkeypatch.setattr(
        LeRobotDatasetImporter,
        "_load_metadata",
        lambda self: SimpleNamespace(total_episodes=4, camera_keys=["front"], fps=20.0),
    )
    monkeypatch.setattr(LeRobotDatasetImporter, "_resolve_frequency", lambda *_: 20.0)

    importer = LeRobotDatasetImporter(
        input_dataset_name="in",
        output_dataset_id="out-dataset-id",
        dataset_dir=".",
        dataset_config=SimpleNamespace(),
        joint_info={},
        urdf_path="/tmp/robot.urdf",
        ik_init_config=[0.0, 1.0],
        dry_run=True,
        suppress_warnings=True,
        skip_on_error="step",
    )

    assert captured["urdf_path"] == "/tmp/robot.urdf"
    assert captured["ik_init_config"] == [0.0, 1.0]
    assert captured["skip_on_error"] == "step"
    assert importer.num_episodes == 4
    assert importer.camera_keys == ["front"]
    assert importer.frequency == 20.0


def test_lerobot_resolve_frequency_prefers_config_and_warns_on_mismatch():
    """Configured frequency should be used even if metadata differs."""
    importer = object.__new__(LeRobotDatasetImporter)
    importer.data_config = SimpleNamespace(frequency=30.0)
    importer.logger = MagicMock()

    frequency = importer._resolve_frequency(meta_frequency=15.0)

    assert frequency == 30.0
    importer.logger.warning.assert_called_once_with(
        "Dataset FPS %s does not match configured FPS %s", 15.0, 30.0
    )


def test_lerobot_resolve_frequency_falls_back_to_metadata():
    """Metadata FPS should be used when config frequency is missing."""
    importer = object.__new__(LeRobotDatasetImporter)
    importer.data_config = SimpleNamespace(frequency=None)
    importer.logger = MagicMock()

    frequency = importer._resolve_frequency(meta_frequency=25.0)

    assert frequency == 25.0
    importer.logger.warning.assert_not_called()


def test_lerobot_resolve_frequency_raises_when_missing_everywhere():
    """Importer should fail if neither config nor metadata provides frequency."""
    importer = object.__new__(LeRobotDatasetImporter)
    importer.data_config = SimpleNamespace(frequency=None)
    importer.logger = MagicMock()

    with pytest.raises(
        ImportError, match="Frequency not provided in config and missing from metadata"
    ):
        importer._resolve_frequency(meta_frequency=None)


def test_lerobot_iter_episode_steps_uses_dataset_indices_when_available():
    """When 'index' exists, importer should fetch rows through dataset indexing."""
    importer = object.__new__(LeRobotDatasetImporter)

    class _FakeDataset:
        def __init__(self):
            self.hf_dataset = _FakeHFDataset([
                {"episode_index": 1, "frame_index": 2, "index": 20},
                {"episode_index": 1, "frame_index": 1, "index": 10},
                {"episode_index": 2, "frame_index": 1, "index": 99},
            ])

        def __getitem__(self, idx):
            return {"from_dataset": idx}

    dataset = _FakeDataset()
    step_iter, total_steps = importer._iter_episode_steps(dataset, episode_id=1)

    assert total_steps == 2
    assert list(step_iter) == [{"from_dataset": 10}, {"from_dataset": 20}]


def test_lerobot_iter_episode_steps_falls_back_to_filtered_rows_without_index():
    """Without 'index', importer should yield rows from the filtered table directly."""
    importer = object.__new__(LeRobotDatasetImporter)

    dataset = SimpleNamespace(
        hf_dataset=_FakeHFDataset([
            {"episode_index": 7, "frame_index": 2, "value": "b"},
            {"episode_index": 7, "frame_index": 1, "value": "a"},
            {"episode_index": 8, "frame_index": 1, "value": "x"},
        ])
    )
    step_iter, total_steps = importer._iter_episode_steps(dataset, episode_id=7)

    assert total_steps == 2
    assert list(step_iter) == [
        {"episode_index": 7, "frame_index": 1, "value": "a"},
        {"episode_index": 7, "frame_index": 2, "value": "b"},
    ]
