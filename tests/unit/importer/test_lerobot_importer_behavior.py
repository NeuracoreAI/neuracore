"""Tests for LeRobot importer behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from neuracore_types import DataType
from neuracore_types.importer.config import LanguageConfig

from neuracore.importer.core.base import ImportItem, WorkerError
from neuracore.importer.core.exceptions import ImportError
from neuracore.importer.lerobot_importer import LeRobotDatasetImporter


class _FakeTensor:
    """Simple test tensor-like object exposing a numpy() method."""

    def __init__(self, value):
        self._value = value

    def numpy(self):
        return self._value


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
    importer._episode_iter = iter([1, 2, 3])
    importer.frequency = 30.0

    state = importer.__getstate__()

    assert "_dataset" not in state
    assert "_episode_iter" not in state
    assert state["dataset_name"] == "demo"
    assert state["frequency"] == 30.0


def test_lerobot_prepare_worker_uses_chunk_bounds(monkeypatch):
    """Worker prep should initialize dataset and episode iterator for its chunk."""
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
    importer._collect_episode_ids = MagicMock(return_value=[10, 20, 30, 40, 50])

    chunk = [ImportItem(index=2), ImportItem(index=3)]
    importer.prepare_worker(worker_id=7, chunk=chunk)

    assert base_calls == [(7, chunk)]
    importer._load_dataset.assert_called_once_with()
    importer._collect_episode_ids.assert_called_once_with(fake_dataset)
    assert importer._dataset is fake_dataset
    assert list(importer._episode_iter) == [30, 40]


def test_lerobot_import_item_step_mode_skips_failing_steps():
    """LeRobot importer should continue when a step fails in step mode."""
    importer = object.__new__(LeRobotDatasetImporter)
    importer._dataset = object()
    importer._episode_iter = iter([123])
    importer.ik_init_config = None
    importer.frequency = 10.0
    importer._worker_id = 2
    importer.num_episodes = 1
    importer.skip_on_error = "step"
    importer.logger = MagicMock()
    importer._error_queue = MagicMock()
    importer._log_worker_error = MagicMock()
    importer._emit_progress = MagicMock()
    importer._iter_episode_steps = MagicMock(
        return_value=(iter([{"v": 1}, {"v": 2}]), 2)
    )
    importer._record_step = MagicMock(side_effect=[ValueError("bad step"), None])

    with patch("neuracore.importer.lerobot_importer.nc.start_recording"), patch(
        "neuracore.importer.lerobot_importer.nc.stop_recording"
    ) as stop_recording:
        importer.import_item(ImportItem(index=0))

    assert importer._record_step.call_count == 2
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
    stop_recording.assert_called_once_with(wait=True)


def test_lerobot_import_item_non_step_mode_re_raises():
    """LeRobot importer should re-raise step errors when not in step mode."""
    importer = object.__new__(LeRobotDatasetImporter)
    importer._dataset = object()
    importer._episode_iter = iter([555])
    importer.ik_init_config = None
    importer.frequency = 10.0
    importer._worker_id = 1
    importer.num_episodes = 1
    importer.skip_on_error = "episode"
    importer.logger = MagicMock()
    importer._error_queue = MagicMock()
    importer._log_worker_error = MagicMock()
    importer._emit_progress = MagicMock()
    importer._iter_episode_steps = MagicMock(return_value=(iter([{"v": 1}]), 1))
    importer._record_step = MagicMock(side_effect=RuntimeError("explode"))

    with patch("neuracore.importer.lerobot_importer.nc.start_recording"), patch(
        "neuracore.importer.lerobot_importer.nc.stop_recording"
    ):
        with pytest.raises(RuntimeError, match="explode"):
            importer.import_item(ImportItem(index=0))

    importer._error_queue.put.assert_not_called()
    importer._log_worker_error.assert_not_called()


def test_lerobot_import_item_requires_initialized_worker_dataset():
    """Importer should fail if worker-local dataset state is missing."""
    importer = object.__new__(LeRobotDatasetImporter)
    importer._dataset = None
    importer._episode_iter = None
    importer.ik_init_config = None

    with pytest.raises(ImportError, match="Worker dataset was not initialized"):
        importer.import_item(ImportItem(index=0))


def test_lerobot_import_item_raises_when_episode_iterator_is_exhausted():
    """Importer should fail when chunk has no remaining episodes."""
    importer = object.__new__(LeRobotDatasetImporter)
    importer._dataset = object()
    importer._episode_iter = iter([])
    importer.ik_init_config = None
    importer.num_episodes = 3

    with pytest.raises(ImportError, match="No episode available for index 4"):
        importer.import_item(ImportItem(index=4))


def test_lerobot_import_item_requires_frequency():
    """Importer should require an explicit episode frequency before logging steps."""
    importer = object.__new__(LeRobotDatasetImporter)
    importer._dataset = object()
    importer._episode_iter = iter([9])
    importer.ik_init_config = None
    importer.frequency = None

    with pytest.raises(
        ImportError, match="Frequency is required for importing episodes"
    ):
        importer.import_item(ImportItem(index=0))


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
    importer._log_data = MagicMock()

    importer._record_step({"instruction": "close gripper"}, timestamp=7.5)

    importer._log_data.assert_called_once_with(
        DataType.LANGUAGE,
        "close gripper",
        mapping_item,
        import_format,
        7.5,
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
    import_format = SimpleNamespace(language_type=LanguageConfig.STRING)
    import_config = SimpleNamespace(
        source="observation",
        mapping=[mapping_item],
        format=import_format,
    )
    importer.dataset_config = SimpleNamespace(
        data_import_config={DataType.JOINT_POSITIONS: import_config}
    )
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
        output_dataset_name="out",
        dataset_dir=".",
        dataset_config=SimpleNamespace(),
        joint_info={},
        ik_urdf_path="/tmp/robot.urdf",
        ik_init_config=[0.0, 1.0],
        dry_run=True,
        suppress_warnings=True,
        skip_on_error="step",
    )

    assert captured["ik_urdf_path"] == "/tmp/robot.urdf"
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


def test_lerobot_collect_episode_ids_returns_sorted_unique_ids():
    """Episode IDs should be sorted and deduplicated."""
    importer = object.__new__(LeRobotDatasetImporter)
    dataset = SimpleNamespace(hf_dataset={"episode_index": [3, 1, 2, 3, 1]})

    episode_ids = importer._collect_episode_ids(dataset)

    assert episode_ids == [1, 2, 3]


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
