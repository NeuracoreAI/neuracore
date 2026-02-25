"""Tests for shared RLDS/TFDS importer behavior and RLDS overrides."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from neuracore_types import DataType
from neuracore_types.importer.config import LanguageConfig

from neuracore.importer.core.base import ImportItem, WorkerError
from neuracore.importer.core.exceptions import ImportError
from neuracore.importer.rlds_tfds_importer import (
    RLDSAndTFDSDatasetImporterBase,
    RLDSDatasetImporter,
)


class _FakeTensor:
    """Simple test tensor-like object exposing a numpy() method."""

    def __init__(self, value):
        self._value = value

    def numpy(self):
        return self._value


def test_base_handle_step_error_default_returns_false():
    """Base importer should re-raise step errors unless subclass handles them."""
    importer = object.__new__(RLDSAndTFDSDatasetImporterBase)
    item = ImportItem(index=1)

    assert importer._handle_step_error(RuntimeError("boom"), item, 2) is False


def test_rlds_handle_step_error_step_mode_enqueues_and_logs():
    """RLDS importer should skip step failures when configured with step mode."""
    importer = object.__new__(RLDSDatasetImporter)
    importer.skip_on_error = "step"
    importer._worker_id = 3
    importer._error_queue = MagicMock()
    importer._log_worker_error = MagicMock()

    try:
        raise ValueError("bad step")
    except ValueError as exc:
        handled = importer._handle_step_error(exc, ImportItem(index=7), 4)

    assert handled is True
    importer._error_queue.put.assert_called_once()
    queued_error = importer._error_queue.put.call_args.args[0]
    assert isinstance(queued_error, WorkerError)
    assert queued_error.worker_id == 3
    assert queued_error.item_index == 7
    assert queued_error.message == "Step 4: bad step"
    assert queued_error.traceback is not None
    assert "ValueError: bad step" in queued_error.traceback
    importer._log_worker_error.assert_called_once_with(3, 7, "Step 4: bad step")


def test_rlds_handle_step_error_non_step_mode_returns_false():
    """RLDS importer should not handle step errors unless step mode is enabled."""
    importer = object.__new__(RLDSDatasetImporter)
    importer.skip_on_error = "episode"
    importer._worker_id = 1
    importer._error_queue = MagicMock()
    importer._log_worker_error = MagicMock()

    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        handled = importer._handle_step_error(exc, ImportItem(index=2), 1)

    assert handled is False
    importer._error_queue.put.assert_not_called()
    importer._log_worker_error.assert_not_called()


def test_rlds_record_step_supports_empty_source_path_for_language():
    """RLDS _record_step should allow empty source path and string language values."""
    importer = object.__new__(RLDSDatasetImporter)
    mapping_item = SimpleNamespace(
        source_name="instruction",
        index=None,
        index_range=None,
        name="instruction",
    )
    import_format = SimpleNamespace(language_type=LanguageConfig.STRING)
    import_config = SimpleNamespace(
        source="",
        mapping=[mapping_item],
        format=import_format,
    )
    importer.dataset_config = SimpleNamespace(
        data_import_config={DataType.LANGUAGE: import_config}
    )
    importer._log_data = MagicMock()

    importer._record_step({"instruction": "pick up block"}, timestamp=12.5)

    importer._log_data.assert_called_once_with(
        DataType.LANGUAGE,
        "pick up block",
        mapping_item,
        import_format,
        12.5,
    )


def test_rlds_record_step_converts_tensor_to_numpy_for_non_language():
    """RLDS _record_step should call numpy() for non-language data."""
    importer = object.__new__(RLDSDatasetImporter)
    mapping_item = SimpleNamespace(
        source_name="joint_positions",
        index=None,
        index_range=None,
        name="joint_positions",
    )
    import_format = SimpleNamespace(language_type=LanguageConfig.STRING)
    import_config = SimpleNamespace(
        source="",
        mapping=[mapping_item],
        format=import_format,
    )
    importer.dataset_config = SimpleNamespace(
        data_import_config={DataType.JOINT_POSITIONS: import_config}
    )
    importer._log_data = MagicMock()

    importer._record_step({"joint_positions": _FakeTensor([1.0, 2.0])}, timestamp=3.0)

    importer._log_data.assert_called_once_with(
        DataType.JOINT_POSITIONS,
        [1.0, 2.0],
        mapping_item,
        import_format,
        3.0,
    )


def test_rlds_init_forwards_ik_args_to_base(monkeypatch):
    """RLDS importer should forward IK initialization args to base class."""
    captured = {}

    def fake_base_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(RLDSAndTFDSDatasetImporterBase, "__init__", fake_base_init)

    RLDSDatasetImporter(
        input_dataset_name="in",
        output_dataset_name="out",
        dataset_dir=SimpleNamespace(),
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


def test_extract_then_convert_dict_raises_import_error_without_nested_search():
    """Dict outputs are now rejected directly instead of nested tensor search."""
    importer = object.__new__(RLDSAndTFDSDatasetImporterBase)
    item = SimpleNamespace(
        source_name="observation",
        index=None,
        index_range=None,
        name="ee_pose",
    )
    extracted = importer._extract_source_data(
        source={"observation": {"state": _FakeTensor([1.0, 2.0])}},
        item=item,
        import_source_path="steps.observation",
        data_type=DataType.POSES,
    )

    with pytest.raises(ImportError, match="Failed to convert data to numpy array"):
        importer._convert_source_data(
            source_data=extracted,
            data_type=DataType.POSES,
            language_type=LanguageConfig.STRING,
            item_name="ee_pose",
            import_source_path="steps.observation",
        )
