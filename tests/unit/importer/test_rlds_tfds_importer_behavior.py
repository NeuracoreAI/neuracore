"""Tests for shared RLDS/TFDS importer behavior and RLDS overrides."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from neuracore_types import DataType, JointPositionInputTypeConfig
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

    def __getitem__(self, key):
        return _FakeTensor(self._value[key])


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
    importer.ordered_import_configs = [(DataType.LANGUAGE, import_config)]
    importer._log_data = MagicMock()

    importer._record_step({"instruction": "pick up block"}, timestamp=12.5)

    importer._log_data.assert_called_once_with(
        DataType.LANGUAGE,
        "pick up block",
        mapping_item,
        import_format,
        12.5,
        extrinsics=None,
        intrinsics=None,
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
    import_format = SimpleNamespace(
        language_type=LanguageConfig.STRING,
        joint_position_input_type=JointPositionInputTypeConfig.CUSTOM,
        ee_pose_input_type=None,
    )
    import_config = SimpleNamespace(
        source="",
        mapping=[mapping_item],
        format=import_format,
    )
    importer.dataset_config = SimpleNamespace(
        data_import_config={DataType.JOINT_POSITIONS: import_config}
    )
    importer.ordered_import_configs = [(DataType.JOINT_POSITIONS, import_config)]
    importer._log_data = MagicMock()

    importer._record_step({"joint_positions": _FakeTensor([1.0, 2.0])}, timestamp=3.0)

    importer._log_data.assert_called_once_with(
        DataType.JOINT_POSITIONS,
        [1.0, 2.0],
        mapping_item,
        import_format,
        3.0,
        extrinsics=None,
        intrinsics=None,
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
        urdf_path="/tmp/robot.urdf",
        ik_init_config=[0.0, 1.0],
        dry_run=True,
        suppress_warnings=True,
        skip_on_error="step",
    )

    assert captured["urdf_path"] == "/tmp/robot.urdf"
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
        source={
            "steps": {
                "observation": {"observation": {"state": _FakeTensor([1.0, 2.0])}}
            }
        },
        item=item,
        import_source_path="steps.observation",
        data_type=DataType.POSES,
    )

    with pytest.raises(ImportError, match="Failed to convert data to numpy array"):
        importer._convert_source_data(
            source_data=extracted,
            data_type=DataType.POSES,
            item_name="ee_pose",
        )


def test_rlds_extract_source_data_uses_split_pose_sources_and_concatenates():
    """Split pose source config should override source_name and concatenate slices."""
    importer = object.__new__(RLDSAndTFDSDatasetImporterBase)
    item = SimpleNamespace(
        pose_position_source_name="position",
        pose_orientation_source_name="orientation",
        pose_position_index_range=SimpleNamespace(start=1, end=4),
        pose_orientation_index_range=SimpleNamespace(start=0, end=4),
    )

    source_data = importer._extract_source_data(
        source={
            "steps": {
                "observation": {
                    "combined_pose": _FakeTensor([9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]),
                    "position": _FakeTensor([10.0, 1.0, 2.0, 3.0, 20.0]),
                    "orientation": _FakeTensor([0.1, 0.2, 0.3, 0.4, 0.5]),
                }
            }
        },
        item=item,
        import_source_path="steps.observation",
        data_type=DataType.POSES,
    )

    converted = importer._convert_source_data(
        source_data=source_data,
        data_type=DataType.POSES,
        item_name="ee_pose",
    )
    assert converted.tolist() == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4]


def test_rlds_extract_source_data_split_pose_requires_all_split_fields():
    """Split pose extraction should reject partially specified configs."""
    importer = object.__new__(RLDSAndTFDSDatasetImporterBase)
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
                "steps": {
                    "observation": {
                        "combined_pose": _FakeTensor(
                            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
                        ),
                        "position": _FakeTensor([1.0, 2.0, 3.0]),
                        "orientation": _FakeTensor([0.1, 0.2, 0.3, 0.4]),
                    }
                }
            },
            item=item,
            import_source_path="steps.observation",
            data_type=DataType.POSES,
        )


@pytest.mark.parametrize(
    ("source_name", "expected"),
    [
        ("observation", {"joint_positions": _FakeTensor([1.0, 2.0, 3.0])}),
        ("steps.observation", {"joint_positions": _FakeTensor([4.0, 5.0, 6.0])}),
        (
            "episode.steps.observation",
            {"joint_positions": _FakeTensor([7.0, 8.0, 9.0])},
        ),
    ],
)
def test_resolve_source_path_supports_dot_delimited_source_depths(
    source_name, expected
):
    """Import source path should resolve 1-3 dot-delimited keys."""
    importer = object.__new__(RLDSAndTFDSDatasetImporterBase)

    step_data = {
        "observation": {"joint_positions": _FakeTensor([1.0, 2.0, 3.0])},
        "steps": {"observation": {"joint_positions": _FakeTensor([4.0, 5.0, 6.0])}},
        "episode": {
            "steps": {"observation": {"joint_positions": _FakeTensor([7.0, 8.0, 9.0])}}
        },
    }

    resolved = importer._resolve_source_path(step_data, source_name)
    assert set(resolved.keys()) == {"joint_positions"}
    assert resolved["joint_positions"].numpy() == expected["joint_positions"].numpy()


@pytest.mark.parametrize(
    ("item_source_name", "expected"),
    [
        ("joint_positions", _FakeTensor([1.0, 2.0, 3.0])),
        ("robot.joint_positions", _FakeTensor([4.0, 5.0, 6.0])),
        ("arm.robot.joint_positions", _FakeTensor([7.0, 8.0, 9.0])),
    ],
)
def test_extract_source_data_supports_dot_delimited_item_source_name_depths(
    item_source_name, expected
):
    """Mapping source_name should resolve 1-3 dot-delimited keys."""
    importer = object.__new__(RLDSAndTFDSDatasetImporterBase)
    item = SimpleNamespace(
        source_name=item_source_name,
        index=None,
        index_range=None,
        name="joint_positions",
    )

    source = {
        "steps": {
            "observation": {
                "joint_positions": _FakeTensor([1.0, 2.0, 3.0]),
                "robot": {"joint_positions": _FakeTensor([4.0, 5.0, 6.0])},
                "arm": {"robot": {"joint_positions": _FakeTensor([7.0, 8.0, 9.0])}},
            }
        }
    }

    resolved = importer._extract_source_data(
        source=source,
        item=item,
        import_source_path="steps.observation",
        data_type=DataType.JOINT_POSITIONS,
    )
    assert resolved.numpy() == expected.numpy()


def test_record_step_resolves_mixed_dot_delimited_source_and_source_name():
    """_record_step should resolve import source and mapping source_name together."""
    importer = object.__new__(RLDSDatasetImporter)
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
        {
            "episode": {
                "steps": {"robot": {"joint_positions": _FakeTensor([0.1, 0.2, 0.3])}}
            }
        },
        timestamp=1.25,
    )

    importer._log_data.assert_called_once_with(
        DataType.JOINT_POSITIONS,
        [0.1, 0.2, 0.3],
        mapping_item,
        import_format,
        1.25,
        extrinsics=None,
        intrinsics=None,
    )
