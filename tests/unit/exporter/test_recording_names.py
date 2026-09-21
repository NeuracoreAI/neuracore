"""Verify display names in export filenames, progress and validation errors."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from neuracore_types import DataType

from neuracore.exporter.export import export_recordings
from neuracore.exporter.mcap import McapExporter


@pytest.fixture
def recording():
    return SimpleNamespace(
        id="internal-recording-id",
        name="Pick and place",
        robot_id="robot-id",
        instance=0,
        start_time=1.0,
        end_time=2.0,
        deleted=False,
        metadata=Mock(model_dump=Mock(return_value={})),
        data_types={DataType.JOINT_POSITIONS},
        sensor_manifest={DataType.JOINT_POSITIONS: ["arm"]},
        download=Mock(return_value=b'[{"timestamp": 1.0, "value": 0.5}]'),
    )


@pytest.fixture
def writer(monkeypatch):
    # Filename and workflow checks do not require the optional MCAP dependency.
    package = ModuleType("mcap")
    module = ModuleType("mcap.writer")
    module.Writer = Mock()
    module.CompressionType = SimpleNamespace(NONE=0)
    package.writer = module
    monkeypatch.setitem(sys.modules, "mcap", package)
    monkeypatch.setitem(sys.modules, "mcap.writer", module)
    return McapExporter()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Pick and place", "nc_Pick and place.mcap"),
        ("../arm\\camera:take?", "nc__arm_camera_take_.mcap"),
        ("", "nc_recording.mcap"),
    ],
)
def test_recording_name_filename(writer, recording, tmp_path, name, expected):
    recording.name = name
    writer.prepare(None, tmp_path)
    files = writer.write_recording(0, recording)
    assert files[0].path == expected
    assert (tmp_path / expected).is_file()
    assert recording.id not in files[0].path


def test_duplicate_names_do_not_overwrite(writer, recording, tmp_path):
    writer.prepare(None, tmp_path)
    first = writer.write_recording(0, recording)[0]
    (tmp_path / first.path).write_bytes(b"original recording")
    second = writer.write_recording(1, recording)[0]
    assert second.path == "nc_Pick and place_2.mcap"
    assert (tmp_path / first.path).read_bytes() == b"original recording"
    recording.name = "Pick and place_2"
    third = writer.write_recording(2, recording)[0]
    assert third.path == "nc_Pick and place_2_2.mcap"


def test_progress_uses_name_and_manifest_retains_id(writer, recording, tmp_path):
    progress = Mock()
    dataset = SimpleNamespace(id="dataset-id", name="Demo")
    path = export_recordings(dataset, [recording], tmp_path / "out", writer, progress)
    progress.assert_called_once_with(0, 1, "Pick and place")
    manifest = json.loads(path.read_text())
    assert manifest["files"] == [
        {"path": "nc_Pick and place.mcap", "recording_id": recording.id}
    ]
    assert manifest["status"] == "succeeded"


def test_validation_error_uses_name(writer, recording):
    recording.end_time = None
    with pytest.raises(ValueError) as error:
        writer.validate_recording(recording)
    assert recording.name in str(error.value)
    assert recording.id not in str(error.value)
