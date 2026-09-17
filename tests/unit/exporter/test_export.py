"""Exercise the shared workflow without MCAP or a one-file-per-recording layout."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from neuracore.exporter import export as workflow
from neuracore.exporter.export import DatasetExporter, ExportFile, export_dataset


class CombinedExporter(DatasetExporter):
    """Test format that buffers episodes and publishes shared files at the end."""

    format_name = "combined-test"

    def __init__(self, fail_at=None):
        self.fail_at = fail_at
        self.ids = []
        self.output = None
        self.aborted = False

    def check_dependencies(self):
        pass

    def validate_recording(self, recording):
        # This format does not require MCAP's raw sensor manifest.
        if self.fail_at == "validate":
            raise ValueError("invalid input")

    def prepare(self, dataset, output):
        self.output = output
        (output / "episodes.partial").write_text("")
        if self.fail_at == "prepare":
            raise RuntimeError("prepare failed")

    def write_recording(self, index, recording):
        if self.fail_at == "write":
            raise KeyboardInterrupt()
        self.ids.append(recording.recording.id)
        return []

    def finalize(self):
        manifest = json.loads((self.output / "manifest.json").read_text())
        assert manifest["status"] == "finalising"
        assert manifest["files"] == []
        assert manifest["completed_recording_ids"] == self.ids
        if self.fail_at == "finalize":
            raise RuntimeError("finalize failed")
        (self.output / "episodes.partial").write_text(json.dumps(self.ids))
        (self.output / "episodes.partial").rename(self.output / "episodes.json")
        (self.output / "info.json").write_text(json.dumps({"count": len(self.ids)}))
        return [ExportFile("episodes.json"), ExportFile("info.json")]

    def abort(self):
        self.aborted = True
        (self.output / "episodes.partial").unlink(missing_ok=True)


@pytest.fixture
def dataset(monkeypatch):
    class Dataset(list):
        id = "dataset-1"
        name = "Demo"
        org_id = "org-1"

    def source(org_id, recording_id):
        return SimpleNamespace(metadata=lambda: {"id": recording_id})

    monkeypatch.setattr(workflow, "RecordingSource", source)
    return Dataset([SimpleNamespace(id="rec-1"), SimpleNamespace(id="rec-2")])


def test_format_can_publish_shared_files_at_finalization(dataset, tmp_path):
    writer = CombinedExporter()
    progress = Mock()
    path = export_dataset(dataset, tmp_path / "out", writer, progress)
    manifest = json.loads(path.read_text())
    assert manifest["format"] == "combined-test"
    assert manifest["status"] == "succeeded"
    assert manifest["files"] == [{"path": "episodes.json"}, {"path": "info.json"}]
    assert json.loads((path.parent / "episodes.json").read_text()) == ["rec-1", "rec-2"]
    assert [call.args for call in progress.call_args_list] == [
        (0, 2, "rec-1"),
        (1, 2, "rec-2"),
    ]
    assert not writer.aborted


@pytest.mark.parametrize(
    "stage,error",
    [
        ("prepare", RuntimeError),
        ("write", KeyboardInterrupt),
        ("finalize", RuntimeError),
    ],
)
def test_lifecycle_failure_aborts_and_marks_incomplete(dataset, tmp_path, stage, error):
    writer = CombinedExporter(fail_at=stage)
    output = tmp_path / "out"
    with pytest.raises(error):
        export_dataset(dataset, output, writer)
    assert writer.aborted
    assert not (output / "episodes.partial").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_format_validation_precedes_output_creation(dataset, tmp_path):
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="invalid input"):
        export_dataset(dataset, output, CombinedExporter(fail_at="validate"))
    assert not output.exists()


def test_cleanup_error_does_not_hide_original_failure(dataset, tmp_path):
    writer = CombinedExporter(fail_at="finalize")
    writer.abort = Mock(side_effect=OSError("cleanup failed"))
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="finalize failed"):
        export_dataset(dataset, output, writer)
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"
