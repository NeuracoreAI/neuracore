# cspell:ignore Fjoint
"""Read exported MCAPs back to verify data fidelity and failure handling."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from neuracore_types import DataType
from typer.testing import CliRunner

from neuracore.exporter import cli
from neuracore.exporter.export import export_recordings
from neuracore.exporter.mcap import McapExporter

mcap_reader = pytest.importorskip("mcap.reader")


@pytest.fixture
def dataset():
    class Dataset(list):
        id = "dataset-1"
        name = "Demo"
        org_id = "org-1"

    return Dataset()


@pytest.fixture
def recording():
    files = {
        "JOINT_POSITIONS/arm\\joint/trace.json": json.dumps([
            {"timestamp": 1.250000001, "value": 0.75},
            {"timestamp": 1.5, "value": -0.1},
        ]).encode(),
        "RGB_IMAGES/front/trace.json": json.dumps(
            [{"timestamp": 1.3, "frame_idx": 0}]
        ).encode(),
        "POINT_CLOUDS/lidar/trace.json": json.dumps(
            [{"timestamp": 1.4, "frame_idx": 0, "offset": 0, "length": 4}]
        ).encode(),
        "RGB_IMAGES/front/lossless.mp4": b"original-video-bytes",
        "POINT_CLOUDS/lidar/trace.bin": b"cloud-bytes",
    }
    return SimpleNamespace(
        id="recording-1",
        name="recording-1",
        robot_id="robot-1",
        instance=0,
        start_time=1.25,
        end_time=2.0,
        deleted=False,
        metadata=Mock(model_dump=Mock(return_value={"name": "recording-1"})),
        data_types={
            DataType.JOINT_POSITIONS,
            DataType.RGB_IMAGES,
            DataType.POINT_CLOUDS,
        },
        sensor_manifest={
            DataType.JOINT_POSITIONS: ["arm/joint"],
            DataType.RGB_IMAGES: ["front"],
            DataType.POINT_CLOUDS: ["lidar"],
        },
        download=Mock(side_effect=files.__getitem__),
        _files=files,
    )


def test_round_trip_raw_timestamps_payloads_and_attachments(
    dataset, recording, tmp_path
):
    writer = McapExporter()
    progress = Mock()
    manifest_path = export_recordings(
        dataset, [recording], tmp_path / "export", writer, progress
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "succeeded"
    assert manifest["recording_ids"] == ["recording-1"]
    assert manifest["files"] == [
        {"path": "nc_recording-1.mcap", "recording_id": "recording-1"}
    ]
    with (manifest_path.parent / "nc_recording-1.mcap").open("rb") as stream:
        reader = mcap_reader.make_reader(stream, validate_crcs=True)
        messages = list(reader.iter_messages())
        joints = [
            message
            for _, channel, message in messages
            if channel.topic == "/neuracore/JOINT_POSITIONS/arm%2Fjoint"
        ]
        assert [m.log_time for m in joints] == [1250000001, 1500000000]
        assert json.loads(joints[0].data) == {"timestamp": 1.250000001, "value": 0.75}
        assert len(messages) == 4
        for schema, channel, message in messages:
            assert schema.encoding == "jsonschema"
            assert channel.message_encoding == "json"
            assert message.publish_time == message.log_time
        attachments = {a.name: a.data for a in reader.iter_attachments()}
        assert attachments == {
            "RGB_IMAGES/front/lossless.mp4": recording._files[
                "RGB_IMAGES/front/lossless.mp4"
            ],
            "POINT_CLOUDS/lidar/trace.bin": recording._files[
                "POINT_CLOUDS/lidar/trace.bin"
            ],
        }
        camera = next(c for _, c, _ in messages if c.metadata["sensor_name"] == "front")
        assert camera.metadata["attachment"] in attachments
        saved_metadata = next(reader.iter_metadata())
        assert json.loads(saved_metadata.metadata["json"])["id"] == "recording-1"
    progress.assert_called_once_with(0, 1, "recording-1")


def test_failure_does_not_publish_partial_mcap(dataset, recording, tmp_path):
    recording.download = Mock(side_effect=RuntimeError("download failed"))
    output = tmp_path / "export"
    with pytest.raises(RuntimeError, match="download failed"):
        export_recordings(dataset, [recording], output, McapExporter())
    assert not list(output.glob("*.partial"))
    assert not list(output.glob("*.mcap"))
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_existing_output_is_untouched(dataset, recording, tmp_path):
    output = tmp_path / "export"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        export_recordings(dataset, [recording], output, McapExporter())
    assert sentinel.read_text() == "keep"
    recording.download.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"sensor_manifest": {}},
        {"end_time": None},
        {"deleted": True},
        {"sensor_manifest": {DataType.JOINT_POSITIONS: ["arm"]}},
    ],
)
def test_rejects_incomplete_recordings_before_writing(
    dataset, recording, tmp_path, changes
):
    for key, value in changes.items():
        setattr(recording, key, value)
    output = tmp_path / "export"
    with pytest.raises(ValueError):
        export_recordings(dataset, [recording], output, McapExporter())
    assert not output.exists()


def test_video_fallback_only_on_not_found(dataset, recording, tmp_path):
    files = dict(recording._files)
    del files["RGB_IMAGES/front/lossless.mp4"]
    files["RGB_IMAGES/front/lossy.mp4"] = b"lossy-original"
    response = requests.Response()
    response.status_code = 404

    def download(path):
        if path.endswith("lossless.mp4"):
            raise requests.HTTPError(response=response)
        return files[path]

    recording.download = Mock(side_effect=download)
    manifest_path = export_recordings(
        dataset, [recording], tmp_path / "export", McapExporter()
    )
    with (manifest_path.parent / "nc_recording-1.mcap").open("rb") as stream:
        attachments = list(mcap_reader.make_reader(stream).iter_attachments())
    assert any(a.name.endswith("lossy.mp4") for a in attachments)

    response.status_code = 403
    recording.download = Mock(side_effect=download)
    with pytest.raises(requests.HTTPError):
        export_recordings(dataset, [recording], tmp_path / "forbidden", McapExporter())


def test_cli_requires_dataset(tmp_path, monkeypatch):
    login = Mock()
    monkeypatch.setattr(cli, "login", login)
    result = CliRunner().invoke(cli.export_app, ["--output", str(tmp_path / "out")])
    assert result.exit_code != 0
    login.assert_not_called()


def test_cli_exports_selected_dataset(dataset, recording, tmp_path, monkeypatch):
    dataset.append(recording)
    monkeypatch.setattr(cli, "login", Mock())
    monkeypatch.setattr(cli, "get_dataset", Mock(return_value=dataset))
    output = tmp_path / "out"
    result = CliRunner().invoke(
        cli.export_app, ["--dataset", "Demo", "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert "Export complete" in result.output
    assert json.loads((output / "manifest.json").read_text())["status"] == "succeeded"


def test_registered_cli_writes_dataset(dataset, recording, tmp_path, monkeypatch):
    from neuracore.core.cli.app import app

    dataset.append(recording)
    monkeypatch.setattr(cli, "login", Mock())
    monkeypatch.setattr(cli, "get_dataset", Mock(return_value=dataset))
    output = tmp_path / "out"
    result = CliRunner().invoke(
        app, ["export", "mcap", "--dataset", "Demo", "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads((output / "manifest.json").read_text())["status"] == "succeeded"
    assert "[1/1] Exporting recording recording-1" in result.output


def test_second_recording_failure_retains_completed_file(dataset, recording, tmp_path):
    second = SimpleNamespace(**{**vars(recording)})
    second.id = "recording-2"
    dataset.extend([recording, second])

    def fail_on_second_recording(completed, total, recording_name):
        if completed == 1:
            second.download = Mock(side_effect=RuntimeError("download failed"))

    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="download failed"):
        export_recordings(
            dataset, list(dataset), output, McapExporter(), fail_on_second_recording
        )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["files"] == [
        {"path": "nc_recording-1.mcap", "recording_id": "recording-1"}
    ]
    assert (output / "nc_recording-1.mcap").exists()
    assert not list(output.glob("*.partial"))
