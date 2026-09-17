# cspell:ignore Fjoint
"""Read exported MCAPs back to verify data fidelity and failure handling."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from typer.testing import CliRunner

from neuracore.exporter import cli
from neuracore.exporter import export as workflow
from neuracore.exporter import mcap as exporter
from neuracore.exporter import source as recording_source

mcap_reader = pytest.importorskip("mcap.reader")


@pytest.fixture
def export_source(monkeypatch):
    metadata = {
        "id": "recording-1",
        "start_time": 1.25,
        "end_time": 2.0,
        "data_types": ["JOINT_POSITIONS", "RGB_IMAGES", "POINT_CLOUDS"],
        "sensor_manifest": {
            "JOINT_POSITIONS": ["arm/joint"],
            "RGB_IMAGES": ["front"],
            "POINT_CLOUDS": ["lidar"],
        },
    }
    samples = {
        "JOINT_POSITIONS/arm\\joint/trace.json": [
            {"timestamp": 1.250000001, "value": 0.75},
            {"timestamp": 1.5, "value": -0.1},
        ],
        "RGB_IMAGES/front/trace.json": [{"timestamp": 1.3, "frame_idx": 0}],
        "POINT_CLOUDS/lidar/trace.json": [
            {"timestamp": 1.4, "frame_idx": 0, "offset": 0, "length": 4}
        ],
    }
    files = {name: json.dumps(trace).encode() for name, trace in samples.items()}
    files["RGB_IMAGES/front/lossless.mp4"] = b"original-video-bytes"
    files["POINT_CLOUDS/lidar/trace.bin"] = b"cloud-bytes"
    source = Mock()
    source.metadata.side_effect = lambda: copy.deepcopy(metadata)
    source.download.side_effect = files.__getitem__
    monkeypatch.setattr(workflow, "RecordingSource", Mock(return_value=source))

    # A small concrete iterable avoids mocking the SDK's pagination internals.
    class Dataset(list):
        id = "dataset-1"
        name = "Demo"
        org_id = "org-1"

    return Dataset([SimpleNamespace(id="recording-1")]), source, metadata, files


def test_round_trip_raw_timestamps_payloads_and_attachments(export_source, tmp_path):
    dataset, source, metadata, files = export_source
    progress = Mock()
    manifest = exporter.export_dataset_mcap(dataset, tmp_path / "export", progress)
    document = json.loads(manifest.read_text())
    assert document["status"] == "succeeded"
    assert document["recording_ids"] == ["recording-1"]
    assert document["files"] == [{"recording_id": "recording-1", "path": "000000.mcap"}]
    with (manifest.parent / "000000.mcap").open("rb") as stream:
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
            "RGB_IMAGES/front/lossless.mp4": files["RGB_IMAGES/front/lossless.mp4"],
            "POINT_CLOUDS/lidar/trace.bin": files["POINT_CLOUDS/lidar/trace.bin"],
        }
        camera = next(c for _, c, _ in messages if c.metadata["sensor_name"] == "front")
        assert camera.metadata["attachment"] in attachments
        saved_metadata = next(reader.iter_metadata())
        assert json.loads(saved_metadata.metadata["json"]) == metadata
    progress.assert_called_once_with(0, 1, "recording-1")


@pytest.mark.parametrize("failure", [RuntimeError("broken"), KeyboardInterrupt()])
def test_failure_does_not_publish_partial_mcap(export_source, tmp_path, failure):
    dataset, source, _, _ = export_source
    source.download.side_effect = failure
    output = tmp_path / "export"
    with pytest.raises(type(failure)):
        exporter.export_dataset_mcap(dataset, output)
    assert list(output.iterdir()) == [output / "manifest.json"]
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_existing_output_is_untouched(export_source, tmp_path):
    dataset, source, _, _ = export_source
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        exporter.export_dataset_mcap(dataset, tmp_path)
    assert sentinel.read_text() == "keep"
    source.metadata.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"sensor_manifest": {}},
        {"end_time": None},
        {"deleted": True},
        {"sensor_manifest": {"JOINT_POSITIONS": ["arm"]}},
    ],
)
def test_rejects_incomplete_recordings_before_writing(export_source, tmp_path, changes):
    dataset, _, metadata, _ = export_source
    metadata.update(changes)
    output = tmp_path / "export"
    with pytest.raises(ValueError):
        exporter.export_dataset_mcap(dataset, output)
    assert not output.exists()


def test_video_fallback_only_on_not_found(export_source, tmp_path):
    dataset, source, _, files = export_source
    files["RGB_IMAGES/front/lossy.mp4"] = b"lossy-original"
    response = requests.Response()
    response.status_code = 404

    def download(path):
        if path.endswith("lossless.mp4"):
            raise requests.HTTPError(response=response)
        return files[path]

    source.download.side_effect = download
    manifest = exporter.export_dataset_mcap(dataset, tmp_path / "export")
    with (manifest.parent / "000000.mcap").open("rb") as stream:
        attachments = list(mcap_reader.make_reader(stream).iter_attachments())
    assert any(a.name.endswith("lossy.mp4") for a in attachments)
    response.status_code = 403
    with pytest.raises(requests.HTTPError):
        exporter.export_dataset_mcap(dataset, tmp_path / "forbidden")


def test_storage_download_does_not_send_api_token(requests_mock, monkeypatch):
    monkeypatch.setattr(
        recording_source,
        "get_auth",
        lambda: SimpleNamespace(get_headers=lambda: {"Authorization": "Bearer secret"}),
    )
    source = recording_source.RecordingSource("org", "rec")
    requests_mock.get(
        f"{source.url}/download_url", json={"url": "https://storage.test/file"}
    )
    storage = requests_mock.get("https://storage.test/file", content=b"data")
    assert source.download("JOINT_POSITIONS/arm/trace.json") == b"data"
    assert "Authorization" not in storage.last_request.headers


@pytest.mark.parametrize("selection", [[], ["--dataset", "Demo", "--dataset-id", "id"]])
def test_cli_requires_exactly_one_selector(tmp_path, selection, monkeypatch):
    login = Mock()
    monkeypatch.setattr(cli, "login", login)
    result = CliRunner().invoke(
        cli.export_app, ["--output", str(tmp_path / "out"), *selection]
    )
    assert result.exit_code == 2
    login.assert_not_called()


@pytest.mark.parametrize(
    "flag,kwargs",
    [
        ("--dataset", {"name": "Demo", "id": None}),
        ("--dataset-id", {"name": None, "id": "Demo"}),
    ],
)
def test_cli_exports_selected_dataset(tmp_path, monkeypatch, flag, kwargs):
    monkeypatch.setattr(cli, "login", Mock())
    get_dataset = Mock(return_value=object())
    export = Mock(return_value=tmp_path / "out" / "manifest.json")
    monkeypatch.setattr(cli, "get_dataset", get_dataset)
    monkeypatch.setattr(cli, "export_dataset", export)
    result = CliRunner().invoke(
        cli.export_app, [flag, "Demo", "-o", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.output
    get_dataset.assert_called_once_with(**kwargs)
    assert "Export complete" in result.output


def test_empty_dataset_rejected(export_source, tmp_path):
    dataset, _, _, _ = export_source
    dataset.clear()
    with pytest.raises(ValueError, match="no recordings"):
        exporter.export_dataset_mcap(dataset, tmp_path / "out")


def test_registered_cli_writes_dataset(export_source, tmp_path, monkeypatch):
    from neuracore.core.cli.app import app

    dataset, _, _, _ = export_source
    monkeypatch.setattr(cli, "login", Mock())
    monkeypatch.setattr(cli, "get_dataset", Mock(return_value=dataset))
    output = tmp_path / "out"
    result = CliRunner().invoke(
        app, ["export", "mcap", "--dataset", "Demo", "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads((output / "manifest.json").read_text())["status"] == "succeeded"
    assert "[1/1] Exporting recording recording-1" in result.output


def test_second_recording_failure_retains_completed_file(export_source, tmp_path):
    dataset, source, metadata, _ = export_source
    dataset.append(SimpleNamespace(id="recording-2"))
    second = dict(metadata, id="recording-2")
    source.metadata.side_effect = [metadata, second]

    def fail_on_second_recording(completed, total, recording_id):
        if completed == 1:
            source.download.side_effect = RuntimeError("download failed")

    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="download failed"):
        exporter.export_dataset_mcap(dataset, output, progress=fail_on_second_recording)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["files"] == [{"recording_id": "recording-1", "path": "000000.mcap"}]
    assert (output / "000000.mcap").exists()
    assert not (output / "000001.mcap").exists()
    assert not list(output.glob("*.partial"))
