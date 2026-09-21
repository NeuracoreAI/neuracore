"""Round-trip and validation tests for the LeRobot exporter."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from neuracore_types import DataType
from typer.testing import CliRunner

from neuracore.exporter import cli
from neuracore.exporter.export import export_recordings
from neuracore.exporter.lerobot import LeRobotExporter

pq = pytest.importorskip("pyarrow.parquet")


def _values(names_to_values: dict) -> dict:
    return {
        name: SimpleNamespace(value=value) for name, value in names_to_values.items()
    }


def _frame(fill: int) -> np.ndarray:
    return np.full((4, 4, 3), fill, dtype=np.uint8)


def _point(
    joints: dict,
    targets: dict | None = None,
    camera_frame: np.ndarray | None = None,
    text: str | None = None,
) -> SimpleNamespace:
    data = {DataType.JOINT_POSITIONS: _values(joints)}
    if targets is not None:
        data[DataType.JOINT_TARGET_POSITIONS] = _values(targets)
    if camera_frame is not None:
        data[DataType.RGB_IMAGES] = {"front": SimpleNamespace(frame=camera_frame)}
    if text is not None:
        data[DataType.LANGUAGE] = {"instruction": SimpleNamespace(text=text)}
    return SimpleNamespace(data=data)


@pytest.fixture
def dataset():
    class Dataset(list):
        id = "dataset-1"
        name = "Demo"
        org_id = "org-1"

    return Dataset()


def _make_recording(recording_id: str, points: list) -> SimpleNamespace:
    return SimpleNamespace(
        id=recording_id,
        name=recording_id,
        robot_id="robot-1",
        instance=0,
        start_time=1.0,
        end_time=2.0,
        deleted=False,
        data_types={
            DataType.JOINT_POSITIONS,
            DataType.JOINT_TARGET_POSITIONS,
            DataType.RGB_IMAGES,
        },
        sensor_manifest={
            DataType.JOINT_POSITIONS: ["shoulder", "elbow"],
            DataType.JOINT_TARGET_POSITIONS: ["shoulder", "elbow"],
            DataType.RGB_IMAGES: ["front"],
        },
        synchronize=Mock(return_value=points),
    )


@pytest.fixture
def recording():
    points = [
        _point(
            {"shoulder": 0.0, "elbow": 1.0},
            {"shoulder": 0.1, "elbow": 1.1},
            _frame(0),
            text="pick it up",
        ),
        _point(
            {"shoulder": 0.2, "elbow": 1.2}, {"shoulder": 0.3, "elbow": 1.3}, _frame(50)
        ),
    ]
    return _make_recording("recording-1", points)


def test_round_trip_writes_parquet_video_and_metadata(dataset, recording, tmp_path):
    manifest_path = export_recordings(
        dataset, [recording], tmp_path / "export", LeRobotExporter(fps=10)
    )
    recording.synchronize.assert_called_once_with(frequency=10)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "succeeded"

    root = manifest_path.parent
    table = pq.read_table(root / "data" / "chunk-000" / "episode_000000.parquet")
    # Columns are ordered by sorted joint name: "elbow" before "shoulder".
    np.testing.assert_allclose(
        table.column("observation.state").to_pylist(),
        [[1.0, 0.0], [1.2, 0.2]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        table.column("action").to_pylist(), [[1.1, 0.1], [1.3, 0.3]], atol=1e-6
    )
    assert table.column("frame_index").to_pylist() == [0, 1]
    assert table.column("episode_index").to_pylist() == [0, 0]
    assert table.column("index").to_pylist() == [0, 1]
    assert table.column("task_index").to_pylist() == [0, 0]
    np.testing.assert_allclose(
        table.column("timestamp").to_pylist(), [0.0, 0.1], atol=1e-6
    )

    video_path = (
        root
        / "videos"
        / "chunk-000"
        / "observation.images.front"
        / "episode_000000.mp4"
    )
    assert video_path.exists()
    av = pytest.importorskip("av")
    with av.open(str(video_path)) as container:
        assert len(list(container.decode(video=0))) == 2

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v2.1"
    assert info["fps"] == 10
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 2
    assert info["features"]["observation.state"]["names"] == ["elbow", "shoulder"]
    assert info["features"]["action"]["names"] == ["elbow", "shoulder"]
    assert info["features"]["observation.images.front"]["shape"] == [4, 4, 3]

    tasks = [
        json.loads(line)
        for line in (root / "meta" / "tasks.jsonl").read_text().splitlines()
    ]
    assert tasks == [{"task_index": 0, "task": "pick it up"}]

    episodes = [
        json.loads(line)
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    assert episodes == [{"episode_index": 0, "tasks": ["pick it up"], "length": 2}]

    stats = [
        json.loads(line)
        for line in (root / "meta" / "episodes_stats.jsonl").read_text().splitlines()
    ]
    assert stats[0]["episode_index"] == 0
    image_stats = stats[0]["stats"]["observation.images.front"]
    assert np.array(image_stats["min"]).shape == (3, 1, 1)
    assert np.array(image_stats["count"]) == [2]

    assert manifest["files"] == [
        {
            "path": "data/chunk-000/episode_000000.parquet",
            "recording_id": "recording-1",
        },
        {
            "path": "videos/chunk-000/observation.images.front/episode_000000.mp4",
            "recording_id": "recording-1",
        },
        {"path": "meta/info.json"},
        {"path": "meta/tasks.jsonl"},
        {"path": "meta/episodes.jsonl"},
        {"path": "meta/episodes_stats.jsonl"},
    ]


def test_second_episode_continues_indices_and_reuses_task(dataset, recording, tmp_path):
    second_points = [
        _point(
            {"shoulder": 1.0, "elbow": 1.0},
            {"shoulder": 1.1, "elbow": 1.1},
            _frame(0),
            text="pick it up",
        )
    ]
    second = _make_recording("recording-2", second_points)

    manifest_path = export_recordings(
        dataset, [recording, second], tmp_path / "export", LeRobotExporter(fps=10)
    )
    root = manifest_path.parent

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 3

    second_table = pq.read_table(root / "data" / "chunk-000" / "episode_000001.parquet")
    assert second_table.column("episode_index").to_pylist() == [1]
    assert second_table.column("frame_index").to_pylist() == [0]
    # "index" keeps counting across episodes: recording-1 already used indices 0-1.
    assert second_table.column("index").to_pylist() == [2]

    tasks = [
        json.loads(line)
        for line in (root / "meta" / "tasks.jsonl").read_text().splitlines()
    ]
    assert tasks == [{"task_index": 0, "task": "pick it up"}]


def test_rejects_mismatched_schema_across_recordings(dataset, recording, tmp_path):
    other_points = [_point({"only_joint": 0.0}, camera_frame=_frame(0))]
    other = _make_recording("recording-2", other_points)
    other.sensor_manifest = {DataType.JOINT_POSITIONS: ["only_joint"]}
    other.data_types = {DataType.JOINT_POSITIONS}

    output = tmp_path / "export"
    with pytest.raises(ValueError, match="different joint/camera layout"):
        export_recordings(dataset, [recording, other], output, LeRobotExporter(fps=10))
    assert not output.exists()


def test_exports_camera_only_recording_without_joint_positions(dataset, tmp_path):
    points = [_point({}, camera_frame=_frame(0))]
    camera_only = _make_recording("recording-1", points)
    camera_only.sensor_manifest = {DataType.RGB_IMAGES: ["front"]}
    camera_only.data_types = {DataType.RGB_IMAGES}

    manifest_path = export_recordings(
        dataset, [camera_only], tmp_path / "export", LeRobotExporter(fps=10)
    )
    root = manifest_path.parent
    assert json.loads(manifest_path.read_text())["status"] == "succeeded"

    info = json.loads((root / "meta" / "info.json").read_text())
    assert "observation.state" not in info["features"]
    assert "action" not in info["features"]
    assert "observation.images.front" in info["features"]

    table = pq.read_table(root / "data" / "chunk-000" / "episode_000000.parquet")
    assert "observation.state" not in table.column_names
    assert "action" not in table.column_names


def test_failure_does_not_publish_partial_episode(dataset, recording, tmp_path):
    recording.synchronize = Mock(side_effect=RuntimeError("sync failed"))
    output = tmp_path / "export"
    with pytest.raises(RuntimeError, match="sync failed"):
        export_recordings(dataset, [recording], output, LeRobotExporter(fps=10))
    assert not list((output / "data" / "chunk-000").glob("*.parquet"))
    assert not list(output.rglob("*.partial"))
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"


def test_cli_exports_lerobot_dataset(dataset, recording, tmp_path, monkeypatch):
    from neuracore.core.cli.app import app

    dataset.append(recording)
    monkeypatch.setattr(cli, "login", Mock())
    monkeypatch.setattr(cli, "get_dataset", Mock(return_value=dataset))
    output = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        [
            "export",
            "lerobot",
            "--dataset",
            "Demo",
            "--output",
            str(output),
            "--fps",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads((output / "manifest.json").read_text())["status"] == "succeeded"
