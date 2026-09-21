"""Export synchronized Neuracore recordings to LeRobot v2.1 dataset files."""

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from neuracore_types import DataType
from neuracore_types.utils.name_utils import to_safe_name

from neuracore.core.data.dataset import Dataset
from neuracore.core.data.recording import Recording
from neuracore.exporter.export import DatasetExporter, ExportFile, export_recordings

LeRobot_Codebase = "v2.1"
CHUNK = "chunk-000"
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)


class LeRobotExporter(DatasetExporter):
    """Write a LeRobot v2.1 dataset: one parquet and video set per recording.

    Every recording is synchronized at a fixed frequency and must share the same
    optional joint names (``JOINT_POSITIONS``, exported as ``observation.state``),
    the same optional target joints (``JOINT_TARGET_POSITIONS``, exported as
    ``action``) and the same optional RGB cameras (``observation.images.<name>``),
    since LeRobot datasets require one feature layout across all episodes. Other
    Neuracore data types (grippers, end-effector poses, point clouds, depth) are
    not exported.
    """

    format_name = "neuracore-lerobot-v2.1"

    def __init__(self, fps: int) -> None:
        """Create a writer that synchronizes every recording at ``fps`` Hz."""
        if fps <= 0:
            raise ValueError("fps must be positive.")
        self.fps = fps
        self.output: Path | None = None
        self._schema: tuple[list[str], list[str], list[str]] | None = None
        self._robot_type: str | None = None
        self._camera_sizes: dict[str, tuple[int, int]] = {}
        self._tasks: dict[str, int] = {}
        self._episodes: list[dict] = []
        self._episode_stats: list[dict] = []
        self._total_frames = 0

    @staticmethod
    def _vector_stats(array: np.ndarray) -> dict[str, list]:
        """Per-column min/max/mean/std/count for an ``(n, d)`` array."""
        return {
            "min": np.min(array, axis=0).tolist(),
            "max": np.max(array, axis=0).tolist(),
            "mean": np.mean(array, axis=0).tolist(),
            "std": np.std(array, axis=0).tolist(),
            "count": [len(array)],
        }

    @staticmethod
    def _scalar_stats(array: np.ndarray) -> dict[str, list]:
        """min/max/mean/std/count for an ``(n,)`` array, kept as length-1 vectors."""
        return {
            "min": [float(array.min())],
            "max": [float(array.max())],
            "mean": [float(array.mean())],
            "std": [float(array.std())],
            "count": [len(array)],
        }

    @staticmethod
    def _image_stats(frames: list[np.ndarray]) -> dict[str, list]:
        """Compute per-channel statistics for HWC uint8 frames normalized to [0, 1]."""
        stacked = np.stack(frames).astype(np.float64) / 255.0
        axes = (0, 1, 2)
        return {
            "min": stacked.min(axis=axes).reshape(-1, 1, 1).tolist(),
            "max": stacked.max(axis=axes).reshape(-1, 1, 1).tolist(),
            "mean": stacked.mean(axis=axes).reshape(-1, 1, 1).tolist(),
            "std": stacked.std(axis=axes).reshape(-1, 1, 1).tolist(),
            "count": [len(frames)],
        }

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    @staticmethod
    def _write_atomic(destination: Path, write_fn: Callable[[Path], None]) -> None:
        """Write via a sibling ``.partial`` file, publishing only on success."""
        partial = destination.with_suffix(destination.suffix + ".partial")
        try:
            write_fn(partial)
            partial.rename(destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    def check_dependencies(self) -> None:
        """Require only parquet support; video encoding uses the core av package."""
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot export requires optional dependencies: "
                'pip install "neuracore[export]"'
            ) from exc

    def validate_recording(self, recording: Recording) -> None:
        """Require a complete sensor manifest with a layout shared across recordings."""
        super().validate_recording(recording)
        manifest = recording.sensor_manifest
        schema = (
            sorted(manifest.get(DataType.JOINT_POSITIONS, [])),
            sorted(manifest.get(DataType.JOINT_TARGET_POSITIONS, [])),
            sorted(manifest.get(DataType.RGB_IMAGES, [])),
        )
        if self._schema is None:
            self._schema = schema
            self._robot_type = recording.robot_id
        elif schema != self._schema:
            raise ValueError(
                f"Recording {recording.name} has a different joint/camera layout than "
                "the rest of the dataset. LeRobot export requires every recording to "
                "share the same observation.state, action and camera names."
            )

    def prepare(self, dataset: Dataset, output: Path) -> None:
        """Create the LeRobot directory layout."""
        if self._schema is None:
            raise RuntimeError("Validate recordings before preparing the exporter.")
        self.output = output
        (output / "meta").mkdir(parents=True)
        (output / "data" / CHUNK).mkdir(parents=True)
        videos = output / "videos" / CHUNK
        _, _, camera_names = self._schema
        for name in camera_names:
            (videos / f"observation.images.{to_safe_name(name)}").mkdir(parents=True)

    def _encode_video(self, frames: list[np.ndarray], path: Path) -> None:
        import av

        height, width = frames[0].shape[:2]
        container = av.open(str(path), mode="w", format="mp4")
        try:
            stream = container.add_stream("libx264", rate=self.fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            for frame in frames:
                packet_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(packet_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()

    def write_recording(self, index: int, recording: Recording) -> list[ExportFile]:
        """Synchronize one recording and publish its parquet and video files."""
        if self.output is None or self._schema is None:
            raise RuntimeError("Prepare the exporter before writing recordings.")
        state_names, action_names, camera_names = self._schema

        synced = recording.synchronize(frequency=self.fps)
        num_frames = len(synced)
        if num_frames == 0:
            raise ValueError(
                f"Recording {recording.name} has no frames at {self.fps} fps."
            )

        states = (
            np.empty((num_frames, len(state_names)), dtype=np.float32)
            if state_names
            else None
        )
        actions = (
            np.empty((num_frames, len(action_names)), dtype=np.float32)
            if action_names
            else None
        )
        camera_frames: dict[str, list[np.ndarray]] = {name: [] for name in camera_names}
        task_text = recording.name

        for i, point in enumerate(synced):
            if states is not None:
                joints = point.data[DataType.JOINT_POSITIONS]
                states[i] = [joints[name].value for name in state_names]
            if actions is not None:
                targets = point.data[DataType.JOINT_TARGET_POSITIONS]
                actions[i] = [targets[name].value for name in action_names]
            for name in camera_names:
                frame = point.data[DataType.RGB_IMAGES][name].frame
                camera_frames[name].append(np.array(frame))
            if i == 0:
                language = point.data.get(DataType.LANGUAGE)
                if language:
                    task_text = next(iter(language.values())).text

        task_index = self._tasks.setdefault(task_text, len(self._tasks))
        frame_index = np.arange(num_frames, dtype=np.int64)
        episode_index = np.full(num_frames, index, dtype=np.int64)
        dataset_index = frame_index + self._total_frames
        timestamp = (frame_index / self.fps).astype(np.float32)
        task_index_col = np.full(num_frames, task_index, dtype=np.int64)

        import pyarrow as pa
        import pyarrow.parquet as pq

        float_list = pa.list_(pa.float32())
        columns: dict[str, pa.Array] = {
            "timestamp": pa.array(timestamp, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(episode_index, type=pa.int64()),
            "index": pa.array(dataset_index, type=pa.int64()),
            "task_index": pa.array(task_index_col, type=pa.int64()),
        }
        if states is not None:
            columns["observation.state"] = pa.array(states.tolist(), type=float_list)
        if actions is not None:
            columns["action"] = pa.array(actions.tolist(), type=float_list)

        parquet_name = f"episode_{index:06d}.parquet"
        parquet_path = self.output / "data" / CHUNK / parquet_name
        video_paths = {
            name: (
                self.output
                / "videos"
                / CHUNK
                / f"observation.images.{to_safe_name(name)}"
                / f"episode_{index:06d}.mp4"
            )
            for name in camera_names
        }

        written: list[Path] = []
        try:
            self._write_atomic(
                parquet_path, lambda p: pq.write_table(pa.table(columns), p)
            )
            written.append(parquet_path)
            for name, video_path in video_paths.items():
                frames = camera_frames[name]
                height, width = frames[0].shape[:2]
                self._camera_sizes.setdefault(name, (width, height))
                self._write_atomic(video_path, partial(self._encode_video, frames))
                written.append(video_path)
        except BaseException:
            for path in written:
                path.unlink(missing_ok=True)
            raise

        self._episodes.append({
            "episode_index": index,
            "tasks": [task_text],
            "length": num_frames,
        })
        stats = {
            "timestamp": self._scalar_stats(timestamp.astype(np.float64)),
            "frame_index": self._scalar_stats(frame_index.astype(np.float64)),
            "episode_index": self._scalar_stats(episode_index.astype(np.float64)),
            "index": self._scalar_stats(dataset_index.astype(np.float64)),
            "task_index": self._scalar_stats(task_index_col.astype(np.float64)),
        }
        if states is not None:
            stats["observation.state"] = self._vector_stats(states)
        if actions is not None:
            stats["action"] = self._vector_stats(actions)
        for name in camera_names:
            stats[f"observation.images.{to_safe_name(name)}"] = self._image_stats(
                camera_frames[name]
            )
        self._episode_stats.append({"episode_index": index, "stats": stats})
        self._total_frames += num_frames

        files = [ExportFile(f"data/{CHUNK}/{parquet_name}", recording.id)]
        for video_path in video_paths.values():
            relative_path = str(video_path.relative_to(self.output))
            files.append(ExportFile(relative_path, recording.id))
        return files

    def finalize(self) -> list[ExportFile]:
        """Write dataset-wide metadata and finish the export."""
        if self.output is None or self._schema is None:
            raise RuntimeError("Prepare the LeRobot exporter before finalizing.")
        state_names, action_names, camera_names = self._schema

        features: dict[str, dict[str, object]] = {}
        if state_names:
            features["observation.state"] = {
                "dtype": "float32",
                "shape": [len(state_names)],
                "names": state_names,
            }
        if action_names:
            features["action"] = {
                "dtype": "float32",
                "shape": [len(action_names)],
                "names": action_names,
            }
        for name in camera_names:
            width, height = self._camera_sizes[name]
            features[f"observation.images.{to_safe_name(name)}"] = {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": self.fps,
                    "video.height": height,
                    "video.width": width,
                    "video.channels": 3,
                    "video.codec": "h264",
                    "video.pix_format": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        features.update({
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        })

        info = {
            "codebase_version": LeRobot_Codebase,
            "robot_type": self._robot_type,
            "total_episodes": len(self._episodes),
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "total_videos": len(self._episodes) * len(camera_names),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self._episodes)}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if camera_names else None,
            "features": features,
        }
        tasks = [{"task_index": idx, "task": text} for text, idx in self._tasks.items()]

        meta = self.output / "meta"
        episode_stats = self._episode_stats
        files = {
            meta / "info.json": lambda p: p.write_text(json.dumps(info, indent=4)),
            meta / "tasks.jsonl": lambda p: self._write_jsonl(p, tasks),
            meta / "episodes.jsonl": lambda p: self._write_jsonl(p, self._episodes),
            meta
            / "episodes_stats.jsonl": lambda p: self._write_jsonl(p, episode_stats),
        }
        written: list[Path] = []
        try:
            for path, write_fn in files.items():
                self._write_atomic(path, write_fn)
                written.append(path)
        except BaseException:
            for path in written:
                path.unlink(missing_ok=True)
            raise

        self.output = None
        return [ExportFile(str(path.relative_to(meta.parent))) for path in files]

    def abort(self) -> None:
        """Reset the writer; individual writes clean up their own partial files."""
        self.output = None


def export_dataset_lerobot(
    dataset: Dataset,
    output: Path,
    fps: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Export LeRobot using the shared workflow, retaining the original Python API."""
    return export_recordings(
        dataset, list(dataset), output, LeRobotExporter(fps), progress=progress
    )
