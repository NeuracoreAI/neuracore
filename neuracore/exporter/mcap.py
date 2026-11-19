"""Export original Neuracore traces and media to generic JSON MCAP files."""

import io
import json
import re
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import av
import requests
from neuracore_types import DATA_TYPE_TO_NC_DATA_CLASS, DataType
from neuracore_types.timestamps import NANOSECONDS_PER_MICROSECOND

from neuracore import __version__
from neuracore.core.data.dataset import Dataset
from neuracore.core.data.frame_cache import get_camera_data_filenames
from neuracore.core.data.recording import Recording
from neuracore.core.data.sensor_paths import sensor_directory
from neuracore.exporter.export import DatasetExporter, ExportFile, export_recordings

VIDEO_TIMESTAMP_TOLERANCE_US = 2
"""Largest gap in microseconds between a video frame time and its trace time."""

TRACE_SCHEMA_NAME_FORMAT = "neuracore.{data_type}.v1"
"""Name of the JSON Schema that describes the trace entries of one data type."""


class McapExporter(DatasetExporter):
    """Write one standalone JSON MCAP per recording.

    Every trace sample becomes one JSON message holding the trace entry unchanged.
    Each data type has its own JSON Schema, generated from its neuracore_types
    model.
    RGB videos, depth frames files (lossless.bin) and point cloud binaries are
    attached as stored. The MCAP chunks are compressed with zstd.
    """

    format_name = "neuracore-mcap-json-v2"

    def __init__(self) -> None:
        """Create a writer; output is assigned when the export is prepared."""
        self.output: Path | None = None

    @staticmethod
    def _timestamp_ns(value: int | float) -> int:
        """Convert Unix seconds to MCAP nanoseconds without float multiplication."""
        return int(Decimal(str(value)) * 1_000_000_000)

    @classmethod
    def _sample_timestamp_ns(cls, sample: dict) -> int:
        """Return a trace sample's capture time in MCAP nanoseconds.

        A sample with ``timestamp_us`` is exact in microseconds; an older
        sample has only ``timestamp`` in seconds.
        """
        if "timestamp_us" in sample:
            return sample["timestamp_us"] * NANOSECONDS_PER_MICROSECOND
        return cls._timestamp_ns(sample["timestamp"])

    @staticmethod
    def _trace_entry_json_schema(data_type: DataType) -> dict:
        """Return the JSON Schema of one stored trace entry of a data type.

        The schema comes from the neuracore_types model of the data type. Fields
        with a default are optional, because stored entries leave some of them
        out. Extra keys, such as the byte range of a depth frame, stay allowed.

        Args:
            data_type: Data type of the trace.

        Returns:
            The JSON Schema as a dictionary.
        """
        model = DATA_TYPE_TO_NC_DATA_CLASS[data_type]
        schema = model.model_json_schema(mode="serialization")
        properties = schema.get("properties", {})
        schema["required"] = [
            name
            for name in schema.get("required", [])
            if "default" not in properties.get(name, {})
        ]
        schema["additionalProperties"] = True
        return {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}

    @staticmethod
    def _download_if_present(recording: Recording, path: str) -> bytes | None:
        """Download a recording file, returning None when storage has no such file."""
        try:
            return recording.download(path)
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise
            return None

    @classmethod
    def _media(
        cls, recording: Recording, data_type: DataType, prefix: str
    ) -> tuple[str, str, bytes] | None:
        """Resolve the stored binary payload for a sensor, if it has one.

        RGB cameras use their lossless video, or their lossy video when no
        lossless video exists. Depth cameras use only their depth frames file.
        Point clouds use trace.bin when the recording has one; otherwise the
        point cloud is inline in trace.json.
        """
        if data_type == DataType.POINT_CLOUDS:
            path = f"{prefix}/trace.bin"
            payload = cls._download_if_present(recording, path)
            if payload is None:
                return None
            return path, "application/octet-stream", payload
        if data_type not in (DataType.RGB_IMAGES, DataType.DEPTH_IMAGES):
            return None
        for filename in get_camera_data_filenames(data_type):
            path = f"{prefix}/{filename}"
            payload = cls._download_if_present(recording, path)
            if payload is not None:
                media_type = (
                    "video/mp4"
                    if filename.endswith(".mp4")
                    else "application/octet-stream"
                )
                return path, media_type, payload
        raise ValueError(f"No camera data found for sensor {prefix}.")

    def check_dependencies(self) -> None:
        """Require only the MCAP optional dependencies."""
        try:
            import mcap.writer  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "MCAP export requires optional dependencies: "
                'pip install "neuracore[export]"'
            ) from exc

    def validate_recording(self, recording: Recording) -> None:
        """Require completed recordings with complete raw sensor manifests."""
        if recording.end_time is None or recording.deleted:
            raise ValueError(
                f"Recording {recording.name} is not a completed recording."
            )
        manifest = recording.sensor_manifest
        if not manifest or set(manifest) != set(recording.data_types):
            raise ValueError(
                f"Recording {recording.name} has no complete sensor manifest. "
                "This exporter requires recordings with a stored sensor manifest. "
                "Please delete this recording or contact support if you believe "
                "this is an error."
            )

        for data_type, names in manifest.items():
            DataType(data_type)
            if not names or len(set(names)) != len(names):
                raise ValueError(f"Invalid sensor manifest for {recording.name}.")
            for name in names:
                if not isinstance(name, str) or not name or name in (".", ".."):
                    raise ValueError("Invalid sensor name in recording manifest.")

    @classmethod
    def _check_video_matches_trace(
        cls, trace: list[dict], media: tuple[str, str, bytes]
    ) -> None:
        """Check that the video holds one frame per trace entry at its trace time.

        Args:
            trace: Trace entries of the camera, in frame order.
            media: Video path, media type and bytes.

        Raises:
            ValueError: If the frame count, a frame index or a frame time differs
                from the trace.
        """
        path, _, payload = media
        with av.open(io.BytesIO(payload)) as container:
            stream = container.streams.video[0]
            packets = container.demux(stream)
            frame_times = sorted(
                packet.pts for packet in packets if packet.pts is not None
            )
            time_base = stream.time_base
        if len(frame_times) != len(trace):
            raise ValueError(
                f"Video {path} holds {len(frame_times)} frames but the trace has "
                f"{len(trace)} entries."
            )
        first_trace_ns = cls._sample_timestamp_ns(trace[0])
        for index, (item, frame_time) in enumerate(zip(trace, frame_times)):
            if item.get("frame_idx") != index:
                raise ValueError(f"Invalid frame index in {path}: {item}.")
            video_us = float((frame_time - frame_times[0]) * time_base * 1_000_000)
            trace_us = (cls._sample_timestamp_ns(item) - first_trace_ns) / 1_000
            if abs(video_us - trace_us) > VIDEO_TIMESTAMP_TOLERANCE_US:
                raise ValueError(
                    f"Frame {index} of {path} is at {video_us:.1f} us but the trace "
                    f"says {trace_us:.1f} us."
                )

    @staticmethod
    def _check_byte_ranges_inside_file(
        trace: list[dict], media: tuple[str, str, bytes]
    ) -> None:
        """Check that every trace entry names a byte range inside the attachment.

        Args:
            trace: Trace entries, each holding the offset and length of its frame.
            media: Attachment path, media type and bytes.

        Raises:
            ValueError: If an entry has no byte range inside the file.
        """
        path, _, payload = media
        for item in trace:
            offset, length = item.get("offset"), item.get("length")
            if offset is None or length is None or offset + length > len(payload):
                raise ValueError(f"Invalid frame range in {path}: {item}.")

    def prepare(self, dataset: Dataset, output: Path) -> None:
        """Set the destination for this dataset's MCAP files."""
        self.output = output

    def write_recording(self, index: int, recording: Recording) -> list[ExportFile]:
        """Publish one MCAP and report its relative filename."""
        if self.output is None:
            raise RuntimeError("Prepare the MCAP exporter before writing recordings.")
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", recording.name).strip(" .")
        name = name or "recording"
        filename = f"nc_{name}.mcap"
        destination = self.output / filename
        suffix = 2
        while destination.exists():
            filename = f"nc_{name}_{suffix}.mcap"
            destination = self.output / filename
            suffix += 1
        from mcap.writer import CompressionType, Writer

        partial = destination.with_suffix(".mcap.partial")
        try:
            with partial.open("xb") as stream:
                writer = Writer(stream, compression=CompressionType.ZSTD)
                writer.start(profile="", library=f"neuracore/{__version__}")
                recording_metadata = {
                    "id": recording.id,
                    "robot_id": recording.robot_id,
                    "instance": recording.instance,
                    "start_time": recording.start_time,
                    "end_time": recording.end_time,
                    "metadata": recording.metadata.model_dump(mode="json"),
                    "data_types": sorted(dt.value for dt in recording.data_types),
                    "sensor_manifest": {
                        kind.value: names
                        for kind, names in recording.sensor_manifest.items()
                    },
                }
                writer.add_metadata(
                    "neuracore.recording",
                    {"json": json.dumps(recording_metadata, allow_nan=False)},
                )
                for data_type, names in sorted(recording.sensor_manifest.items()):
                    schema_id = writer.register_schema(
                        name=TRACE_SCHEMA_NAME_FORMAT.format(data_type=data_type.value),
                        encoding="jsonschema",
                        data=json.dumps(
                            self._trace_entry_json_schema(data_type)
                        ).encode(),
                    )
                    for name in sorted(names):
                        prefix = sensor_directory(data_type, name)
                        trace = json.loads(recording.download(f"{prefix}/trace.json"))
                        if not isinstance(trace, list) or not trace:
                            raise ValueError(f"Empty or invalid trace: {prefix}.")
                        media = self._media(recording, data_type, prefix)
                        channel_metadata = {
                            "data_type": data_type.value,
                            "sensor_name": name,
                        }
                        if media is not None:
                            if media[1] == "video/mp4":
                                self._check_video_matches_trace(trace, media)
                            else:
                                self._check_byte_ranges_inside_file(trace, media)
                            path, media_type, payload = media
                            timestamp = (
                                recording.start_timestamp_us
                                * NANOSECONDS_PER_MICROSECOND
                                if recording.start_timestamp_us is not None
                                else self._timestamp_ns(recording.start_time)
                            )
                            writer.add_attachment(
                                create_time=timestamp,
                                log_time=timestamp,
                                name=path,
                                media_type=media_type,
                                data=payload,
                            )
                            channel_metadata["attachment"] = path
                            del payload
                        channel_id = writer.register_channel(
                            topic=f"/neuracore/{prefix}",
                            message_encoding="json",
                            schema_id=schema_id,
                            metadata=channel_metadata,
                        )
                        for sequence, item in enumerate(trace):
                            if not isinstance(item, dict) or not (
                                "timestamp_us" in item or "timestamp" in item
                            ):
                                raise ValueError(f"Invalid trace sample: {prefix}.")
                            timestamp = self._sample_timestamp_ns(item)
                            writer.add_message(
                                channel_id=channel_id,
                                log_time=timestamp,
                                publish_time=timestamp,
                                sequence=sequence,
                                data=json.dumps(item, allow_nan=False).encode(),
                            )
                writer.finish()
            partial.rename(destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return [ExportFile(filename, recording.id)]

    def finalize(self) -> list[ExportFile]:
        """Finish the export; individual MCAP files are already finalised."""
        self.output = None
        return []

    def abort(self) -> None:
        """Reset the writer; individual writes clean up their partial files."""
        self.output = None


def export_dataset_mcap(
    dataset: Dataset,
    output: Path,
    progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Export MCAP using the shared workflow, retaining the original Python API."""
    return export_recordings(
        dataset, list(dataset), output, McapExporter(), progress=progress
    )
