"""Export original Neuracore traces and media to generic JSON MCAP files."""

import json
import re
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from urllib.parse import quote

import requests
from neuracore_types import DataType
from neuracore_types.utils.name_utils import to_safe_name

from neuracore import __version__
from neuracore.core.data.dataset import Dataset
from neuracore.core.data.frame_cache import video_filename_preference
from neuracore.core.data.recording import Recording
from neuracore.exporter.export import DatasetExporter, ExportFile, export_recordings


class McapExporter(DatasetExporter):
    """Write one standalone JSON MCAP per recording."""

    format_name = "neuracore-mcap-json-v1"

    def __init__(self) -> None:
        """Create a writer; output is assigned when the export is prepared."""
        self.output: Path | None = None

    @staticmethod
    def _timestamp_ns(value: int | float) -> int:
        """Convert Unix seconds to MCAP nanoseconds without float multiplication."""
        return int(Decimal(str(value)) * 1_000_000_000)

    @staticmethod
    def _media(
        recording: Recording, data_type: DataType, prefix: str
    ) -> tuple[str, str, bytes] | None:
        """Resolve the stored binary payload for a sensor, if it has one."""
        if data_type == DataType.POINT_CLOUDS:
            path = f"{prefix}/trace.bin"
            return path, "application/octet-stream", recording.download(path)
        if data_type not in (DataType.RGB_IMAGES, DataType.DEPTH_IMAGES):
            return None
        for filename in video_filename_preference(data_type):
            path = f"{prefix}/{filename}"
            try:
                return path, "video/mp4", recording.download(path)
            except requests.HTTPError as exc:
                if exc.response is None or exc.response.status_code != 404:
                    raise
        raise ValueError(f"No video payload found for sensor {prefix}.")

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
                writer = Writer(stream, compression=CompressionType.NONE)
                writer.start(profile="", library=f"neuracore/{__version__}")
                schema_id = writer.register_schema(
                    name="neuracore.raw_trace.v1",
                    encoding="jsonschema",
                    data=json.dumps({
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {"timestamp": {"type": "number"}},
                        "required": ["timestamp"],
                        "additionalProperties": True,
                    }).encode(),
                )
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
                    for name in sorted(names):
                        prefix = f"{data_type.value}/{to_safe_name(name)}"
                        trace = json.loads(recording.download(f"{prefix}/trace.json"))
                        if not isinstance(trace, list) or not trace:
                            raise ValueError(f"Empty or invalid trace: {prefix}.")
                        media = self._media(recording, data_type, prefix)
                        channel_metadata = {
                            "data_type": data_type.value,
                            "sensor_name": name,
                        }
                        if media is not None:
                            path, media_type, payload = media
                            timestamp = self._timestamp_ns(recording.start_time)
                            writer.add_attachment(
                                create_time=timestamp,
                                log_time=timestamp,
                                name=path,
                                media_type=media_type,
                                data=payload,
                            )
                            channel_metadata["attachment"] = path
                            del payload, media
                        channel_id = writer.register_channel(
                            topic=(
                                f"/neuracore/{data_type.value}/"
                                f"{quote(name, safe='')}"
                            ),
                            message_encoding="json",
                            schema_id=schema_id,
                            metadata=channel_metadata,
                        )
                        for sequence, item in enumerate(trace):
                            if not isinstance(item, dict) or "timestamp" not in item:
                                raise ValueError(f"Invalid trace sample: {prefix}.")
                            timestamp = self._timestamp_ns(item["timestamp"])
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
