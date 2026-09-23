"""Format-independent dataset export orchestration."""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuracore import __version__
from neuracore.core.data.dataset import Dataset
from neuracore.core.data.recording import Recording

logger = logging.getLogger(__name__)


@dataclass
class ExportFile:
    """Published output path relative to the export directory.

    Omit recording_id for dataset-wide artifacts or files shared by episodes.
    """

    path: str
    recording_id: str | None = None


class DatasetExporter(ABC):
    """Lifecycle implemented by each output format.

    Dependency and recording validation run before output creation. prepare starts
    the writer; write_recording may return zero, one or many published files.
    finalize flushes dataset-wide files. abort releases resources on failure while
    preserving completed files. Neither validation method should write output.
    """

    format_name: str

    @abstractmethod
    def check_dependencies(self) -> None:
        """Raise an actionable error if format dependencies are unavailable."""
        ...

    @abstractmethod
    def validate_recording(self, recording: Recording) -> None:
        """Check the format's requirements for one recording."""
        ...

    @abstractmethod
    def prepare(self, dataset: Dataset, output: Path) -> None:
        """Initialise the writer in an existing, newly created output directory."""
        ...

    @abstractmethod
    def write_recording(self, index: int, recording: Recording) -> list[ExportFile]:
        """Write an episode and return any files already published."""
        ...

    @abstractmethod
    def finalize(self) -> list[ExportFile]:
        """Finish the dataset and return any additional published files."""
        ...

    @abstractmethod
    def abort(self) -> None:
        """Release resources and remove unfinished format-specific output."""
        ...


def export_recordings(
    dataset: Dataset,
    recordings: list[Recording],
    output: Path,
    exporter: DatasetExporter,
    progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Export a dataset using a format writer and maintain its progress manifest.

    Args:
        dataset: Dataset accessible with the caller's authentication.
        recordings: Recordings to export, already populated from the dataset.
        output: New output directory; existing directories are never overwritten.
        exporter: Format implementation owning validation and output layout.
        progress: Callback receiving completed count, total and next recording name.

    Returns:
        Path to the export manifest. Success requires format finalisation.
    """
    exporter.check_dependencies()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    for recording in recordings:
        exporter.validate_recording(recording)

    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    manifest: dict[str, Any] = {
        "format": exporter.format_name,
        "exporter_version": __version__,
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "status": "running",
        "recording_ids": [r.id for r in recordings],
        "completed_recording_ids": [],
        "files": [],
    }

    def save_manifest() -> None:
        """Write the manifest atomically to avoid partial files on failure."""
        temporary = output / "manifest.json.partial"
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)

    def add_files(files: list[ExportFile]) -> None:
        """Add published files to the manifest and save it."""
        for file in files:
            entry = {"path": file.path}
            if file.recording_id is not None:
                entry["recording_id"] = file.recording_id
            manifest["files"].append(entry)

    save_manifest()
    try:
        exporter.prepare(dataset, output)
        for index, recording in enumerate(recordings):
            if progress:
                progress(index, len(recordings), recording.name)
            add_files(exporter.write_recording(index, recording))
            manifest["completed_recording_ids"].append(recording.id)
            save_manifest()
        manifest["status"] = "finalising"
        save_manifest()
        add_files(exporter.finalize())
        manifest["status"] = "succeeded"
        save_manifest()
    except BaseException:
        # Cleanup failures should not replace the original conversion error.
        try:
            exporter.abort()
        except Exception:
            logger.exception("Failed to clean up exporter resources")
        manifest["status"] = "incomplete"
        try:
            save_manifest()
        except Exception:
            logger.exception("Failed to save the incomplete export manifest")
        raise
    return manifest_path
