"""Dataset type detection based on file markers."""

from collections.abc import Iterator
from pathlib import Path

from neuracore_types.importer.config import DatasetTypeConfig

from .exceptions import DatasetDetectionError

TFDS_MARKERS = {
    "dataset_info.json",
    "features.json",
    "dataset_state.json",
}
MCAP_MARKERS = {".mcap"}
RLDS_MARKERS = {
    "rlds_metadata.json",
    "rlds_metadata.jsonl",
    "rlds_description.pbtxt",
    "rlds_description.pb",
    "episode_metadata.jsonl",
    "step_metadata.jsonl",
}
LEROBOT_MARKERS = {
    "dataset_infos.json",
    "dataset_dict.json",
    "state.json",
}
LEROBOT_META_MARKERS = {"info.json", "stats.json", "tasks.parquet"}


def iter_first_two_levels(root: Path) -> Iterator[Path]:
    """Yield paths in ``root`` and its first-level subdirectories.

    Stream results to avoid materializing the full directory, and skip entries
    that can't be accessed (e.g., permission issues, concurrent deletes).
    """
    try:
        for child in root.iterdir():
            yield child
            try:
                is_dir = child.is_dir()
            except OSError:
                # Skip entries whose metadata can't be read
                continue

            if is_dir:
                try:
                    yield from child.iterdir()
                except OSError:
                    # Subdirectory not readable or vanished
                    continue
    except OSError:
        return


class DatasetDetector:
    """Identify dataset layout (TFDS, RLDS, or LeRobot) based on file markers."""

    def __init__(self) -> None:
        """Initialize dataset detector."""

    def detect(self, dataset_dir: Path) -> DatasetTypeConfig:
        """Detect whether the dataset is TFDS, RLDS, or LeRobot."""
        if dataset_dir.is_file():
            if dataset_dir.suffix.lower() in MCAP_MARKERS:
                return DatasetTypeConfig.MCAP
            raise DatasetDetectionError(
                f"Dataset path '{dataset_dir}' is not a directory."
            )
        if not dataset_dir.is_dir():
            raise DatasetDetectionError(
                f"Dataset path '{dataset_dir}' is not a directory."
            )

        lower_dir_name = dataset_dir.name.lower()
        has_tfds_marker = False
        has_rlds_marker = False
        has_lerobot_marker = False
        has_mcap_marker = False
        has_arrow_data = False
        has_parquet_data = False

        for path in iter_first_two_levels(dataset_dir):
            if not path.is_file():
                continue
            name = path.name
            lower_name = name.lower()
            has_tfds_marker = has_tfds_marker or lower_name in TFDS_MARKERS
            has_rlds_marker = has_rlds_marker or lower_name in RLDS_MARKERS
            has_lerobot_marker = has_lerobot_marker or lower_name in LEROBOT_MARKERS
            has_arrow_data = has_arrow_data or path.suffix.lower() == ".arrow"
            has_parquet_data = has_parquet_data or path.suffix.lower() == ".parquet"
            has_mcap_marker = has_mcap_marker or path.suffix.lower() in MCAP_MARKERS
            if path.parent.name == "meta" and lower_name in LEROBOT_META_MARKERS:
                has_lerobot_marker = True

        has_rlds_marker = has_rlds_marker or "rlds" in lower_dir_name
        has_lerobot_marker = has_lerobot_marker or "lerobot" in lower_dir_name

        if has_mcap_marker:
            return DatasetTypeConfig.MCAP
        if has_rlds_marker:
            return DatasetTypeConfig.RLDS
        if has_lerobot_marker and (has_arrow_data or has_parquet_data):
            return DatasetTypeConfig.LEROBOT
        if has_tfds_marker:
            return DatasetTypeConfig.TFDS
        if (
            has_arrow_data or has_parquet_data
        ) and "lerobot" in dataset_dir.name.lower():
            return DatasetTypeConfig.LEROBOT

        raise DatasetDetectionError(
            f"Unable to determine dataset type at '{dataset_dir}'. "
            "Expected MCAP, TFDS/RLDS, or LeRobot layout."
        )
