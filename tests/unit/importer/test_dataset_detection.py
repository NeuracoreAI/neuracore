from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from neuracore_types.importer.config import DatasetTypeConfig

from neuracore.importer.core.dataset_detector import DatasetDetector
from neuracore.importer.core.exceptions import DatasetDetectionError
from neuracore.importer.importer import detect_dataset_type

DetectorFn = Callable[[Path], object]


def _create_files(root: Path, files: Sequence[str]) -> Path:
    """Create empty files (and parent dirs) under the given root for testing."""
    root.mkdir(parents=True, exist_ok=True)
    for relative_path in files:
        file_path = root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
    return root


@pytest.fixture()
def importer_detect() -> DetectorFn:
    return detect_dataset_type


@pytest.fixture()
def detectors(importer_detect: DetectorFn) -> list[tuple[str, DetectorFn]]:
    return [
        ("importer_module", importer_detect),
        ("detector_class", DatasetDetector().detect),
    ]


def test_detect_prefers_rlds_when_multiple_markers(
    tmp_path: Path, detectors: list[tuple[str, DetectorFn]]
):
    dataset_dir = _create_files(
        tmp_path / "example_dataset",
        ["rlds_metadata.json", "dataset_infos.json", "episodes/data.arrow"],
    )

    for _, detector in detectors:
        assert detector(dataset_dir) == DatasetTypeConfig.RLDS


def test_detect_lerobot_with_parquet_and_meta(
    importer_detect: DetectorFn, tmp_path: Path
):
    dataset_dir = _create_files(
        tmp_path / "lerobot_dataset",
        ["meta/info.json", "meta/stats.json", "data/data.parquet"],
    )

    assert importer_detect(dataset_dir) == DatasetTypeConfig.LEROBOT


def test_detect_lerobot_by_dir_name_and_arrow(
    tmp_path: Path, detectors: list[tuple[str, DetectorFn]]
):
    dataset_dir = _create_files(
        tmp_path / "cool_lerobot_dataset",
        ["shards/episode_0001.arrow"],
    )

    for _, detector in detectors:
        assert detector(dataset_dir) == DatasetTypeConfig.LEROBOT


def test_detect_tfds_case_insensitive_markers(
    tmp_path: Path, detectors: list[tuple[str, DetectorFn]]
):
    dataset_dir = _create_files(
        tmp_path / "tfds_style_dataset",
        ["DATASET_INFO.JSON", "features.JSON"],
    )

    for _, detector in detectors:
        assert detector(dataset_dir) == DatasetTypeConfig.TFDS


def test_detect_mcap_file(tmp_path: Path, detectors: list[tuple[str, DetectorFn]]):
    file_path = tmp_path / "example.mcap"
    file_path.touch()

    for _, detector in detectors:
        assert detector(file_path) == DatasetTypeConfig.MCAP


def test_detect_mcap_in_directory(
    tmp_path: Path, detectors: list[tuple[str, DetectorFn]]
):
    dataset_dir = _create_files(
        tmp_path / "mcap_dataset",
        ["session/example.mcap"],
    )

    for _, detector in detectors:
        assert detector(dataset_dir) == DatasetTypeConfig.MCAP


def test_detect_rlds_by_directory_name(
    tmp_path: Path, detectors: list[tuple[str, DetectorFn]]
):
    dataset_dir = tmp_path / "my_rlds_dataset"
    dataset_dir.mkdir()

    for _, detector in detectors:
        assert detector(dataset_dir) == DatasetTypeConfig.RLDS


def test_detect_unknown_layout_raises(
    tmp_path: Path, detectors: list[tuple[str, DetectorFn]]
):
    dataset_dir = _create_files(tmp_path / "unknown_dataset", ["notes.txt"])

    for name, detector in detectors:
        expected_exc = (
            ValueError if name == "importer_module" else DatasetDetectionError
        )
        with pytest.raises(expected_exc):
            detector(dataset_dir)


def test_detect_path_that_is_not_directory(
    tmp_path: Path, detectors: list[tuple[str, DetectorFn]]
):
    file_path = tmp_path / "not_a_directory.arrow"
    file_path.touch()

    for name, detector in detectors:
        expected_exc = (
            ValueError if name == "importer_module" else DatasetDetectionError
        )
        with pytest.raises(expected_exc):
            detector(file_path)


def test_arrow_without_markers_does_not_guess_type(
    tmp_path: Path, detectors: list[tuple[str, DetectorFn]]
):
    dataset_dir = _create_files(tmp_path / "generic_dataset", ["episodes/data.arrow"])

    for name, detector in detectors:
        expected_exc = (
            ValueError if name == "importer_module" else DatasetDetectionError
        )
        with pytest.raises(expected_exc):
            detector(dataset_dir)
