"""Unit tests for parse_storage_size and NeuracoreDatasetImporter disk usage helpers."""

from pathlib import Path

from neuracore.importer.core.base import ImportItem, NeuracoreDatasetImporter
from neuracore.importer.core.utils import parse_storage_size


class _DiskUsageTestImporter(NeuracoreDatasetImporter):
    """Concrete importer overriding _get_disk_check_path for deterministic tests."""

    def __init__(self, disk_check_path: Path):
        self._disk_check_path = disk_check_path

    def build_work_items(self):
        return [ImportItem(index=0)]

    def import_item(self, item: ImportItem) -> None:
        return None

    def _record_step(self, step: dict, timestamp: float) -> None:
        return None

    def _get_disk_check_path(self) -> Path:
        return self._disk_check_path


def test_parse_storage_size_parses_valid_values():
    """Human-readable sizes map to byte counts (binary units)."""
    assert parse_storage_size("1kb") == 1024
    assert parse_storage_size("500 mb") == 500 * 1024**2
    assert parse_storage_size("2GB") == 2 * 1024**3
    assert parse_storage_size("1.5gb") == int(1.5 * 1024**3)


def test_parse_storage_size_rejects_invalid_values():
    """Malformed strings raise ValueError with an invalid-size message."""
    for value in ("", "10", "5tb", "abc"):
        try:
            parse_storage_size(value)
            assert False, f"Expected ValueError for value: {value}"
        except ValueError as exc:
            assert "Invalid storage size" in str(exc)


def test_parse_storage_size_rejects_non_positive_values():
    """Zero or negative numeric values raise ValueError."""
    for value in ("0kb", "0mb", "0gb"):
        try:
            parse_storage_size(value)
            assert False, f"Expected ValueError for value: {value}"
        except ValueError as exc:
            assert "Storage size must be positive" in str(exc)


def test_format_bytes_human_readable_units():
    """_format_bytes steps through B, KiB, GiB-style labels."""
    assert NeuracoreDatasetImporter._format_bytes(0) == "0.0 B"
    assert NeuracoreDatasetImporter._format_bytes(1023) == "1023.0 B"
    assert NeuracoreDatasetImporter._format_bytes(1024) == "1.0 KiB"
    assert NeuracoreDatasetImporter._format_bytes(1536) == "1.5 KiB"
    assert NeuracoreDatasetImporter._format_bytes(1024**3) == "1.0 GiB"


def test_get_disk_usage_returns_zero_for_missing_path(tmp_path):
    """Missing monitored path reports zero bytes used."""
    importer = _DiskUsageTestImporter(tmp_path / "missing")
    assert importer._get_disk_usage() == 0


def test_get_disk_usage_for_file_path(tmp_path):
    """When the path is a file, usage is that file's size."""
    payload = b"hello importer"
    file_path = tmp_path / "single.bin"
    file_path.write_bytes(payload)

    importer = _DiskUsageTestImporter(file_path)
    assert importer._get_disk_usage() == len(payload)


def test_get_disk_usage_sums_nested_directory_files(tmp_path):
    """Directory usage sums all nested file sizes."""
    first = tmp_path / "a.bin"
    first.write_bytes(b"abc")
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    second = nested_dir / "b.bin"
    second.write_bytes(b"12345")

    importer = _DiskUsageTestImporter(tmp_path)
    assert importer._get_disk_usage() == 8
