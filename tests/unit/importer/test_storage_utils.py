"""Unit tests for parse_storage_size."""

from neuracore.importer.core.utils import parse_storage_size


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
