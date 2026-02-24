import argparse

import pytest

from neuracore.data_daemon.config_manager.helpers import parse_bytes


@pytest.mark.parametrize(
    "value, expected",
    [
        ("0b", 0),
        ("1b", 1),
        ("1", 1),
        ("1k", 1024),
        ("1kb", 1024),
        ("2kb", 2 * 1024),
        ("1mb", 1024 * 1024),
        ("50mb", 50 * 1024 * 1024),
        ("300m", 300 * 1024 * 1024),
        ("1gb", 1024 * 1024 * 1024),
        ("2gb", 2 * 1024 * 1024 * 1024),
        ("9122488b", 9_122_488),
    ],
)
def test_parse_bytes_valid(value: str, expected: int) -> None:
    assert parse_bytes(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "  1kb  ",
        "1KB",
    ],
)
def test_parse_bytes_optional_tolerances(value: str) -> None:
    try:
        out = parse_bytes(value)
    except (argparse.ArgumentTypeError, ValueError):
        pytest.skip("Parser does not accept whitespace/case variations.")
    assert isinstance(out, int)
    assert out > 0


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "nope",
        "kb",
        "1KiB",
        "1gbps",
        "mb",
        "-1kb",
        "1.5gb",
    ],
)
def test_parse_bytes_invalid_raises(value: str) -> None:
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        parse_bytes(value)
