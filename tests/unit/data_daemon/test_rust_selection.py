from __future__ import annotations

import pytest

import neuracore.data_daemon.rust_selection as rust_selection
from neuracore.data_daemon.rust_selection import is_rust_daemon_enabled


@pytest.fixture(autouse=True)
def clear_availability_cache() -> None:
    """Drop the cached binary lookup so each test controls availability itself."""
    rust_selection._bundled_binary_available.cache_clear()
    yield
    rust_selection._bundled_binary_available.cache_clear()


@pytest.fixture
def binary_available(monkeypatch: pytest.MonkeyPatch):
    """Return a setter for whether the bundled daemon binary appears present."""

    def _set(present: bool) -> None:
        monkeypatch.setattr(
            rust_selection,
            "rust_daemon_binary_path",
            lambda: object() if present else None,
        )
        rust_selection._bundled_binary_available.cache_clear()

    return _set


def test_defaults_to_rust_when_binary_is_bundled(
    monkeypatch: pytest.MonkeyPatch, binary_available
) -> None:
    monkeypatch.delenv("NCD_RUST_DAEMON", raising=False)
    binary_available(True)
    assert is_rust_daemon_enabled() is True


def test_defaults_to_python_when_binary_is_missing(
    monkeypatch: pytest.MonkeyPatch, binary_available
) -> None:
    """A bridge-only source install must not split across the two daemons."""
    monkeypatch.delenv("NCD_RUST_DAEMON", raising=False)
    binary_available(False)
    assert is_rust_daemon_enabled() is False


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "n", " 0 "])
def test_falsy_flag_pins_to_python_even_with_binary(
    monkeypatch: pytest.MonkeyPatch, binary_available, value: str
) -> None:
    monkeypatch.setenv("NCD_RUST_DAEMON", value)
    binary_available(True)
    assert is_rust_daemon_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "y", " 1 "])
def test_truthy_flag_forces_rust_without_binary(
    monkeypatch: pytest.MonkeyPatch, binary_available, value: str
) -> None:
    """An explicit opt-in still wins so CI can demand Rust and fail loudly."""
    monkeypatch.setenv("NCD_RUST_DAEMON", value)
    binary_available(False)
    assert is_rust_daemon_enabled() is True


def test_unrecognised_value_falls_through_to_availability(
    monkeypatch: pytest.MonkeyPatch, binary_available
) -> None:
    monkeypatch.setenv("NCD_RUST_DAEMON", "maybe")
    binary_available(True)
    assert is_rust_daemon_enabled() is True
    binary_available(False)
    assert is_rust_daemon_enabled() is False


def test_availability_lookup_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check sits on per-frame logging paths, so it must not re-stat."""
    monkeypatch.delenv("NCD_RUST_DAEMON", raising=False)
    calls = 0

    def _counting_path():
        nonlocal calls
        calls += 1
        return object()

    monkeypatch.setattr(rust_selection, "rust_daemon_binary_path", _counting_path)
    rust_selection._bundled_binary_available.cache_clear()

    for _ in range(5):
        assert is_rust_daemon_enabled() is True

    assert calls == 1
