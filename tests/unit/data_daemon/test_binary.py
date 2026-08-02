"""Resolution of the bundled data-daemon binary."""

from __future__ import annotations

from pathlib import Path

import pytest

import neuracore.data_daemon.binary as binary
from neuracore.data_daemon.binary import (
    DaemonBinaryNotFoundError,
    data_daemon_binary_path,
    require_data_daemon_binary,
)


def test_binary_path_is_none_when_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(binary, "files", lambda _package: tmp_path)

    assert data_daemon_binary_path() is None


def test_binary_path_resolves_bundled_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundled = tmp_path / "bin" / "data-daemon"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"")
    monkeypatch.setattr(binary, "files", lambda _package: tmp_path)

    assert data_daemon_binary_path() == bundled


def test_require_binary_raises_with_install_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing binary must fail loudly — there is no fallback daemon."""
    monkeypatch.setattr(binary, "files", lambda _package: tmp_path)

    with pytest.raises(DaemonBinaryNotFoundError) as exc_info:
        require_data_daemon_binary()

    assert "build_wheel_artefacts.sh" in str(exc_info.value)


def test_require_binary_returns_path_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundled = tmp_path / "bin" / "data-daemon"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"")
    monkeypatch.setattr(binary, "files", lambda _package: tmp_path)

    assert require_data_daemon_binary() == bundled
