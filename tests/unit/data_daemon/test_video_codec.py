"""Video-codec writes are delegated to the daemon binary's profile CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import neuracore.data_daemon.video_codec as video_codec
from neuracore.data_daemon.binary import DaemonBinaryNotFoundError
from neuracore.data_daemon.const import DEFAULT_PROFILE_NAME
from neuracore.data_daemon.video_codec import set_active_profile_video_codec


@pytest.fixture
def bundled_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    binary_path = tmp_path / "data-daemon"
    monkeypatch.setattr(video_codec, "require_data_daemon_binary", lambda: binary_path)
    return binary_path


def _record_runs(
    monkeypatch: pytest.MonkeyPatch, returncode: int = 0, stderr: str = ""
) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        return subprocess.CompletedProcess(command, returncode, "", stderr)

    monkeypatch.setattr(video_codec.subprocess, "run", fake_run)
    return commands


def test_creates_then_updates_the_active_profile(
    monkeypatch: pytest.MonkeyPatch, bundled_binary: Path
) -> None:
    commands = _record_runs(monkeypatch)

    set_active_profile_video_codec("h264_medium")

    assert commands == [
        [str(bundled_binary), "profile", "create", DEFAULT_PROFILE_NAME],
        [
            str(bundled_binary),
            "profile",
            "update",
            DEFAULT_PROFILE_NAME,
            "--video-codec",
            "h264_medium",
        ],
    ]


def test_targets_the_profile_named_by_the_environment(
    monkeypatch: pytest.MonkeyPatch, bundled_binary: Path
) -> None:
    monkeypatch.setenv("NEURACORE_DAEMON_PROFILE", "demo")
    commands = _record_runs(monkeypatch)

    set_active_profile_video_codec("h264_lossless")

    assert [command[3] for command in commands] == ["demo", "demo"]


def test_an_existing_profile_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, bundled_binary: Path
) -> None:
    """`profile create` failing because the profile exists must not propagate."""
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        if command[2] == "create":
            return subprocess.CompletedProcess(command, 1, "", "already exists")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(video_codec.subprocess, "run", fake_run)

    set_active_profile_video_codec("h264_medium")

    assert len(calls) == 2


def test_raises_when_the_update_fails(
    monkeypatch: pytest.MonkeyPatch, bundled_binary: Path
) -> None:
    _record_runs(monkeypatch, returncode=1, stderr="Profile 'demo' not found.")

    with pytest.raises(RuntimeError) as exc_info:
        set_active_profile_video_codec("h264_medium")

    assert "not found" in str(exc_info.value)


def _raise_missing_binary() -> Path:
    raise DaemonBinaryNotFoundError("missing")


def test_raises_without_the_bundled_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        video_codec, "require_data_daemon_binary", _raise_missing_binary
    )

    with pytest.raises(DaemonBinaryNotFoundError):
        set_active_profile_video_codec("h264_medium")
