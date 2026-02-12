from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from neuracore.data_daemon.communications_management import management_channel


class _DummyRDM:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class _DummyBootstrap:
    def start(self):
        class _Context:
            recording_disk_manager = _DummyRDM()
            comm_manager = object()

        return _Context()


class _DummyStore:
    async def list_traces(self):
        return []


def test_get_ndd_context_runs_startup_and_writes_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    recordings_root = tmp_path / "recordings"
    socket_path = tmp_path / "daemon.sock"
    events_path = tmp_path / "events.sock"

    monkeypatch.setenv("NEURACORE_DAEMON_PID_PATH", str(pid_path))
    monkeypatch.setenv("NEURACORE_DAEMON_DB_PATH", str(db_path))
    monkeypatch.setenv("NEURACORE_DAEMON_RECORDINGS_ROOT", str(recordings_root))

    monkeypatch.setattr(management_channel, "SOCKET_PATH", socket_path)
    monkeypatch.setattr(management_channel, "RECORDING_EVENTS_SOCKET_PATH", events_path)
    monkeypatch.setattr(management_channel, "RecordingDiskManager", _DummyRDM)
    monkeypatch.setattr(management_channel, "DaemonBootstrap", _DummyBootstrap)
    monkeypatch.setattr(
        management_channel, "SqliteStateStore", lambda _path: _DummyStore()
    )

    channel = management_channel.ManagementChannel()
    daemon = asyncio.run(channel.get_ndd_context())

    assert daemon is not None
    assert pid_path.exists()


def test_get_ndd_context_rejects_running_pid_with_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_path = tmp_path / "daemon.pid"
    socket_path = tmp_path / "daemon.sock"
    socket_path.write_text("stale", encoding="utf-8")
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    monkeypatch.setenv("NEURACORE_DAEMON_PID_PATH", str(pid_path))
    monkeypatch.setattr(management_channel, "SOCKET_PATH", socket_path)
    monkeypatch.setattr(management_channel, "RecordingDiskManager", _DummyRDM)

    channel = management_channel.ManagementChannel()
    daemon = asyncio.run(channel.get_ndd_context())

    assert daemon is None
