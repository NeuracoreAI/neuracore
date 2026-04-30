"""Tests for the daemon runner entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import neuracore.data_daemon.runner_entry as runner_entry
from neuracore.data_daemon.runtime import DaemonContext


@pytest.fixture
def pid_path(tmp_path: Path) -> Path:
    return tmp_path / "daemon.pid"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


def test_main_logs_error_when_runtime_initialize_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pid_path: Path,
    db_path: Path,
) -> None:
    runtime = MagicMock()
    runtime.initialize.return_value = None

    monkeypatch.setattr(runner_entry, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(runner_entry, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(runner_entry, "DaemonRuntime", lambda **_: runtime)
    monkeypatch.setattr(runner_entry, "install_signal_handlers", lambda *_: None)
    shutdown_calls: list[tuple[Path, tuple[Path, ...], Path]] = []
    monkeypatch.setattr(
        runner_entry,
        "shutdown",
        lambda *, pid_path, socket_paths, db_path: shutdown_calls.append(
            (pid_path, socket_paths, db_path)
        ),
    )

    with caplog.at_level(logging.ERROR):
        runner_entry.main()

    runtime.initialize.assert_called_once()
    runtime.run_forever.assert_not_called()
    runtime.shutdown.assert_called_once()
    assert "Failed to start daemon" in caplog.text
    assert shutdown_calls == [(pid_path, (runner_entry.SOCKET_PATH,), db_path)]


def test_main_runs_runtime_forever_when_initialize_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    pid_path: Path,
    db_path: Path,
) -> None:
    runtime = MagicMock()
    runtime.initialize.return_value = DaemonContext(
        config=MagicMock(),
        loop_manager=MagicMock(),
        comm_manager=MagicMock(),
        services=MagicMock(),
        recording_disk_manager=MagicMock(),
    )

    monkeypatch.setattr(runner_entry, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(runner_entry, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(runner_entry, "DaemonRuntime", lambda **_: runtime)
    monkeypatch.setattr(runner_entry, "install_signal_handlers", lambda *_: None)
    monkeypatch.setattr(runner_entry, "shutdown", lambda **_: None)

    runner_entry.main()

    runtime.initialize.assert_called_once()
    runtime.run_forever.assert_called_once()
    runtime.shutdown.assert_called_once()


def test_main_still_shuts_down_after_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pid_path: Path,
    db_path: Path,
) -> None:
    runtime = MagicMock()
    runtime.initialize.return_value = DaemonContext(
        config=MagicMock(),
        loop_manager=MagicMock(),
        comm_manager=MagicMock(),
        services=MagicMock(),
        recording_disk_manager=MagicMock(),
    )
    runtime.run_forever.side_effect = KeyboardInterrupt()

    monkeypatch.setattr(runner_entry, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(runner_entry, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(runner_entry, "DaemonRuntime", lambda **_: runtime)
    monkeypatch.setattr(runner_entry, "install_signal_handlers", lambda *_: None)
    shutdown_calls: list[tuple[Path, tuple[Path, ...], Path]] = []
    monkeypatch.setattr(
        runner_entry,
        "shutdown",
        lambda *, pid_path, socket_paths, db_path: shutdown_calls.append(
            (pid_path, socket_paths, db_path)
        ),
    )

    with caplog.at_level(logging.INFO):
        runner_entry.main()

    runtime.shutdown.assert_called_once()
    assert "Received keyboard interrupt" in caplog.text
    assert shutdown_calls == [(pid_path, (runner_entry.SOCKET_PATH,), db_path)]


def test_main_logs_fatal_error_when_run_forever_crashes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    pid_path: Path,
    db_path: Path,
) -> None:
    runtime = MagicMock()
    runtime.initialize.return_value = DaemonContext(
        config=MagicMock(),
        loop_manager=MagicMock(),
        comm_manager=MagicMock(),
        services=MagicMock(),
        recording_disk_manager=MagicMock(),
    )
    runtime.run_forever.side_effect = RuntimeError("boom")

    monkeypatch.setattr(runner_entry, "get_daemon_pid_path", lambda: pid_path)
    monkeypatch.setattr(runner_entry, "get_daemon_db_path", lambda: db_path)
    monkeypatch.setattr(runner_entry, "DaemonRuntime", lambda **_: runtime)
    monkeypatch.setattr(runner_entry, "install_signal_handlers", lambda *_: None)
    shutdown_calls: list[tuple[Path, tuple[Path, ...], Path]] = []
    monkeypatch.setattr(
        runner_entry,
        "shutdown",
        lambda *, pid_path, socket_paths, db_path: shutdown_calls.append(
            (pid_path, socket_paths, db_path)
        ),
    )

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="boom"):
        runner_entry.main()

    runtime.shutdown.assert_called_once()
    assert "Fatal error while daemon main loop was running" in caplog.text
    assert shutdown_calls == [(pid_path, (runner_entry.SOCKET_PATH,), db_path)]
