import sqlite3
from pathlib import Path

import recording_playback_shared as shared


def _write_cleanup_files(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    pid_path = tmp_path / "daemon.pid"
    db_path = tmp_path / "state.db"
    wal_path = tmp_path / "state.db.wal"
    shm_path = tmp_path / "state.db.shm"
    socket_path = tmp_path / "daemon.sock"

    pid_path.write_text("not-a-pid", encoding="utf-8")
    db_path.write_text("sqlite", encoding="utf-8")
    wal_path.write_text("wal", encoding="utf-8")
    shm_path.write_text("shm", encoding="utf-8")
    socket_path.write_text("socket", encoding="utf-8")

    return pid_path, db_path, wal_path, shm_path, socket_path


def test_daemon_cleanup_preserves_sqlite_state_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pid_path, db_path, wal_path, shm_path, socket_path = _write_cleanup_files(tmp_path)

    monkeypatch.setenv("NEURACORE_DAEMON_PID_PATH", str(pid_path))
    monkeypatch.setenv("NEURACORE_DAEMON_DB_PATH", str(db_path))
    monkeypatch.setattr(shared, "SOCKET_PATH", str(socket_path))
    monkeypatch.setattr(shared, "get_runner_pids", lambda: set())

    shared.daemon_cleanup()

    assert not pid_path.exists()
    assert not socket_path.exists()
    assert db_path.exists()
    assert wal_path.exists()
    assert shm_path.exists()


def test_daemon_cleanup_can_empty_sqlite_state_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pid_path, db_path, wal_path, shm_path, socket_path = _write_cleanup_files(tmp_path)

    db_path.unlink(missing_ok=True)

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE recordings ("
            "recording_id TEXT PRIMARY KEY, trace_count INTEGER"
            ")"
        )
        conn.execute(
            "INSERT INTO recordings(recording_id, trace_count) VALUES (?, ?)",
            ("rec_1", 5),
        )
        conn.commit()

    monkeypatch.setenv("NEURACORE_DAEMON_PID_PATH", str(pid_path))
    monkeypatch.setenv("NEURACORE_DAEMON_DB_PATH", str(db_path))
    monkeypatch.setattr(shared, "SOCKET_PATH", str(socket_path))
    monkeypatch.setattr(shared, "get_runner_pids", lambda: set())

    shared.daemon_cleanup(state_db_action="empty")

    assert not pid_path.exists()
    assert not socket_path.exists()
    assert db_path.exists()
    with sqlite3.connect(str(db_path)) as conn:
        row_count = conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]
    assert row_count == 0
    assert not wal_path.exists()
    assert not shm_path.exists()


def test_daemon_cleanup_can_destroy_sqlite_state_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pid_path, db_path, wal_path, shm_path, socket_path = _write_cleanup_files(tmp_path)

    monkeypatch.setenv("NEURACORE_DAEMON_PID_PATH", str(pid_path))
    monkeypatch.setenv("NEURACORE_DAEMON_DB_PATH", str(db_path))
    monkeypatch.setattr(shared, "SOCKET_PATH", str(socket_path))
    monkeypatch.setattr(shared, "get_runner_pids", lambda: set())

    shared.daemon_cleanup(state_db_action="delete")

    assert not pid_path.exists()
    assert not socket_path.exists()
    assert not db_path.exists()
    assert not wal_path.exists()
    assert not shm_path.exists()
