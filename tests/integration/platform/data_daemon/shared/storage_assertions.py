"""Assertions about daemon on-disk storage state after cleanup.

Sits below :mod:`assertions` in the import graph — depends only on
neuracore helpers, stdlib, and test constants.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from neuracore.data_daemon.helpers import (
    get_daemon_db_path,
    get_daemon_recordings_root_path,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    STORAGE_STATE_DELETE,
    STORAGE_STATE_EMPTY,
)


def _is_retryable_sqlite_error(exc: sqlite3.Error) -> bool:
    message = str(exc).lower()
    return any(
        text in message
        for text in (
            "database is locked",
            "disk i/o error",
            "database disk image is malformed",
            "unable to open database file",
            "no such table",
        )
    )


def _with_sqlite_retry(
    operation: Callable[[], Any],
    *,
    timeout_s: float = 20.0,
    poll_interval_s: float = 0.1,
) -> Any:
    deadline = time.monotonic() + timeout_s
    last_error: sqlite3.Error | None = None

    while time.monotonic() < deadline:
        try:
            return operation()
        except sqlite3.Error as exc:
            if not _is_retryable_sqlite_error(exc):
                raise

            last_error = exc
            time.sleep(poll_interval_s)

    raise AssertionError(
        f"Daemon DB was not queryable during storage assertion after {timeout_s}s. "
        f"db_path={get_daemon_db_path()} last_error={last_error!r}"
    ) from last_error


def assert_db_absent() -> None:
    """Fail if the active daemon DB file or its WAL/SHM sidecars still exist."""
    db_path = get_daemon_db_path()
    for candidate in (
        db_path,
        Path(str(db_path) + "-wal"),
        Path(str(db_path) + "-shm"),
    ):
        assert (
            not candidate.exists()
        ), f"DB artefact was not removed after cleanup: {candidate}"


def assert_recordings_folder_absent() -> None:
    """Fail if the active daemon recordings root directory still exists."""
    recordings_root = get_daemon_recordings_root_path()
    assert (
        not recordings_root.exists()
    ), f"Recordings folder still present: {recordings_root}"


def assert_db_empty() -> None:
    """Fail if any known daemon DB tables contain rows."""
    db_path = get_daemon_db_path()
    if not db_path.exists():
        return

    def _fetch_non_empty_tables() -> list[str]:
        with sqlite3.connect(str(db_path), timeout=5.0) as conn:
            conn.execute("PRAGMA busy_timeout = 5000")

            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            if quick_check and quick_check[0] != "ok":
                raise sqlite3.DatabaseError(
                    f"SQLite quick_check failed: {quick_check[0]}"
                )

            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            non_empty: list[str] = []
            for table in sorted(tables):
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[
                    0
                ]  # noqa: S608
                if count:
                    non_empty.append(f"  {table}: {count} row(s)")

            return non_empty

    non_empty = _with_sqlite_retry(_fetch_non_empty_tables)

    assert (
        not non_empty
    ), "Daemon DB is not empty — unexpected rows found:\n" + "\n".join(non_empty)


def assert_recordings_folder_empty() -> None:
    """Fail if the recordings root contains any files."""
    recordings_root = get_daemon_recordings_root_path()
    if not recordings_root.exists():
        return
    leftover = [p for p in recordings_root.rglob("*") if p.is_file()]
    assert not leftover, (
        f"Recordings folder is not empty after cleanup: {recordings_root}\n"
        f"  {len(leftover)} file(s) remain, e.g. {leftover[0]}"
    )


def assert_post_test_storage_state(storage_state_action: str) -> None:
    """Assert on-disk artefact state matches what the test configuration demands."""
    if storage_state_action == STORAGE_STATE_DELETE:
        assert_db_absent()
        assert_recordings_folder_absent()
    elif storage_state_action == STORAGE_STATE_EMPTY:
        assert_db_empty()
        assert_recordings_folder_empty()
