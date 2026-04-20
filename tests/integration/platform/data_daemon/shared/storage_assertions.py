"""Assertions about daemon on-disk storage state after cleanup.

Sits below :mod:`assertions` in the import graph — depends only on
neuracore helpers, stdlib, and test constants.
"""

from __future__ import annotations

import sqlite3

from neuracore.data_daemon.helpers import (
    get_daemon_db_path,
    get_daemon_recordings_root_path,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    STORAGE_STATE_DELETE,
    STORAGE_STATE_EMPTY,
)


def assert_db_absent() -> None:
    """Fail if the active daemon DB file or its WAL/SHM sidecars still exist."""
    db_path = get_daemon_db_path()
    for candidate in (
        db_path,
        db_path.with_suffix(db_path.suffix + ".wal"),
        db_path.with_suffix(db_path.suffix + ".shm"),
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
    with sqlite3.connect(str(db_path)) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    non_empty: list[str] = []
    for table in sorted(tables):
        with sqlite3.connect(str(db_path)) as conn:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[
                0
            ]  # noqa: S608
        if count:
            non_empty.append(f"  {table}: {count} row(s)")
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
