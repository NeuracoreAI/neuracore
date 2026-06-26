"""Assertions about daemon on-disk storage state after cleanup.

Sits below :mod:`assertions` in the import graph — depends only on
neuracore helpers, stdlib, and test constants.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from neuracore.data_daemon.helpers import (
    get_daemon_db_path,
    get_daemon_recordings_root_path,
)
from neuracore.data_daemon.rust_selection import is_rust_daemon_enabled
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    OFFLINE_DB_PATH,
    OFFLINE_RECORDINGS_ROOT,
    STORAGE_STATE_DELETE,
    STORAGE_STATE_EMPTY,
)


def harness_db_path() -> Path:
    """Return the DB path the harness should clean and assert against.

    The Rust daemon only sees ``NEURACORE_DAEMON_DB_PATH`` while it runs (set by
    ``scoped_daemon_storage_env``); the clean/check helpers run *outside* that
    scope, where the production getter falls back to ``~/.neuracore`` and would
    target the wrong folder. When the Rust daemon is active and the env var is
    unset, resolve the real shared test-state path the daemon actually used.
    """
    if is_rust_daemon_enabled() and not os.getenv("NEURACORE_DAEMON_DB_PATH"):
        return OFFLINE_DB_PATH
    return get_daemon_db_path()


def harness_recordings_root() -> Path:
    """Return the recordings root the harness should clean and assert against.

    See :func:`harness_db_path` for why the Rust daemon needs special handling.
    """
    if is_rust_daemon_enabled() and not os.getenv("NEURACORE_DAEMON_RECORDINGS_ROOT"):
        return OFFLINE_RECORDINGS_ROOT
    return get_daemon_recordings_root_path()


def assert_db_absent() -> None:
    """Fail if the active daemon DB file or its WAL/SHM sidecars still exist."""
    db_path = harness_db_path()
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
    recordings_root = harness_recordings_root()
    assert (
        not recordings_root.exists()
    ), f"Recordings folder still present: {recordings_root}"


_INFRA_TABLES = frozenset({
    # sqlx migration bookkeeping (Rust daemon)
    "_sqlx_migrations",
    # Alembic migration bookkeeping (legacy Python daemon)
    "alembic_version",
    # SQLite internal sequence table
    "sqlite_sequence",
})


def assert_db_empty() -> None:
    """Fail if any user-data daemon DB tables contain rows.

    Migration-bookkeeping tables (``_sqlx_migrations``, ``alembic_version``)
    and SQLite's internal ``sqlite_sequence`` are excluded — they're owned
    by the migration framework, not by the daemon's domain model, so a
    non-zero row count there is expected after the daemon has started even
    once.
    """
    db_path = harness_db_path()
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
        if table in _INFRA_TABLES:
            continue
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
    recordings_root = harness_recordings_root()
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
