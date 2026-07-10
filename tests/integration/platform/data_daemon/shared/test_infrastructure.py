"""Test infrastructure: Timer, path constants, and storage lifecycle helpers.

Provides :class:`Timer` for timing assertions, test-local path constants,
per-test artifact directory setup, and the :func:`scoped_storage_state` /
:func:`apply_storage_state_action` helpers used by all test suites.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import neuracore as nc
from neuracore.core.robot import Robot, get_robot_id_from_name
from tests.integration.platform.data_daemon.shared.storage_assertions import (
    assert_post_test_storage_state,
    harness_db_path,
    harness_recordings_root,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    case_id,
    log_run_analysis,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DATA_DAEMON_TEST_ARTIFACTS_DIR,
    STORAGE_STATE_DELETE,
    STORAGE_STATE_EMPTY,
)

if TYPE_CHECKING:
    from tests.integration.platform.data_daemon.shared.test_case import build_test_case
    from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
        ContextResult,
        ContextSpec,
    )

    DataDaemonTestCase = build_test_case.DataDaemonTestCase

# Add examples dir to path so recording-worker helpers can import from it.
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent.parent.parent.parent.parent / "examples"))
# ruff: noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared mutable test state
# ---------------------------------------------------------------------------

ISOLATION_TEST_STARTED: dict[str, bool] = {"value": False}
"""Flag indicating whether at least one isolation test has run in this session."""

DATA_DAEMON_TEST_ARTIFACT_COUNTER: dict[str, int] = {"value": 0}
"""Monotonic counter used to number per-test artifact directories."""


# ---------------------------------------------------------------------------
# Per-test artifact directories
# ---------------------------------------------------------------------------


def setup_per_test_artifact_dirs(
    test_label: str,
) -> tuple[Path, Path]:
    """Create a numbered per-test artifact directory and configure env vars.

    Args:
        test_label: A short human-readable label appended to the directory
            name (e.g. the case ID).

    Returns:
        A ``(per_test_artifacts_dir, per_test_recordings_dir)`` tuple.
    """
    DATA_DAEMON_TEST_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DAEMON_TEST_ARTIFACT_COUNTER["value"] += 1
    test_number = DATA_DAEMON_TEST_ARTIFACT_COUNTER["value"]
    per_test_artifacts_dir = (
        DATA_DAEMON_TEST_ARTIFACTS_DIR / f"[{test_number}]-{test_label}"
    )
    per_test_recordings_dir = per_test_artifacts_dir / "recordings"
    per_test_recordings_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NEURACORE_DAEMON_DB_PATH"] = str(per_test_artifacts_dir / "state.db")
    os.environ["NEURACORE_DAEMON_RECORDINGS_ROOT"] = str(per_test_recordings_dir)
    os.environ["NCD_PATH_TO_STORE_RECORD"] = str(per_test_recordings_dir)
    return per_test_artifacts_dir, per_test_recordings_dir


# ---------------------------------------------------------------------------
# Storage lifecycle helpers
# ---------------------------------------------------------------------------


@contextmanager
def scoped_storage_state(
    case: DataDaemonTestCase,
    *,
    specs: Sequence[ContextSpec] = (),
) -> Generator[None]:
    """Apply storage cleanup before and after the block.

    ``"delete"`` removes the DB file and recordings folder.  ``"empty"`` clears
    DB tables and deletes recordings folder contents.  ``"preserve"`` leaves all
    local storage untouched.  The cloud dataset and robots named by ``specs``
    are deleted on exit, independently of the local storage action.

    Does **not** start, stop, or signal any daemon process.

    Args:
        case: Test case whose ``storage_state_action`` controls local cleanup.
        specs: Context specs naming the cloud dataset and robots to delete.

    Yields:
        ``None``.
    """
    apply_storage_state_action(case.storage_state_action)
    try:
        yield
    finally:
        apply_storage_state_action(case.storage_state_action)
        delete_cloud_resources(specs)
        assert_post_test_storage_state(
            storage_state_action=case.storage_state_action,
        )


def apply_storage_state_action(storage_state_action: str) -> None:
    """Apply a storage state action to the DB and recordings folder.

    Args:
        storage_state_action: One of ``"preserve"``, ``"empty"``, or ``"delete"``.
    """
    import shutil

    db_path = harness_db_path()
    recordings_root = harness_recordings_root()

    if storage_state_action == STORAGE_STATE_EMPTY:
        if db_path.exists():
            connection = sqlite3.connect(str(db_path))
            try:
                for table in ("traces", "recordings"):
                    try:
                        connection.execute(f"DELETE FROM {table}")
                    except sqlite3.OperationalError:
                        pass
                connection.commit()
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                connection.close()
        if recordings_root.exists():
            shutil.rmtree(recordings_root, ignore_errors=True)
        recordings_root.mkdir(parents=True, exist_ok=True)
    elif storage_state_action == STORAGE_STATE_DELETE:
        try:
            db_path.unlink(missing_ok=True)
        except OSError:
            pass
        if recordings_root.exists():
            shutil.rmtree(recordings_root, ignore_errors=True)

    if storage_state_action in {STORAGE_STATE_EMPTY, STORAGE_STATE_DELETE}:
        for suffix in ("-shm", "-wal"):
            try:
                Path(str(db_path) + suffix).unlink(missing_ok=True)
            except OSError:
                pass


def delete_cloud_dataset(dataset_name: str) -> None:
    """Delete a cloud dataset, logging a warning when the delete fails.

    Args:
        dataset_name: Name of the cloud dataset to delete.
    """
    try:
        nc.login()
        nc.get_dataset(dataset_name).delete()
        logger.info("Deleted cloud dataset %r", dataset_name)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to delete cloud dataset %r", dataset_name, exc_info=True)


def delete_cloud_robot(robot_name: str) -> None:
    """Delete a cloud robot by name, logging a warning when the delete fails.

    Args:
        robot_name: Name of the cloud robot to delete.
    """
    try:
        nc.login()
        robot = Robot(robot_name, instance=0)
        robot.id = get_robot_id_from_name(robot_name)
        robot.delete()
        logger.info("Deleted cloud robot %r", robot_name)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to delete cloud robot %r", robot_name, exc_info=True)


def delete_cloud_resources(specs: Sequence[ContextSpec]) -> None:
    """Delete the cloud dataset and robots named by the context specs.

    Args:
        specs: Context specs naming the cloud dataset and robots to delete.
    """
    if not specs:
        return
    delete_cloud_dataset(specs[0].dataset_name)
    for spec in specs:
        delete_cloud_robot(spec.robot_name)


# ---------------------------------------------------------------------------
# Analysis-report helpers
# ---------------------------------------------------------------------------


def build_isolation_run_analysis(
    *,
    case: DataDaemonTestCase,
    results: list[ContextResult],
    daemon_shutdown_s: float | None = None,
    final_cleanup_s: float | None = None,
    status: str = "generated",
    disk_durations: dict[str, float] | None = None,
    label_prefix: str | None = None,
    test_wall_s: float | None = None,
) -> str:
    """Build isolation analysis with daemon shutdown timings.

    Delegates to :func:`log_run_analysis` after appending optional
    shutdown-timing lines to the extra sections block.

    Args:
        case: The test case that ran.
        results: List of per-context result dicts collected during the run.
        daemon_shutdown_s: Optional measured daemon shutdown duration in seconds.
        final_cleanup_s: Optional measured total cleanup duration in seconds.
        status: Free-form status string embedded in the report header.
        label_prefix: Optional prefix (e.g. ``"offline"``, ``"online"``) to
            disambiguate multi-run summaries for the same case.

    Returns:
        The formatted analysis report as a multi-line string.
    """
    daemon_lines: list[str] = []
    if daemon_shutdown_s is not None:
        daemon_lines.append(f"    profile shutdown: {daemon_shutdown_s:.3f}s")
    if final_cleanup_s is not None:
        daemon_lines.append(f"    final cleanup:    {final_cleanup_s:.3f}s")

    display_case_id = (
        f"{label_prefix}/{case_id(case)}" if label_prefix else case_id(case)
    )
    extra_sections = ["", "  Daemon shutdown:", *daemon_lines] if daemon_lines else None
    return log_run_analysis(
        case=case,
        results=results,
        title=f"Isolation run analysis: {display_case_id}",
        status=status,
        note="Timing diagnostics are informational only.",
        extra_sections=extra_sections,
        include_in_session_summary=True,
        disk_durations=disk_durations,
        label_prefix=label_prefix,
        test_wall_s=test_wall_s,
    )


def set_case_analysis_report(
    *,
    request: pytest.FixtureRequest,
    case: DataDaemonTestCase,
    results: list[ContextResult],
    daemon_shutdown_s: float | None = None,
    final_cleanup_s: float | None = None,
    disk_durations: dict[str, float] | None = None,
    label_prefix: str | None = None,
    test_wall_s: float | None = None,
) -> None:
    """Attach an isolation analysis report to the pytest node for terminal output.

    Stores the formatted analysis string on ``request.node.run_analysis_report``
    so the conftest terminal reporter can display it.  On failure, a minimal
    fallback string is stored instead.

    Args:
        request: The active :class:`pytest.FixtureRequest`.
        case: The test case that ran.
        results: Per-context result dicts collected during the run.
        daemon_shutdown_s: Optional measured daemon shutdown duration.
        final_cleanup_s: Optional measured total cleanup duration.
        label_prefix: Optional prefix (e.g. ``"offline"``, ``"online"``) to
            disambiguate multi-run summaries for the same case.
    """
    display_case_id = (
        f"{label_prefix}-{case_id(case)}" if label_prefix else case_id(case)
    )
    try:
        request.node.run_analysis_report = build_isolation_run_analysis(
            case=case,
            results=results,
            daemon_shutdown_s=daemon_shutdown_s,
            final_cleanup_s=final_cleanup_s,
            status="generated",
            disk_durations=disk_durations,
            label_prefix=label_prefix,
            test_wall_s=test_wall_s,
        )
    except Exception as exc:  # noqa: BLE001
        request.node.run_analysis_report = "\n".join([
            "=" * 64,
            f"Isolation run analysis: {display_case_id}",
            "=" * 64,
            f"  Analysis status: failed ({exc})",
            "  Timing diagnostics are informational only.",
            "=" * 64,
        ])
