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
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import neuracore as nc
from neuracore.data_daemon.helpers import (
    get_daemon_db_path,
    get_daemon_recordings_root_path,
)
from tests.integration.platform.data_daemon.shared.storage_assertions import (
    assert_post_test_storage_state,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    case_id,
    log_run_analysis,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    STORAGE_STATE_DELETE,
    STORAGE_STATE_EMPTY,
)

if TYPE_CHECKING:
    from tests.integration.platform.data_daemon.shared.test_case import build_test_case
    from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
        ContextResult,
    )

    DataDaemonTestCase = build_test_case.DataDaemonTestCase

# Add examples dir to path so recording-worker helpers can import from it.
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent.parent.parent.parent.parent / "examples"))
# ruff: noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test-state directories and path constants
# ---------------------------------------------------------------------------

DATA_DAEMON_TEST_STATE_ROOT = Path(".data_daemon_test_state")
"""Root directory for all test-local daemon state (DB, recordings, artifacts)."""

DATA_DAEMON_TEST_ARTIFACTS_DIR = (
    DATA_DAEMON_TEST_STATE_ROOT / "artifacts" / time.strftime("%Y%m%d_%H%M%S")
)
"""Timestamped directory where per-test artifact copies are stored."""

OFFLINE_RECORDINGS_ROOT = DATA_DAEMON_TEST_STATE_ROOT / "recordings"
"""Directory used as the offline daemon's recordings root in tests."""

OFFLINE_DB_PATH = DATA_DAEMON_TEST_STATE_ROOT / "state.db"
"""Path used for the offline daemon's SQLite state DB in tests."""

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
    dataset_name: str | None = None,
) -> Generator[None, None, None]:
    """Apply storage cleanup before and after the block.

    ``"delete"`` removes the DB file, recordings folder, and the cloud dataset
    (when ``dataset_name`` is provided).  ``"empty"`` clears DB tables and
    deletes recordings folder contents.  ``"preserve"`` leaves all storage
    untouched.

    Does **not** start, stop, or signal any daemon process.

    Args:
        case: Test case whose ``storage_state_action`` controls cleanup.
        dataset_name: Optional cloud dataset name to delete when
            ``storage_state_action`` is ``"delete"``.

    Yields:
        ``None``.
    """
    apply_storage_state_action(case.storage_state_action)
    try:
        yield
    finally:
        apply_storage_state_action(case.storage_state_action)
        if (
            dataset_name is not None
            and case.storage_state_action == STORAGE_STATE_DELETE
        ):
            delete_cloud_dataset(dataset_name)
        assert_post_test_storage_state(
            storage_state_action=case.storage_state_action,
        )


def apply_storage_state_action(storage_state_action: str) -> None:
    """Apply a storage state action to the DB and recordings folder.

    Args:
        storage_state_action: One of ``"preserve"``, ``"empty"``, or ``"delete"``.
    """
    import shutil

    db_path = get_daemon_db_path()
    recordings_root = get_daemon_recordings_root_path()

    if storage_state_action == STORAGE_STATE_EMPTY:
        if db_path.exists():
            with sqlite3.connect(str(db_path)) as connection:
                for table in ("traces", "recordings"):
                    try:
                        connection.execute(f"DELETE FROM {table}")
                    except sqlite3.OperationalError:
                        pass
                connection.commit()
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
        for suffix in (".shm", ".wal"):
            try:
                db_path.with_suffix(db_path.suffix + suffix).unlink(missing_ok=True)
            except OSError:
                pass


def delete_cloud_dataset(dataset_name: str) -> None:
    """Delete a cloud dataset when the storage action demands it.

    Args:
        dataset_name: Name of the cloud dataset to delete.
    """
    try:
        nc.login()
        nc.get_dataset(dataset_name).delete()
        logger.info(
            "Deleted cloud dataset %r (storage_state_action=delete)", dataset_name
        )
    except Exception:  # noqa: BLE001
        logger.warning("Failed to delete cloud dataset %r", dataset_name, exc_info=True)


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
    case_id_prefix: str | None = None,
    test_wall_s: float | None = None,
) -> str:
    """Build isolation analysis with daemon shutdown timings.

    Delegates to :func:`log_run_analysis` after appending optional
    shutdown-timing lines to the extra sections block. The title shows the
    overarching test wall time when ``test_wall_s`` is provided; otherwise
    falls back to the max-context span plus shutdown overhead.

    Args:
        case: The test case that ran.
        results: List of per-context result dicts collected during the run.
        daemon_shutdown_s: Optional measured daemon shutdown duration in seconds.
        final_cleanup_s: Optional measured total cleanup duration in seconds.
        status: Free-form status string embedded in the report header.
        test_wall_s: Total elapsed wall time for the entire test, from the
            first line of the test function to the finally block. When
            provided, used as ``total_wall_s`` in the session summary.

    Returns:
        The formatted analysis report as a multi-line string.
    """
    daemon_lines: list[str] = []
    if daemon_shutdown_s is not None:
        daemon_lines.append(f"    profile shutdown: {daemon_shutdown_s:.3f}s")
    if final_cleanup_s is not None:
        daemon_lines.append(f"    final cleanup:    {final_cleanup_s:.3f}s")

    extra_sections = ["", "  Daemon shutdown:", *daemon_lines] if daemon_lines else None

    display_wall_s = test_wall_s
    if display_wall_s is None:
        display_wall_s = (
            max(ctx.wall_stopped_at for ctx in results)
            - min(ctx.wall_started_at or 0.0 for ctx in results)
            if results
            else 0.0
        )
        if daemon_shutdown_s is not None:
            display_wall_s += daemon_shutdown_s
        if final_cleanup_s is not None:
            display_wall_s += final_cleanup_s
    title_suffix = f"  (wall={display_wall_s:.1f}s)" if (results or test_wall_s) else ""

    return log_run_analysis(
        case=case,
        results=results,
        title=f"Isolation run analysis: {case_id(case)}{title_suffix}",
        status=status,
        note="Timing diagnostics are informational only.",
        extra_sections=extra_sections,
        include_in_session_summary=True,
        disk_durations=disk_durations,
        case_id_prefix=case_id_prefix,
        daemon_shutdown_s=daemon_shutdown_s,
        final_cleanup_s=final_cleanup_s,
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
    case_id_prefix: str | None = None,
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
        test_wall_s: Total elapsed wall time from the first line of the test
            function to the finally block.
    """
    try:
        request.node.run_analysis_report = build_isolation_run_analysis(
            case=case,
            results=results,
            daemon_shutdown_s=daemon_shutdown_s,
            final_cleanup_s=final_cleanup_s,
            status="generated",
            disk_durations=disk_durations,
            case_id_prefix=case_id_prefix,
            test_wall_s=test_wall_s,
        )
    except Exception as exc:  # noqa: BLE001
        request.node.run_analysis_report = "\n".join([
            "=" * 64,
            f"Isolation run analysis: {case_id(case)}",
            "=" * 64,
            f"  Analysis status: failed ({exc})",
            "  Timing diagnostics are informational only.",
            "=" * 64,
        ])


def build_terminal_summary(
    session_runs: list[dict[str, object]],
) -> list[str]:
    """Build terminal summary lines for session test runs.

    Consolidates session run analysis with wall clock times and infrastructure
    timings. Wall times are computed from each run's context results, which are
    already calculated and stored during test execution.

    Args:
        session_runs: List of session run dicts from SESSION_RUNS, each containing
            case_id, dataset_name, timer_stats, and context_results.

    Returns:
        List of formatted summary lines ready for terminal output.
    """
    from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (  # noqa: E501
        _format_timer_stats_line,
    )
    from tests.integration.platform.data_daemon.shared.timer import Timer

    if not session_runs:
        return []

    separator = "=" * 64
    lines = [
        "",
        separator,
        f"Session summary  ({len(session_runs)} test(s) completed)",
        separator,
    ]

    all_labels = sorted({label for run in session_runs for label in run["timer_stats"]})
    for run in session_runs:
        total_wall_s = run.get("total_wall_s", 0.0)
        dataset_suffix = (
            f"  dataset={run['dataset_name']!r}" if run.get("dataset_name") else ""
        )
        lines.append(
            f"\n  {run['case_id']}  (wall={total_wall_s:.1f}s){dataset_suffix}"
        )
        context_results = run.get("context_results", [])
        if context_results:
            ctx_parts = [
                f"ctx[{c['context_index']}]={c['wall_s']:.1f}s" for c in context_results
            ]
            lines.append(f"    contexts: {', '.join(ctx_parts)}")
        for label in all_labels:
            stats = run["timer_stats"].get(label)
            if stats is not None:
                lines.append(_format_timer_stats_line(label, stats))
            else:
                lines.append(f"    {label:<42}  ---")

    infra_labels = sorted(label for label in Timer._stats if label not in all_labels)
    if infra_labels:
        lines.append("\n  Infrastructure timings:")
        for label in infra_labels:
            lines.append(_format_timer_stats_line(label, Timer._stats[label]))

    lines.append(separator)
    return lines
