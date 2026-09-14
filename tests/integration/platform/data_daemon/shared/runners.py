"""Composite daemon lifecycle context managers for integration tests.

Sits at the top of the shared-module import graph: combines process control
(:mod:`process_control`), profile management (:mod:`profiles`), and
process/socket assertions (:mod:`assertions`) into convenience wrappers
used by every test suite.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

from neuracore.data_daemon.const import DEFAULT_DAEMON_STARTUP_TIMEOUT_SECONDS
from neuracore.data_daemon.daemon_control import ensure_daemon_running
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_daemon_cleanup,
)
from tests.integration.platform.data_daemon.shared.process_control import (
    Timer,
    reclaim_leftover_daemons,
    stop_daemon,
)
from tests.integration.platform.data_daemon.shared.profiles import (
    scoped_offline_profile,
    scoped_online_mode,
)
from tests.integration.platform.data_daemon.shared.reporting import report_step
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    OFFLINE_DB_PATH,
    OFFLINE_RECORDINGS_ROOT,
)


@contextmanager
def scoped_daemon_storage_env() -> Generator[None]:
    """Point the daemon at the shared ``.data_daemon_test_state`` directory for the
    duration of the block.

    Yields:
        ``None`` — the storage env vars are configured while the body runs.
    """
    OFFLINE_RECORDINGS_ROOT.mkdir(parents=True, exist_ok=True)
    previous_recordings_root = os.environ.get("NEURACORE_DAEMON_RECORDINGS_ROOT")
    previous_db_path = os.environ.get("NEURACORE_DAEMON_DB_PATH")
    if previous_recordings_root is None:
        os.environ["NEURACORE_DAEMON_RECORDINGS_ROOT"] = str(OFFLINE_RECORDINGS_ROOT)
    if previous_db_path is None:
        os.environ["NEURACORE_DAEMON_DB_PATH"] = str(OFFLINE_DB_PATH)
    try:
        yield
    finally:
        if previous_recordings_root is None:
            os.environ.pop("NEURACORE_DAEMON_RECORDINGS_ROOT", None)
        if previous_db_path is None:
            os.environ.pop("NEURACORE_DAEMON_DB_PATH", None)


def stop_daemon_and_verify() -> None:
    """Stop the daemon and assert nothing of it is left behind.

    Reclaims leftovers only when that assertion fails. A leak is exactly what
    this assertion catches, so scanning for one on every test's happy path is
    wasted work; reclaiming on the way out of the failure keeps the leak to the
    test that caused it rather than to every test after it.

    Raises:
        AssertionError: When a daemon process, PID file, or producer subprocess
            survives the stop.
    """
    stop_daemon()
    try:
        assert_daemon_cleanup()
    except AssertionError:
        reclaim_leftover_daemons()
        raise


@contextmanager
def offline_daemon(start: bool = False) -> Generator[None]:
    """Configure the daemon for offline mode for the duration of the block.

    Asserts clean process/socket state on entry and again after the daemon
    stops, so tests do not need to call :func:`assert_daemon_cleanup`
    themselves.

    Composes :func:`~profiles.scoped_offline_profile` (profile env)
    with :func:`~process_control.stop_daemon` /
    ``ensure_daemon_running`` (process lifecycle).

    Args:
        start: Start the daemon on entry. Otherwise the SDK starts it, as it
            does for a user.

    Yields:
        ``None``, with offline mode configured while the body executes.
    """
    with scoped_daemon_storage_env(), scoped_offline_profile():
        try:
            stop_daemon_and_verify()
            if start:
                ensure_daemon_running(timeout_s=DEFAULT_DAEMON_STARTUP_TIMEOUT_SECONDS)
            yield
        finally:
            stop_daemon_and_verify()


@contextmanager
def online_daemon(start: bool = False) -> Generator[None]:
    """Configure the daemon for online mode for the duration of the block.

    Stops any suite-owned leftover daemon, asserts clean process/socket state,
    and asserts cleanup again after the daemon stops.

    Forces ``NCD_OFFLINE=0`` and clears ``NEURACORE_DAEMON_PROFILE`` so
    callers cannot inherit a temporary offline profile from prior tests.

    Args:
        start: Start a fresh daemon on entry. Otherwise the SDK starts it, as
            it does for a user.

    Yields:
        ``None``, with online mode configured while the body executes.
    """
    with scoped_daemon_storage_env(), scoped_online_mode():
        try:
            if start:
                with report_step("Start clean online daemon"):
                    stop_daemon_and_verify()
                    ensure_daemon_running(
                        timeout_s=DEFAULT_DAEMON_STARTUP_TIMEOUT_SECONDS
                    )
            else:
                stop_daemon_and_verify()
            yield
        finally:
            with report_step("Gracefully stop online daemon"):
                with Timer(
                    30.0,
                    label="daemon.online_shutdown",
                    always_log=True,
                    assert_deadline=False,
                ):
                    stop_daemon_and_verify()
