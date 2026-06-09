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

import neuracore as nc
from neuracore.data_daemon.const import DEFAULT_DAEMON_STARTUP_TIMEOUT_SECONDS
from neuracore.data_daemon.lifecycle.daemon_os_control import ensure_daemon_running
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_daemon_cleanup,
)
from tests.integration.platform.data_daemon.shared.process_control import (
    Timer,
    stop_daemon,
)
from tests.integration.platform.data_daemon.shared.profiles import (
    scoped_offline_profile,
    scoped_online_mode,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    MAX_TIME_TO_START_S,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    OFFLINE_DB_PATH,
    OFFLINE_RECORDINGS_ROOT,
)


@contextmanager
def scoped_daemon_storage_env() -> Generator[None, None, None]:
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


@contextmanager
def offline_daemon_running() -> Generator[None, None, None]:
    """Run the daemon in offline mode for the duration of the block.

    Asserts clean process/socket state before starting the daemon and again
    after it stops, so tests do not need to call :func:`assert_daemon_cleanup`
    themselves.

    Composes :func:`~profiles.scoped_offline_profile` (profile env)
    with :func:`~process_control.stop_daemon` /
    ``ensure_daemon_running`` (process lifecycle).

    Yields:
        ``None`` — the daemon is running in offline mode while the body
        executes.
    """
    with scoped_daemon_storage_env(), scoped_offline_profile():
        try:
            stop_daemon()
            assert_daemon_cleanup()
            ensure_daemon_running(timeout_s=DEFAULT_DAEMON_STARTUP_TIMEOUT_SECONDS)
            with Timer(MAX_TIME_TO_START_S, label="nc.login", always_log=True):
                nc.login()
            yield
        finally:
            stop_daemon()
            assert_daemon_cleanup()


@contextmanager
def online_daemon_running() -> Generator[None, None, None]:
    """Run the daemon in online mode for the duration of the block.

    Stops any suite-owned leftover daemon, asserts clean process/socket state,
    then starts a fresh daemon and asserts cleanup again after it stops.

    Forces ``NCD_OFFLINE=0`` and clears ``NEURACORE_DAEMON_PROFILE`` so
    callers cannot inherit a temporary offline profile from prior tests.

    Yields:
        ``None`` — the daemon is running in online mode while the body
        executes.
    """
    with scoped_daemon_storage_env(), scoped_online_mode():
        try:
            stop_daemon()
            assert_daemon_cleanup()
            ensure_daemon_running(timeout_s=DEFAULT_DAEMON_STARTUP_TIMEOUT_SECONDS)

            with Timer(MAX_TIME_TO_START_S, label="nc.login", always_log=True):
                nc.login()
            yield
        finally:
            stop_daemon()
            assert_daemon_cleanup()
