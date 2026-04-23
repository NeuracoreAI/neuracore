"""Composite daemon lifecycle context managers for integration tests.

Sits at the top of the shared-module import graph: combines process control
(:mod:`process_control`), profile management (:mod:`profiles`), and
process/socket assertions (:mod:`assertions`) into convenience wrappers
used by every test suite.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import neuracore as nc
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
    with scoped_offline_profile():
        assert_daemon_cleanup()
        ensure_daemon_running(timeout_s=10.0)
        try:
            with Timer(MAX_TIME_TO_START_S, label="nc.login", always_log=True):
                nc.login()
            yield
        finally:
            stop_daemon()
            assert_daemon_cleanup()


@contextmanager
def online_daemon_running() -> Generator[None, None, None]:
    """Run the daemon in online mode for the duration of the block.

    Asserts clean process/socket state before starting the daemon and again
    after it stops, so tests do not need to call :func:`assert_daemon_cleanup`
    themselves.

    Forces ``NCD_OFFLINE=0`` and clears ``NEURACORE_DAEMON_PROFILE`` so
    callers cannot inherit a temporary offline profile from prior tests.

    Yields:
        ``None`` — the daemon is running in online mode while the body
        executes.
    """
    with scoped_online_mode():
        print("RUNNER: before initial assert_daemon_cleanup", flush=True)
        assert_daemon_cleanup()
        print("RUNNER: after initial assert_daemon_cleanup", flush=True)

        print("RUNNER: before initial stop_daemon", flush=True)
        stop_daemon()
        print("RUNNER: after initial stop_daemon", flush=True)

        print("RUNNER: before ensure_daemon_running", flush=True)
        daemon_pid = ensure_daemon_running(timeout_s=10.0)
        print(f"RUNNER: after ensure_daemon_running pid={daemon_pid}", flush=True)

        try:
            with Timer(MAX_TIME_TO_START_S, label="nc.login", always_log=True):
                nc.login()
            print("RUNNER: yielding to test body", flush=True)
            yield
            print("RUNNER: test body returned normally", flush=True)
        finally:
            print("RUNNER: finally before stop_daemon", flush=True)
            stop_daemon()
            print("RUNNER: finally after stop_daemon", flush=True)

            print("RUNNER: finally before assert_daemon_cleanup", flush=True)
            assert_daemon_cleanup()
            print("RUNNER: finally after assert_daemon_cleanup", flush=True)
