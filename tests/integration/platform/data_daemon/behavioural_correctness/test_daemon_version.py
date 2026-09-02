"""Behavioural correctness tests for daemon version signalling.

Verifies that a live daemon answers the iceoryx2 version query with the
neuracore version it was built from, that this matches the installed neuracore
package, and that a daemon built from a different version is refused with a
message naming both versions. These tests force online mode so startup never
inherits an offline profile.
"""

from __future__ import annotations

from unittest import mock

import pytest

import neuracore.data_daemon.daemon_control as daemon_control
from neuracore.data_daemon.daemon_control import (
    DaemonVersionMismatchError,
    ensure_daemon_running,
    pid_is_running,
)
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running

STALE_SDK_VERSION = "0.0.0+stale"

VERSION_QUERY_TIMEOUT_SECONDS = 5.0


def _query_daemon_version() -> str | None:
    """Ask the running daemon for its build version over iceoryx2."""
    data_bridge = daemon_control._import_data_bridge()
    assert data_bridge is not None, "The native extension must import here"
    return data_bridge.daemon_version(VERSION_QUERY_TIMEOUT_SECONDS)


def test_running_daemon_reports_the_installed_neuracore_version() -> None:
    """Verify the daemon answers with the version of the installed package.

    The daemon comes from the binary this install ships, so the version it
    reports over iceoryx2 must equal the installed neuracore version.
    """
    with online_daemon_running():
        sdk_version = daemon_control._sdk_version()
        reported_version = _query_daemon_version()
        assert reported_version is not None and (
            daemon_control._pep440(reported_version) == sdk_version
        ), (
            f"Daemon reported neuracore version {reported_version!r}, "
            f"expected the installed version {sdk_version!r}"
        )

        daemon_pid = ensure_daemon_running()
        assert pid_is_running(
            daemon_pid
        ), f"ensure_daemon_running returned pid {daemon_pid}, which is not running"


def test_daemon_from_another_neuracore_version_is_rejected() -> None:
    """Verify a mismatched daemon is refused with both versions in the message.

    The daemon stays real and only the installed version is replaced, so the
    comparison runs over the true iceoryx2 round trip. The replacement starts
    inside the block because ``online_daemon_running`` itself calls
    ``ensure_daemon_running``, which the mismatch would then break.
    """
    with online_daemon_running():
        reported_version = _query_daemon_version()
        assert (
            reported_version is not None
        ), "Daemon did not answer the version query, so no mismatch can be shown"

        with mock.patch.object(
            daemon_control, "_sdk_version", return_value=STALE_SDK_VERSION
        ):
            with pytest.raises(DaemonVersionMismatchError) as exc_info:
                ensure_daemon_running()

        message = str(exc_info.value)
        assert (
            reported_version in message
        ), f"Mismatch message does not name the daemon version: {message!r}"
        assert (
            STALE_SDK_VERSION in message
        ), f"Mismatch message does not name the installed version: {message!r}"
        assert (
            "neuracore data-daemon stop" in message
        ), f"Mismatch message does not give the stop command: {message!r}"
