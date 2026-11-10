"""Behavioural correctness tests for daemon notification-stream signalling.

Verifies that a live daemon reports over iceoryx2 whether it is subscribed to
the backend's recording notifications: subscribed when online, never when
offline.
"""

from __future__ import annotations

import pytest

import neuracore.data_daemon.daemon_control as daemon_control
from tests.integration.platform.data_daemon.shared.runners import (
    offline_daemon_running,
    online_daemon_running,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    has_configured_org,
)

SUBSCRIBE_TIMEOUT_SECONDS = 20.0

OFFLINE_WAIT_SECONDS = 2.0


def _wait_until_subscribed(timeout_s: float) -> bool:
    """Ask the running daemon whether it hears recording notifications."""
    data_bridge = daemon_control._import_data_bridge()
    assert data_bridge is not None, "The native extension must import here"
    return (
        data_bridge.wait_until_ready(timeout_s, notifications_subscribed=True)
        is not None
    )


def test_online_daemon_reports_it_is_subscribed() -> None:
    """Verify an online daemon reports the notification stream as open."""
    if not has_configured_org():
        pytest.skip(
            "Notification-stream tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )
    with online_daemon_running():
        assert _wait_until_subscribed(
            SUBSCRIBE_TIMEOUT_SECONDS
        ), f"Daemon did not report a subscription in {SUBSCRIBE_TIMEOUT_SECONDS}s"


def test_offline_daemon_never_reports_it_is_subscribed() -> None:
    """Verify an offline daemon never reports the notification stream as open."""
    with offline_daemon_running():
        assert not _wait_until_subscribed(
            OFFLINE_WAIT_SECONDS
        ), "Offline daemon reported a notification subscription"
