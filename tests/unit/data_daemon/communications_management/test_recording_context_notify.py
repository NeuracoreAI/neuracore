"""Tests for notify_daemon_config_changed (the best-effort daemon reload nudge)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from neuracore.data_daemon.communications_management.shared_transport import (
    recording_context,
)

MODULE = (
    "neuracore.data_daemon.communications_management.shared_transport.recording_context"
)


def test_notify_is_noop_when_rust_daemon_disabled() -> None:
    native = MagicMock()
    with (
        patch(f"{MODULE}.rust_daemon_enabled", return_value=False),
        patch(f"{MODULE}._load_native", return_value=native) as load_native,
    ):
        recording_context.notify_daemon_config_changed()
    load_native.assert_not_called()
    native.refresh_config.assert_not_called()


def test_notify_calls_refresh_config_when_rust_daemon_enabled() -> None:
    native = MagicMock()
    with (
        patch(f"{MODULE}.rust_daemon_enabled", return_value=True),
        patch(f"{MODULE}._load_native", return_value=native),
    ):
        recording_context.notify_daemon_config_changed()
    native.refresh_config.assert_called_once_with()


def test_notify_swallows_native_errors() -> None:
    # Contract: never raises to the SDK caller even if the native call fails.
    native = MagicMock()
    native.refresh_config.side_effect = RuntimeError("daemon gone")
    with (
        patch(f"{MODULE}.rust_daemon_enabled", return_value=True),
        patch(f"{MODULE}._load_native", return_value=native),
    ):
        recording_context.notify_daemon_config_changed()  # must not raise
    native.refresh_config.assert_called_once_with()
