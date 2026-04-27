"""Daemon profile and environment context managers for integration tests.

Manages temporary offline YAML profiles and online-mode env overrides.
No process control and no assertions — composes with :mod:`process_control`
and :mod:`runners`.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

TEST_PROFILE_PATHS: set[Path] = set()
"""Tracks daemon profile files created by tests for cleanup on teardown."""


def cleanup_test_profiles() -> None:
    """Delete all daemon profile files created by tests in this session."""
    for profile_path in list(TEST_PROFILE_PATHS):
        try:
            profile_path.unlink(missing_ok=True)
        except OSError:
            pass
    TEST_PROFILE_PATHS.clear()


@contextmanager
def scoped_offline_profile() -> Generator[None, None, None]:
    """Activate a temporary offline daemon profile for the duration of the block.

    Creates a uniquely-named YAML profile under
    ``~/.neuracore/data_daemon/profiles/`` that sets ``offline: true``, points
    ``NEURACORE_DAEMON_PROFILE`` at it, then cleans up on exit.

    Does **not** start, stop, or clean up any daemon processes or storage.

    Yields:
        ``None`` — ``NEURACORE_DAEMON_PROFILE`` is set to the offline profile
        while the body executes.
    """
    profile_name = f"offline_profile_{uuid.uuid4().hex[:8]}"
    profile_path = (
        Path.home() / ".neuracore" / "data_daemon" / "profiles" / f"{profile_name}.yaml"
    )
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text("offline: true\n", encoding="utf-8")
    TEST_PROFILE_PATHS.add(profile_path)

    previous_profile = os.environ.get("NEURACORE_DAEMON_PROFILE")
    os.environ["NEURACORE_DAEMON_PROFILE"] = profile_name
    try:
        yield
    finally:
        if previous_profile is None:
            os.environ.pop("NEURACORE_DAEMON_PROFILE", None)
        else:
            os.environ["NEURACORE_DAEMON_PROFILE"] = previous_profile
        try:
            profile_path.unlink(missing_ok=True)
        except OSError:
            pass
        finally:
            cleanup_test_profiles()


@contextmanager
def scoped_online_mode() -> Generator[None, None, None]:
    """Force online daemon config for the block and clean up test profiles.

    Sets ``NCD_OFFLINE=0`` and clears ``NEURACORE_DAEMON_PROFILE`` so callers
    cannot inherit an offline profile or mode from prior tests. Cleans up any
    test profiles created during test execution on exit.

    Yields:
        ``None`` — online daemon configuration is active while the body runs.
    """
    previous_offline = os.environ.get("NCD_OFFLINE")
    previous_profile = os.environ.get("NEURACORE_DAEMON_PROFILE")

    os.environ["NCD_OFFLINE"] = "0"
    os.environ.pop("NEURACORE_DAEMON_PROFILE", None)
    try:
        yield
    finally:
        if previous_offline is None:
            os.environ.pop("NCD_OFFLINE", None)
        else:
            os.environ["NCD_OFFLINE"] = previous_offline
        if previous_profile is None:
            os.environ.pop("NEURACORE_DAEMON_PROFILE", None)
        else:
            os.environ["NEURACORE_DAEMON_PROFILE"] = previous_profile
        cleanup_test_profiles()
