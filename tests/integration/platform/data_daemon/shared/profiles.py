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


def profile_path(profile_name: str) -> Path:
    """Return the on-disk YAML path for a daemon profile.

    Mirrors the daemon's own resolution (see
    ``rust/data_daemon_shared/src/config/profile.rs``).
    """
    return (
        Path.home() / ".neuracore" / "data_daemon" / "profiles" / f"{profile_name}.yaml"
    )


def cleanup_test_profiles() -> None:
    """Delete all daemon profile files created by tests in this session."""
    for profile_path in list(TEST_PROFILE_PATHS):
        try:
            profile_path.unlink(missing_ok=True)
        except OSError:
            pass
    TEST_PROFILE_PATHS.clear()


@contextmanager
def scoped_offline_profile() -> Generator[None]:
    """Activate a temporary offline daemon profile for the duration of the block.

    Creates a uniquely-named YAML profile under
    ``~/.neuracore/data_daemon/profiles/`` that sets ``offline: true``, points
    ``NEURACORE_DAEMON_PROFILE`` at it, then restores the previous value on
    exit.  The profile path is added to :data:`TEST_PROFILE_PATHS` so that
    :func:`cleanup_test_profiles` can remove it at teardown.

    Does **not** start, stop, or clean up any daemon processes or storage.

    Yields:
        ``None`` — ``NEURACORE_DAEMON_PROFILE`` is set to the offline profile
        while the body executes.
    """
    profile_name = f"offline_profile_{uuid.uuid4().hex[:8]}"
    offline_profile_path = profile_path(profile_name)
    offline_profile_path.parent.mkdir(parents=True, exist_ok=True)
    offline_profile_path.write_text("offline: true\n", encoding="utf-8")
    TEST_PROFILE_PATHS.add(offline_profile_path)

    previous_profile = os.environ.get("NEURACORE_DAEMON_PROFILE")
    os.environ["NEURACORE_DAEMON_PROFILE"] = profile_name
    try:
        yield
    finally:
        if previous_profile is None:
            os.environ.pop("NEURACORE_DAEMON_PROFILE", None)
        else:
            os.environ["NEURACORE_DAEMON_PROFILE"] = previous_profile


@contextmanager
def scoped_holdback_ms(milliseconds: int) -> Generator[None]:
    """Pin the dispatcher's routing holdback for the block.

    The daemon reads ``NCD_HOLDBACK_MS`` once at startup, so this must wrap the
    daemon start rather than the workload. An internal override with no
    profile-config representation: it makes a timing-dependent reproduction
    deterministic rather than covering a deployed configuration.
    """
    previous = os.environ.get("NCD_HOLDBACK_MS")
    os.environ["NCD_HOLDBACK_MS"] = str(milliseconds)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("NCD_HOLDBACK_MS", None)
        else:
            os.environ["NCD_HOLDBACK_MS"] = previous


@contextmanager
def scoped_recording_reaper_disabled() -> Generator[None]:
    """Keep uploaded recordings' files and rows on the machine for the block.

    The reaper deletes exactly what a test asserting on disk after an upload
    needs to read. Read once at startup, so this must wrap the daemon start
    rather than the workload; the value is ``"0"`` rather than the empty string,
    which the daemon treats as unset.
    """
    previous = os.environ.get("NCD_RECORDING_REAPER")
    os.environ["NCD_RECORDING_REAPER"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("NCD_RECORDING_REAPER", None)
        else:
            os.environ["NCD_RECORDING_REAPER"] = previous


@contextmanager
def scoped_online_mode() -> Generator[None]:
    """Force online daemon config for the block.

    Sets ``NCD_OFFLINE=0`` and clears ``NEURACORE_DAEMON_PROFILE`` so callers
    cannot inherit an offline profile or mode from prior tests.

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
