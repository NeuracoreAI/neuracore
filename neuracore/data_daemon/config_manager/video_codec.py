"""Persist the global video codec via the daemon profile system.

The video codec is a daemon-profile setting (the same profile/`DaemonConfig`
system as the other ``NCD_*`` options), not part of the SDK's ``config.json``.
The SDK's ``set_video_encoding_options`` writes it into the active profile; the
daemon reads it back from its in-memory config
(:class:`~neuracore.data_daemon.config_manager.config_watcher.ConfigWatcher`),
which is refreshed on a poll (and, on the Rust daemon, on a ``RefreshConfig``
command), so the setting can change between recordings without restarting the
daemon.

The active profile is ``NEURACORE_DAEMON_PROFILE`` (or ``DEFAULT_PROFILE_NAME``
when unset), matching how the daemon resolves its configuration at launch.
"""

from __future__ import annotations

import os

from neuracore.data_daemon.config_manager.profiles import (
    ProfileAlreadyExist,
    ProfileManager,
)
from neuracore.data_daemon.const import DEFAULT_PROFILE_NAME


def _active_profile_name() -> str:
    """Return the active daemon profile name, mirroring the launch resolution."""
    return os.environ.get("NEURACORE_DAEMON_PROFILE") or DEFAULT_PROFILE_NAME


def set_active_profile_video_codec(codec: str) -> None:
    """Persist the video codec into the active daemon profile.

    Creates the active profile if it does not exist yet (whatever
    ``NEURACORE_DAEMON_PROFILE`` names, or the default when unset) so the setting
    always has somewhere to persist, then writes ``video_codec`` so the change is
    picked up by the daemon for the next recording. Note this is broader than the
    daemon's launch behaviour, which only auto-creates the *default* profile;
    here we materialise the active profile with defaults if the user pointed
    ``NEURACORE_DAEMON_PROFILE`` at one they never created. Pass the
    ``h264_lossless`` default to switch back from a lossy mode.

    Args:
        codec: The codec identifier to store (e.g. ``"h264_medium"`` or the
            ``"h264_lossless"`` default).
    """
    profile_name = _active_profile_name()
    manager = ProfileManager()
    try:
        manager.create_profile(profile_name)
    except ProfileAlreadyExist:
        pass

    manager.update_profile(profile_name, {"video_codec": codec})
