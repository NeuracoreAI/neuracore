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

from neuracore.data_daemon.config_manager.profiles import (
    ProfileAlreadyExist,
    ProfileManager,
)
from neuracore.data_daemon.const import active_profile_name


def set_active_profile_video_codec(codec: str) -> None:
    """Persist the video codec into the active daemon profile.

    Creates the active profile if missing, then writes ``video_codec`` so the
    change is picked up by the daemon for the next recording.

    Args:
        codec: The codec identifier to store (e.g. ``"h264_medium"`` or the
            ``"h264_lossless"`` default).
    """
    profile_name = active_profile_name()
    manager = ProfileManager()
    try:
        manager.create_profile(profile_name)
    except ProfileAlreadyExist:
        pass

    manager.update_profile(profile_name, {"video_codec": codec})
