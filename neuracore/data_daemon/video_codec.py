"""Persist the global video codec via the daemon profile system.

The video codec is a daemon-profile setting (the same profile/``DaemonConfig``
system as the other ``NCD_*`` options), not part of the SDK's ``config.json``.
The SDK's ``set_video_encoding_options`` writes it into the active profile; the
daemon reads it back from its in-memory config, which is refreshed on a poll
(and on a ``RefreshConfig`` command), so the setting can change between
recordings without restarting the daemon.

Writes are delegated to the daemon binary's ``profile`` subcommand rather than
re-implementing the YAML format here: the binary owns the profile schema, and a
Python round-trip through a partial model would silently drop fields it does
not know about.

The active profile is ``NEURACORE_DAEMON_PROFILE`` (or ``DEFAULT_PROFILE_NAME``
when unset), matching how the daemon resolves its configuration at launch.
"""

from __future__ import annotations

import subprocess

from neuracore.data_daemon.binary import require_data_daemon_binary
from neuracore.data_daemon.const import active_profile_name


def set_active_profile_video_codec(codec: str) -> None:
    """Persist the video codec into the active daemon profile.

    Creates the active profile if missing, then writes ``video_codec`` so the
    change is picked up by the daemon for the next recording.

    Args:
        codec: The codec identifier to store (e.g. ``"h264_medium"`` or the
            ``"h264_lossless"`` default).

    Raises:
        RuntimeError: If the profile could not be written.
    """
    binary = str(require_data_daemon_binary())
    profile_name = active_profile_name()

    # An already-existing profile is the common case and not an error here.
    subprocess.run(  # noqa: S603 - bundled binary, fixed argv
        [binary, "profile", "create", profile_name],
        capture_output=True,
        check=False,
    )

    completed = subprocess.run(  # noqa: S603 - bundled binary, fixed argv
        [binary, "profile", "update", profile_name, "--video-codec", codec],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"Failed to set video codec on profile {profile_name!r}: {detail}"
        )
