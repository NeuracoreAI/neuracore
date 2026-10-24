"""Data daemon settings that an import applies to its workers."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from neuracore_types import Codec

from neuracore.data_daemon.daemon_control import pid_is_running, read_pid_from_file
from neuracore.data_daemon.helpers import (
    get_daemon_pid_path,
    get_daemon_recordings_root_path,
)

logger = logging.getLogger(__name__)

MIN_SPOOL_LIMIT_BYTES = 2 * 1024**3
# Half the daemon's default storage_limit (50% of free disk), so spool plus
# finished traces stay under it and the daemon never drops frames for space.
SPOOL_FREE_DISK_FRACTION = 0.25


def auto_spool_limit_bytes(recordings_root: Path | None = None) -> int:
    """Return a large spool cap from the free disk, so imports rarely stall."""
    root = recordings_root or get_daemon_recordings_root_path()
    probe = root
    while not probe.exists():
        probe = probe.parent
    free_bytes = shutil.disk_usage(probe).free
    return max(MIN_SPOOL_LIMIT_BYTES, int(free_bytes * SPOOL_FREE_DISK_FRACTION))


@dataclass(frozen=True)
class ImporterDaemonConfig:
    """Daemon settings for an import.

    A spool_limit of None means auto_spool_limit_bytes(). Other None values
    keep the daemon defaults.
    """

    spool_limit: int | None = None
    transcode_workers: int | None = None
    upload_workers: int | None = None
    video_codec: Codec | None = None

    @property
    def sets_daemon_launch_options(self) -> bool:
        """Whether a setting only takes effect when the daemon is launched."""
        return any(
            value is not None
            for value in (self.transcode_workers, self.upload_workers, self.video_codec)
        )

    def env(self) -> dict[str, str]:
        """Return the NCD_* environment variables for these settings."""
        spool_limit = (
            auto_spool_limit_bytes() if self.spool_limit is None else self.spool_limit
        )
        values = {
            "NCD_SPOOL_LIMIT": spool_limit,
            "NCD_TRANSCODE_WORKERS": self.transcode_workers,
            "NCD_UPLOAD_WORKERS": self.upload_workers,
            "NCD_VIDEO_CODEC": self.video_codec.value if self.video_codec else None,
        }
        return {name: str(value) for name, value in values.items() if value is not None}

    def apply_to_environment(self) -> None:
        """Set the NCD_* variables for this process and the workers it spawns.

        Call this before anything starts the daemon.
        """
        if self.sets_daemon_launch_options:
            pid = read_pid_from_file(get_daemon_pid_path())
            if pid is not None and pid_is_running(pid):
                logger.warning(
                    "The data daemon is already running (pid=%s). Daemon worker "
                    "and codec settings apply only when the daemon starts. Stop "
                    "the daemon to apply them.",
                    pid,
                )
        if self.sets_daemon_launch_options:
            logger.info(
                "A daemon started by this import keeps its worker and codec "
                "settings until it restarts."
            )
        os.environ.update(self.env())
