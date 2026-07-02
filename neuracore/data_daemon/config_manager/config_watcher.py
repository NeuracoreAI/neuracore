"""In-memory daemon-config cache with periodic + on-demand refresh."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from neuracore.data_daemon.config_manager.config import ConfigManager
from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.config_manager.profiles import ProfileManager
from neuracore.data_daemon.const import active_profile_name

logger = logging.getLogger(__name__)

# Matches the Rust daemon's CONFIG_POLL and ORG_CONFIG_POLL cadence: the profile
# file is tiny and rarely changes, so a coarse one-second re-parse is ample.
DEFAULT_CONFIG_POLL_SECONDS = 1.0


class ConfigWatcher:
    """Cache the effective daemon config, refreshed periodically in the background."""

    def __init__(
        self,
        initial_config: DaemonConfig,
        profile_name: str | None = None,
        check_interval: float = DEFAULT_CONFIG_POLL_SECONDS,
        resolver: Callable[[], DaemonConfig] | None = None,
    ) -> None:
        """Initialise the watcher.

        Args:
            initial_config: The launch-resolved effective config, used as the
                seed so the cached value is valid before the first poll.
            profile_name: Active profile to re-resolve; defaults to the launch
                resolution (``NEURACORE_DAEMON_PROFILE`` or the default profile).
            check_interval: Seconds between background re-resolves.
            resolver: Optional injectable resolver returning the effective config.
        """
        self._config = initial_config
        self._profile_name = profile_name or active_profile_name()
        self._check_interval = check_interval
        self._resolver = resolver or self._default_resolver
        self._stopped = False
        self._task: asyncio.Task | None = None

    def _default_resolver(self) -> DaemonConfig:
        """Resolve the effective config from the active profile + env overlay."""
        config_manager = ConfigManager(ProfileManager(), profile=self._profile_name)
        return config_manager.resolve_effective_config()

    async def start(self) -> None:
        """Start the background refresh loop."""
        self._stopped = False
        self._task = asyncio.create_task(self._check_loop())
        logger.debug("ConfigWatcher started")

    async def stop(self) -> None:
        """Stop the background refresh loop."""
        self._stopped = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.debug("ConfigWatcher stopped")

    async def _check_loop(self) -> None:
        """Periodically re-resolve the effective config into the cache."""
        loop = asyncio.get_running_loop()
        while not self._stopped:
            await asyncio.sleep(self._check_interval)
            await self._reload(loop)

    async def _reload(self, loop: asyncio.AbstractEventLoop) -> None:
        """Re-resolve off the loop; keep the last-good value on failure."""
        try:
            self._config = await loop.run_in_executor(None, self._resolver)
        except Exception as error:  # noqa: BLE001 - never crash the loop
            logger.warning(
                "Failed to resolve daemon config; keeping last-good value: %s",
                error,
            )

    async def refresh(self) -> None:
        """Force an immediate re-resolve (awaitable).

        The Python daemon has no SDK-triggered refresh command, so production
        relies on the periodic poll; this exists for tests and for parity with
        the Rust daemon's on-demand refresh.
        """
        await self._reload(asyncio.get_running_loop())

    def refresh_now(self) -> None:
        """Synchronously re-resolve the effective config into the cache.

        Unlike :meth:`refresh` (awaitable, driven off the General Loop), this
        resolves inline so a caller on any thread can read config written moments
        earlier without waiting for the periodic poll. The encoding path uses it
        when building a new video encoder so a codec set via the SDK's
        ``set_video_encoding_options`` immediately before ``start_recording`` is
        honoured for that recording rather than lagging the ~1s poll.

        Resolution failures keep the last-good value (never raised), so a caller
        on the recording hot path is never broken by a transient profile read.
        """
        try:
            self._config = self._resolver()
        except Exception as error:  # noqa: BLE001 - never break the caller
            logger.warning(
                "Failed to resolve daemon config; keeping last-good value: %s",
                error,
            )

    def video_codec(self) -> str | None:
        """Return the currently-cached video codec id, or ``None`` for default."""
        return self._config.video_codec
