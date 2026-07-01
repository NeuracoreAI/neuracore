"""In-memory daemon-config cache with periodic + on-demand refresh.

Holds the effective :class:`DaemonConfig` (profile YAML + ``NCD_*`` env overlay)
in memory and re-resolves it on a background asyncio task, so consumers — chiefly
the encoder manager reading the video codec — read from memory instead of
re-parsing the profile YAML on every trace. Mirrors
:class:`~neuracore.data_daemon.connection_management.connection_manager.ConnectionManager`'s
poll-and-cache loop.

Unlike the Rust daemon, there is no on-demand refresh command from the SDK (the
SDK has no live daemon connection when ``set_video_encoding_options`` is called),
so a between-recordings profile change is picked up on the next poll — well
within the "applies to the next recording" contract.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from neuracore.data_daemon.config_manager.config import ConfigManager
from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.config_manager.profiles import ProfileManager
from neuracore.data_daemon.config_manager.video_codec import _active_profile_name

logger = logging.getLogger(__name__)

# Matches the Rust daemon's CONFIG_POLL and ORG_CONFIG_POLL cadence: the profile
# file is tiny and rarely changes, so a coarse one-second re-parse is ample.
DEFAULT_CONFIG_POLL_SECONDS = 1.0


class ConfigWatcher:
    """Cache the effective daemon config, refreshed periodically in the background.

    The cached value is read synchronously via :meth:`config` / :meth:`video_codec`
    (cheap attribute reads); the potentially-blocking YAML resolve happens on the
    executor so it never stalls the event loop.
    """

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
            resolver: Optional injectable resolver returning the effective config
                (defaults to the profile + env resolution). Lets tests supply a
                stub instead of mutating process-global env / home.
        """
        self._config = initial_config
        self._profile_name = profile_name or _active_profile_name()
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
        """Force an immediate re-resolve (awaitable). Used by any on-demand path."""
        await self._reload(asyncio.get_running_loop())

    def config(self) -> DaemonConfig:
        """Return the currently-cached effective config."""
        return self._config

    def video_codec(self) -> str | None:
        """Return the currently-cached video codec id, or ``None`` for default."""
        return self._config.video_codec
