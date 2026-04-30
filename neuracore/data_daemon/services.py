"""Daemon async services lifecycle management."""

from __future__ import annotations

import logging
from pathlib import Path

import aiohttp

from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.connection_management.connection_manager import (
    ConnectionManager,
)
from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.progress_reporter import ProgressReporter
from neuracore.data_daemon.registration_management.registration_manager import (
    RegistrationManager,
)
from neuracore.data_daemon.state_management.state_manager import StateManager
from neuracore.data_daemon.state_management.state_store_sqlite import SqliteStateStore
from neuracore.data_daemon.upload_management.trace_status_updater import (
    TraceStatusUpdater,
)
from neuracore.data_daemon.upload_management.upload_manager import UploadManager

logger = logging.getLogger(__name__)


class DaemonServices:
    """Async services running on the General Loop.

    Owns the full lifecycle of all I/O-bound async services:
    - HTTP requests (uploads, progress reports, connectivity checks)
    - SQLite database operations
    - Event handling and coordination

    Use ``DaemonServices.create()`` to construct and start all services,
    and ``shutdown()`` to tear them down gracefully.
    """

    def __init__(
        self,
        client_session: aiohttp.ClientSession,
        state_store: SqliteStateStore,
        state_manager: StateManager,
        registration_manager: RegistrationManager,
        trace_status_updater: TraceStatusUpdater,
        upload_manager: UploadManager,
        connection_manager: ConnectionManager,
        progress_reporter: ProgressReporter,
    ) -> None:
        """Store fully-initialised service instances."""
        self.client_session = client_session
        self.state_store = state_store
        self.state_manager = state_manager
        self.registration_manager = registration_manager
        self.trace_status_updater = trace_status_updater
        self.upload_manager = upload_manager
        self.connection_manager = connection_manager
        self.progress_reporter = progress_reporter

    @classmethod
    async def create(
        cls,
        config: DaemonConfig,
        emitter: Emitter,
        db_path: Path,
    ) -> DaemonServices:
        """Initialise and start all async services on the General Loop."""
        logger.debug("Bootstrapping async services on General Loop...")

        client_session = aiohttp.ClientSession()
        logger.debug("Created aiohttp.ClientSession")

        state_store = SqliteStateStore(db_path)
        await state_store.init_async_store()
        logger.debug("SqliteStateStore initialized at %s", db_path)
        await state_store.reset_retrying_to_written()

        state_manager = StateManager(state_store, config, emitter=emitter)
        await state_manager.recover_startup_state()
        logger.debug("StateManager initialized")

        registration_manager = RegistrationManager(
            client_session=client_session,
            state_api=state_manager,
            emitter=emitter,
        )
        if not config.offline:
            registration_manager.start()
            logger.debug("RegistrationManager started")

        trace_status_updater = TraceStatusUpdater(client_session)
        logger.debug("TraceStatusUpdater initialized")

        upload_manager = UploadManager(
            config=config,
            client_session=client_session,
            emitter=emitter,
            trace_status_updater=trace_status_updater,
        )
        logger.debug("UploadManager initialized")

        connection_manager = ConnectionManager(client_session, emitter)
        await connection_manager.start()
        logger.debug("ConnectionManager started")

        progress_reporter = ProgressReporter(client_session, emitter)
        logger.debug("ProgressReporter initialized")

        logger.debug("Async services bootstrap complete")

        return cls(
            client_session=client_session,
            state_store=state_store,
            state_manager=state_manager,
            registration_manager=registration_manager,
            trace_status_updater=trace_status_updater,
            upload_manager=upload_manager,
            connection_manager=connection_manager,
            progress_reporter=progress_reporter,
        )

    async def shutdown(self) -> None:
        """Gracefully shut down all async services."""
        logger.debug("Shutting down async services...")

        try:
            await self.connection_manager.stop()
            logger.debug("ConnectionManager stopped")
        except Exception:
            logger.exception("Error stopping ConnectionManager")

        try:
            await self.registration_manager.shutdown()
            logger.debug("RegistrationManager stopped")
        except Exception:
            logger.exception("Error stopping RegistrationManager")

        try:
            await self.upload_manager.shutdown()
            logger.debug("UploadManager shutdown")
        except Exception:
            logger.exception("Error shutting down UploadManager")

        try:
            await self.state_store.reset_retrying_to_written()
            await self.state_store.close()
            logger.debug("SqliteStateStore closed")
        except Exception:
            logger.exception("Error closing SqliteStateStore")

        try:
            await self.client_session.close()
            logger.debug("aiohttp session closed")
        except Exception:
            logger.exception("Error closing aiohttp session")

        logger.debug("Async services shutdown complete")


__all__ = ["DaemonServices"]
