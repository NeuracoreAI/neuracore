"""Daemon bootstrap and lifecycle management.

This module provides a clean, modular initialization sequence for the
data daemon. It coordinates the startup of all subsystems in the correct
order across the three execution contexts.

INITIALIZATION SEQUENCE
=======================

    DaemonBootstrap.start()
         │
         ├─[1] Configuration
         │     └── ProfileManager → ConfigManager → DaemonConfig
         │
         ├─[2] Authentication
         │     └── Auth.login(api_key) - Initialize Auth singleton
         │
         ├─[3] Event Loops (EventLoopManager)
         │     ├── General Loop Thread started
         │     ├── Encoder Loop Thread started
         │     └── init_emitter(loop=general_loop)
         │
         ├─[4] Async Services (on General Loop)
         │     ├── aiohttp.ClientSession
         │     ├── SqliteStateStore + init_async_store()
         │     ├── StateManager (registers event listeners)
         │     ├── UploadManager (listens for READY_FOR_UPLOAD)
         │     ├── ConnectionManager + start() (monitors API)
         │     └── ProgressReporter (listens for PROGRESS_REPORT)
         │
         ├─[5] Recording & Encoding (RecordingDiskManager)
         │     ├── _TraceFilesystem (path management)
         │     ├── _TraceController (trace lifecycle)
         │     ├── _EncoderManager (encoder factory)
         │     ├── StorageBudget (disk space tracking)
         │     ├── _RawBatchWriter → schedule_on_general_loop()
         │     └── _BatchEncoderWorker → schedule_on_encoder_loop()
         │
         ├─[6] ZMQ Communications
         │     └── CommunicationsManager
         │
         └─[7] Return DaemonContext
               └── Daemon created with context, calls run()


MODULE REGISTRY
===============

Main Thread:
    EventLoopManager     - DaemonBootstrap       - Manages async loops
    CommunicationsManager- DaemonBootstrap       - ZMQ sockets
    Daemon               - runner_entry          - Message loop

General Loop:
    Emitter              - EventLoopManager      - Event coordination
    AuthManager          - bootstrap_async  - API auth (singleton)
    SqliteStateStore     - bootstrap_async  - Trace state persistence
    StateManager         - bootstrap_async  - State coordination
    UploadManager        - bootstrap_async  - Cloud uploads
    ConnectionManager    - bootstrap_async  - API monitoring
    ProgressReporter     - bootstrap_async  - Progress reporting
    _RawBatchWriter      - RecordingDiskManager  - Raw file I/O

Encoder Loop:
    _BatchEncoderWorker  - RecordingDiskManager  - Video/JSON encoding
    VideoTrace           - _EncoderManager       - H.264 encoding
    JsonTrace            - _EncoderManager       - JSON encoding

"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from neuracore.core.auth import login
from neuracore.data_daemon.communications_management.communications_manager import (
    CommunicationsManager,
)
from neuracore.data_daemon.config_manager.config import ConfigManager
from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.config_manager.profiles import ProfileManager
from neuracore.data_daemon.connection_management.connection_manager import (
    ConnectionManager,
)
from neuracore.data_daemon.const import CONFIG_DIR
from neuracore.data_daemon.event_loop_manager import EventLoopManager
from neuracore.data_daemon.progress_reporter import ProgressReporter
from neuracore.data_daemon.recording_encoding_disk_manager import (
    recording_disk_manager as rdm,
)
from neuracore.data_daemon.state_management.state_manager import StateManager
from neuracore.data_daemon.state_management.state_store_sqlite import SqliteStateStore
from neuracore.data_daemon.upload_management.upload_manager import UploadManager

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = CONFIG_DIR / "data_daemon" / "state.db"


@dataclass
class AsyncServices:
    """Services running on the General Loop.

    These services handle I/O-bound async work:
    - HTTP requests (uploads, progress reports, connectivity checks)
    - SQLite database operations
    - Event handling and coordination
    """

    client_session: aiohttp.ClientSession
    state_store: SqliteStateStore
    state_manager: StateManager
    upload_manager: UploadManager
    connection_manager: ConnectionManager
    progress_reporter: ProgressReporter


@dataclass
class DaemonContext:
    """Complete daemon context with all initialized components.

    This is returned by DaemonBootstrap.start() and contains
    references to all subsystems for the Daemon to use.
    """

    config: DaemonConfig
    loop_manager: EventLoopManager
    comm_manager: CommunicationsManager
    services: AsyncServices
    recording_disk_manager: rdm.RecordingDiskManager


async def bootstrap_async_services(
    config: DaemonConfig,
    db_path: Path = DEFAULT_DB_PATH,
) -> AsyncServices:
    """Initialize async services on the General Loop.

    This coroutine MUST be scheduled on the General Loop via:
        loop_manager.schedule_on_general_loop(bootstrap_async_services(...))

    Initialization order (dependencies flow downward):

        aiohttp.ClientSession
              │
              ├──────────────────┬─────────────────┐
              ▼                  ▼                 ▼
        ConnectionManager  UploadManager  ProgressReporter
              │                  │
              └────────┬─────────┘
                       ▼
              SqliteStateStore
                       │
                       ▼
                 StateManager
                       │
                       └── Registers event listeners:
                           • TRACE_WRITTEN → update state, emit READY_FOR_UPLOAD
                           • UPLOAD_COMPLETE → mark trace uploaded
                           • STOP_RECORDING → finalize recording
                           • IS_CONNECTED → track connectivity
                           • PROGRESS_REPORTED → mark as reported

    Args:
        config: Daemon configuration.
        db_path: Path to SQLite database.

    Returns:
        AsyncServices with all initialized services.
    """
    logger.info("Bootstrapping async services on General Loop...")

    client_session = aiohttp.ClientSession()
    logger.debug("Created aiohttp.ClientSession")

    state_store = SqliteStateStore(db_path)
    await state_store.init_async_store()
    logger.info("SqliteStateStore initialized at %s", db_path)

    state_manager = StateManager(state_store)
    logger.info("StateManager initialized")

    upload_manager = UploadManager(config, client_session)
    logger.info("UploadManager initialized")

    connection_manager = ConnectionManager(client_session)
    await connection_manager.start()
    logger.info("ConnectionManager started")

    progress_reporter = ProgressReporter(client_session)
    logger.info("ProgressReporter initialized")

    logger.info("Async services bootstrap complete")

    return AsyncServices(
        client_session=client_session,
        state_store=state_store,
        state_manager=state_manager,
        upload_manager=upload_manager,
        connection_manager=connection_manager,
        progress_reporter=progress_reporter,
    )


async def shutdown_async_services(services: AsyncServices) -> None:
    """Gracefully shutdown async services.

    Shutdown order is reverse of initialization to ensure
    dependencies are cleaned up properly.

    Args:
        services: AsyncServices to shutdown.
    """
    logger.info("Shutting down async services...")

    try:
        await services.connection_manager.stop()
        logger.debug("ConnectionManager stopped")
    except Exception:
        logger.exception("Error stopping ConnectionManager")

    try:
        await services.upload_manager.shutdown()
        logger.debug("UploadManager shutdown")
    except Exception:
        logger.exception("Error shutting down UploadManager")

    try:
        await services.state_store.close()
        logger.debug("SqliteStateStore closed")
    except Exception:
        logger.exception("Error closing SqliteStateStore")

    try:
        await services.client_session.close()
        logger.debug("aiohttp session closed")
    except Exception:
        logger.exception("Error closing aiohttp session")

    logger.info("Async services shutdown complete")


class DaemonBootstrap:
    """Coordinates daemon initialization and shutdown.

    This class manages the complete lifecycle of the daemon across
    all three execution contexts (Main Thread, General Loop, Encoder Loop).

    Usage:
        bootstrap = DaemonBootstrap()
        context = bootstrap.start()

        if context:
            daemon = Daemon(
                recording_disk_manager=context.recording_disk_manager,
                comm_manager=context.comm_manager,
                loop_manager=context.loop_manager,
            )
            try:
                daemon.run()  # Blocking
            finally:
                bootstrap.stop()
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        """Initialize bootstrap.

        Args:
            db_path: Path to SQLite database file.
        """
        self._db_path = db_path
        self._context: DaemonContext | None = None

    def _resolve_configuration(self) -> DaemonConfig | None:
        """Resolve daemon configuration from profiles, env, and CLI."""
        try:
            profile_name = os.environ.get("NEURACORE_DAEMON_PROFILE") or None

            profile_manager = ProfileManager()
            config_manager = ConfigManager(profile_manager, profile=profile_name)

            config = config_manager.resolve_effective_config()

            logger.info("Configuration resolved")
            return config
        except Exception:
            logger.exception("Failed to resolve configuration")
            return None

    def _start_event_loops(self) -> EventLoopManager | None:
        """Start the EventLoopManager with General and Encoder loops."""
        loop_manager = EventLoopManager()
        try:
            loop_manager.start()
            logger.info("       General Loop: started (I/O-bound work)")
            logger.info("       Encoder Loop: started (CPU-bound work)")
            return loop_manager
        except Exception as e:
            logger.exception(f"Failed to start EventLoopManager: {str(e)}")
            return None

    def _bootstrap_async_services(
        self,
        config: DaemonConfig,
        loop_manager: EventLoopManager,
    ) -> AsyncServices | None:
        """Bootstrap async services on the General Loop."""
        try:
            future = loop_manager.schedule_on_general_loop(
                bootstrap_async_services(config, self._db_path)
            )
            services = future.result(timeout=30.0)
            logger.info("       SqliteStateStore: initialized")
            logger.info("       StateManager: listening for events")
            logger.info("       UploadManager: ready for uploads")
            logger.info("       ConnectionManager: monitoring API")
            logger.info("       ProgressReporter: ready to report")
            return services
        except Exception:
            logger.exception("Failed to bootstrap async services")
            loop_manager.stop()
            return None

    def _init_recording_disk_manager(
        self,
        config: DaemonConfig,
        loop_manager: EventLoopManager,
        services: AsyncServices,
    ) -> rdm.RecordingDiskManager | None:
        """Initialize the RecordingDiskManager with workers on respective loops."""
        try:
            recording_disk_manager = rdm.RecordingDiskManager(
                loop_manager=loop_manager,
                recordings_root=config.path_to_store_record,
            )
            logger.info("       _RawBatchWriter: scheduled on General Loop")
            logger.info("       _BatchEncoderWorker: scheduled on Encoder Loop")
            return recording_disk_manager
        except Exception:
            logger.exception("Failed to initialize RecordingDiskManager")
            loop_manager.schedule_on_general_loop(
                shutdown_async_services(services)
            ).result(timeout=10.0)
            loop_manager.stop()
            return None

    def _shutdown_recording_disk_manager(self, ctx: DaemonContext) -> None:
        """Shutdown the RecordingDiskManager."""
        try:
            future = ctx.loop_manager.schedule_on_general_loop(
                ctx.recording_disk_manager.shutdown()
            )
            future.result(timeout=30.0)
            logger.info("       _RawBatchWriter: stopped")
            logger.info("       _BatchEncoderWorker: stopped")
        except Exception:
            logger.exception("Error shutting down RecordingDiskManager")

    def _shutdown_async_services(self, ctx: DaemonContext) -> None:
        """Shutdown async services on the General Loop."""
        try:
            future = ctx.loop_manager.schedule_on_general_loop(
                shutdown_async_services(ctx.services)
            )
            future.result(timeout=10.0)
        except Exception:
            logger.exception("Error shutting down async services")

    def _stop_event_loops(self, ctx: DaemonContext) -> None:
        """Stop the EventLoopManager."""
        try:
            ctx.loop_manager.stop()
            logger.info("       General Loop: stopped")
            logger.info("       Encoder Loop: stopped")
        except Exception:
            logger.exception("Error stopping EventLoopManager")

    def _initialize_auth(self, config: DaemonConfig) -> bool:
        """Initialize authentication using API key from config or environment.

        Args:
            config: Daemon configuration containing optional API key.

        Returns:
            True if authentication succeeded or is not required (offline mode),
            False if authentication failed.
        """
        if config.offline:
            logger.info("Offline mode: skipping authentication")
            return True
        try:
            api_key = config.api_key
            login(api_key)
            logger.info("Authentication: successful")
            return True
        except Exception:
            logger.exception("Failed to initialize authentication")
            return False

    def start(self) -> DaemonContext | None:
        """Start all daemon subsystems in correct order.

        Initialization sequence:
        1. Configuration - resolve from profiles, env, CLI
        2. Authentication - initialize Auth singleton with API key
        3. EventLoopManager - start General + Encoder loop threads
        4. AsyncServices - bootstrap on General Loop
        5. RecordingDiskManager - initialize with workers on respective loops
        6. CommunicationsManager - ZMQ socket management

        Returns:
            DaemonContext if successful, None if startup failed.
        """
        logger.info("=" * 60)
        logger.info("DAEMON BOOTSTRAP STARTING")
        logger.info("=" * 60)

        logger.info("[1/6] Resolving configuration...")
        config = self._resolve_configuration()
        if config is None:
            return None

        logger.info("[2/6] Initializing authentication...")
        if not self._initialize_auth(config):
            return None

        logger.info("[3/6] Starting EventLoopManager...")
        loop_manager = self._start_event_loops()
        if loop_manager is None:
            return None

        logger.info("[4/6] Bootstrapping async services on General Loop...")
        services = self._bootstrap_async_services(config, loop_manager)
        if services is None:
            return None

        logger.info("[5/6] Initializing RecordingDiskManager...")
        recording_disk_manager = self._init_recording_disk_manager(
            config, loop_manager, services
        )
        if recording_disk_manager is None:
            return None

        logger.info("[6/6] Creating CommunicationsManager...")
        comm_manager = CommunicationsManager()
        logger.info("       ZMQ sockets ready")

        self._context = DaemonContext(
            config=config,
            loop_manager=loop_manager,
            comm_manager=comm_manager,
            services=services,
            recording_disk_manager=recording_disk_manager,
        )

        logger.info("=" * 60)
        logger.info("DAEMON BOOTSTRAP COMPLETE")
        logger.info("=" * 60)
        return self._context

    def stop(self) -> None:
        """Gracefully shutdown all daemon subsystems.

        Shutdown order is reverse of initialization:
        1. RecordingDiskManager (flush pending writes, stop workers)
        2. AsyncServices (stop uploads, close connections)
        3. EventLoopManager (stop loop threads)
        """
        if self._context is None:
            logger.warning("Cannot stop: daemon not started")
            return

        logger.info("=" * 60)
        logger.info("DAEMON SHUTDOWN STARTING")
        logger.info("=" * 60)

        ctx = self._context

        logger.info("[1/3] Shutting down RecordingDiskManager...")
        self._shutdown_recording_disk_manager(ctx)

        logger.info("[2/3] Shutting down async services...")
        self._shutdown_async_services(ctx)

        logger.info("[3/3] Stopping EventLoopManager...")
        self._stop_event_loops(ctx)

        self._context = None

        logger.info("=" * 60)
        logger.info("DAEMON SHUTDOWN COMPLETE")
        logger.info("=" * 60)

    @property
    def context(self) -> DaemonContext | None:
        """Return the current daemon context."""
        return self._context
