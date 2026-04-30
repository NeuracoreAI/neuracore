"""Daemon runtime orchestration and lifecycle management."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from neuracore.core.auth import login
from neuracore.data_daemon.communications_management.shared_transport.communications_manager import (
    CommunicationsManager,
)
from neuracore.data_daemon.communications_management.consumer.data_bridge import (
    DataBridge,
)
from neuracore.data_daemon.config_manager.config import ConfigManager
from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.config_manager.profiles import (
    ProfileAlreadyExist,
    ProfileManager,
)
from neuracore.data_daemon.const import (
    DEFAULT_PROFILE_NAME,
    DEFAULT_SHARED_MEMORY_SIZE,
    DEFAULT_VIDEO_SLOT_COUNT,
    DEFAULT_VIDEO_SLOT_SIZE,
)
from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.event_loop_manager import EventLoopManager
from neuracore.data_daemon.helpers import is_debug_mode
from neuracore.data_daemon.lifecycle.daemon_os_control import acquire_pid_file
from neuracore.data_daemon.lifecycle.runtime_recovery import (
    cleanup_stale_shared_slot_segments,
    cleanup_socket_files,
    cleanup_stale_shared_memory_buffers,
    reconcile_state_with_filesystem,
    shared_memory_free_bytes,
    shared_memory_required_bytes,
    validate_or_recover_sqlite,
)
from neuracore.data_daemon.recording_encoding_disk_manager import (
    recording_disk_manager as rdm,
)
from neuracore.data_daemon.services import DaemonServices
from neuracore.data_daemon.tools.event_logger import EventLogger

logger = logging.getLogger(__name__)


@dataclass
class DaemonContext:
    """Complete daemon context with all initialized components."""

    config: DaemonConfig
    loop_manager: EventLoopManager
    comm_manager: CommunicationsManager
    services: DaemonServices
    recording_disk_manager: rdm.RecordingDiskManager


class DaemonRuntime:
    """Coordinates daemon initialization, runtime execution, and shutdown."""

    def __init__(
        self,
        db_path: Path,
        *,
        pid_path: Path,
        socket_paths: tuple[Path, ...],
    ) -> None:
        """Initialize runtime paths and state for a daemon process instance."""
        self._db_path = db_path
        self._pid_path = pid_path
        self._socket_paths = socket_paths
        self._manage_pid = os.environ.get("NEURACORE_DAEMON_MANAGE_PID", "1") != "0"
        self._context: DaemonContext | None = None
        self._daemon: DataBridge | None = None
        self._event_logger: EventLogger | None = None

    def _get_recordings_root(self, config: DaemonConfig) -> Path:
        if config.path_to_store_record:
            return Path(config.path_to_store_record)
        return self._db_path.parent / "recordings"

    def _prepare_runtime_state(self, config: DaemonConfig) -> Path:
        if self._manage_pid:
            acquire_pid_file(self._pid_path)

        cleaned_shared_buffers = cleanup_stale_shared_memory_buffers()
        if cleaned_shared_buffers:
            logger.info(
                "Recovered %d stale shared-memory allocation(s) from /dev/shm",
                cleaned_shared_buffers,
            )

        cleaned_shared_slots = cleanup_stale_shared_slot_segments()
        if cleaned_shared_slots:
            logger.info(
                "Recovered %d stale shared-slot segment(s) from /dev/shm",
                cleaned_shared_slots,
            )

        cleanup_socket_files(self._socket_paths)

        sqlite_ok = validate_or_recover_sqlite(self._db_path, recover=True)
        if not sqlite_ok:
            logger.warning("SQLite recovered by rotation; new DB will be created.")

        try:
            free_shared_bytes = shared_memory_free_bytes()
            min_required_bytes = shared_memory_required_bytes(
                DEFAULT_SHARED_MEMORY_SIZE,
                metadata_size=4096,
            )
            video_required_bytes = shared_memory_required_bytes(
                DEFAULT_VIDEO_SLOT_SIZE * DEFAULT_VIDEO_SLOT_COUNT,
                metadata_size=4096,
            )
            if free_shared_bytes < min_required_bytes:
                logger.warning(
                    "Shared-memory startup preflight: only %d bytes free in /dev/shm; "
                    "%d bytes are required for default shared memory. "
                    "New shared-memory sessions will fail until space is reclaimed.",
                    free_shared_bytes,
                    min_required_bytes,
                )
            elif free_shared_bytes < video_required_bytes:
                logger.warning(
                    "Shared-memory startup preflight: %d bytes free in /dev/shm; "
                    "default non-video traffic fits, but default video shared "
                    "memory needs %d bytes.",
                    free_shared_bytes,
                    video_required_bytes,
                )
        except Exception:
            logger.exception("Failed shared-memory startup preflight")

        recordings_root = self._get_recordings_root(config)
        recordings_root.mkdir(parents=True, exist_ok=True)
        return recordings_root

    def _reconcile_runtime_state(
        self,
        loop_manager: EventLoopManager,
        services: DaemonServices,
        recordings_root: Path,
    ) -> bool:
        try:
            future = loop_manager.schedule_on_general_loop(
                reconcile_state_with_filesystem(services.state_store, recordings_root)
            )
            future.result(timeout=30.0)
            logger.debug("       Runtime state reconciled with filesystem")
            return True
        except Exception:
            logger.exception("Failed to reconcile runtime state")
            loop_manager.schedule_on_general_loop(services.shutdown()).result(
                timeout=10.0
            )
            loop_manager.stop()
            return False

    def _resolve_configuration(self) -> DaemonConfig | None:
        """Resolve daemon configuration from profiles, env, and CLI."""
        try:
            profile_name = (
                os.environ.get("NEURACORE_DAEMON_PROFILE") or DEFAULT_PROFILE_NAME
            )

            profile_manager = ProfileManager()

            try:
                profile_manager.create_profile(DEFAULT_PROFILE_NAME)
            except ProfileAlreadyExist:
                pass
            except Exception:
                logger.exception(
                    "Failed to create default profile %r", DEFAULT_PROFILE_NAME
                )

            config_manager = ConfigManager(profile_manager, profile=profile_name)

            config = config_manager.resolve_effective_config()

            logger.debug("Configuration resolved")
            return config
        except Exception:
            logger.exception("Failed to resolve configuration")
            return None

    def _start_event_loops(self) -> tuple[EventLoopManager, Emitter] | None:
        loop_manager = EventLoopManager()
        try:
            emitter = loop_manager.start()
            logger.debug("       General Loop: started (I/O-bound work)")
            logger.debug("       Encoder Loop: started (CPU-bound work)")
            return loop_manager, emitter
        except Exception as error:
            logger.exception(f"Failed to start EventLoopManager: {str(error)}")
            return None

    def _bootstrap_async_services(
        self,
        config: DaemonConfig,
        loop_manager: EventLoopManager,
        emitter: Emitter,
    ) -> DaemonServices | None:
        try:
            future = loop_manager.schedule_on_general_loop(
                DaemonServices.create(config, emitter, self._db_path)
            )
            services = future.result(timeout=30.0)
            logger.debug("       SqliteStateStore: initialized")
            logger.debug("       StateManager: listening for events")
            logger.debug("       UploadManager: ready for uploads")
            logger.debug("       ConnectionManager: monitoring API")
            logger.debug("       ProgressReporter: ready to report")
            return services
        except Exception:
            logger.exception("Failed to bootstrap async services")
            loop_manager.stop()
            return None

    def _init_recording_disk_manager(
        self,
        config: DaemonConfig,
        loop_manager: EventLoopManager,
        emitter: Emitter,
        services: DaemonServices,
    ) -> rdm.RecordingDiskManager | None:
        try:
            recording_disk_manager = rdm.RecordingDiskManager(
                loop_manager=loop_manager,
                emitter=emitter,
                recordings_root=config.path_to_store_record,
            )
            logger.debug("       _RawBatchWriter: scheduled on General Loop")
            logger.debug("       _BatchEncoderWorker: scheduled on Encoder Loop")
            return recording_disk_manager
        except Exception:
            logger.exception("Failed to initialize RecordingDiskManager")
            loop_manager.schedule_on_general_loop(services.shutdown()).result(
                timeout=10.0
            )
            loop_manager.stop()
            return None

    def _shutdown_recording_disk_manager(self, ctx: DaemonContext) -> None:
        try:
            future = ctx.loop_manager.schedule_on_general_loop(
                ctx.recording_disk_manager.shutdown()
            )
            future.result(timeout=30.0)
            logger.debug("       _RawBatchWriter: stopped")
            logger.debug("       _BatchEncoderWorker: stopped")
        except Exception:
            logger.exception("Error shutting down RecordingDiskManager")

    def _shutdown_async_services(self, ctx: DaemonContext) -> None:
        try:
            future = ctx.loop_manager.schedule_on_general_loop(ctx.services.shutdown())
            future.result(timeout=10.0)
        except Exception:
            logger.exception("Error shutting down async services")

    def _stop_event_loops(self, ctx: DaemonContext) -> None:
        try:
            ctx.loop_manager.stop()
            logger.debug("       General Loop: stopped (I/O-bound work)")
            logger.debug("       Encoder Loop: stopped (CPU-bound work)")
        except Exception:
            logger.exception("Error stopping EventLoopManager")

    def _initialize_auth(self, config: DaemonConfig) -> bool:
        if config.offline:
            logger.info("Offline mode: skipping authentication")
            return True
        try:
            login(config.api_key)
            logger.info("Authentication: successful")
            return True
        except Exception:
            logger.exception("Failed to initialize authentication")
            return False

    def initialize(self) -> DaemonContext | None:
        """Resolve configuration, start services, and build the daemon context."""
        logger.info("Daemon runtime initialization starting")

        logger.debug("[1/8] Resolving configuration...")
        config = self._resolve_configuration()
        if config is None:
            return None

        logger.debug("[2/8] Preparing runtime state...")
        try:
            recordings_root = self._prepare_runtime_state(config)
        except Exception:
            logger.exception("Failed to prepare runtime state")
            return None

        logger.debug("[3/8] Initializing authentication...")
        if not self._initialize_auth(config):
            return None

        logger.debug("[4/8] Starting EventLoopManager...")
        loop_result = self._start_event_loops()
        if loop_result is None:
            return None
        loop_manager, emitter = loop_result

        debug_mode = is_debug_mode()
        if debug_mode:
            log_path = self._db_path.parent / "daemon_events_timeline.csv"
            self._event_logger = EventLogger(log_path)
            self._event_logger.attach(emitter)
            logger.info("Debug Mode enabled")

        logger.debug("[5/8] Bootstrapping async services on General Loop...")
        services = self._bootstrap_async_services(config, loop_manager, emitter)
        if services is None:
            return None

        logger.debug("[6/8] Reconciling runtime state...")
        if not self._reconcile_runtime_state(loop_manager, services, recordings_root):
            return None

        logger.debug("[7/8] Initializing RecordingDiskManager...")
        recording_disk_manager = self._init_recording_disk_manager(
            config, loop_manager, emitter, services
        )
        if recording_disk_manager is None:
            return None

        logger.debug("[8/8] Creating communications runtime...")
        comm_manager = CommunicationsManager()
        self._daemon = DataBridge(
            recording_disk_manager=recording_disk_manager,
            emitter=emitter,
            comm_manager=comm_manager,
        )
        logger.debug("       ZMQ sockets ready")

        self._context = DaemonContext(
            config=config,
            loop_manager=loop_manager,
            comm_manager=comm_manager,
            services=services,
            recording_disk_manager=recording_disk_manager,
        )

        logger.info("Daemon runtime initialization complete")
        return self._context

    def run_forever(self) -> None:
        """Run the blocking daemon loop after successful initialization."""
        if self._context is None or self._daemon is None:
            raise RuntimeError("Cannot run daemon before initialize()")

        logger.info("Daemon starting main loop...")
        self._daemon.run()

    def shutdown(self) -> None:
        """Stop the daemon and tear down all initialized runtime resources."""
        if self._daemon is not None:
            self._daemon.stop()

        if self._context is None:
            logger.warning("Cannot shut down: daemon not started")
            self._daemon = None
            return

        logger.info("Daemon runtime shutdown starting")
        ctx = self._context

        logger.debug("[1/3] Shutting down RecordingDiskManager...")
        self._shutdown_recording_disk_manager(ctx)

        logger.debug("[2/3] Shutting down async services...")
        self._shutdown_async_services(ctx)

        logger.debug("[3/3] Stopping EventLoopManager...")
        self._stop_event_loops(ctx)

        if self._event_logger is not None:
            self._event_logger.close()
            self._event_logger = None

        self._daemon = None
        self._context = None

        cleaned_shared_slots = cleanup_stale_shared_slot_segments()
        if cleaned_shared_slots:
            logger.info(
                "Cleaned %d shared-slot segment(s) during daemon shutdown",
                cleaned_shared_slots,
            )

        logger.info("Daemon runtime shutdown complete")

    @property
    def context(self) -> DaemonContext | None:
        """Return the active daemon context when initialization has succeeded."""
        return self._context


__all__ = [
    "DaemonContext",
    "DaemonRuntime",
    "DaemonServices",
]
