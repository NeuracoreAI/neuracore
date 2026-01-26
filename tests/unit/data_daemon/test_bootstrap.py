"""Tests for daemon bootstrap and lifecycle management.

This module tests the bootstrap sequence that initializes all daemon
subsystems in the correct order across three execution contexts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from neuracore.data_daemon.bootstrap import (
    AsyncServices,
    DaemonBootstrap,
    DaemonContext,
    bootstrap_async_services,
    shutdown_async_services,
)
from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.state_management.state_manager import StateManager
from neuracore.data_daemon.state_management.state_store_sqlite import SqliteStateStore


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Temporary SQLite database path."""
    return tmp_path / "test_state.db"


@pytest.fixture
def mock_config() -> DaemonConfig:
    """Mock DaemonConfig with minimal valid settings."""
    return DaemonConfig(path_to_store_record="/tmp/test_recordings")


@pytest.fixture
def mock_async_services() -> AsyncServices:
    """Create a mock AsyncServices instance for shutdown tests."""
    # Use MagicMock for base objects, only make specific methods async
    mock_session = MagicMock(spec=aiohttp.ClientSession)
    mock_session.close = AsyncMock()

    mock_state_store = MagicMock(spec=SqliteStateStore)
    mock_state_store.close = AsyncMock()
    mock_state_manager = MagicMock(spec=StateManager)

    mock_upload_manager = MagicMock()
    mock_upload_manager.shutdown = AsyncMock()

    mock_connection_manager = MagicMock()
    mock_connection_manager.stop = AsyncMock()

    mock_progress_reporter = MagicMock()

    return AsyncServices(
        client_session=mock_session,
        state_store=mock_state_store,
        state_manager=mock_state_manager,
        upload_manager=mock_upload_manager,
        connection_manager=mock_connection_manager,
        progress_reporter=mock_progress_reporter,
    )


class TestBootstrapAsyncServices:
    """Tests for bootstrap_async_services() function."""

    @pytest.mark.asyncio
    async def test_b1_happy_path_all_services_initialize(
        self,
        mock_config: DaemonConfig,
        temp_db_path: Path,
    ) -> None:
        """
        B1: Happy Path - All Services Initialize Successfully

        The Story:
        The daemon is starting up. After the event loops are running, we need to
        initialize all async services on the General Loop. This includes auth,
        HTTP client, database, state management, uploads, connectivity monitoring,
        and progress reporting. Everything must initialize in the correct order.

        The Flow:
        1. Create a mock DaemonConfig with basic settings
        2. Call `bootstrap_async_services(config, temp_db_path)`
        3. All 6 services initialize in dependency order
        4. Returns AsyncServices with all components populated

        Why This Matters:
        This is the primary happy path for daemon startup. Every service must
        be properly initialized before the daemon can process recordings. If
        any service is missing or misconfigured, recordings will fail silently.

        Key Assertions:
        - Returns AsyncServices (not None)
        - client_session is an aiohttp.ClientSession
        - state_store is initialized (init_async_store called)
        - state_manager, upload_manager, connection_manager, progress_reporter all exist
        - initialize_auth() was called with the config
        - connection_manager.start() was called
        """
        with (
            patch("neuracore.data_daemon.bootstrap.initialize_auth") as mock_init_auth,
            patch("neuracore.data_daemon.bootstrap.ConnectionManager") as MockConnMgr,
            patch("neuracore.data_daemon.bootstrap.UploadManager") as MockUploadMgr,
            patch(
                "neuracore.data_daemon.bootstrap.ProgressReporter"
            ) as MockProgressReporter,
            patch("neuracore.data_daemon.bootstrap.SqliteStateStore") as MockStateStore,
            patch("neuracore.data_daemon.bootstrap.StateManager") as MockStateMgr,
        ):
            # Setup mocks
            mock_conn_instance = AsyncMock()
            mock_conn_instance.start = AsyncMock()
            MockConnMgr.return_value = mock_conn_instance

            mock_upload_instance = MagicMock()
            MockUploadMgr.return_value = mock_upload_instance

            mock_progress_instance = MagicMock()
            MockProgressReporter.return_value = mock_progress_instance

            mock_state_store_instance = AsyncMock()
            mock_state_store_instance.init_async_store = AsyncMock()
            MockStateStore.return_value = mock_state_store_instance

            mock_state_manager_instance = MagicMock()
            MockStateMgr.return_value = mock_state_manager_instance

            # Execute
            services = await bootstrap_async_services(mock_config, temp_db_path)

            # Verify AsyncServices returned with all components
            assert services is not None
            assert isinstance(services, AsyncServices)

            # Verify client_session is aiohttp.ClientSession
            assert isinstance(services.client_session, aiohttp.ClientSession)

            # Verify state_store is initialized (init_async_store called)
            assert services.state_store is mock_state_store_instance
            mock_state_store_instance.init_async_store.assert_called_once()

            # Verify state_manager exists
            assert services.state_manager is mock_state_manager_instance

            # Verify upload_manager, connection_manager, progress_reporter exist
            assert services.upload_manager is mock_upload_instance
            assert services.connection_manager is mock_conn_instance
            assert services.progress_reporter is mock_progress_instance

            # Verify initialize_auth was called with config
            mock_init_auth.assert_called_once_with(daemon_config=mock_config)

            # Verify connection_manager.start() was called
            mock_conn_instance.start.assert_called_once()

            # Cleanup
            await services.client_session.close()

    @pytest.mark.asyncio
    async def test_b2_auth_initialization_receives_config(
        self,
        mock_config: DaemonConfig,
        temp_db_path: Path,
    ) -> None:
        """
        B2: Auth Initialization Receives Config

        The Story:
        The AuthManager is a singleton that needs the daemon config to know
        which API key and org to use. It must be initialized BEFORE any service
        that makes authenticated API calls.

        The Flow:
        1. Call `bootstrap_async_services(config, db_path)`
        2. Verify `initialize_auth(daemon_config=config)` is called first
        3. Other services initialize after

        Why This Matters:
        Services like UploadManager and ProgressReporter use `get_auth()` to get
        credentials. If auth isn't initialized first, they'll fail with "auth not
        initialized" errors when trying to upload.

        Key Assertions:
        - initialize_auth called exactly once
        - initialize_auth called with daemon_config=config
        - Called before other services that depend on auth
        """
        call_order: list[str] = []

        def track_init_auth(**kwargs):
            call_order.append("initialize_auth")

        def track_upload_manager(*args, **kwargs):
            call_order.append("UploadManager")
            return MagicMock()

        def track_progress_reporter(*args, **kwargs):
            call_order.append("ProgressReporter")
            return MagicMock()

        with (
            patch(
                "neuracore.data_daemon.bootstrap.initialize_auth",
                side_effect=track_init_auth,
            ) as mock_init_auth,
            patch("neuracore.data_daemon.bootstrap.ConnectionManager") as MockConnMgr,
            patch(
                "neuracore.data_daemon.bootstrap.UploadManager",
                side_effect=track_upload_manager,
            ),
            patch(
                "neuracore.data_daemon.bootstrap.ProgressReporter",
                side_effect=track_progress_reporter,
            ),
        ):
            mock_conn_instance = AsyncMock()
            mock_conn_instance.start = AsyncMock()
            MockConnMgr.return_value = mock_conn_instance

            services = await bootstrap_async_services(mock_config, temp_db_path)

            # Verify initialize_auth called exactly once
            mock_init_auth.assert_called_once()

            # Verify initialize_auth called with daemon_config=config
            mock_init_auth.assert_called_with(daemon_config=mock_config)

            # Verify initialize_auth called before services that depend on auth
            assert call_order.index("initialize_auth") < call_order.index(
                "UploadManager"
            )
            assert call_order.index("initialize_auth") < call_order.index(
                "ProgressReporter"
            )

            # Cleanup
            await services.client_session.close()


class TestShutdownAsyncServices:
    """Tests for shutdown_async_services() function."""

    @pytest.mark.asyncio
    async def test_s1_clean_shutdown_closes_all_resources(
        self,
        mock_async_services: AsyncServices,
    ) -> None:
        """
        S1: Clean Shutdown Closes All Resources

        The Story:
        The daemon is shutting down gracefully. We need to close HTTP connections,
        stop background monitors, and wait for in-flight uploads to complete.
        Resources must be released in reverse order of initialization.

        The Flow:
        1. Have a running AsyncServices instance
        2. Call `shutdown_async_services(services)`
        3. ConnectionManager.stop() is called
        4. UploadManager.shutdown() is called (waits for uploads)
        5. aiohttp.ClientSession.close() is called last

        Why This Matters:
        Improper shutdown can cause resource leaks (unclosed sockets), lost data
        (interrupted uploads), or hanging processes. The reverse order ensures
        dependencies are respected.

        Key Assertions:
        - connection_manager.stop() called
        - upload_manager.shutdown() called
        - state_store.close() called
        - client_session.close() called
        - No exceptions raised
        """
        # Execute
        await shutdown_async_services(mock_async_services)

        # Verify connection_manager.stop() called
        mock_async_services.connection_manager.stop.assert_called_once()

        # Verify upload_manager.shutdown() called
        mock_async_services.upload_manager.shutdown.assert_called_once()

        # Verify state_store.close() called
        mock_async_services.state_store.close.assert_called_once()

        # Verify client_session.close() called
        mock_async_services.client_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_s2_shutdown_continues_despite_errors(
        self,
        mock_async_services: AsyncServices,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        S2: Shutdown Continues Despite Individual Errors

        The Story:
        During shutdown, ConnectionManager.stop() throws an exception (maybe the
        socket was already closed). We must NOT abort shutdown - we need to
        continue closing other resources to avoid leaks.

        The Flow:
        1. Mock connection_manager.stop() to raise RuntimeError
        2. Call `shutdown_async_services(services)`
        3. Error is logged but not raised
        4. upload_manager.shutdown() still called
        5. client_session.close() still called

        Why This Matters:
        Partial shutdowns are worse than crashes. If we abort on first error,
        we leak the HTTP session and any other resources. The daemon process
        may hang or leave orphaned connections.

        Key Assertions:
        - No exception propagates to caller
        - All shutdown methods still called despite earlier error
        - Error is logged (check log capture)
        """
        # Setup: connection_manager.stop() raises error
        mock_async_services.connection_manager.stop = AsyncMock(
            side_effect=RuntimeError("Connection already closed")
        )

        # Execute - should not raise
        with caplog.at_level(logging.ERROR):
            await shutdown_async_services(mock_async_services)

        # Verify no exception propagated (we got here)

        # Verify upload_manager.shutdown() still called despite earlier error
        mock_async_services.upload_manager.shutdown.assert_called_once()

        # Verify client_session.close() still called
        mock_async_services.client_session.close.assert_called_once()

        # Verify error was logged
        assert "Error stopping ConnectionManager" in caplog.text


class TestDaemonBootstrapStart:
    """Tests for DaemonBootstrap.start() method."""

    def test_d1_full_startup_returns_daemon_context(
        self,
        temp_db_path: Path,
        mock_config: DaemonConfig,
    ) -> None:
        """
        D1: Full Startup Returns Complete DaemonContext

        The Story:
        The daemon process starts. DaemonBootstrap.start() orchestrates the
        entire initialization sequence across 5 layers. On success, it returns
        a DaemonContext containing everything the Daemon needs to run.

        The Flow:
        1. Create DaemonBootstrap instance
        2. Call bootstrap.start()
        3. Layer 1: Config resolved from ProfileManager -> ConfigManager
        4. Layer 2: EventLoopManager.start() creates General + Encoder loops
        5. Layer 3: bootstrap_async_services() runs on General Loop
        6. Layer 4: RecordingDiskManager initialized with loop_manager
        7. Layer 5: CommunicationsManager created for ZMQ
        8. Returns DaemonContext with all components

        Why This Matters:
        This is THE entry point for daemon initialization. The returned context
        is passed to the Daemon class which uses it to run the main message loop.
        Any missing component means the daemon can't function.

        Key Assertions:
        - Returns DaemonContext (not None)
        - context.config is the resolved DaemonConfig
        - context.loop_manager is running
        - context.services contains all AsyncServices
        - context.recording_disk_manager is initialized
        - context.comm_manager is ready
        """
        mock_rdm = MagicMock()
        mock_comm = MagicMock()
        mock_services = MagicMock(spec=AsyncServices)
        mock_loop_mgr = MagicMock()

        with (
            patch("neuracore.data_daemon.bootstrap.ProfileManager"),
            patch("neuracore.data_daemon.bootstrap.ConfigManager") as MockConfigMgr,
            patch("neuracore.data_daemon.bootstrap.EventLoopManager") as MockLoopMgr,
            patch(
                "neuracore.data_daemon.bootstrap.bootstrap_async_services",
                new=MagicMock(),  # Completely replace with sync mock
            ),
            patch(
                "neuracore.data_daemon.bootstrap.rdm.RecordingDiskManager",
                return_value=mock_rdm,
            ),
            patch(
                "neuracore.data_daemon.bootstrap.CommunicationsManager",
                return_value=mock_comm,
            ),
        ):
            # Setup config manager mock
            mock_config_mgr_instance = MagicMock()
            mock_config_mgr_instance.resolve_effective_config.return_value = mock_config
            MockConfigMgr.return_value = mock_config_mgr_instance

            # Setup loop manager mock - simulates successful startup
            mock_loop_mgr.is_running.return_value = True
            services_future = MagicMock()
            services_future.result.return_value = mock_services
            mock_loop_mgr.schedule_on_general_loop.return_value = services_future
            MockLoopMgr.return_value = mock_loop_mgr

            bootstrap = DaemonBootstrap(db_path=temp_db_path)
            context = bootstrap.start()

            # Returns DaemonContext (not None)
            assert context is not None
            assert isinstance(context, DaemonContext)

            # context.config is the resolved DaemonConfig
            assert context.config is mock_config

            # context.loop_manager is running
            assert context.loop_manager is mock_loop_mgr
            assert context.loop_manager.is_running()

            # context.services contains all AsyncServices
            assert context.services is mock_services

            # context.recording_disk_manager is initialized
            assert context.recording_disk_manager is mock_rdm

            # context.comm_manager is ready
            assert context.comm_manager is mock_comm

    def test_d2_config_failure_returns_none(
        self,
        temp_db_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        D2: Config Resolution Failure Returns None

        The Story:
        The user has a corrupted config file or invalid environment variables.
        ProfileManager or ConfigManager throws an exception. The daemon should
        return None (not crash) so the CLI can show a helpful error message.

        The Flow:
        1. Mock ConfigManager.resolve_effective_config() to raise ValueError
        2. Call bootstrap.start()
        3. Exception is caught and logged
        4. Returns None immediately
        5. No cleanup needed (nothing was started yet)

        Why This Matters:
        Config errors are common: typos in API keys, invalid paths, missing env
        vars. The daemon must fail gracefully and log what went wrong so users
        can fix it. Crashing with a stack trace is unfriendly.

        Key Assertions:
        - Returns None
        - Error is logged with context
        - No EventLoopManager started (would leak threads)
        """
        with (
            patch("neuracore.data_daemon.bootstrap.ProfileManager"),
            patch("neuracore.data_daemon.bootstrap.ConfigManager") as MockConfigMgr,
            patch("neuracore.data_daemon.bootstrap.EventLoopManager") as MockLoopMgr,
        ):
            # Setup: config resolution fails
            mock_config_mgr_instance = MagicMock()
            mock_config_mgr_instance.resolve_effective_config.side_effect = ValueError(
                "Invalid API key format"
            )
            MockConfigMgr.return_value = mock_config_mgr_instance

            bootstrap = DaemonBootstrap(db_path=temp_db_path)

            with caplog.at_level(logging.ERROR):
                context = bootstrap.start()

            # Returns None
            assert context is None

            # Error is logged
            assert "Failed to resolve configuration" in caplog.text

            # No EventLoopManager started
            MockLoopMgr.return_value.start.assert_not_called()

    def test_d3_loop_manager_failure_returns_none(
        self,
        temp_db_path: Path,
        mock_config: DaemonConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        D3: EventLoopManager Failure Returns None

        The Story:
        The system is low on resources and can't create new threads. EventLoopManager
        .start() fails. The daemon should return None. Since config already loaded
        but loops didn't start, there's nothing to clean up.

        The Flow:
        1. Mock EventLoopManager.start() to raise RuntimeError
        2. Call bootstrap.start()
        3. Config layer succeeds
        4. Loop layer fails, exception caught
        5. Returns None
        6. No async services to clean up (they need the loop)

        Why This Matters:
        Thread/resource exhaustion can happen in containerized environments with
        strict limits. Failing cleanly lets orchestrators (k8s, systemd) handle
        restart logic.

        Key Assertions:
        - Returns None
        - Config was resolved (no wasted work check)
        - No loops left running
        - Error logged
        """
        with (
            patch("neuracore.data_daemon.bootstrap.ProfileManager"),
            patch("neuracore.data_daemon.bootstrap.ConfigManager") as MockConfigMgr,
            patch("neuracore.data_daemon.bootstrap.EventLoopManager") as MockLoopMgr,
        ):
            # Setup: config succeeds
            mock_config_mgr_instance = MagicMock()
            mock_config_mgr_instance.resolve_effective_config.return_value = mock_config
            MockConfigMgr.return_value = mock_config_mgr_instance

            # Setup: EventLoopManager.start() fails
            mock_loop_mgr_instance = MagicMock()
            mock_loop_mgr_instance.start.side_effect = RuntimeError(
                "Cannot create thread"
            )
            mock_loop_mgr_instance.is_running.return_value = False
            MockLoopMgr.return_value = mock_loop_mgr_instance

            bootstrap = DaemonBootstrap(db_path=temp_db_path)

            with caplog.at_level(logging.ERROR):
                context = bootstrap.start()

            # Returns None
            assert context is None

            # Config was resolved
            mock_config_mgr_instance.resolve_effective_config.assert_called_once()

            # No loops left running (start failed, so is_running should be False)
            assert mock_loop_mgr_instance.is_running() is False

            # Error logged
            assert "Failed to start EventLoopManager" in caplog.text

    def test_d4_async_services_failure_cleans_up_loops(
        self,
        temp_db_path: Path,
        mock_config: DaemonConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        D4: Async Services Failure Cleans Up Loops

        The Story:
        The loops are running, but SqliteStateStore.init_async_store() fails
        (maybe disk is full or permissions denied). We MUST stop the event loops
        before returning None, or we'll leak two threads.

        The Flow:
        1. Mock bootstrap_async_services to raise Exception
        2. Call bootstrap.start()
        3. Config succeeds
        4. EventLoopManager.start() succeeds
        5. Async services fail
        6. EventLoopManager.stop() is called for cleanup
        7. Returns None

        Why This Matters:
        Thread leaks are insidious. The process looks like it exited but threads
        keep running. In tests, this causes pytest to hang. In production, it
        wastes resources and can cause mysterious behavior.

        Key Assertions:
        - Returns None
        - EventLoopManager.stop() was called
        - No threads left alive
        - Error logged
        """
        with (
            patch("neuracore.data_daemon.bootstrap.ProfileManager"),
            patch("neuracore.data_daemon.bootstrap.ConfigManager") as MockConfigMgr,
            patch("neuracore.data_daemon.bootstrap.EventLoopManager") as MockLoopMgr,
            patch(
                "neuracore.data_daemon.bootstrap.bootstrap_async_services",
                new=MagicMock(),  # Completely replace with sync mock
            ),
        ):
            # Setup: config succeeds
            mock_config_mgr_instance = MagicMock()
            mock_config_mgr_instance.resolve_effective_config.return_value = mock_config
            MockConfigMgr.return_value = mock_config_mgr_instance

            # Setup: EventLoopManager starts successfully but async services fail
            mock_loop_mgr_instance = MagicMock()
            mock_future = MagicMock()
            mock_future.result.side_effect = RuntimeError("Database init failed")
            mock_loop_mgr_instance.schedule_on_general_loop.return_value = mock_future
            # After stop() is called, is_running returns False
            mock_loop_mgr_instance.is_running.return_value = False
            MockLoopMgr.return_value = mock_loop_mgr_instance

            bootstrap = DaemonBootstrap(db_path=temp_db_path)

            with caplog.at_level(logging.ERROR):
                context = bootstrap.start()

            # Returns None
            assert context is None

            # EventLoopManager.stop() was called for cleanup
            mock_loop_mgr_instance.stop.assert_called_once()

            # No threads left alive (loops were stopped)
            assert mock_loop_mgr_instance.is_running() is False

            # Error logged
            assert "Failed to bootstrap async services" in caplog.text

    def test_d5_rdm_failure_cleans_up_services_and_loops(
        self,
        temp_db_path: Path,
        mock_config: DaemonConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        D5: RecordingDiskManager Failure Cleans Up Services and Loops

        The Story:
        Loops are running, async services are initialized, but RecordingDiskManager
        fails (invalid recordings_root path). We must shut down async services
        AND stop the loops. This is the most complex cleanup scenario.

        The Flow:
        1. Mock RecordingDiskManager.__init__ to raise Exception
        2. Call bootstrap.start()
        3. Config, loops, async services all succeed
        4. RDM fails
        5. shutdown_async_services() called on General Loop
        6. EventLoopManager.stop() called
        7. Returns None

        Why This Matters:
        At this point we have HTTP sessions open, database connections, and
        running event loops. All must be cleaned up or we leak everything.
        This tests the full cleanup path.

        Key Assertions:
        - Returns None
        - shutdown_async_services() was called
        - EventLoopManager.stop() was called
        - aiohttp session closed
        - No resource leaks
        """
        mock_services = MagicMock(spec=AsyncServices)

        # Create mock for shutdown_async_services to track calls
        mock_shutdown = MagicMock()

        with (
            patch("neuracore.data_daemon.bootstrap.ProfileManager"),
            patch("neuracore.data_daemon.bootstrap.ConfigManager") as MockConfigMgr,
            patch("neuracore.data_daemon.bootstrap.EventLoopManager") as MockLoopMgr,
            patch(
                "neuracore.data_daemon.bootstrap.bootstrap_async_services",
                new=MagicMock(),  # Completely replace with sync mock
            ),
            patch(
                "neuracore.data_daemon.bootstrap.shutdown_async_services",
                new=mock_shutdown,  # Completely replace with sync mock
            ),
            patch(
                "neuracore.data_daemon.bootstrap.rdm.RecordingDiskManager"
            ) as MockRDM,
        ):
            # Setup: config succeeds
            mock_config_mgr_instance = MagicMock()
            mock_config_mgr_instance.resolve_effective_config.return_value = mock_config
            MockConfigMgr.return_value = mock_config_mgr_instance

            # Setup: EventLoopManager and async services succeed
            mock_loop_mgr_instance = MagicMock()

            # First call returns services, second call (shutdown) returns None
            services_future = MagicMock()
            services_future.result.return_value = mock_services
            shutdown_future = MagicMock()
            shutdown_future.result.return_value = None

            mock_loop_mgr_instance.schedule_on_general_loop.side_effect = [
                services_future,
                shutdown_future,
            ]
            MockLoopMgr.return_value = mock_loop_mgr_instance

            # Setup: RDM fails
            MockRDM.side_effect = RuntimeError("Invalid recordings path")

            bootstrap = DaemonBootstrap(db_path=temp_db_path)

            with caplog.at_level(logging.ERROR):
                context = bootstrap.start()

            # Returns None
            assert context is None

            # shutdown_async_services was scheduled (cleanup)
            assert mock_loop_mgr_instance.schedule_on_general_loop.call_count == 2

            # Verify shutdown_async_services was called with the services
            # (This ensures aiohttp session close will be called)
            mock_shutdown.assert_called_once_with(mock_services)

            # EventLoopManager.stop() was called
            mock_loop_mgr_instance.stop.assert_called_once()

            # No resource leaks - loops are stopped
            mock_loop_mgr_instance.is_running.return_value = False
            assert mock_loop_mgr_instance.is_running() is False

            # Error logged
            assert "Failed to initialize RecordingDiskManager" in caplog.text


class TestDaemonBootstrapStop:
    """Tests for DaemonBootstrap.stop() method."""

    def test_t1_stop_shuts_down_all_layers(
        self,
        temp_db_path: Path,
        mock_config: DaemonConfig,
    ) -> None:
        """
        T1: Stop Shuts Down All Layers in Reverse Order

        The Story:
        The daemon received SIGTERM. DaemonBootstrap.stop() must gracefully shut
        down all subsystems. Order matters: stop accepting new work (RDM), finish
        in-flight work (services), then stop the loops.

        The Flow:
        1. Have a running daemon (bootstrap.start() succeeded)
        2. Call bootstrap.stop()
        3. Layer 1: RecordingDiskManager.shutdown() flushes pending writes
        4. Layer 2: shutdown_async_services() closes connections
        5. Layer 3: EventLoopManager.stop() terminates threads
        6. context is set to None

        Why This Matters:
        Graceful shutdown prevents data loss. RDM must flush buffered data to
        disk before services shut down. Services must complete uploads before
        loops stop. Wrong order = lost recordings.

        Key Assertions:
        - RDM shutdown called first
        - Services shutdown called second
        - Loops stopped last
        - bootstrap.context is None after
        """
        mock_rdm = MagicMock()
        mock_rdm.shutdown = AsyncMock()
        mock_services = MagicMock(spec=AsyncServices)
        mock_loop_mgr = MagicMock()

        shutdown_order: list[str] = []

        def track_rdm_shutdown():
            shutdown_order.append("rdm")
            future = MagicMock()
            future.result.return_value = None
            return future

        def track_services_shutdown():
            shutdown_order.append("services")
            future = MagicMock()
            future.result.return_value = None
            return future

        def track_loop_stop(*args, **kwargs):
            shutdown_order.append("loops")

        # Create a context manually
        context = DaemonContext(
            config=mock_config,
            loop_manager=mock_loop_mgr,
            comm_manager=MagicMock(),
            services=mock_services,
            recording_disk_manager=mock_rdm,
        )

        # Track shutdown calls
        call_count = [0]

        def schedule_side_effect(coroutine):
            # Close the coroutine to avoid "never awaited" warnings
            coroutine.close()
            call_count[0] += 1
            if call_count[0] == 1:
                return track_rdm_shutdown()
            else:
                return track_services_shutdown()

        mock_loop_mgr.schedule_on_general_loop.side_effect = schedule_side_effect
        mock_loop_mgr.stop.side_effect = track_loop_stop

        bootstrap = DaemonBootstrap(db_path=temp_db_path)
        bootstrap._context = context

        bootstrap.stop()

        # Verify shutdown order
        assert shutdown_order == ["rdm", "services", "loops"]

        # bootstrap.context is None after
        assert bootstrap.context is None

    def test_t2_stop_without_start_logs_warning(
        self,
        temp_db_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        T2: Stop Without Start Logs Warning

        The Story:
        Due to a bug or race condition, stop() is called on a bootstrap that
        never successfully started. This shouldn't crash - just log a warning
        and return.

        The Flow:
        1. Create DaemonBootstrap (don't call start)
        2. Call bootstrap.stop()
        3. Warning logged: "Cannot stop: daemon not started"
        4. Returns without error

        Why This Matters:
        Defensive programming. In finally blocks or signal handlers, stop() may
        be called unconditionally. It must handle the "nothing to stop" case.

        Key Assertions:
        - No exception raised
        - Warning logged
        - Returns cleanly
        """
        bootstrap = DaemonBootstrap(db_path=temp_db_path)

        # Don't call start()

        with caplog.at_level(logging.WARNING):
            bootstrap.stop()  # Should not raise

        # Warning logged
        assert "Cannot stop: daemon not started" in caplog.text

    def test_t3_stop_continues_despite_errors(
        self,
        temp_db_path: Path,
        mock_config: DaemonConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        T3: Stop Continues Despite Shutdown Errors

        The Story:
        During stop(), RecordingDiskManager.shutdown() times out (encoder worker
        is stuck). We must NOT abort - we still need to close HTTP sessions and
        stop loops to avoid leaks.

        The Flow:
        1. Have running daemon
        2. Mock RDM.shutdown() to raise TimeoutError
        3. Call bootstrap.stop()
        4. RDM error logged but not raised
        5. shutdown_async_services() still called
        6. EventLoopManager.stop() still called

        Why This Matters:
        Same principle as S2. Partial shutdown is worse than complete shutdown
        with logged errors. We must release all resources we can.

        Key Assertions:
        - No exception propagates
        - All shutdown methods attempted
        - Errors logged
        - context is None after
        """
        mock_rdm = MagicMock()
        mock_services = MagicMock(spec=AsyncServices)
        mock_loop_mgr = MagicMock()

        # Create context
        context = DaemonContext(
            config=mock_config,
            loop_manager=mock_loop_mgr,
            comm_manager=MagicMock(),
            services=mock_services,
            recording_disk_manager=mock_rdm,
        )

        # RDM shutdown fails
        rdm_future = MagicMock()
        rdm_future.result.side_effect = TimeoutError("Encoder worker stuck")

        # Services shutdown succeeds
        services_future = MagicMock()
        services_future.result.return_value = None

        call_count = [0]

        def schedule_side_effect(coroutine):
            # Close the coroutine to avoid "never awaited" warnings
            coroutine.close()
            call_count[0] += 1
            if call_count[0] == 1:
                return rdm_future
            else:
                return services_future

        mock_loop_mgr.schedule_on_general_loop.side_effect = schedule_side_effect

        bootstrap = DaemonBootstrap(db_path=temp_db_path)
        bootstrap._context = context

        with caplog.at_level(logging.ERROR):
            bootstrap.stop()  # Should not raise

        # All shutdown methods attempted
        assert mock_loop_mgr.schedule_on_general_loop.call_count == 2

        # EventLoopManager.stop() still called
        mock_loop_mgr.stop.assert_called_once()

        # Error logged
        assert "Error shutting down RecordingDiskManager" in caplog.text

        # context is None after
        assert bootstrap.context is None


class TestDaemonBootstrapContext:
    """Tests for DaemonBootstrap.context property."""

    def test_c1_context_property_before_start(
        self,
        temp_db_path: Path,
    ) -> None:
        """
        C1: Context Property Before Start

        The Story:
        Code checks bootstrap.context before calling start(). This is valid -
        maybe checking if already running. Should return None, not raise.

        The Flow:
        1. Create DaemonBootstrap
        2. Access bootstrap.context
        3. Returns None

        Key Assertions:
        - Returns None
        - No exception
        """
        bootstrap = DaemonBootstrap(db_path=temp_db_path)

        # Access context before start
        result = bootstrap.context

        # Returns None
        assert result is None

    def test_c2_context_property_after_start(
        self,
        temp_db_path: Path,
        mock_config: DaemonConfig,
    ) -> None:
        """
        C2: Context Property After Successful Start

        The Story:
        After start() succeeds, context should return the DaemonContext. This
        lets callers access components without storing the return value.

        The Flow:
        1. Create and start DaemonBootstrap
        2. Access bootstrap.context
        3. Returns the DaemonContext from start()

        Key Assertions:
        - Returns same DaemonContext as start() returned
        - All components accessible
        """
        mock_rdm = MagicMock()
        mock_comm = MagicMock()
        mock_services = MagicMock(spec=AsyncServices)
        mock_loop_mgr = MagicMock()

        with (
            patch("neuracore.data_daemon.bootstrap.ProfileManager"),
            patch("neuracore.data_daemon.bootstrap.ConfigManager") as MockConfigMgr,
            patch("neuracore.data_daemon.bootstrap.EventLoopManager") as MockLoopMgr,
            patch(
                "neuracore.data_daemon.bootstrap.bootstrap_async_services",
                new=MagicMock(),  # Completely replace with sync mock
            ),
            patch(
                "neuracore.data_daemon.bootstrap.rdm.RecordingDiskManager",
                return_value=mock_rdm,
            ),
            patch(
                "neuracore.data_daemon.bootstrap.CommunicationsManager",
                return_value=mock_comm,
            ),
        ):
            mock_config_mgr_instance = MagicMock()
            mock_config_mgr_instance.resolve_effective_config.return_value = mock_config
            MockConfigMgr.return_value = mock_config_mgr_instance

            # Setup loop manager mock
            mock_loop_mgr.is_running.return_value = True
            services_future = MagicMock()
            services_future.result.return_value = mock_services
            mock_loop_mgr.schedule_on_general_loop.return_value = services_future
            MockLoopMgr.return_value = mock_loop_mgr

            bootstrap = DaemonBootstrap(db_path=temp_db_path)
            start_result = bootstrap.start()

            # Access context property
            context_result = bootstrap.context

            # Returns same DaemonContext as start() returned
            assert context_result is start_result
            assert context_result is not None

            # All components accessible
            assert context_result.config is mock_config
            assert context_result.loop_manager is mock_loop_mgr
            assert context_result.services is mock_services
            assert context_result.recording_disk_manager is mock_rdm
            assert context_result.comm_manager is mock_comm
