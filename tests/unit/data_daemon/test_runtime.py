"""Tests for daemon runtime orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.runtime import DaemonContext, DaemonRuntime
from neuracore.data_daemon.services import DaemonServices


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Temporary SQLite database path."""
    return tmp_path / "test_state.db"


@pytest.fixture
def mock_config() -> DaemonConfig:
    """Mock DaemonConfig with minimal valid settings."""
    return DaemonConfig(path_to_store_record="/tmp/test_recordings")


class TestDaemonRuntimeInitialize:
    """Tests for DaemonRuntime.initialize() method."""

    def test_d1_full_startup_returns_daemon_context(
        self,
        temp_db_path: Path,
        mock_config: DaemonConfig,
    ) -> None:
        """
        D1: Full Startup Returns Complete DaemonContext

        The Story:
        The daemon process starts. DaemonRuntime.initialize() orchestrates the
        entire initialization sequence across 5 layers. On success, it returns
        a DaemonContext containing everything the Daemon needs to run.

        The Flow:
        1. Create DaemonRuntime instance
        2. Call runtime.initialize()
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
        - context.services contains all DaemonServices dependencies
        - context.recording_disk_manager is initialized
        - context.comm_manager is ready
        """
        mock_rdm = MagicMock()
        mock_comm = MagicMock()
        mock_services = MagicMock(spec=DaemonServices)
        mock_services.state_store = MagicMock()
        mock_loop_mgr = MagicMock()

        with (
            patch("neuracore.data_daemon.runtime.ProfileManager"),
            patch("neuracore.data_daemon.runtime.ConfigManager") as MockConfigMgr,
            patch("neuracore.data_daemon.runtime.login"),
            patch("neuracore.data_daemon.runtime.EventLoopManager") as MockLoopMgr,
            patch(
                "neuracore.data_daemon.runtime.rdm.RecordingDiskManager",
                return_value=mock_rdm,
            ),
            patch(
                "neuracore.data_daemon.runtime.CommunicationsManager",
                return_value=mock_comm,
            ),
        ):
            mock_config_mgr_instance = MagicMock()
            mock_config_mgr_instance.resolve_effective_config.return_value = mock_config
            MockConfigMgr.return_value = mock_config_mgr_instance

            mock_loop_mgr.is_running.return_value = True
            services_future = MagicMock()
            services_future.result.return_value = mock_services
            reconcile_future = MagicMock()
            reconcile_future.result.return_value = None
            scheduled_calls = [0]

            def schedule_side_effect(coroutine):
                coroutine.close()
                scheduled_calls[0] += 1
                if scheduled_calls[0] == 1:
                    return services_future
                return reconcile_future

            mock_loop_mgr.schedule_on_general_loop.side_effect = schedule_side_effect
            MockLoopMgr.return_value = mock_loop_mgr

            runtime = DaemonRuntime(
                db_path=temp_db_path,
                pid_path=temp_db_path.with_suffix(".pid"),
                socket_paths=(),
            )
            context = runtime.initialize()

            assert context is not None
            assert isinstance(context, DaemonContext)

            assert context.config is mock_config

            assert context.loop_manager is mock_loop_mgr
            assert context.loop_manager.is_running()

            assert context.services is mock_services

            assert context.recording_disk_manager is mock_rdm

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
        2. Call runtime.initialize()
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
            patch("neuracore.data_daemon.runtime.ProfileManager"),
            patch("neuracore.data_daemon.runtime.ConfigManager") as MockConfigMgr,
            patch("neuracore.data_daemon.runtime.EventLoopManager") as MockLoopMgr,
        ):
            mock_config_mgr_instance = MagicMock()
            mock_config_mgr_instance.resolve_effective_config.side_effect = ValueError(
                "Invalid API key format"
            )
            MockConfigMgr.return_value = mock_config_mgr_instance

            runtime = DaemonRuntime(
                db_path=temp_db_path,
                pid_path=temp_db_path.with_suffix(".pid"),
                socket_paths=(),
            )

            with caplog.at_level(logging.ERROR):
                context = runtime.initialize()

            assert context is None

            assert "Failed to resolve configuration" in caplog.text

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
        2. Call runtime.initialize()
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
            patch("neuracore.data_daemon.runtime.ProfileManager"),
            patch("neuracore.data_daemon.runtime.ConfigManager") as MockConfigMgr,
            patch("neuracore.data_daemon.runtime.login"),
            patch("neuracore.data_daemon.runtime.EventLoopManager") as MockLoopMgr,
        ):
            mock_config_mgr_instance = MagicMock()
            mock_config_mgr_instance.resolve_effective_config.return_value = mock_config
            MockConfigMgr.return_value = mock_config_mgr_instance

            mock_loop_mgr_instance = MagicMock()
            mock_loop_mgr_instance.start.side_effect = RuntimeError(
                "Cannot create thread"
            )
            mock_loop_mgr_instance.is_running.return_value = False
            MockLoopMgr.return_value = mock_loop_mgr_instance

            runtime = DaemonRuntime(
                db_path=temp_db_path,
                pid_path=temp_db_path.with_suffix(".pid"),
                socket_paths=(),
            )

            with caplog.at_level(logging.ERROR):
                context = runtime.initialize()

            assert context is None

            mock_config_mgr_instance.resolve_effective_config.assert_called_once()

            assert mock_loop_mgr_instance.is_running() is False

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
        1. Mock general-loop bootstrap work to raise Exception
        2. Call runtime.initialize()
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
            patch("neuracore.data_daemon.runtime.ProfileManager"),
            patch("neuracore.data_daemon.runtime.ConfigManager") as MockConfigMgr,
            patch("neuracore.data_daemon.runtime.login"),
            patch("neuracore.data_daemon.runtime.EventLoopManager") as MockLoopMgr,
        ):
            mock_config_mgr_instance = MagicMock()
            mock_config_mgr_instance.resolve_effective_config.return_value = mock_config
            MockConfigMgr.return_value = mock_config_mgr_instance

            mock_loop_mgr_instance = MagicMock()
            mock_future = MagicMock()
            mock_future.result.side_effect = RuntimeError("Database init failed")

            def schedule_side_effect(coroutine):
                coroutine.close()
                return mock_future

            mock_loop_mgr_instance.schedule_on_general_loop.side_effect = (
                schedule_side_effect
            )
            mock_loop_mgr_instance.is_running.return_value = False
            MockLoopMgr.return_value = mock_loop_mgr_instance

            runtime = DaemonRuntime(
                db_path=temp_db_path,
                pid_path=temp_db_path.with_suffix(".pid"),
                socket_paths=(),
            )

            with caplog.at_level(logging.ERROR):
                context = runtime.initialize()

            assert context is None

            mock_loop_mgr_instance.stop.assert_called_once()

            assert mock_loop_mgr_instance.is_running() is False

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
        2. Call runtime.initialize()
        3. Config, loops, async services all succeed
        4. RDM fails
        5. services.shutdown() scheduled on General Loop
        6. EventLoopManager.stop() called
        7. Returns None

        Why This Matters:
        At this point we have HTTP sessions open, database connections, and
        running event loops. All must be cleaned up or we leak everything.
        This tests the full cleanup path.

        Key Assertions:
        - Returns None
        - services.shutdown() was called
        - EventLoopManager.stop() was called
        - aiohttp session closed
        - No resource leaks
        """
        mock_services = SimpleNamespace(
            state_store=MagicMock(),
            shutdown=AsyncMock(),
        )

        with (
            patch("neuracore.data_daemon.runtime.ProfileManager"),
            patch("neuracore.data_daemon.runtime.ConfigManager") as MockConfigMgr,
            patch("neuracore.data_daemon.runtime.login"),
            patch("neuracore.data_daemon.runtime.EventLoopManager") as MockLoopMgr,
            patch(
                "neuracore.data_daemon.runtime.DaemonServices.create",
                new_callable=AsyncMock,
            ) as mock_create_services,
            patch("neuracore.data_daemon.runtime.rdm.RecordingDiskManager") as MockRDM,
        ):
            mock_config_mgr_instance = MagicMock()
            mock_config_mgr_instance.resolve_effective_config.return_value = mock_config
            MockConfigMgr.return_value = mock_config_mgr_instance

            mock_loop_mgr_instance = MagicMock()
            mock_create_services.return_value = mock_services

            services_future = MagicMock()
            services_future.result.return_value = mock_services
            reconcile_future = MagicMock()
            reconcile_future.result.return_value = None
            shutdown_future = MagicMock()
            shutdown_future.result.return_value = None

            scheduled_futures = [
                services_future,
                reconcile_future,
                shutdown_future,
            ]

            def schedule_side_effect(coroutine):
                coroutine.close()
                return scheduled_futures.pop(0)

            mock_loop_mgr_instance.schedule_on_general_loop.side_effect = (
                schedule_side_effect
            )
            MockLoopMgr.return_value = mock_loop_mgr_instance

            MockRDM.side_effect = RuntimeError("Invalid recordings path")

            runtime = DaemonRuntime(
                db_path=temp_db_path,
                pid_path=temp_db_path.with_suffix(".pid"),
                socket_paths=(),
            )

            with caplog.at_level(logging.ERROR):
                context = runtime.initialize()

            assert context is None

            assert mock_loop_mgr_instance.schedule_on_general_loop.call_count == 3

            mock_create_services.assert_called_once()

            mock_services.shutdown.assert_called_once_with()

            mock_loop_mgr_instance.stop.assert_called_once()

            mock_loop_mgr_instance.is_running.return_value = False
            assert mock_loop_mgr_instance.is_running() is False

            assert "Failed to initialize RecordingDiskManager" in caplog.text


class TestDaemonRuntimeShutdown:
    """Tests for DaemonRuntime.shutdown() method."""

    def test_t1_stop_shuts_down_all_layers(
        self,
        temp_db_path: Path,
        mock_config: DaemonConfig,
    ) -> None:
        """
        T1: Stop Shuts Down All Layers in Reverse Order

        The Story:
        The daemon received SIGTERM. DaemonRuntime.shutdown() must gracefully shut
        down all subsystems. Order matters: stop accepting new work (RDM), finish
        in-flight work (services), then stop the loops.

        The Flow:
        1. Have a running daemon (runtime.initialize() succeeded)
        2. Call runtime.shutdown()
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
        - runtime.context is None after
        """
        mock_rdm = MagicMock()
        mock_rdm.shutdown = AsyncMock()
        mock_services = MagicMock(spec=DaemonServices)
        mock_services.state_store = MagicMock()
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

        context = DaemonContext(
            config=mock_config,
            loop_manager=mock_loop_mgr,
            comm_manager=MagicMock(),
            services=mock_services,
            recording_disk_manager=mock_rdm,
        )

        call_count = [0]

        def schedule_side_effect(coroutine):
            coroutine.close()
            call_count[0] += 1
            if call_count[0] == 1:
                return track_rdm_shutdown()
            else:
                return track_services_shutdown()

        mock_loop_mgr.schedule_on_general_loop.side_effect = schedule_side_effect
        mock_loop_mgr.stop.side_effect = track_loop_stop

        runtime = DaemonRuntime(
            db_path=temp_db_path,
            pid_path=temp_db_path.with_suffix(".pid"),
            socket_paths=(),
        )
        runtime._context = context

        runtime.shutdown()

        assert shutdown_order == ["rdm", "services", "loops"]

        assert runtime.context is None

    def test_t2_stop_without_start_logs_warning(
        self,
        temp_db_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        T2: Stop Without Start Logs Warning

        The Story:
        Due to a bug or race condition, stop() is called on a runtime that
        never successfully started. This shouldn't crash - just log a warning
        and return.

        The Flow:
        1. Create DaemonRuntime (don't call initialize)
        2. Call runtime.shutdown()
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
        runtime = DaemonRuntime(
            db_path=temp_db_path,
            pid_path=temp_db_path.with_suffix(".pid"),
            socket_paths=(),
        )

        with caplog.at_level(logging.WARNING):
            runtime.shutdown()

        assert "Cannot shut down: daemon not started" in caplog.text

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
        3. Call runtime.shutdown()
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
        mock_services = MagicMock(spec=DaemonServices)
        mock_loop_mgr = MagicMock()

        context = DaemonContext(
            config=mock_config,
            loop_manager=mock_loop_mgr,
            comm_manager=MagicMock(),
            services=mock_services,
            recording_disk_manager=mock_rdm,
        )

        rdm_future = MagicMock()
        rdm_future.result.side_effect = TimeoutError("Encoder worker stuck")

        services_future = MagicMock()
        services_future.result.return_value = None

        call_count = [0]

        def schedule_side_effect(coroutine):
            coroutine.close()
            call_count[0] += 1
            if call_count[0] == 1:
                return rdm_future
            else:
                return services_future

        mock_loop_mgr.schedule_on_general_loop.side_effect = schedule_side_effect

        runtime = DaemonRuntime(
            db_path=temp_db_path,
            pid_path=temp_db_path.with_suffix(".pid"),
            socket_paths=(),
        )
        runtime._context = context

        with caplog.at_level(logging.ERROR):
            runtime.shutdown()

        assert mock_loop_mgr.schedule_on_general_loop.call_count == 2

        mock_loop_mgr.stop.assert_called_once()

        assert "Error shutting down RecordingDiskManager" in caplog.text

        assert runtime.context is None


class TestDaemonRuntimeContext:
    """Tests for DaemonRuntime.context property."""

    def test_c1_context_property_before_start(
        self,
        temp_db_path: Path,
    ) -> None:
        """
        C1: Context Property Before Start

        The Story:
        Code checks runtime.context before calling initialize(). This is valid -
        maybe checking if already running. Should return None, not raise.

        The Flow:
        1. Create DaemonRuntime
        2. Access runtime.context
        3. Returns None

        Key Assertions:
        - Returns None
        - No exception
        """
        runtime = DaemonRuntime(
            db_path=temp_db_path,
            pid_path=temp_db_path.with_suffix(".pid"),
            socket_paths=(),
        )

        result = runtime.context

        assert result is None

    def test_c2_context_property_after_start(
        self,
        temp_db_path: Path,
        mock_config: DaemonConfig,
    ) -> None:
        """
        C2: Context Property After Successful Start

        The Story:
        After initialize() succeeds, context should return the DaemonContext. This
        lets callers access components without storing the return value.

        The Flow:
        1. Create and initialize DaemonRuntime
        2. Access runtime.context
        3. Returns the DaemonContext from initialize()

        Key Assertions:
        - Returns same DaemonContext as initialize() returned
        - All components accessible
        """
        mock_rdm = MagicMock()
        mock_comm = MagicMock()
        mock_services = MagicMock(spec=DaemonServices)
        mock_services.state_store = MagicMock()
        mock_loop_mgr = MagicMock()

        with (
            patch("neuracore.data_daemon.runtime.ProfileManager"),
            patch("neuracore.data_daemon.runtime.ConfigManager") as MockConfigMgr,
            patch("neuracore.data_daemon.runtime.login"),
            patch("neuracore.data_daemon.runtime.EventLoopManager") as MockLoopMgr,
            patch(
                "neuracore.data_daemon.runtime.rdm.RecordingDiskManager",
                return_value=mock_rdm,
            ),
            patch(
                "neuracore.data_daemon.runtime.CommunicationsManager",
                return_value=mock_comm,
            ),
        ):
            mock_config_mgr_instance = MagicMock()
            mock_config_mgr_instance.resolve_effective_config.return_value = mock_config
            MockConfigMgr.return_value = mock_config_mgr_instance

            mock_loop_mgr.is_running.return_value = True
            services_future = MagicMock()
            services_future.result.return_value = mock_services
            reconcile_future = MagicMock()
            reconcile_future.result.return_value = None
            scheduled_calls = [0]

            def schedule_side_effect(coroutine):
                coroutine.close()
                scheduled_calls[0] += 1
                if scheduled_calls[0] == 1:
                    return services_future
                return reconcile_future

            mock_loop_mgr.schedule_on_general_loop.side_effect = schedule_side_effect
            MockLoopMgr.return_value = mock_loop_mgr

            runtime = DaemonRuntime(
                db_path=temp_db_path,
                pid_path=temp_db_path.with_suffix(".pid"),
                socket_paths=(),
            )
            start_result = runtime.initialize()

            context_result = runtime.context

            assert context_result is start_result
            assert context_result is not None

            assert context_result.config is mock_config
            assert context_result.loop_manager is mock_loop_mgr
            assert context_result.services is mock_services
            assert context_result.recording_disk_manager is mock_rdm
            assert context_result.comm_manager is mock_comm
