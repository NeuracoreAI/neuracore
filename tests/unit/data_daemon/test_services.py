"""Tests for daemon async service construction and shutdown."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.services import DaemonServices
from neuracore.data_daemon.state_management.state_manager import StateManager
from neuracore.data_daemon.state_management.state_store_sqlite import SqliteStateStore
from neuracore.data_daemon.upload_management.trace_status_updater import (
    TraceStatusUpdater,
)


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_state.db"


@pytest.fixture
def mock_config() -> DaemonConfig:
    return DaemonConfig(path_to_store_record="/tmp/test_recordings")


@pytest.fixture
def mock_daemon_services() -> DaemonServices:
    mock_session = MagicMock(spec=aiohttp.ClientSession)
    mock_session.close = AsyncMock()

    mock_state_store = MagicMock(spec=SqliteStateStore)
    mock_state_store.close = AsyncMock()
    mock_state_manager = MagicMock(spec=StateManager)

    mock_registration_manager = MagicMock()
    mock_registration_manager.shutdown = AsyncMock()

    mock_trace_status_updater = MagicMock(spec=TraceStatusUpdater)

    mock_upload_manager = MagicMock()
    mock_upload_manager.shutdown = AsyncMock()

    mock_connection_manager = MagicMock()
    mock_connection_manager.stop = AsyncMock()

    return DaemonServices(
        client_session=mock_session,
        state_store=mock_state_store,
        state_manager=mock_state_manager,
        registration_manager=mock_registration_manager,
        trace_status_updater=mock_trace_status_updater,
        upload_manager=mock_upload_manager,
        connection_manager=mock_connection_manager,
        progress_reporter=MagicMock(),
    )


@pytest.fixture(autouse=True)
def _mock_registration_manager_ctor():
    with patch("neuracore.data_daemon.services.RegistrationManager") as mock_ctor:
        instance = MagicMock()
        instance.start = MagicMock()
        instance.shutdown = AsyncMock()
        mock_ctor.return_value = instance
        yield mock_ctor


class TestDaemonServicesCreate:
    @pytest.mark.asyncio
    async def test_b1_happy_path_all_services_initialize(
        self,
        mock_config: DaemonConfig,
        temp_db_path: Path,
        emitter: Emitter,
    ) -> None:
        with (
            patch("neuracore.data_daemon.services.ConnectionManager") as MockConnMgr,
            patch("neuracore.data_daemon.services.UploadManager") as MockUploadMgr,
            patch(
                "neuracore.data_daemon.services.ProgressReporter"
            ) as MockProgressReporter,
            patch("neuracore.data_daemon.services.SqliteStateStore") as MockStateStore,
            patch("neuracore.data_daemon.services.StateManager") as MockStateMgr,
        ):
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
            mock_state_manager_instance.recover_startup_state = AsyncMock()
            MockStateMgr.return_value = mock_state_manager_instance

            services = await DaemonServices.create(mock_config, emitter, temp_db_path)

            assert services is not None
            assert isinstance(services, DaemonServices)
            assert isinstance(services.client_session, aiohttp.ClientSession)
            assert services.state_store is mock_state_store_instance
            assert services.state_manager is mock_state_manager_instance
            assert services.upload_manager is mock_upload_instance
            assert services.connection_manager is mock_conn_instance
            assert services.progress_reporter is mock_progress_instance

            mock_state_store_instance.init_async_store.assert_called_once()
            mock_conn_instance.start.assert_called_once()
            await services.client_session.close()


class TestDaemonServicesShutdown:
    @pytest.mark.asyncio
    async def test_s1_clean_shutdown_closes_all_resources(
        self,
        mock_daemon_services: DaemonServices,
    ) -> None:
        await mock_daemon_services.shutdown()

        mock_daemon_services.connection_manager.stop.assert_called_once()
        mock_daemon_services.upload_manager.shutdown.assert_called_once()
        mock_daemon_services.state_store.close.assert_called_once()
        mock_daemon_services.client_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_s2_shutdown_continues_despite_errors(
        self,
        mock_daemon_services: DaemonServices,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_daemon_services.connection_manager.stop = AsyncMock(
            side_effect=RuntimeError("Connection already closed")
        )

        with caplog.at_level(logging.ERROR):
            await mock_daemon_services.shutdown()

        mock_daemon_services.upload_manager.shutdown.assert_called_once()
        mock_daemon_services.client_session.close.assert_called_once()
        assert "Error stopping ConnectionManager" in caplog.text
