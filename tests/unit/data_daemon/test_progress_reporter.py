"""Unit tests for progress_reporter module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from neuracore.data_daemon.const import BACKEND_API_MAX_RETRIES
from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.progress_reporter import ProgressReporter


@pytest.fixture
def mock_emitter():
    """Create a mock emitter."""
    emitter = MagicMock()
    emitter.on = MagicMock()
    emitter.emit = MagicMock()
    return emitter


@pytest.fixture
def mock_trace():
    """Create a mock trace record."""
    trace = MagicMock()
    trace.recording_id = "recording-123"
    trace.trace_id = "trace-456"
    trace.total_bytes = 1024
    trace.robot_id = "robot-1"
    trace.robot_name = "TestRobot"
    trace.robot_instance = 1
    trace.dataset_id = "dataset-1"
    trace.dataset_name = "TestDataset"
    return trace


@pytest.fixture
def mock_auth():
    """Create a mock auth manager."""
    auth = MagicMock()
    auth.get_headers = MagicMock(return_value={"Authorization": "Bearer token"})
    return auth


class TestProgressReporterSuccess:
    """Test successful progress reporting."""

    @pytest.mark.asyncio
    async def test_report_progress_success(self, mock_emitter, mock_trace, mock_auth):
        """Test successful progress report emits PROGRESS_REPORTED."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with (
            patch(
                "neuracore.data_daemon.progress_reporter.get_emitter",
                return_value=mock_emitter,
            ),
            patch(
                "neuracore.data_daemon.progress_reporter.get_auth",
                return_value=mock_auth,
            ),
            patch(
                "neuracore.data_daemon.progress_reporter.get_current_org",
                return_value="org-123",
            ),
        ):
            reporter = ProgressReporter(mock_session)
            await reporter.report_progress(1000.0, 2000.0, [mock_trace])

            mock_emitter.emit.assert_called_once_with(
                Emitter.PROGRESS_REPORTED, "recording-123"
            )


class TestProgressReporterRetry:
    """Test retry logic for progress reporting."""

    @pytest.mark.asyncio
    async def test_retries_on_retryable_status_code(
        self, mock_emitter, mock_trace, mock_auth
    ):
        """Test that retryable status codes trigger retries."""
        call_count = 0

        def mock_post_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = AsyncMock()
            if call_count < 3:
                mock_response.status = 503
                mock_response.text = AsyncMock(return_value="Service Unavailable")
            else:
                mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            return mock_response

        mock_session = MagicMock()
        mock_session.post = mock_post_side_effect

        with (
            patch(
                "neuracore.data_daemon.progress_reporter.get_emitter",
                return_value=mock_emitter,
            ),
            patch(
                "neuracore.data_daemon.progress_reporter.get_auth",
                return_value=mock_auth,
            ),
            patch(
                "neuracore.data_daemon.progress_reporter.get_current_org",
                return_value="org-123",
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            reporter = ProgressReporter(mock_session)
            await reporter.report_progress(1000.0, 2000.0, [mock_trace])

            assert call_count == 3
            mock_emitter.emit.assert_called_once_with(
                Emitter.PROGRESS_REPORTED, "recording-123"
            )

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_status(
        self, mock_emitter, mock_trace, mock_auth
    ):
        """Test that non-retryable status codes don't trigger retries."""
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with (
            patch(
                "neuracore.data_daemon.progress_reporter.get_emitter",
                return_value=mock_emitter,
            ),
            patch(
                "neuracore.data_daemon.progress_reporter.get_auth",
                return_value=mock_auth,
            ),
            patch(
                "neuracore.data_daemon.progress_reporter.get_current_org",
                return_value="org-123",
            ),
        ):
            reporter = ProgressReporter(mock_session)
            await reporter.report_progress(1000.0, 2000.0, [mock_trace])

            mock_session.post.assert_called_once()
            mock_emitter.emit.assert_called_once_with(
                Emitter.PROGRESS_REPORT_FAILED,
                "recording-123",
                "HTTP 400: Bad Request",
            )

    @pytest.mark.asyncio
    async def test_retries_on_network_error(self, mock_emitter, mock_trace, mock_auth):
        """Test that network errors trigger retries."""
        call_count = 0

        def mock_post_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise aiohttp.ClientError("Connection failed")
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            return mock_response

        mock_session = MagicMock()
        mock_session.post = mock_post_side_effect

        with (
            patch(
                "neuracore.data_daemon.progress_reporter.get_emitter",
                return_value=mock_emitter,
            ),
            patch(
                "neuracore.data_daemon.progress_reporter.get_auth",
                return_value=mock_auth,
            ),
            patch(
                "neuracore.data_daemon.progress_reporter.get_current_org",
                return_value="org-123",
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            reporter = ProgressReporter(mock_session)
            await reporter.report_progress(1000.0, 2000.0, [mock_trace])

            assert call_count == 3
            mock_emitter.emit.assert_called_once_with(
                Emitter.PROGRESS_REPORTED, "recording-123"
            )

    @pytest.mark.asyncio
    async def test_emits_failed_after_all_retries_exhausted(
        self, mock_emitter, mock_trace, mock_auth
    ):
        """Test that PROGRESS_REPORT_FAILED is emitted after all retries fail."""
        mock_response = AsyncMock()
        mock_response.status = 503
        mock_response.text = AsyncMock(return_value="Service Unavailable")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        with (
            patch(
                "neuracore.data_daemon.progress_reporter.get_emitter",
                return_value=mock_emitter,
            ),
            patch(
                "neuracore.data_daemon.progress_reporter.get_auth",
                return_value=mock_auth,
            ),
            patch(
                "neuracore.data_daemon.progress_reporter.get_current_org",
                return_value="org-123",
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            reporter = ProgressReporter(mock_session)
            await reporter.report_progress(1000.0, 2000.0, [mock_trace])

            assert mock_session.post.call_count == BACKEND_API_MAX_RETRIES
            mock_emitter.emit.assert_called_once_with(
                Emitter.PROGRESS_REPORT_FAILED,
                "recording-123",
                "HTTP 503: Service Unavailable",
            )


class TestProgressReporterEdgeCases:
    """Test edge cases for progress reporting."""

    @pytest.mark.asyncio
    async def test_empty_traces_returns_early(self, mock_emitter):
        """Test that empty traces list returns without action."""
        mock_session = MagicMock()

        with patch(
            "neuracore.data_daemon.progress_reporter.get_emitter",
            return_value=mock_emitter,
        ):
            reporter = ProgressReporter(mock_session)
            await reporter.report_progress(1000.0, 2000.0, [])

            mock_session.post.assert_not_called()
            mock_emitter.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_recording_id_returns_early(self, mock_emitter, mock_trace):
        """Test that missing recording_id returns without action."""
        mock_trace.recording_id = None
        mock_session = MagicMock()

        with patch(
            "neuracore.data_daemon.progress_reporter.get_emitter",
            return_value=mock_emitter,
        ):
            reporter = ProgressReporter(mock_session)
            await reporter.report_progress(1000.0, 2000.0, [mock_trace])

            mock_session.post.assert_not_called()
            mock_emitter.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_triggers_retry(self, mock_emitter, mock_trace, mock_auth):
        """Test that timeout errors trigger retries."""
        call_count = 0

        def mock_post_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise asyncio.TimeoutError("Request timed out")
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            return mock_response

        mock_session = MagicMock()
        mock_session.post = mock_post_side_effect

        with (
            patch(
                "neuracore.data_daemon.progress_reporter.get_emitter",
                return_value=mock_emitter,
            ),
            patch(
                "neuracore.data_daemon.progress_reporter.get_auth",
                return_value=mock_auth,
            ),
            patch(
                "neuracore.data_daemon.progress_reporter.get_current_org",
                return_value="org-123",
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            reporter = ProgressReporter(mock_session)
            await reporter.report_progress(1000.0, 2000.0, [mock_trace])

            assert call_count == 2
            mock_emitter.emit.assert_called_once_with(
                Emitter.PROGRESS_REPORTED, "recording-123"
            )
