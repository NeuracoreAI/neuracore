"""Tests for ResumableFileUploader.

Tests chunked file uploads, resumable sessions, retry logic, and error handling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from neuracore.data_daemon.upload_management.resumable_file_uploader import (
    ResumableFileUploader,
)


@pytest.fixture
def test_file(tmp_path: Path) -> Path:
    """Create a 5MB test file."""
    test_file = tmp_path / "test_video.mp4"
    test_file.write_bytes(b"X" * (5 * 1024 * 1024))
    return test_file


@pytest.fixture
def large_test_file(tmp_path: Path) -> Path:
    """Create a 10MB test file."""
    test_file = tmp_path / "large_file.mp4"
    test_file.write_bytes(b"X" * (10 * 1024 * 1024))
    return test_file


@pytest.fixture
def very_large_test_file(tmp_path: Path) -> Path:
    """Create a 200MB test file for multi-chunk upload testing."""
    test_file = tmp_path / "very_large_file.mp4"
    test_file.write_bytes(b"X" * (200 * 1024 * 1024))
    return test_file


@pytest.fixture
def mock_auth():
    """Mock authentication."""
    with patch(
        "neuracore.data_daemon.upload_management.resumable_file_uploader.get_auth"
    ) as mock_get_auth:
        auth_instance = MagicMock()
        auth_instance.get_org_id.return_value = "test-org"
        auth_instance.get_headers.return_value = {"Authorization": "Bearer test-token"}
        mock_get_auth.return_value = auth_instance
        yield mock_get_auth


@pytest.fixture
def uploader(test_file: Path, mock_auth) -> ResumableFileUploader:
    """Create a basic ResumableFileUploader instance."""
    return ResumableFileUploader(
        recording_id="rec-123",
        filepath=str(test_file),
        cloud_filepath="RGB_IMAGES/camera/trace.mp4",
        content_type="video/mp4",
        bytes_uploaded=0,
    )


def test_uploader_initializes_correctly(
    uploader: ResumableFileUploader, test_file: Path
) -> None:
    """Test ResumableFileUploader initialization."""
    assert uploader._recording_id == "rec-123"
    assert uploader._filepath == str(test_file)
    assert uploader._cloud_filepath == "RGB_IMAGES/camera/trace.mp4"
    assert uploader._content_type == "video/mp4"
    assert uploader._bytes_uploaded == 0


def test_uploader_gets_session_uri(uploader: ResumableFileUploader) -> None:
    """Test obtaining upload session URI from backend."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "url": "https://storage.googleapis.com/upload/session/123"
        }
        mock_get.return_value.raise_for_status = MagicMock()

        session_uri = uploader._get_upload_session_uri()

        assert session_uri == "https://storage.googleapis.com/upload/session/123"
        mock_get.assert_called_once()


def test_uploader_handles_successful_upload(uploader: ResumableFileUploader) -> None:
    """Test successful file upload."""
    with patch("requests.get") as mock_get, patch("requests.put") as mock_put:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"url": "https://upload.url"}
        mock_get.return_value.raise_for_status = MagicMock()

        mock_put.return_value.status_code = 200

        success, bytes_uploaded, error_message = uploader.upload()

        assert success is True
        assert bytes_uploaded == 5 * 1024 * 1024
        assert error_message is None


def test_uploader_tracks_progress_with_callback(test_file: Path, mock_auth) -> None:
    """Test progress callback is called with byte deltas."""
    progress_updates = []

    def progress_callback(bytes_delta: int) -> None:
        progress_updates.append(bytes_delta)

    uploader = ResumableFileUploader(
        recording_id="rec-123",
        filepath=str(test_file),
        cloud_filepath="RGB_IMAGES/camera/trace.mp4",
        content_type="video/mp4",
        progress_callback=progress_callback,
    )

    with patch("requests.get") as mock_get, patch("requests.put") as mock_put:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"url": "https://upload.url"}
        mock_get.return_value.raise_for_status = MagicMock()

        mock_put.return_value.status_code = 200

        uploader.upload()

        assert len(progress_updates) > 0
        assert sum(progress_updates) == 5 * 1024 * 1024


def test_uploader_resumes_from_offset(large_test_file: Path, mock_auth) -> None:
    """Test resuming upload from a specific offset."""
    uploader = ResumableFileUploader(
        recording_id="rec-123",
        filepath=str(large_test_file),
        cloud_filepath="RGB_IMAGES/camera/trace.mp4",
        content_type="video/mp4",
        bytes_uploaded=5 * 1024 * 1024,
    )

    with patch("requests.get") as mock_get, patch("requests.put") as mock_put:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"url": "https://upload.url"}
        mock_get.return_value.raise_for_status = MagicMock()

        mock_put.return_value.status_code = 200

        success, bytes_uploaded, error_message = uploader.upload()

        assert success is True
        assert bytes_uploaded == 10 * 1024 * 1024

        first_put_call = mock_put.call_args_list[0]
        content_range = first_put_call[1]["headers"]["Content-Range"]
        assert content_range.startswith("bytes 5242880-")


def test_uploader_handles_session_expiration(uploader: ResumableFileUploader) -> None:
    """Test handling of 410 Gone (session expiration)."""
    with patch("requests.get") as mock_get, patch("requests.put") as mock_put:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.side_effect = [
            {"url": "https://upload.url/session1"},
            {"url": "https://upload.url/session2"},
        ]
        mock_get.return_value.raise_for_status = MagicMock()

        mock_put.return_value.status_code = 410
        mock_put.side_effect = [
            MagicMock(status_code=410),  # Session expired
            MagicMock(status_code=200),  # Success with new session
        ]

        success, bytes_uploaded, error_message = uploader.upload()

        assert success is True
        assert mock_get.call_count == 2


def test_uploader_handles_network_error(uploader: ResumableFileUploader) -> None:
    """Test handling of network connection errors."""
    with patch("requests.get") as mock_get, patch("requests.put") as mock_put:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"url": "https://upload.url"}
        mock_get.return_value.raise_for_status = MagicMock()

        # Mock network error
        mock_put.side_effect = requests.exceptions.ConnectionError("Network error")

        success, bytes_uploaded, error_message = uploader.upload()

        assert success is False
        assert "Network connection error" in error_message


def test_uploader_handles_file_not_found(mock_auth) -> None:
    """Test handling when file doesn't exist."""
    uploader = ResumableFileUploader(
        recording_id="rec-123",
        filepath="/nonexistent/file.mp4",
        cloud_filepath="RGB_IMAGES/camera/trace.mp4",
        content_type="video/mp4",
    )

    with pytest.raises(FileNotFoundError):
        uploader.upload()


def test_uploader_retries_on_timeout(uploader: ResumableFileUploader) -> None:
    """Test retry logic on timeout errors."""
    with (
        patch("requests.get") as mock_get,
        patch("requests.put") as mock_put,
        patch("time.sleep"),
    ):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"url": "https://upload.url"}
        mock_get.return_value.raise_for_status = MagicMock()

        mock_put.side_effect = [
            requests.exceptions.Timeout("Timeout"),
            requests.exceptions.Timeout("Timeout"),
            MagicMock(status_code=200),
        ]

        success, bytes_uploaded, error_message = uploader.upload()

        assert success is True
        assert mock_put.call_count == 3


def test_uploader_fails_after_max_retries(uploader: ResumableFileUploader) -> None:
    """Test upload fails after MAX_RETRIES attempts."""
    with (
        patch("requests.get") as mock_get,
        patch("requests.put") as mock_put,
        patch("time.sleep"),
    ):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"url": "https://upload.url"}
        mock_get.return_value.raise_for_status = MagicMock()

        mock_put.side_effect = requests.exceptions.Timeout("Timeout")

        success, bytes_uploaded, error_message = uploader.upload()

        assert success is False
        assert "failed after" in error_message
        assert mock_put.call_count == ResumableFileUploader.MAX_RETRIES


def test_uploader_handles_http_errors(uploader: ResumableFileUploader) -> None:
    """Test handling of HTTP error responses."""
    with (
        patch("requests.get") as mock_get,
        patch("requests.put") as mock_put,
        patch("time.sleep"),
    ):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"url": "https://upload.url"}
        mock_get.return_value.raise_for_status = MagicMock()

        mock_put.return_value.status_code = 500

        success, bytes_uploaded, error_message = uploader.upload()

        assert success is False
        assert "failed after" in error_message


def test_uploader_sets_correct_content_range_headers(
    uploader: ResumableFileUploader,
) -> None:
    """Test Content-Range headers are set correctly."""
    with patch("requests.get") as mock_get, patch("requests.put") as mock_put:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"url": "https://upload.url"}
        mock_get.return_value.raise_for_status = MagicMock()

        mock_put.return_value.status_code = 200

        uploader.upload()

        last_put_call = mock_put.call_args_list[-1]
        headers = last_put_call[1]["headers"]
        content_range = headers["Content-Range"]

        assert content_range.endswith(f"/{5 * 1024 * 1024}")


def test_uploader_handles_session_uri_fetch_failure(
    uploader: ResumableFileUploader,
) -> None:
    """Test handling when fetching session URI fails."""
    with patch("requests.get") as mock_get:
        # Mock session URI fetch failure
        mock_get.side_effect = requests.exceptions.RequestException("API Error")

        success, bytes_uploaded, error_message = uploader.upload()

        assert success is False
        assert "Failed to get upload session URI" in error_message


def test_uploader_handles_large_file(very_large_test_file: Path, mock_auth) -> None:
    """Test uploading large file (multiple chunks)."""
    progress_updates = []

    def progress_callback(bytes_delta: int) -> None:
        progress_updates.append(bytes_delta)

    uploader = ResumableFileUploader(
        recording_id="rec-123",
        filepath=str(very_large_test_file),
        cloud_filepath="RGB_IMAGES/camera/trace.mp4",
        content_type="video/mp4",
        progress_callback=progress_callback,
    )

    with patch("requests.get") as mock_get, patch("requests.put") as mock_put:
        # Mock session URI
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"url": "https://upload.url"}
        mock_get.return_value.raise_for_status = MagicMock()

        def mock_put_response(*args, **kwargs):
            if mock_put.call_count < 4:
                return MagicMock(status_code=308)
            return MagicMock(status_code=200)

        mock_put.side_effect = mock_put_response

        success, bytes_uploaded, error_message = uploader.upload()

        assert success is True
        assert bytes_uploaded == 200 * 1024 * 1024
        assert len(progress_updates) == 4
        assert sum(progress_updates) == 200 * 1024 * 1024


def test_uploader_exponential_backoff(uploader: ResumableFileUploader) -> None:
    """Test exponential backoff between retries."""
    with (
        patch("requests.get") as mock_get,
        patch("requests.put") as mock_put,
        patch("time.sleep") as mock_sleep,
    ):
        # Mock session URI
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"url": "https://upload.url"}
        mock_get.return_value.raise_for_status = MagicMock()

        # Mock timeouts then success
        mock_put.side_effect = [
            requests.exceptions.Timeout("Timeout"),
            requests.exceptions.Timeout("Timeout"),
            MagicMock(status_code=200),
        ]

        uploader.upload()

        sleep_calls = [call(1), call(2)]
        mock_sleep.assert_has_calls(sleep_calls)
