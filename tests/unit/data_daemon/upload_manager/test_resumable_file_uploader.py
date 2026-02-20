"""Tests for ResumableFileUploader.

Tests chunked file uploads, resumable sessions, retry logic, and error handling.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ssl
from pathlib import Path
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
import pytest_asyncio

from neuracore.data_daemon.upload_management.resumable_file_uploader import (
    ResumableFileUploader,
)


def _compute_file_md5_b64(filepath: Path) -> str:
    """Compute base64-encoded MD5 hash for a file."""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5_hash.update(chunk)
    return base64.b64encode(md5_hash.digest()).decode()


@pytest_asyncio.fixture
async def client_session():
    session = aiohttp.ClientSession()
    yield session
    await session.close()


@pytest.fixture
def test_file(tmp_path: Path) -> Path:
    test_file = tmp_path / "test_video.mp4"
    test_file.write_bytes(b"X" * (5 * 1024 * 1024))
    return test_file


@pytest.fixture
def large_test_file(tmp_path: Path) -> Path:
    test_file = tmp_path / "large_file.mp4"
    test_file.write_bytes(b"X" * (10 * 1024 * 1024))
    return test_file


@pytest.fixture
def very_large_test_file(tmp_path: Path) -> Path:
    test_file = tmp_path / "very_large_file.mp4"
    test_file.write_bytes(b"X" * (200 * 1024 * 1024))
    return test_file


@pytest.fixture
def mock_auth():
    with (
        patch(
            "neuracore.data_daemon.upload_management.resumable_file_uploader.get_auth"
        ) as mock_get_auth,
        patch(
            "neuracore.data_daemon.upload_management.resumable_file_uploader.get_current_org",
            return_value="test-org",
        ),
    ):
        auth_instance = MagicMock()
        auth_instance.get_headers = MagicMock(
            return_value={"Authorization": "Bearer test-token"}
        )
        mock_get_auth.return_value = auth_instance
        yield mock_get_auth


@pytest.fixture
def uploader(
    test_file: Path, mock_auth, client_session: aiohttp.ClientSession
) -> ResumableFileUploader:
    return ResumableFileUploader(
        recording_id="rec-123",
        filepath=str(test_file),
        cloud_filepath="RGB_IMAGES/camera/trace.mp4",
        content_type="video/mp4",
        client_session=client_session,
        bytes_uploaded=0,
    )


class _MockAioHTTPResponse:
    def __init__(
        self,
        *,
        status: int,
        json_data=None,
        exc: Exception | None = None,
        headers: dict | None = None,
        text_data: str = "",
    ):
        self.status = status
        self._json_data = json_data
        self._exc = exc
        self.headers = headers or {}
        self._text_data = text_data
        self.request_info = MagicMock()
        self.history = ()

    async def __aenter__(self):
        if self._exc is not None:
            raise self._exc
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=self.request_info,
                history=self.history,
                status=self.status,
                message="error",
                headers=self.headers,
            )

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text_data


def test_uploader_initializes_correctly(
    uploader: ResumableFileUploader, test_file: Path
) -> None:
    assert uploader._recording_id == "rec-123"
    assert uploader._filepath == str(test_file)
    assert uploader._cloud_filepath == "RGB_IMAGES/camera/trace.mp4"
    assert uploader._content_type == "video/mp4"
    assert uploader._bytes_uploaded == 0


@pytest.mark.asyncio
async def test_uploader_gets_session_uri(uploader: ResumableFileUploader) -> None:
    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200,
            json_data={"url": "https://storage.googleapis.com/upload/session/123"},
        ),
    ) as mock_get:
        session_uri = await uploader._get_upload_session_uri()
        assert session_uri == "https://storage.googleapis.com/upload/session/123"
        assert mock_get.call_count == 1


@pytest.mark.asyncio
async def test_uploader_handles_successful_upload(
    uploader: ResumableFileUploader, test_file: Path
) -> None:
    md5_b64 = _compute_file_md5_b64(test_file)

    put_responses = [
        _MockAioHTTPResponse(status=308),
        _MockAioHTTPResponse(
            status=200,
            headers={"x-goog-hash": f"md5={md5_b64}"},
        ),
    ]

    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):
        with patch.object(
            uploader._session,
            "put",
            side_effect=put_responses,
        ):
            success, bytes_uploaded, error_message = await uploader.upload()

    assert success is True
    assert bytes_uploaded == 5 * 1024 * 1024
    assert error_message is None


@pytest.mark.asyncio
async def test_uploader_tracks_progress_with_callback(
    test_file: Path, mock_auth, client_session: aiohttp.ClientSession
) -> None:
    progress_updates: list[int] = []
    md5_b64 = _compute_file_md5_b64(test_file)

    async def progress_callback(bytes_delta: int) -> None:
        progress_updates.append(bytes_delta)

    uploader = ResumableFileUploader(
        recording_id="rec-123",
        filepath=str(test_file),
        cloud_filepath="RGB_IMAGES/camera/trace.mp4",
        content_type="video/mp4",
        client_session=client_session,
        progress_callback=progress_callback,
    )

    put_responses = [
        _MockAioHTTPResponse(status=308),
        _MockAioHTTPResponse(status=200, headers={"x-goog-hash": f"md5={md5_b64}"}),
    ]

    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):
        with patch.object(
            uploader._session,
            "put",
            side_effect=put_responses,
        ):
            await uploader.upload()

    assert len(progress_updates) > 0
    assert sum(progress_updates) == 5 * 1024 * 1024


@pytest.mark.asyncio
async def test_uploader_resumes_from_offset(
    large_test_file: Path, mock_auth, client_session: aiohttp.ClientSession
) -> None:
    md5_b64 = _compute_file_md5_b64(large_test_file)

    uploader = ResumableFileUploader(
        recording_id="rec-123",
        filepath=str(large_test_file),
        cloud_filepath="RGB_IMAGES/camera/trace.mp4",
        content_type="video/mp4",
        client_session=client_session,
        bytes_uploaded=5 * 1024 * 1024,
    )

    put_responses = [
        _MockAioHTTPResponse(status=308, headers={"Range": "bytes=0-5242879"}),
        _MockAioHTTPResponse(status=200, headers={"x-goog-hash": f"md5={md5_b64}"}),
    ]

    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):
        with patch.object(
            uploader._session,
            "put",
            side_effect=put_responses,
        ) as mock_put:
            success, bytes_uploaded, error_message = await uploader.upload()

    assert success is True
    assert bytes_uploaded == 10 * 1024 * 1024
    second_put_call = mock_put.call_args_list[1]
    content_range = second_put_call.kwargs["headers"]["Content-Range"]
    assert content_range.startswith("bytes 5242880-")


@pytest.mark.asyncio
async def test_uploader_handles_session_expiration(
    uploader: ResumableFileUploader, test_file: Path
) -> None:
    md5_b64 = _compute_file_md5_b64(test_file)

    with patch.object(
        uploader._session,
        "get",
        side_effect=[
            _MockAioHTTPResponse(
                status=200, json_data={"url": "https://upload.url/session1"}
            ),
            _MockAioHTTPResponse(
                status=200, json_data={"url": "https://upload.url/session2"}
            ),
        ],
    ) as mock_get:
        with patch.object(
            uploader._session,
            "put",
            side_effect=[
                _MockAioHTTPResponse(status=308),
                _MockAioHTTPResponse(status=410),
                _MockAioHTTPResponse(
                    status=200, headers={"x-goog-hash": f"md5={md5_b64}"}
                ),
            ],
        ):
            success, bytes_uploaded, error_message = await uploader.upload()

    assert success is True
    assert mock_get.call_count == 2


@pytest.mark.asyncio
async def test_uploader_handles_network_error(uploader: ResumableFileUploader) -> None:
    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):
        with patch.object(
            uploader._session,
            "put",
            side_effect=[
                _MockAioHTTPResponse(
                    status=0,
                    exc=aiohttp.ClientConnectorError(MagicMock(), OSError("boom")),
                ),
            ],
        ):
            success, bytes_uploaded, error_message = await uploader.upload()

    assert success is False
    assert error_message is not None
    assert (
        "Failed to check upload status" in error_message
        or "Network connection error" in error_message
    )


@pytest.mark.asyncio
async def test_uploader_handles_file_not_found(
    mock_auth, client_session: aiohttp.ClientSession
) -> None:
    uploader = ResumableFileUploader(
        recording_id="rec-123",
        filepath="/nonexistent/file.mp4",
        cloud_filepath="RGB_IMAGES/camera/trace.mp4",
        content_type="video/mp4",
        client_session=client_session,
    )

    with pytest.raises(FileNotFoundError):
        await uploader.upload()


@pytest.mark.asyncio
async def test_uploader_retries_on_timeout(
    uploader: ResumableFileUploader, test_file: Path
) -> None:
    md5_b64 = _compute_file_md5_b64(test_file)

    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):
        with patch.object(
            uploader._session,
            "put",
            side_effect=[
                _MockAioHTTPResponse(status=308),
                _MockAioHTTPResponse(status=0, exc=asyncio.TimeoutError()),
                _MockAioHTTPResponse(status=0, exc=asyncio.TimeoutError()),
                _MockAioHTTPResponse(
                    status=200, headers={"x-goog-hash": f"md5={md5_b64}"}
                ),
            ],
        ) as mock_put:

            async def _sleep(_: float) -> None:
                return None

            with patch("asyncio.sleep", side_effect=_sleep):
                success, bytes_uploaded, error_message = await uploader.upload()

    assert success is True
    assert mock_put.call_count == 4


@pytest.mark.asyncio
async def test_uploader_fails_after_max_retries(
    uploader: ResumableFileUploader,
) -> None:
    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):
        put_responses = [_MockAioHTTPResponse(status=308)] + [
            _MockAioHTTPResponse(status=0, exc=asyncio.TimeoutError())
        ] * ResumableFileUploader.MAX_RETRIES

        with patch.object(
            uploader._session,
            "put",
            side_effect=put_responses,
        ) as mock_put:

            async def _sleep(_: float) -> None:
                return None

            with patch("asyncio.sleep", side_effect=_sleep):
                success, bytes_uploaded, error_message = await uploader.upload()

    assert success is False
    assert error_message is not None
    assert "failed after" in error_message
    assert mock_put.call_count == ResumableFileUploader.MAX_RETRIES + 1


@pytest.mark.asyncio
async def test_uploader_handles_http_errors(uploader: ResumableFileUploader) -> None:
    call_count = 0

    def put_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _MockAioHTTPResponse(status=308)
        return _MockAioHTTPResponse(status=500)

    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):
        with patch.object(
            uploader._session,
            "put",
            side_effect=put_side_effect,
        ):

            async def _sleep(_: float) -> None:
                return None

            with patch("asyncio.sleep", side_effect=_sleep):
                success, bytes_uploaded, error_message = await uploader.upload()

    assert success is False
    assert error_message is not None
    assert "failed after" in error_message


@pytest.mark.asyncio
async def test_uploader_sets_correct_content_range_headers(
    uploader: ResumableFileUploader, test_file: Path
) -> None:
    md5_b64 = _compute_file_md5_b64(test_file)

    put_responses = [
        _MockAioHTTPResponse(status=308),
        _MockAioHTTPResponse(status=200, headers={"x-goog-hash": f"md5={md5_b64}"}),
    ]

    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):
        with patch.object(
            uploader._session,
            "put",
            side_effect=put_responses,
        ) as mock_put:
            await uploader.upload()

    upload_put_call = mock_put.call_args_list[1]
    headers = upload_put_call.kwargs["headers"]
    content_range = headers["Content-Range"]
    assert content_range.endswith(f"/{5 * 1024 * 1024}")


@pytest.mark.asyncio
async def test_uploader_handles_session_uri_fetch_failure(
    uploader: ResumableFileUploader,
) -> None:
    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=0, exc=aiohttp.ClientError("API Error")
        ),
    ):
        success, bytes_uploaded, error_message = await uploader.upload()

    assert success is False
    assert error_message is not None
    assert "Failed to get upload session URI" in error_message


@pytest.mark.asyncio
async def test_uploader_handles_large_file(
    very_large_test_file: Path, mock_auth, client_session: aiohttp.ClientSession
) -> None:
    progress_updates: list[int] = []
    md5_b64 = _compute_file_md5_b64(very_large_test_file)

    async def progress_callback(bytes_delta: int) -> None:
        progress_updates.append(bytes_delta)

    uploader = ResumableFileUploader(
        recording_id="rec-123",
        filepath=str(very_large_test_file),
        cloud_filepath="RGB_IMAGES/camera/trace.mp4",
        content_type="video/mp4",
        client_session=client_session,
        progress_callback=progress_callback,
    )

    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):

        def put_side_effect(*args, **kwargs):
            if put_side_effect.calls == 0:
                put_side_effect.calls += 1
                return _MockAioHTTPResponse(status=308)
            if put_side_effect.calls < 4:
                put_side_effect.calls += 1
                return _MockAioHTTPResponse(status=308)
            return _MockAioHTTPResponse(
                status=200, headers={"x-goog-hash": f"md5={md5_b64}"}
            )

        put_side_effect.calls = 0

        with patch.object(uploader._session, "put", side_effect=put_side_effect):
            success, bytes_uploaded, error_message = await uploader.upload()

    assert success is True
    assert bytes_uploaded == 200 * 1024 * 1024
    assert len(progress_updates) == 4
    assert sum(progress_updates) == 200 * 1024 * 1024


@pytest.mark.asyncio
async def test_uploader_exponential_backoff(
    uploader: ResumableFileUploader, test_file: Path
) -> None:
    md5_b64 = _compute_file_md5_b64(test_file)

    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):
        with patch.object(
            uploader._session,
            "put",
            side_effect=[
                _MockAioHTTPResponse(status=308),
                _MockAioHTTPResponse(status=0, exc=asyncio.TimeoutError()),
                _MockAioHTTPResponse(status=0, exc=asyncio.TimeoutError()),
                _MockAioHTTPResponse(
                    status=200, headers={"x-goog-hash": f"md5={md5_b64}"}
                ),
            ],
        ):
            sleep_calls: list[float] = []

            async def _sleep(t: float) -> None:
                sleep_calls.append(t)

            with patch("asyncio.sleep", side_effect=_sleep):
                await uploader.upload()

    assert sleep_calls[:2] == [1, 2]


@pytest.mark.asyncio
async def test_uploader_permission_denied_fails_fast(
    uploader: ResumableFileUploader,
) -> None:
    """Test that 403 without signed URL expiration fails immediately without retry."""
    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):
        with patch.object(
            uploader._session,
            "put",
            side_effect=[
                _MockAioHTTPResponse(status=308),
                _MockAioHTTPResponse(status=403),
            ],
        ) as mock_put:
            success, bytes_uploaded, error_message = await uploader.upload()

    assert success is False
    assert error_message is not None
    assert "Permission denied" in error_message
    assert mock_put.call_count == 2


@pytest.mark.asyncio
async def test_uploader_bucket_not_found_fails_fast(
    uploader: ResumableFileUploader,
) -> None:
    """Test that 404 fails immediately without retry."""
    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):
        with patch.object(
            uploader._session,
            "put",
            side_effect=[
                _MockAioHTTPResponse(status=308),
                _MockAioHTTPResponse(status=404),
            ],
        ) as mock_put:
            success, bytes_uploaded, error_message = await uploader.upload()

    assert success is False
    assert error_message is not None
    assert "Bucket not found" in error_message
    assert mock_put.call_count == 2


@pytest.mark.asyncio
async def test_uploader_retries_on_429_rate_limit(
    uploader: ResumableFileUploader, test_file: Path
) -> None:
    """Test that 429 (rate limiting) triggers retry with backoff."""
    md5_b64 = _compute_file_md5_b64(test_file)

    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):
        with patch.object(
            uploader._session,
            "put",
            side_effect=[
                _MockAioHTTPResponse(status=308),
                _MockAioHTTPResponse(status=429),
                _MockAioHTTPResponse(status=429),
                _MockAioHTTPResponse(
                    status=200, headers={"x-goog-hash": f"md5={md5_b64}"}
                ),
            ],
        ) as mock_put:
            sleep_calls: list[float] = []

            async def _sleep(t: float) -> None:
                sleep_calls.append(t)

            with patch("asyncio.sleep", side_effect=_sleep):
                success, bytes_uploaded, error_message = await uploader.upload()

    assert success is True
    assert mock_put.call_count == 4
    assert len(sleep_calls) >= 1
    assert sleep_calls[0] == 1


@pytest.mark.asyncio
async def test_uploader_signed_url_expired_reacquires_session(
    uploader: ResumableFileUploader, test_file: Path
) -> None:
    """Test that 403 with X-Signed-Url-Expired header triggers new session."""
    md5_b64 = _compute_file_md5_b64(test_file)

    with patch.object(
        uploader._session,
        "get",
        side_effect=[
            _MockAioHTTPResponse(
                status=200, json_data={"url": "https://upload.url/session1"}
            ),
            _MockAioHTTPResponse(
                status=200, json_data={"url": "https://upload.url/session2"}
            ),
        ],
    ) as mock_get:
        with patch.object(
            uploader._session,
            "put",
            side_effect=[
                _MockAioHTTPResponse(status=308),
                _MockAioHTTPResponse(
                    status=403,
                    headers={"X-Signed-Url-Expired": "true"},
                ),
                _MockAioHTTPResponse(
                    status=200, headers={"x-goog-hash": f"md5={md5_b64}"}
                ),
            ],
        ):
            success, bytes_uploaded, error_message = await uploader.upload()

    assert success is True
    assert mock_get.call_count == 2


@pytest.mark.asyncio
async def test_uploader_finalization_retry_without_reupload(
    uploader: ResumableFileUploader, test_file: Path
) -> None:
    """Test that finalization retries don't re-upload data."""
    md5_b64 = _compute_file_md5_b64(test_file)
    file_size = 5 * 1024 * 1024

    call_count = 0

    def put_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        headers = kwargs.get("headers", {})
        content_range = headers.get("Content-Range", "")

        if call_count == 1:
            return _MockAioHTTPResponse(status=308)

        if call_count == 2:
            return _MockAioHTTPResponse(
                status=308,
                headers={"Range": f"bytes=0-{file_size - 1}"},
            )

        if call_count == 3 and "bytes */*" in content_range:
            return _MockAioHTTPResponse(
                status=308,
                headers={"Range": f"bytes=0-{file_size - 1}"},
            )

        if f"bytes */{file_size}" in content_range:
            if call_count == 4:
                return _MockAioHTTPResponse(status=500)
            return _MockAioHTTPResponse(
                status=200, headers={"x-goog-hash": f"md5={md5_b64}"}
            )

        return _MockAioHTTPResponse(
            status=200, headers={"x-goog-hash": f"md5={md5_b64}"}
        )

    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):
        with patch.object(
            uploader._session,
            "put",
            side_effect=put_side_effect,
        ):

            async def _sleep(_: float) -> None:
                return None

            with patch("asyncio.sleep", side_effect=_sleep):
                success, bytes_uploaded, error_message = await uploader.upload()

    assert success is True
    assert bytes_uploaded == file_size


@pytest.mark.asyncio
async def test_uploader_ssl_error_handling(
    uploader: ResumableFileUploader,
) -> None:
    """Test that SSL errors are handled properly."""
    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):
        with patch.object(
            uploader._session,
            "put",
            side_effect=[
                _MockAioHTTPResponse(
                    status=0,
                    exc=aiohttp.ClientSSLError(
                        MagicMock(), ssl.SSLError(1, "TLS failure")
                    ),
                ),
            ],
        ):
            success, bytes_uploaded, error_message = await uploader.upload()

    assert success is False
    assert error_message is not None
    assert "SSL error" in error_message


@pytest.mark.asyncio
async def test_uploader_backoff_caps_at_five_minutes(
    mock_auth, client_session: aiohttp.ClientSession
) -> None:
    """Test that exponential backoff caps at MAX_BACKOFF_SECONDS (300)."""
    uploader = ResumableFileUploader(
        recording_id="rec-123",
        filepath="/tmp/does-not-matter",
        cloud_filepath="trace.mp4",
        content_type="video/mp4",
        client_session=client_session,
    )

    sleep_calls: list[float] = []

    async def _sleep(t: float) -> None:
        sleep_calls.append(t)

    with patch("asyncio.sleep", side_effect=_sleep):
        await uploader._sleep_backoff(10)

    assert sleep_calls == [300]


@pytest.mark.asyncio
async def test_uploader_resume_no_duplicate_data(
    large_test_file: Path, mock_auth, client_session: aiohttp.ClientSession
) -> None:
    """Test that resuming from server offset doesn't send duplicate bytes."""
    md5_b64 = _compute_file_md5_b64(large_test_file)
    file_size = 10 * 1024 * 1024
    server_offset = 5 * 1024 * 1024

    uploader = ResumableFileUploader(
        recording_id="rec-123",
        filepath=str(large_test_file),
        cloud_filepath="RGB_IMAGES/camera/trace.mp4",
        content_type="video/mp4",
        client_session=client_session,
        bytes_uploaded=0,
    )

    put_responses = [
        _MockAioHTTPResponse(
            status=308,
            headers={"Range": f"bytes=0-{server_offset - 1}"},
        ),
        _MockAioHTTPResponse(status=200, headers={"x-goog-hash": f"md5={md5_b64}"}),
    ]

    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url"}
        ),
    ):
        with patch.object(
            uploader._session,
            "put",
            side_effect=put_responses,
        ) as mock_put:
            success, bytes_uploaded, error_message = await uploader.upload()

    assert success is True
    assert bytes_uploaded == file_size

    upload_call = mock_put.call_args_list[1]
    content_range = upload_call.kwargs["headers"]["Content-Range"]
    assert content_range.startswith(f"bytes {server_offset}-")


@pytest.mark.asyncio
async def test_uploader_session_preserved_on_retry(
    uploader: ResumableFileUploader, test_file: Path
) -> None:
    """Test that session is reused across retries (not recreated)."""
    md5_b64 = _compute_file_md5_b64(test_file)

    with patch.object(
        uploader._session,
        "get",
        return_value=_MockAioHTTPResponse(
            status=200, json_data={"url": "https://upload.url/session1"}
        ),
    ) as mock_get:
        with patch.object(
            uploader._session,
            "put",
            side_effect=[
                _MockAioHTTPResponse(status=308),
                _MockAioHTTPResponse(status=503),
                _MockAioHTTPResponse(status=503),
                _MockAioHTTPResponse(
                    status=200, headers={"x-goog-hash": f"md5={md5_b64}"}
                ),
            ],
        ):

            async def _sleep(_: float) -> None:
                return None

            with patch("asyncio.sleep", side_effect=_sleep):
                success, bytes_uploaded, error_message = await uploader.upload()

    assert success is True
    assert mock_get.call_count == 1


class TestBandwidthThrottling:
    """Tests for bandwidth throttling via BandwidthLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_called_per_chunk(
        self, test_file: Path, mock_auth, client_session: aiohttp.ClientSession
    ) -> None:
        """Test that should call bandwidth_limiter.acquire for each chunk."""
        from neuracore.data_daemon.upload_management.bandwidth_limiter import (
            BandwidthLimiter,
        )

        md5_b64 = _compute_file_md5_b64(test_file)
        file_size = 5 * 1024 * 1024

        limiter = BandwidthLimiter(1 * 1024 * 1024)
        acquire_calls: list[int] = []

        async def mock_acquire(n_bytes: int) -> None:
            acquire_calls.append(n_bytes)

        limiter.acquire = mock_acquire

        uploader = ResumableFileUploader(
            recording_id="rec-123",
            filepath=str(test_file),
            cloud_filepath="RGB_IMAGES/camera/trace.mp4",
            content_type="video/mp4",
            client_session=client_session,
            bandwidth_limiter=limiter,
        )

        with patch.object(
            uploader._session,
            "get",
            return_value=_MockAioHTTPResponse(
                status=200, json_data={"url": "https://upload.url"}
            ),
        ):
            with patch.object(
                uploader._session,
                "put",
                side_effect=[
                    _MockAioHTTPResponse(status=308),
                    _MockAioHTTPResponse(
                        status=200, headers={"x-goog-hash": f"md5={md5_b64}"}
                    ),
                ],
            ):
                success, _, _ = await uploader.upload()

        assert success is True
        assert len(acquire_calls) == 1
        assert acquire_calls[0] == file_size

    @pytest.mark.asyncio
    async def test_no_throttle_when_limiter_none(
        self, test_file: Path, mock_auth, client_session: aiohttp.ClientSession
    ) -> None:
        """Test that no throttling occurs when bandwidth_limiter is None."""
        md5_b64 = _compute_file_md5_b64(test_file)

        uploader = ResumableFileUploader(
            recording_id="rec-123",
            filepath=str(test_file),
            cloud_filepath="RGB_IMAGES/camera/trace.mp4",
            content_type="video/mp4",
            client_session=client_session,
            bandwidth_limiter=None,
        )

        sleep_calls: list[float] = []

        async def capture_sleep(duration: float) -> None:
            sleep_calls.append(duration)

        with patch.object(
            uploader._session,
            "get",
            return_value=_MockAioHTTPResponse(
                status=200, json_data={"url": "https://upload.url"}
            ),
        ):
            with patch.object(
                uploader._session,
                "put",
                side_effect=[
                    _MockAioHTTPResponse(status=308),
                    _MockAioHTTPResponse(
                        status=200, headers={"x-goog-hash": f"md5={md5_b64}"}
                    ),
                ],
            ):
                with patch(
                    "neuracore.data_daemon.upload_management"
                    ".bandwidth_limiter.asyncio.sleep",
                    side_effect=capture_sleep,
                ):
                    success, _, _ = await uploader.upload()

        assert success is True
        assert len(sleep_calls) == 0

    @pytest.mark.asyncio
    async def test_throttle_calculates_correct_delay(self) -> None:
        """Test that BandwidthLimiter.acquire sleeps for the correct duration."""
        import time as real_time

        from neuracore.data_daemon.upload_management.bandwidth_limiter import (
            BandwidthLimiter,
        )

        bandwidth = 1000  # bytes/sec
        sleep_calls: list[float] = []
        monotonic_time = [real_time.monotonic()]

        def fake_monotonic() -> float:
            return monotonic_time[0]

        async def fake_sleep(duration: float) -> None:
            if duration > 0:
                monotonic_time[0] += duration
                sleep_calls.append(duration)

        with (
            patch(
                "neuracore.data_daemon.upload_management"
                ".bandwidth_limiter.time.monotonic",
                new=fake_monotonic,
            ),
            patch(
                "neuracore.data_daemon.upload_management"
                ".bandwidth_limiter.asyncio.sleep",
                new=fake_sleep,
            ),
        ):
            limiter = BandwidthLimiter(bandwidth)
            await limiter.acquire(2000)

        assert len(sleep_calls) == 1
        assert abs(sleep_calls[0] - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_throttle_no_delay_when_tokens_available(self) -> None:
        """Test that no delay is added when the token bucket has sufficient tokens."""
        from neuracore.data_daemon.upload_management.bandwidth_limiter import (
            BandwidthLimiter,
        )

        bandwidth = 1000
        sleep_calls: list[float] = []

        async def fake_sleep(duration: float) -> None:
            if duration > 0:
                sleep_calls.append(duration)

        with patch(
            "neuracore.data_daemon.upload_management"
            ".bandwidth_limiter.asyncio.sleep",
            new=fake_sleep,
        ):
            limiter = BandwidthLimiter(bandwidth)
            await limiter.acquire(500)

        assert len(sleep_calls) == 0
