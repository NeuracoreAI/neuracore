"""Tests for ResumableFileUploader.

Tests chunked file uploads, resumable sessions, retry logic, and error handling.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
import pytest_asyncio

from neuracore.data_daemon.upload_management.resumable_file_uploader import (
    ResumableFileUploader,
)


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
    with patch(
        "neuracore.data_daemon.upload_management.resumable_file_uploader.get_auth"
    ) as mock_get_auth:
        auth_instance = MagicMock()
        auth_instance.get_org_id.return_value = "test-org"
        auth_instance.get_headers.return_value = {"Authorization": "Bearer test-token"}
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
    def __init__(self, *, status: int, json_data=None, exc: Exception | None = None):
        self.status = status
        self._json_data = json_data
        self._exc = exc

    async def __aenter__(self):
        if self._exc is not None:
            raise self._exc
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=self.status,
                message="error",
                headers={},
            )

    async def json(self):
        return self._json_data


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
    uploader: ResumableFileUploader,
) -> None:
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
            return_value=_MockAioHTTPResponse(status=200),
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

    def progress_callback(bytes_delta: int) -> None:
        progress_updates.append(bytes_delta)

    uploader = ResumableFileUploader(
        recording_id="rec-123",
        filepath=str(test_file),
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
        with patch.object(
            uploader._session,
            "put",
            return_value=_MockAioHTTPResponse(status=200),
        ):
            await uploader.upload()

    assert len(progress_updates) > 0
    assert sum(progress_updates) == 5 * 1024 * 1024


@pytest.mark.asyncio
async def test_uploader_resumes_from_offset(
    large_test_file: Path, mock_auth, client_session: aiohttp.ClientSession
) -> None:
    uploader = ResumableFileUploader(
        recording_id="rec-123",
        filepath=str(large_test_file),
        cloud_filepath="RGB_IMAGES/camera/trace.mp4",
        content_type="video/mp4",
        client_session=client_session,
        bytes_uploaded=5 * 1024 * 1024,
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
            return_value=_MockAioHTTPResponse(status=200),
        ) as mock_put:
            success, bytes_uploaded, error_message = await uploader.upload()

    assert success is True
    assert bytes_uploaded == 10 * 1024 * 1024
    first_put_call = mock_put.call_args_list[0]
    content_range = first_put_call.kwargs["headers"]["Content-Range"]
    assert content_range.startswith("bytes 5242880-")


@pytest.mark.asyncio
async def test_uploader_handles_session_expiration(
    uploader: ResumableFileUploader,
) -> None:
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
                _MockAioHTTPResponse(status=410),
                _MockAioHTTPResponse(status=200),
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
            return_value=_MockAioHTTPResponse(
                status=0, exc=aiohttp.ClientConnectorError(MagicMock(), OSError("boom"))
            ),
        ):
            success, bytes_uploaded, error_message = await uploader.upload()

    assert success is False
    assert error_message is not None
    assert "Network connection error" in error_message


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
async def test_uploader_retries_on_timeout(uploader: ResumableFileUploader) -> None:
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
                _MockAioHTTPResponse(status=0, exc=asyncio.TimeoutError()),
                _MockAioHTTPResponse(status=0, exc=asyncio.TimeoutError()),
                _MockAioHTTPResponse(status=200),
            ],
        ) as mock_put:

            async def _sleep(_: float) -> None:
                return None

            with patch("asyncio.sleep", side_effect=_sleep):
                success, bytes_uploaded, error_message = await uploader.upload()

    assert success is True
    assert mock_put.call_count == 3


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
        with patch.object(
            uploader._session,
            "put",
            side_effect=[_MockAioHTTPResponse(status=0, exc=asyncio.TimeoutError())]
            * ResumableFileUploader.MAX_RETRIES,
        ) as mock_put:

            async def _sleep(_: float) -> None:
                return None

            with patch("asyncio.sleep", side_effect=_sleep):
                success, bytes_uploaded, error_message = await uploader.upload()

    assert success is False
    assert error_message is not None
    assert "failed after" in error_message
    assert mock_put.call_count == ResumableFileUploader.MAX_RETRIES


@pytest.mark.asyncio
async def test_uploader_handles_http_errors(uploader: ResumableFileUploader) -> None:
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
            return_value=_MockAioHTTPResponse(status=500),
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
    uploader: ResumableFileUploader,
) -> None:
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
            return_value=_MockAioHTTPResponse(status=200),
        ) as mock_put:
            await uploader.upload()

    last_put_call = mock_put.call_args_list[-1]
    headers = last_put_call.kwargs["headers"]
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

    def progress_callback(bytes_delta: int) -> None:
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
            if put_side_effect.calls < 3:
                put_side_effect.calls += 1
                return _MockAioHTTPResponse(status=308)
            return _MockAioHTTPResponse(status=200)

        put_side_effect.calls = 0

        with patch.object(uploader._session, "put", side_effect=put_side_effect):
            success, bytes_uploaded, error_message = await uploader.upload()

    assert success is True
    assert bytes_uploaded == 200 * 1024 * 1024
    assert len(progress_updates) == 4
    assert sum(progress_updates) == 200 * 1024 * 1024


@pytest.mark.asyncio
async def test_uploader_exponential_backoff(uploader: ResumableFileUploader) -> None:
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
                _MockAioHTTPResponse(status=0, exc=asyncio.TimeoutError()),
                _MockAioHTTPResponse(status=0, exc=asyncio.TimeoutError()),
                _MockAioHTTPResponse(status=200),
            ],
        ):
            sleep_calls: list[float] = []

            async def _sleep(t: float) -> None:
                sleep_calls.append(t)

            with patch("asyncio.sleep", side_effect=_sleep):
                await uploader.upload()

    assert sleep_calls[:2] == [1, 2]
