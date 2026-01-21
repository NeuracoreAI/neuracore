"""Tests for ResumableFileUploader with a local GCS-like server."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest
import requests

from neuracore.data_daemon.upload_management.resumable_file_uploader import (
    ResumableFileUploader,
)


@dataclass
class RequestInfo:
    method: str
    path: str
    content_length: int
    content_range: str
    is_status_check: bool
    is_finalize: bool
    is_final_chunk: bool
    session_id: str | None


@dataclass
class ResponseAction:
    status: int
    headers: dict[str, str] | None = None
    body: bytes | None = None
    drop: bool = False


class UploadSession:
    def __init__(self) -> None:
        self.uploaded_bytes = 0
        self.total_bytes: int | None = None
        self.data = bytearray()
        self.finalized = False


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        text: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._json_body = json_body
        self.text = text or ""

    def json(self) -> dict[str, Any]:
        if self._json_body is None:
            raise ValueError("No JSON body")
        return self._json_body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeResumableTransport:
    def __init__(self) -> None:
        self.sessions: dict[str, UploadSession] = {}
        self.session_count = 0
        self.last_session_id: str | None = None
        self.request_log: list[dict[str, Any]] = []
        self.pre_request: callable | None = None
        self.post_store: callable | None = None
        self.base_url = "http://fake"

    def _create_session(self) -> str:
        self.session_count += 1
        session_id = f"sess-{self.session_count}"
        self.sessions[session_id] = UploadSession()
        self.last_session_id = session_id
        return session_id

    def _record_request(self, info: RequestInfo) -> None:
        self.request_log.append({
            "method": info.method,
            "path": info.path,
            "content_length": info.content_length,
            "content_range": info.content_range,
            "is_status_check": info.is_status_check,
            "is_finalize": info.is_finalize,
            "is_final_chunk": info.is_final_chunk,
            "session_id": info.session_id,
        })

    def get(
        self, url: str, params: dict[str, Any] | None = None, **_: Any
    ) -> FakeResponse:
        parsed = urlparse(url)
        if parsed.path.endswith("/resumable_upload_url"):
            session_id = self._create_session()
            session_url = f"{self.base_url}/upload/{session_id}"
            return FakeResponse(200, json_body={"url": session_url})
        return FakeResponse(404)

    def put(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        data: bytes | None = None,
        **_: Any,
    ) -> FakeResponse:
        parsed = urlparse(url)
        if not parsed.path.startswith("/upload/"):
            return FakeResponse(404)

        session_id = parsed.path.split("/")[-1]
        session = self.sessions.get(session_id)
        if session is None:
            return FakeResponse(404)

        headers = headers or {}
        content_length = int(headers.get("Content-Length", "0"))
        content_range = headers.get("Content-Range", "")
        is_status_check = content_length == 0 and content_range.startswith("bytes */")
        is_finalize = (
            is_status_check
            and session.total_bytes is not None
            and content_range == f"bytes */{session.total_bytes}"
        )
        is_final_chunk = "/" in content_range and content_range.split("/")[-1] != "*"

        info = RequestInfo(
            method="PUT",
            path=parsed.path,
            content_length=content_length,
            content_range=content_range,
            is_status_check=is_status_check,
            is_finalize=is_finalize,
            is_final_chunk=is_final_chunk,
            session_id=session_id,
        )
        self._record_request(info)

        if self.pre_request:
            action = self.pre_request(info)
            if action:
                if action.drop:
                    raise requests.exceptions.ConnectionError("Dropped connection")
                return FakeResponse(
                    action.status,
                    headers=action.headers,
                    text=action.body.decode() if action.body else "",
                )

        if is_status_check:
            if is_finalize and session.total_bytes is not None:
                if session.uploaded_bytes >= session.total_bytes:
                    session.finalized = True
                    md5_hash = hashlib.md5(session.data).hexdigest()
                    return FakeResponse(200, headers={"X-Checksum-MD5": md5_hash})
            if session.finalized:
                return FakeResponse(200)
            headers_out: dict[str, str] = {}
            if session.uploaded_bytes > 0:
                headers_out["Range"] = f"bytes=0-{session.uploaded_bytes - 1}"
            return FakeResponse(308, headers=headers_out)

        data = data or b""
        try:
            _, range_spec = content_range.split(" ", 1)
            range_part, total_part = range_spec.split("/")
            range_start, range_end = range_part.split("-")
            start = int(range_start)
            end = int(range_end)
            total = None if total_part == "*" else int(total_part)
        except ValueError:
            return FakeResponse(400)

        if total is not None:
            session.total_bytes = total

        if start != session.uploaded_bytes:
            headers_out: dict[str, str] = {}
            if session.uploaded_bytes > 0:
                headers_out["Range"] = f"bytes=0-{session.uploaded_bytes - 1}"
            return FakeResponse(308, headers=headers_out)

        session.data.extend(data)
        session.uploaded_bytes += len(data)

        if self.post_store:
            action = self.post_store(info, session)
            if action:
                if action.drop:
                    raise requests.exceptions.ConnectionError("Dropped connection")
                return FakeResponse(
                    action.status,
                    headers=action.headers,
                    text=action.body.decode() if action.body else "",
                )

        if (
            session.total_bytes is not None
            and session.uploaded_bytes >= session.total_bytes
        ):
            session.finalized = True
            md5_hash = hashlib.md5(session.data).hexdigest()
            return FakeResponse(200, headers={"X-Checksum-MD5": md5_hash})

        if end >= session.uploaded_bytes:
            headers_out = {"Range": f"bytes=0-{session.uploaded_bytes - 1}"}
            return FakeResponse(308, headers=headers_out)

        return FakeResponse(200)


@pytest.fixture
def transport(monkeypatch) -> FakeResumableTransport:
    fake_transport = FakeResumableTransport()
    monkeypatch.setattr(
        "neuracore.data_daemon.upload_management.resumable_file_uploader.requests.get",
        fake_transport.get,
    )
    monkeypatch.setattr(
        "neuracore.data_daemon.upload_management.resumable_file_uploader.requests.put",
        fake_transport.put,
    )
    return fake_transport


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
def test_file(tmp_path: Path) -> Path:
    test_file = tmp_path / "test_video.mp4"
    test_file.write_bytes(b"X" * (3 * 1024 * 1024))
    return test_file


def _make_uploader(test_file: Path) -> ResumableFileUploader:
    return ResumableFileUploader(
        recording_id="rec-123",
        filepath=str(test_file),
        cloud_filepath="RGB_IMAGES/camera/trace.mp4",
        content_type="video/mp4",
        bytes_uploaded=0,
    )


def _set_api_url(monkeypatch, transport: FakeResumableTransport) -> None:
    monkeypatch.setattr(
        "neuracore.data_daemon.upload_management.resumable_file_uploader.API_URL",
        transport.base_url,
    )


def test_successful_upload_and_checksum_verification(
    transport: FakeResumableTransport, mock_auth, test_file: Path, monkeypatch
) -> None:
    _set_api_url(monkeypatch, transport)
    monkeypatch.setattr(ResumableFileUploader, "CHUNK_SIZE", 1024 * 1024)

    uploader = _make_uploader(test_file)
    success, bytes_uploaded, error_message = uploader.upload()

    assert success is True
    assert bytes_uploaded == 3 * 1024 * 1024
    assert error_message is None
    session = transport.sessions[transport.last_session_id]  # type: ignore[index]
    assert session.finalized is True


def test_resume_after_interruption_no_duplicate_data(
    transport: FakeResumableTransport, mock_auth, test_file: Path, monkeypatch
) -> None:
    _set_api_url(monkeypatch, transport)
    monkeypatch.setattr(ResumableFileUploader, "CHUNK_SIZE", 1024 * 1024)

    drop_once = {"done": False}

    def pre_request(info: RequestInfo) -> ResponseAction | None:
        if info.is_status_check:
            return None
        if not drop_once["done"] and info.content_range.startswith("bytes 1048576-"):
            drop_once["done"] = True
            return ResponseAction(status=500, drop=True)
        return None

    transport.pre_request = pre_request

    uploader = _make_uploader(test_file)
    success, bytes_uploaded, _ = uploader.upload()

    assert success is True
    assert bytes_uploaded == 3 * 1024 * 1024
    retry_ranges = [
        entry["content_range"]
        for entry in transport.request_log
        if entry["content_range"].startswith("bytes 1048576-")
    ]
    assert len(retry_ranges) >= 1
    session = transport.sessions[transport.last_session_id]  # type: ignore[index]
    assert session.uploaded_bytes == 3 * 1024 * 1024
    assert len(session.data) == 3 * 1024 * 1024
    assert transport.session_count == 1


def test_resumable_session_preserved_on_retry(
    transport: FakeResumableTransport, mock_auth, test_file: Path, monkeypatch
) -> None:
    _set_api_url(monkeypatch, transport)
    monkeypatch.setattr(ResumableFileUploader, "CHUNK_SIZE", 1024 * 1024)

    failures = {"count": 0}

    def pre_request(info: RequestInfo) -> ResponseAction | None:
        if info.is_status_check:
            return None
        if failures["count"] < 2:
            failures["count"] += 1
            return ResponseAction(status=503)
        return None

    transport.pre_request = pre_request

    uploader = _make_uploader(test_file)
    success, bytes_uploaded, _ = uploader.upload()

    assert success is True
    assert bytes_uploaded == 3 * 1024 * 1024
    assert transport.session_count == 1


def test_session_expiration_refreshes_session(
    transport: FakeResumableTransport, mock_auth, test_file: Path, monkeypatch
) -> None:
    _set_api_url(monkeypatch, transport)
    monkeypatch.setattr(ResumableFileUploader, "CHUNK_SIZE", 1024 * 1024)

    expired_once = {"done": False}

    def pre_request(info: RequestInfo) -> ResponseAction | None:
        if info.is_status_check:
            return None
        if not expired_once["done"]:
            expired_once["done"] = True
            return ResponseAction(status=410)
        return None

    transport.pre_request = pre_request

    uploader = _make_uploader(test_file)
    success, bytes_uploaded, _ = uploader.upload()

    assert success is True
    assert bytes_uploaded == 3 * 1024 * 1024
    assert transport.session_count == 2
    assert mock_auth.call_count >= 2


def test_signed_url_expired_reacquires_session(
    transport: FakeResumableTransport, mock_auth, test_file: Path, monkeypatch
) -> None:
    _set_api_url(monkeypatch, transport)
    monkeypatch.setattr(ResumableFileUploader, "CHUNK_SIZE", 1024 * 1024)

    expired_once = {"done": False}

    def pre_request(info: RequestInfo) -> ResponseAction | None:
        if info.is_status_check:
            return None
        if not expired_once["done"]:
            expired_once["done"] = True
            return ResponseAction(status=403, headers={"X-Signed-Url-Expired": "true"})
        return None

    transport.pre_request = pre_request

    uploader = _make_uploader(test_file)
    success, bytes_uploaded, _ = uploader.upload()

    assert success is True
    assert bytes_uploaded == 3 * 1024 * 1024
    assert transport.session_count == 2
    assert mock_auth.call_count >= 2


def test_permission_denied_fails_fast(
    transport: FakeResumableTransport, mock_auth, test_file: Path, monkeypatch
) -> None:
    _set_api_url(monkeypatch, transport)
    monkeypatch.setattr(ResumableFileUploader, "CHUNK_SIZE", 1024 * 1024)

    def pre_request(info: RequestInfo) -> ResponseAction | None:
        if info.is_status_check:
            return None
        return ResponseAction(status=403)

    transport.pre_request = pre_request

    uploader = _make_uploader(test_file)
    success, _, error_message = uploader.upload()

    assert success is False
    assert "Permission denied" in (error_message or "")
    assert transport.session_count == 1


def test_bucket_not_found_fails_fast(
    transport: FakeResumableTransport, mock_auth, test_file: Path, monkeypatch
) -> None:
    _set_api_url(monkeypatch, transport)
    monkeypatch.setattr(ResumableFileUploader, "CHUNK_SIZE", 1024 * 1024)

    def pre_request(info: RequestInfo) -> ResponseAction | None:
        if info.is_status_check:
            return None
        return ResponseAction(status=404)

    transport.pre_request = pre_request

    uploader = _make_uploader(test_file)
    success, _, error_message = uploader.upload()

    assert success is False
    assert "Bucket not found" in (error_message or "")


def test_retry_on_5xx_with_backoff(
    transport: FakeResumableTransport, mock_auth, test_file: Path, monkeypatch
) -> None:
    _set_api_url(monkeypatch, transport)
    monkeypatch.setattr(ResumableFileUploader, "CHUNK_SIZE", 1024 * 1024)

    failures = {"count": 0}

    def pre_request(info: RequestInfo) -> ResponseAction | None:
        if info.is_status_check:
            return None
        if failures["count"] < 2:
            failures["count"] += 1
            return ResponseAction(status=500)
        return None

    transport.pre_request = pre_request

    with patch("time.sleep") as mock_sleep:
        uploader = _make_uploader(test_file)
        success, _, _ = uploader.upload()

    assert success is True
    mock_sleep.assert_any_call(1)
    mock_sleep.assert_any_call(2)


def test_retry_on_429_with_backoff(
    transport: FakeResumableTransport, mock_auth, test_file: Path, monkeypatch
) -> None:
    _set_api_url(monkeypatch, transport)
    monkeypatch.setattr(ResumableFileUploader, "CHUNK_SIZE", 1024 * 1024)

    failures = {"count": 0}

    def pre_request(info: RequestInfo) -> ResponseAction | None:
        if info.is_status_check:
            return None
        if failures["count"] < 2:
            failures["count"] += 1
            return ResponseAction(status=429)
        return None

    transport.pre_request = pre_request

    with patch("time.sleep") as mock_sleep:
        uploader = _make_uploader(test_file)
        success, _, _ = uploader.upload()

    assert success is True
    mock_sleep.assert_any_call(1)
    mock_sleep.assert_any_call(2)


def test_finalization_retry_without_reupload(
    transport: FakeResumableTransport, mock_auth, test_file: Path, monkeypatch
) -> None:
    _set_api_url(monkeypatch, transport)
    monkeypatch.setattr(ResumableFileUploader, "CHUNK_SIZE", 1024 * 1024)

    fail_finalize_once = {"done": False}

    def post_store(info: RequestInfo, session: UploadSession) -> ResponseAction | None:
        if (
            info.is_final_chunk
            and not info.is_status_check
            and not fail_finalize_once["done"]
        ):
            fail_finalize_once["done"] = True
            return ResponseAction(status=500)
        return None

    transport.post_store = post_store

    uploader = _make_uploader(test_file)
    success, bytes_uploaded, _ = uploader.upload()

    assert success is True
    assert bytes_uploaded == 3 * 1024 * 1024
    session = transport.sessions[transport.last_session_id]  # type: ignore[index]
    assert len(session.data) == 3 * 1024 * 1024
    finalize_calls = [entry for entry in transport.request_log if entry["is_finalize"]]
    assert len(finalize_calls) >= 1


def test_bucket_unreachable_at_start(mock_auth, test_file: Path) -> None:
    uploader = _make_uploader(test_file)
    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError("Unreachable")
        success, _, error_message = uploader.upload()
    assert success is False
    assert "Failed to get upload session URI" in (error_message or "")


def test_dns_resolution_failure(mock_auth, test_file: Path) -> None:
    uploader = _make_uploader(test_file)
    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError("DNS failure")
        success, _, error_message = uploader.upload()
    assert success is False
    assert "Failed to get upload session URI" in (error_message or "")


def test_ssl_handshake_failure_on_upload(
    transport: FakeResumableTransport, mock_auth, test_file: Path, monkeypatch
) -> None:
    _set_api_url(monkeypatch, transport)
    monkeypatch.setattr(ResumableFileUploader, "CHUNK_SIZE", 1024 * 1024)

    def pre_request(info: RequestInfo) -> ResponseAction | None:
        if info.is_status_check:
            return None
        return ResponseAction(status=200)

    transport.pre_request = pre_request

    with patch("requests.put") as mock_put:
        mock_put.side_effect = requests.exceptions.SSLError("TLS failure")
        uploader = _make_uploader(test_file)
        success, _, error_message = uploader.upload()

    assert success is False
    assert "SSL error" in (error_message or "")


def test_max_retries_exceeded(
    transport: FakeResumableTransport, mock_auth, test_file: Path, monkeypatch
) -> None:
    _set_api_url(monkeypatch, transport)
    monkeypatch.setattr(ResumableFileUploader, "CHUNK_SIZE", 1024 * 1024)

    def pre_request(info: RequestInfo) -> ResponseAction | None:
        if info.is_status_check:
            return None
        return ResponseAction(status=500)

    transport.pre_request = pre_request

    uploader = _make_uploader(test_file)
    success, _, error_message = uploader.upload()

    assert success is False
    assert "failed after" in (error_message or "")


def test_backoff_caps_at_five_minutes() -> None:
    uploader = ResumableFileUploader(
        recording_id="rec-123",
        filepath="/tmp/does-not-matter",
        cloud_filepath="trace.mp4",
        content_type="video/mp4",
    )
    with patch("time.sleep") as mock_sleep:
        uploader._sleep_backoff(10)
    mock_sleep.assert_called_once_with(300)


def test_resume_from_server_offset(
    transport: FakeResumableTransport, mock_auth, test_file: Path, monkeypatch
) -> None:
    _set_api_url(monkeypatch, transport)
    monkeypatch.setattr(ResumableFileUploader, "CHUNK_SIZE", 1024 * 1024)

    primed = {"done": False}

    def pre_request(info: RequestInfo) -> ResponseAction | None:
        if info.is_status_check and not primed["done"]:
            primed["done"] = True
            session = transport.sessions[info.session_id]  # type: ignore[index]
            session.data.extend(b"X" * (1024 * 1024))
            session.uploaded_bytes = 1024 * 1024
            return None
        return None

    transport.pre_request = pre_request

    uploader = _make_uploader(test_file)
    success, bytes_uploaded, _ = uploader.upload()

    assert success is True
    assert bytes_uploaded == 3 * 1024 * 1024
    upload_starts = [
        entry["content_range"]
        for entry in transport.request_log
        if entry["content_range"].startswith("bytes 1048576-")
    ]
    assert upload_starts
