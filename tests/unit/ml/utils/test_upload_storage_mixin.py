"""Tests for the chunked/resumable upload logic in UploadStorageMixin."""

import pytest

from neuracore.ml.utils import upload_storage_mixin
from neuracore.ml.utils.upload_storage_mixin import UploadStorageMixin

UPLOAD_URL = "https://storage.example.com/session-1"
REFRESHED_UPLOAD_URL = "https://storage.example.com/session-2"


class _FakeHandler(UploadStorageMixin):
    """Minimal concrete UploadStorageMixin for exercising the chunk loop."""

    def __init__(self, upload_url: str = UPLOAD_URL):
        self.log_to_cloud = True
        self._upload_url = upload_url
        self.get_upload_url_calls = 0

    def _get_upload_url(self, filepath: str, content_type: str) -> str:
        self.get_upload_url_calls += 1
        return self._upload_url


@pytest.fixture(autouse=True)
def _no_real_sleeps(monkeypatch):
    """Skip real backoff sleeps so retry tests run fast."""
    monkeypatch.setattr(upload_storage_mixin.time, "sleep", lambda *_: None)


@pytest.fixture(autouse=True)
def _small_chunk_size(monkeypatch):
    """Shrink CHUNK_SIZE so small test payloads still span multiple chunks."""
    monkeypatch.setattr(upload_storage_mixin, "CHUNK_SIZE", 4)


class TestSingleChunkUpload:
    def test_payload_smaller_than_one_chunk_sends_exactly_one_put(self, requests_mock):
        requests_mock.put(UPLOAD_URL, status_code=200)
        handler = _FakeHandler()

        result = handler.upload_bytes(b"ab", "dest.bin")

        assert result is True
        put_requests = [r for r in requests_mock.request_history if r.method == "PUT"]
        assert len(put_requests) == 1
        assert put_requests[0].headers["Content-Range"] == "bytes 0-1/2"


class TestMultiChunkUpload:
    def test_large_payload_is_split_into_multiple_chunks_in_order(self, requests_mock):
        requests_mock.put(UPLOAD_URL, status_code=200)
        handler = _FakeHandler()
        payload = b"0123456789AB"  # 12 bytes / CHUNK_SIZE=4 -> 3 chunks

        result = handler.upload_bytes(payload, "dest.bin")

        assert result is True
        put_requests = [r for r in requests_mock.request_history if r.method == "PUT"]
        assert len(put_requests) == 3
        assert [r.headers["Content-Range"] for r in put_requests] == [
            "bytes 0-3/*",
            "bytes 4-7/*",
            "bytes 8-11/12",
        ]
        # Bytes actually sent reassemble to the original payload, in order.
        assert b"".join(r.body for r in put_requests) == payload

    def test_308_partial_commit_resumes_from_server_reported_offset(
        self, requests_mock
    ):
        # First chunk (bytes 0-3) is only partially committed by the server
        # (2 of 4 bytes); the client must resume at byte 2, not byte 4.
        requests_mock.put(
            UPLOAD_URL,
            [
                {"status_code": 308, "headers": {"Range": "bytes=0-1"}},
                {"status_code": 200},
                {"status_code": 200},
            ],
        )
        handler = _FakeHandler()
        payload = b"0123456789AB"

        result = handler.upload_bytes(payload, "dest.bin")

        assert result is True
        put_requests = [r for r in requests_mock.request_history if r.method == "PUT"]
        # attempt 1: 0-3 (partial commit -> resume at 2), attempt 2: 2-5,
        # attempt 3: 6-9 (still capped at CHUNK_SIZE=4), attempt 4: 10-11 (final)
        assert [r.headers["Content-Range"] for r in put_requests] == [
            "bytes 0-3/*",
            "bytes 2-5/*",
            "bytes 6-9/*",
            "bytes 10-11/12",
        ]


class TestSessionExpiry:
    @pytest.mark.parametrize("expired_status", [404, 410])
    def test_expired_session_fetches_new_url_and_restarts_from_zero(
        self, requests_mock, expired_status
    ):
        handler = _FakeHandler()
        urls = [UPLOAD_URL, REFRESHED_UPLOAD_URL]

        def _get_url(filepath: str, content_type: str) -> str:
            handler.get_upload_url_calls += 1
            return urls[handler.get_upload_url_calls - 1]

        handler._get_upload_url = _get_url
        requests_mock.put(UPLOAD_URL, status_code=expired_status)
        requests_mock.put(REFRESHED_UPLOAD_URL, status_code=200)

        result = handler.upload_bytes(b"ab", "dest.bin")

        assert result is True
        assert handler.get_upload_url_calls == 2
        expired_puts = [
            r
            for r in requests_mock.request_history
            if r.method == "PUT" and r.url == UPLOAD_URL
        ]
        refreshed_puts = [
            r
            for r in requests_mock.request_history
            if r.method == "PUT" and r.url == REFRESHED_UPLOAD_URL
        ]
        assert len(expired_puts) == 1
        assert len(refreshed_puts) == 1


class TestRetryExhaustion:
    def test_persistent_failure_returns_false_without_raising(self, requests_mock):
        requests_mock.put(UPLOAD_URL, status_code=500, text="Server Error")
        handler = _FakeHandler()

        result = handler.upload_bytes(b"ab", "dest.bin")

        assert result is False
        put_requests = [r for r in requests_mock.request_history if r.method == "PUT"]
        assert len(put_requests) == upload_storage_mixin._MAX_CHUNK_RETRIES

    def test_connection_error_during_send_is_retried_then_gives_up(self, requests_mock):
        import requests as requests_lib

        requests_mock.put(UPLOAD_URL, exc=requests_lib.exceptions.ConnectionError)
        handler = _FakeHandler()

        result = handler.upload_bytes(b"ab", "dest.bin")

        assert result is False


class TestNotLoggingToCloud:
    def test_returns_false_and_makes_no_requests_when_cloud_logging_disabled(
        self, requests_mock
    ):
        handler = _FakeHandler()
        handler.log_to_cloud = False

        result = handler.upload_bytes(b"ab", "dest.bin")

        assert result is False
        assert len(requests_mock.request_history) == 0


class TestUploadFile:
    def test_upload_file_reads_from_disk_and_chunks_it(self, tmp_path, requests_mock):
        requests_mock.put(UPLOAD_URL, status_code=200)
        handler = _FakeHandler()
        local_path = tmp_path / "checkpoint.pt"
        local_path.write_bytes(b"0123456789AB")

        result = handler.upload_file(local_path, "checkpoints/checkpoint.pt")

        assert result is True
        put_requests = [r for r in requests_mock.request_history if r.method == "PUT"]
        assert len(put_requests) == 3

    def test_upload_file_returns_false_when_local_file_missing(self, tmp_path):
        handler = _FakeHandler()

        result = handler.upload_file(tmp_path / "missing.pt", "checkpoints/x.pt")

        assert result is False
