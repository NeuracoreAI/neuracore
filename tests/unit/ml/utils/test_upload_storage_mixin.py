"""Tests for the multipart upload helpers in UploadStorageMixin."""

from unittest.mock import MagicMock

import pytest

from neuracore.core.const import API_URL
from neuracore.core.utils import multipart_upload
from neuracore.ml.utils.upload_storage_mixin import UploadStorageMixin

ORG_ID = "test-org-id"
BASE_URL = f"{API_URL}/v2/org/{ORG_ID}"
START_URL = f"{BASE_URL}/training/jobs/job-1/uploads"
PART_URL = "https://storage.example.com/part-1"


class _FakeHandler(UploadStorageMixin):
    """Minimal concrete UploadStorageMixin with a fixed start route."""

    def __init__(self) -> None:
        self.log_to_cloud = True
        self.org_id = ORG_ID

    def _upload_start_url(self) -> str:
        return START_URL


@pytest.fixture(autouse=True)
def _auth(monkeypatch):
    auth = MagicMock()
    auth.get_headers.return_value = {"Authorization": "Bearer test-token"}
    monkeypatch.setattr(multipart_upload, "get_auth", lambda: auth)


@pytest.fixture
def upload_routes(requests_mock):
    """Register a one-part upload on the start, part and complete routes."""
    return {
        "start": requests_mock.post(
            START_URL,
            json={
                "upload_id": "upload-1",
                "part_size_bytes": 16 * 1024 * 1024,
                "part_urls": {"1": PART_URL},
            },
        ),
        "put": requests_mock.put(PART_URL, status_code=200),
        "complete": requests_mock.post(f"{BASE_URL}/uploads/complete", status_code=204),
    }


class TestUploadBytes:
    def test_uploads_to_the_remote_filepath_through_the_v2_routes(self, upload_routes):
        result = _FakeHandler().upload_bytes(b"ab", "logs/train.log", "text/plain")

        assert result is True
        assert upload_routes["start"].last_request.json() == {
            "filepath": "logs/train.log",
            "content_type": "text/plain",
            "size_bytes": 2,
        }
        assert upload_routes["put"].last_request.body == b"ab"
        assert upload_routes["complete"].call_count == 1

    def test_failed_upload_returns_false_without_raising(self, requests_mock):
        requests_mock.post(START_URL, status_code=404, json={"detail": "Not found"})

        result = _FakeHandler().upload_bytes(b"ab", "dest.bin")

        assert result is False

    def test_returns_false_and_makes_no_requests_when_cloud_logging_disabled(
        self, requests_mock
    ):
        handler = _FakeHandler()
        handler.log_to_cloud = False

        result = handler.upload_bytes(b"ab", "dest.bin")

        assert result is False
        assert len(requests_mock.request_history) == 0


class TestUploadFile:
    def test_upload_file_reads_from_disk(self, tmp_path, upload_routes):
        local_path = tmp_path / "checkpoint.pt"
        local_path.write_bytes(b"0123456789AB")

        result = _FakeHandler().upload_file(local_path, "checkpoints/checkpoint.pt")

        assert result is True
        assert upload_routes["put"].last_request.body == b"0123456789AB"

    def test_upload_file_returns_false_when_local_file_missing(self, tmp_path):
        result = _FakeHandler().upload_file(tmp_path / "missing.pt", "checkpoints/x.pt")

        assert result is False
