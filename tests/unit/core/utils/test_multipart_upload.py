"""Tests for MultipartUploader."""

import io
import re
from unittest.mock import MagicMock

import pytest
import requests

from neuracore.core.utils import multipart_upload
from neuracore.core.utils.multipart_upload import MultipartUploader, UploadError

MIB = 1024 * 1024
BASE_URL = "https://api.example.com/api/v2/org/org-1"
START_URL = f"{BASE_URL}/training/jobs/job-1/uploads"
PARTS_URL = f"{BASE_URL}/uploads/parts"
COMPLETE_URL = f"{BASE_URL}/uploads/complete"
STORE_URLS = re.compile(r"https://store\.example\.com/")
CHECKSUM_MISMATCH = {
    "detail": {
        "error": "Uploaded file does not match its checksum.",
        "status": 409,
        "error_type": "CHECKSUM_MISMATCH",
    }
}

GCS_EXPIRED_TOKEN = (
    "<?xml version='1.0' encoding='UTF-8'?><Error><Code>ExpiredToken</Code>"
    "<Message>The provided token has expired.</Message>"
    "<Details>Request signature expired at: 2026-11-19T20:00:00+00:00</Details>"
    "</Error>"
)


def part_url(upload_id: str, part_number: int, signature: str = "first") -> str:
    return f"https://store.example.com/{signature}/{upload_id}/{part_number}"


def started(upload_id: str, part_size_bytes: int, part_count: int) -> dict:
    return {
        "upload_id": upload_id,
        "part_size_bytes": part_size_bytes,
        "part_urls": {
            str(part_number): part_url(upload_id, part_number)
            for part_number in range(1, min(part_count, 8) + 1)
        },
    }


def sign_parts(request, context) -> dict:
    upload = request.json()["uploads"][0]
    return {
        "uploads": [{
            "upload_id": upload["upload_id"],
            "part_urls": {
                str(part_number): part_url(upload["upload_id"], part_number, "fresh")
                for part_number in upload["part_numbers"]
            },
            "uploaded_parts": [],
        }],
        "missing_upload_ids": [],
    }


def upload(data: bytes) -> None:
    MultipartUploader(BASE_URL).upload(
        START_URL, io.BytesIO(data), "text/plain", {"filepath": "logs/train.log"}
    )


@pytest.fixture(autouse=True)
def _auth(monkeypatch):
    auth = MagicMock()
    auth.get_headers.return_value = {"Authorization": "Bearer test-token"}
    monkeypatch.setattr(multipart_upload, "get_auth", lambda: auth)


@pytest.fixture(autouse=True)
def _no_real_sleeps(monkeypatch):
    monkeypatch.setattr(multipart_upload.time, "sleep", lambda *_: None)


def test_parts_are_sent_in_order_and_completed_with_the_file_crc32c(requests_mock):
    data = b"123456789"
    start = requests_mock.post(START_URL, json=started("upload-1", 1, len(data)))
    parts = requests_mock.post(PARTS_URL, json=sign_parts)
    puts = requests_mock.put(STORE_URLS, status_code=200)
    complete = requests_mock.post(COMPLETE_URL, status_code=204)

    upload(data)

    assert start.last_request.json() == {
        "filepath": "logs/train.log",
        "content_type": "text/plain",
        "size_bytes": 9,
    }
    assert parts.last_request.json() == {
        "uploads": [{"upload_id": "upload-1", "part_numbers": [9]}]
    }
    assert b"".join(put.body for put in puts.request_history) == data
    assert all("Authorization" not in put.headers for put in puts.request_history)
    assert complete.last_request.json() == {
        "upload_id": "upload-1",
        "crc32c": "4waSgw==",
    }


def test_empty_file_is_one_empty_part(requests_mock):
    start = requests_mock.post(START_URL, json=started("upload-1", 16 * MIB, 1))
    puts = requests_mock.put(STORE_URLS, status_code=200)
    complete = requests_mock.post(COMPLETE_URL, status_code=204)

    upload(b"")

    assert start.last_request.json()["size_bytes"] == 0
    assert puts.call_count == 1
    assert puts.last_request.headers["Content-Length"] == "0"
    assert complete.last_request.json()["crc32c"] == "AAAAAA=="


def test_parts_use_the_part_size_the_backend_returns(requests_mock):
    requests_mock.post(START_URL, json=started("upload-1", 20 * MIB, 2))
    puts = requests_mock.put(STORE_URLS, status_code=200)
    requests_mock.post(COMPLETE_URL, status_code=204)

    upload(bytes(25 * MIB))

    assert [len(put.body) for put in puts.request_history] == [20 * MIB, 5 * MIB]


@pytest.mark.parametrize(
    ("status_code", "body"),
    [(403, "AccessDenied"), (400, GCS_EXPIRED_TOKEN)],
    ids=["forbidden", "gcs_expired_token"],
)
def test_expired_part_url_is_refreshed(requests_mock, status_code, body):
    requests_mock.post(START_URL, json=started("upload-1", 16 * MIB, 1))
    parts = requests_mock.post(PARTS_URL, json=sign_parts)
    puts = requests_mock.put(STORE_URLS, status_code=200)
    expired = requests_mock.put(
        part_url("upload-1", 1), status_code=status_code, text=body
    )
    requests_mock.post(COMPLETE_URL, status_code=204)

    upload(b"abc")

    assert expired.call_count == 1
    assert parts.last_request.json() == {
        "uploads": [{"upload_id": "upload-1", "part_numbers": [1]}]
    }
    assert puts.last_request.url == part_url("upload-1", 1, "fresh")


def test_transient_part_failures_are_retried(requests_mock):
    requests_mock.post(START_URL, json=started("upload-1", 16 * MIB, 1))
    puts = requests_mock.put(
        STORE_URLS,
        [
            {"exc": requests.ConnectionError},
            {"status_code": 503},
            {"status_code": 200},
        ],
    )
    complete = requests_mock.post(COMPLETE_URL, status_code=204)

    upload(b"abc")

    assert puts.call_count == 3
    assert complete.call_count == 1


def test_checksum_mismatch_restarts_the_file(requests_mock):
    start = requests_mock.post(
        START_URL,
        [
            {"json": started("upload-1", 16 * MIB, 1)},
            {"json": started("upload-2", 16 * MIB, 1)},
        ],
    )
    requests_mock.put(STORE_URLS, status_code=200)
    complete = requests_mock.post(
        COMPLETE_URL,
        [{"status_code": 409, "json": CHECKSUM_MISMATCH}, {"status_code": 204}],
    )

    upload(b"abc")

    assert start.call_count == 2
    assert [request.json()["upload_id"] for request in complete.request_history] == [
        "upload-1",
        "upload-2",
    ]


def test_missing_upload_restarts_once_then_raises(requests_mock):
    data = b"123456789"
    start = requests_mock.post(
        START_URL,
        [
            {"json": started("upload-1", 1, len(data))},
            {"json": started("upload-2", 1, len(data))},
        ],
    )
    requests_mock.post(
        PARTS_URL,
        json=lambda request, context: {
            "uploads": [],
            "missing_upload_ids": [request.json()["uploads"][0]["upload_id"]],
        },
    )
    requests_mock.put(STORE_URLS, status_code=200)

    with pytest.raises(UploadError):
        upload(data)

    assert start.call_count == 2
