"""Tests for MultipartUploader."""

import base64
import io
import re
from unittest.mock import MagicMock

import google_crc32c
import pytest
import requests

from neuracore.core.utils import multipart_upload
from neuracore.core.utils.multipart_upload import MultipartUploader, UploadError

MIB = 1024 * 1024
BASE_URL = "https://api.example.com/api/v2/org/org-1"
START_URL = f"{BASE_URL}/training/jobs/job-1/uploads"
PARTS_URL = f"{BASE_URL}/uploads/parts"
STORE_URLS = re.compile(r"https://store\.example\.com/")
CRC32C_HEADER = "x-amz-checksum-crc32c"

GCS_EXPIRED_TOKEN = (
    "<?xml version='1.0' encoding='UTF-8'?><Error><Code>ExpiredToken</Code>"
    "<Message>The provided token has expired.</Message>"
    "<Details>Request signature expired at: 2026-11-19T20:00:00+00:00</Details>"
    "</Error>"
)


def part_url(upload_id: str, part_number: int, signature: str = "first") -> str:
    return f"https://store.example.com/{signature}/{upload_id}/{part_number}"


def complete_url(upload_id: str) -> str:
    return f"https://store.example.com/complete/{upload_id}"


def started(upload_id: str, part_size_bytes: int, part_count: int) -> dict:
    return {
        "upload_id": upload_id,
        "part_size_bytes": part_size_bytes,
        "part_urls": {
            str(part_number): part_url(upload_id, part_number)
            for part_number in range(1, min(part_count, 8) + 1)
        },
        "complete_url": complete_url(upload_id),
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


def store_part(request, context) -> str:
    """Accept a part and return the ETag ``"{upload_id}-{part_number}"``."""
    upload_id, part_number = request.path.rsplit("/", 2)[1:]
    context.headers["ETag"] = f'"{upload_id}-{part_number}"'
    return ""


def crc32c(data: bytes) -> str:
    return base64.b64encode(google_crc32c.value(data).to_bytes(4, "big")).decode()


def completed(data: bytes) -> dict:
    """A Complete response with the CRC32C of ``data`` in the header, as on GCS."""
    return {
        "headers": {CRC32C_HEADER: crc32c(data)},
        "text": "<CompleteMultipartUploadResult/>",
    }


def complete_body(etags: list[str]) -> str:
    parts = "".join(
        f"<Part><PartNumber>{part_number}</PartNumber>"
        f"<ETag>{etag.replace(chr(34), '&quot;')}</ETag></Part>"
        for part_number, etag in enumerate(etags, start=1)
    )
    return f"<CompleteMultipartUpload>{parts}</CompleteMultipartUpload>"


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


def test_parts_are_sent_in_order_and_completed_with_their_etags(requests_mock):
    data = b"123456789"
    start = requests_mock.post(START_URL, json=started("upload-1", 1, len(data)))
    parts = requests_mock.post(PARTS_URL, json=sign_parts)
    puts = requests_mock.put(STORE_URLS, text=store_part)
    complete = requests_mock.post(complete_url("upload-1"), **completed(data))

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
    assert complete.call_count == 1
    assert complete.last_request.text == complete_body(
        [f'"upload-1-{part_number}"' for part_number in range(1, 10)]
    )
    store_requests = puts.request_history + complete.request_history
    assert all("Authorization" not in request.headers for request in store_requests)


def test_empty_file_is_one_empty_part(requests_mock):
    start = requests_mock.post(START_URL, json=started("upload-1", 16 * MIB, 1))
    puts = requests_mock.put(STORE_URLS, text=store_part)
    complete = requests_mock.post(complete_url("upload-1"), **completed(b""))

    upload(b"")

    assert start.last_request.json()["size_bytes"] == 0
    assert puts.call_count == 1
    assert puts.last_request.headers["Content-Length"] == "0"
    assert complete.last_request.text == complete_body(['"upload-1-1"'])


def test_parts_use_the_part_size_the_backend_returns(requests_mock):
    data = bytes(25 * MIB)
    requests_mock.post(START_URL, json=started("upload-1", 20 * MIB, 2))
    puts = requests_mock.put(STORE_URLS, text=store_part)
    requests_mock.post(complete_url("upload-1"), **completed(data))

    upload(data)

    assert [len(put.body) for put in puts.request_history] == [20 * MIB, 5 * MIB]


@pytest.mark.parametrize(
    ("status_code", "body"),
    [(403, "AccessDenied"), (400, GCS_EXPIRED_TOKEN)],
    ids=["forbidden", "gcs_expired_token"],
)
def test_expired_part_url_is_refreshed(requests_mock, status_code, body):
    requests_mock.post(START_URL, json=started("upload-1", 16 * MIB, 1))
    parts = requests_mock.post(PARTS_URL, json=sign_parts)
    puts = requests_mock.put(STORE_URLS, text=store_part)
    expired = requests_mock.put(
        part_url("upload-1", 1), status_code=status_code, text=body
    )
    requests_mock.post(complete_url("upload-1"), **completed(b"abc"))

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
            {"status_code": 200, "headers": {"ETag": '"etag-1"'}},
        ],
    )
    complete = requests_mock.post(complete_url("upload-1"), **completed(b"abc"))

    upload(b"abc")

    assert puts.call_count == 3
    assert complete.last_request.text == complete_body(['"etag-1"'])


def test_part_without_an_etag_fails_the_upload(requests_mock):
    requests_mock.post(START_URL, json=started("upload-1", 16 * MIB, 1))
    requests_mock.put(STORE_URLS, status_code=200)
    complete = requests_mock.post(complete_url("upload-1"), **completed(b"abc"))

    with pytest.raises(UploadError, match="no ETag"):
        upload(b"abc")

    assert complete.call_count == 0


@pytest.mark.parametrize(
    "response",
    [
        {
            "headers": {CRC32C_HEADER: crc32c(b"abc")},
            "text": "<CompleteMultipartUploadResult/>",
        },
        {
            "text": (
                "<CompleteMultipartUploadResult>"
                f"<ChecksumCRC32C>{crc32c(b'abc')}</ChecksumCRC32C>"
                "</CompleteMultipartUploadResult>"
            )
        },
    ],
    ids=["header", "body"],
)
def test_the_crc32c_is_read_from_the_complete_header_or_body(requests_mock, response):
    start = requests_mock.post(START_URL, json=started("upload-1", 16 * MIB, 1))
    requests_mock.put(STORE_URLS, text=store_part)
    complete = requests_mock.post(complete_url("upload-1"), **response)

    upload(b"abc")

    assert start.call_count == 1
    assert complete.call_count == 1


@pytest.mark.parametrize(
    "first_complete",
    [
        completed(b"other bytes"),
        {"text": "<CompleteMultipartUploadResult/>"},
        {"status_code": 404, "text": "<Error><Code>NoSuchUpload</Code></Error>"},
        {"status_code": 404, "text": "<Error><Code>NoSuchPart</Code></Error>"},
        {"status_code": 400, "text": "<Error><Code>InvalidPart</Code></Error>"},
        {"status_code": 400, "text": GCS_EXPIRED_TOKEN},
        {"status_code": 403, "text": "<Error><Code>AccessDenied</Code></Error>"},
        {"text": "<Error><Code>NoSuchUpload</Code></Error>"},
    ],
    ids=[
        "crc32c_mismatch",
        "no_crc32c",
        "no_such_upload",
        "no_such_part",
        "invalid_part",
        "expired_url",
        "forbidden",
        "200_with_no_such_upload",
    ],
)
def test_a_failed_check_or_lost_upload_restarts_the_file(requests_mock, first_complete):
    start = requests_mock.post(
        START_URL,
        [
            {"json": started("upload-1", 16 * MIB, 1)},
            {"json": started("upload-2", 16 * MIB, 1)},
        ],
    )
    requests_mock.put(STORE_URLS, text=store_part)
    first = requests_mock.post(complete_url("upload-1"), **first_complete)
    second = requests_mock.post(complete_url("upload-2"), **completed(b"abc"))

    upload(b"abc")

    assert start.call_count == 2
    assert first.call_count == 1
    assert second.last_request.text == complete_body(['"upload-2-1"'])


def test_a_crc32c_mismatch_on_the_restart_raises(requests_mock):
    start = requests_mock.post(
        START_URL,
        [
            {"json": started("upload-1", 16 * MIB, 1)},
            {"json": started("upload-2", 16 * MIB, 1)},
        ],
    )
    requests_mock.put(STORE_URLS, text=store_part)
    requests_mock.post(complete_url("upload-1"), **completed(b"other bytes"))
    requests_mock.post(complete_url("upload-2"), **completed(b"other bytes"))

    with pytest.raises(UploadError, match="CRC32C mismatch"):
        upload(b"abc")

    assert start.call_count == 2


def test_transient_complete_failures_are_retried(requests_mock):
    start = requests_mock.post(START_URL, json=started("upload-1", 16 * MIB, 1))
    requests_mock.put(STORE_URLS, text=store_part)
    complete = requests_mock.post(
        complete_url("upload-1"),
        [
            {"exc": requests.ConnectionError},
            {"status_code": 503},
            {"text": "<Error><Code>InternalError</Code></Error>"},
            completed(b"abc"),
        ],
    )

    upload(b"abc")

    assert start.call_count == 1
    assert complete.call_count == 4
    assert {request.text for request in complete.request_history} == {
        complete_body(['"upload-1-1"'])
    }


def test_a_rejected_complete_fails_the_upload(requests_mock):
    requests_mock.post(START_URL, json=started("upload-1", 16 * MIB, 1))
    requests_mock.put(STORE_URLS, text=store_part)
    complete = requests_mock.post(
        complete_url("upload-1"),
        status_code=400,
        text="<Error><Code>MalformedXML</Code></Error>",
    )

    with pytest.raises(UploadError, match="last status 400"):
        upload(b"abc")

    assert complete.call_count == 1


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
    requests_mock.put(STORE_URLS, text=store_part)

    with pytest.raises(UploadError):
        upload(data)

    assert start.call_count == 2
