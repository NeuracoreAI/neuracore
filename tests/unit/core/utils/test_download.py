"""Tests for the download helpers in neuracore.core.utils.download."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests

from neuracore.core.utils.download import download_to_cache, download_with_progress

BODY = b"x" * 4096


class FakeResponse:
    """Streamed response that optionally fails part way through its body."""

    def __init__(self, body: bytes, fail_after: int | None, headers: dict[str, str]):
        self.body = body
        self.fail_after = fail_after
        self.headers = headers

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        sent = 0
        for start in range(0, len(self.body), 256):
            if self.fail_after is not None and sent >= self.fail_after:
                raise ConnectionError("connection reset")
            chunk = self.body[start : start + 256]
            sent += len(chunk)
            yield chunk


class FakeSession:
    """Session handing out a queued response per request and counting GETs."""

    def __init__(self, responses: list[FakeResponse | Exception]):
        self.responses = responses
        self.calls = 0

    @contextmanager
    def get(self, url: str, **kwargs: object) -> Iterator[FakeResponse]:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        yield response


@pytest.fixture
def fake_session(monkeypatch):
    """Install a queue of fake responses on the download module's session."""

    def install(*responses: FakeResponse | Exception) -> FakeSession:
        session = FakeSession(list(responses))
        monkeypatch.setattr(
            "neuracore.core.utils.download.thread_local_session",
            lambda **kwargs: session,
        )
        return session

    return install


def complete() -> FakeResponse:
    """Return a response that delivers the whole body."""
    return FakeResponse(BODY, None, {"Content-Length": str(len(BODY))})


def interrupted() -> FakeResponse:
    """Return a response that dies after 1024 bytes."""
    return FakeResponse(BODY, 1024, {"Content-Length": str(len(BODY))})


def test_interrupted_download_publishes_nothing(fake_session, tmp_path: Path) -> None:
    """Leave neither the destination nor a staging file after a failed download."""
    fake_session(interrupted())
    destination = tmp_path / "model.nc.zip"

    with pytest.raises(ConnectionError):
        download_with_progress("https://example.test/model", "model", destination)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_retry_after_interruption_downloads_in_full(
    fake_session, tmp_path: Path
) -> None:
    """Download the complete body when a cached download is retried."""
    fake_session(interrupted(), complete())
    destination = tmp_path / "job" / "model.nc.zip"

    with pytest.raises(ConnectionError):
        download_to_cache("https://example.test/model", destination, "model")
    download_to_cache("https://example.test/model", destination, "model")

    assert destination.read_bytes() == BODY


def test_cache_hit_skips_download(fake_session, tmp_path: Path) -> None:
    """Return the cached file without downloading when its size matches."""
    session = fake_session(complete())
    destination = tmp_path / "job" / "model.nc.zip"
    destination.parent.mkdir()
    destination.write_bytes(BODY)

    assert download_to_cache("https://example.test/model", destination, "model") == (
        destination
    )
    assert destination.read_bytes() == BODY
    assert session.calls == 1


def test_partial_cache_entry_is_downloaded_again(fake_session, tmp_path: Path) -> None:
    """Download again when the cached file is shorter than the server reports."""
    fake_session(complete())
    destination = tmp_path / "job" / "model.nc.zip"
    destination.parent.mkdir()
    destination.write_bytes(BODY[:1024])

    download_to_cache("https://example.test/model", destination, "model")

    assert destination.read_bytes() == BODY


def test_unknown_content_length_downloads_again(fake_session, tmp_path: Path) -> None:
    """Download again when the server reports no usable length."""
    fake_session(FakeResponse(BODY, None, {}))
    destination = tmp_path / "job" / "model.nc.zip"
    destination.parent.mkdir()
    destination.write_bytes(BODY[:1024])

    download_to_cache("https://example.test/model", destination, "model")

    assert destination.read_bytes() == BODY


def test_probe_failure_propagates(fake_session, tmp_path: Path) -> None:
    """Raise when the length probe fails instead of downloading again."""
    fake_session(requests.ConnectionError("probe failed"), complete())
    destination = tmp_path / "job" / "model.nc.zip"
    destination.parent.mkdir()
    destination.write_bytes(BODY[:1024])

    with pytest.raises(requests.ConnectionError):
        download_to_cache("https://example.test/model", destination, "model")

    assert destination.read_bytes() == BODY[:1024]
