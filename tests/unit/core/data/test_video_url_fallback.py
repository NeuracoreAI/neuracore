"""Tests for the lossless -> lossy video URL fallback in SynchronizedRecording."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests
from neuracore_types import DataType

from neuracore.core.data.synced_recording import SynchronizedRecording

MODULE = "neuracore.core.data.synced_recording"


def _response(status_code: int, url: str | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(
            f"HTTP {status_code}"
        )
    else:
        response.raise_for_status.return_value = None
        response.json.return_value = {"url": url}
    return response


def _call_get_video_url(
    session: MagicMock, camera_type: DataType = DataType.RGB_IMAGES
) -> str:
    fake_self = SimpleNamespace(id="rec-1", dataset=SimpleNamespace(org_id="org-1"))
    auth = MagicMock()
    auth.get_headers = MagicMock(return_value={})
    with (
        patch(f"{MODULE}.thread_local_session", return_value=session),
        patch(f"{MODULE}.get_auth", return_value=auth),
    ):
        return SynchronizedRecording._get_video_url(fake_self, camera_type, "cam")


def test_falls_back_to_lossy_when_lossless_missing() -> None:
    requested: list[str] = []

    def fake_get(url, params=None, headers=None):
        filepath = params["filepath"]
        requested.append(filepath)
        if filepath.endswith("lossless.mp4"):
            return _response(404)
        return _response(200, url="https://example.com/lossy.mp4")

    session = MagicMock()
    session.get.side_effect = fake_get

    assert _call_get_video_url(session) == "https://example.com/lossy.mp4"
    # Lossless is tried first, lossy second.
    assert requested == [
        "RGB_IMAGES/cam/lossless.mp4",
        "RGB_IMAGES/cam/lossy.mp4",
    ]


def test_uses_lossless_when_present() -> None:
    requested: list[str] = []

    def fake_get(url, params=None, headers=None):
        requested.append(params["filepath"])
        return _response(200, url="https://example.com/lossless.mp4")

    session = MagicMock()
    session.get.side_effect = fake_get

    assert _call_get_video_url(session) == "https://example.com/lossless.mp4"
    # The lossless hit short-circuits — lossy is never requested.
    assert requested == ["RGB_IMAGES/cam/lossless.mp4"]


def test_raises_when_neither_lossless_nor_lossy_found() -> None:
    session = MagicMock()
    session.get.side_effect = lambda *args, **kwargs: _response(404)

    with pytest.raises(requests.HTTPError):
        _call_get_video_url(session)


def test_non_404_on_lossless_raises_without_falling_back() -> None:
    # A server/auth error (here 500) on the lossless request is a real failure
    # and must propagate as-is — it must NOT be masked by trying the lossy
    # fallback (which could otherwise silently downgrade to lossy data).
    requested: list[str] = []

    def fake_get(url, params=None, headers=None):
        requested.append(params["filepath"])
        return _response(500)

    session = MagicMock()
    session.get.side_effect = fake_get

    with pytest.raises(requests.HTTPError):
        _call_get_video_url(session)
    # Only the lossless request is made; the lossy fallback is never attempted.
    assert requested == ["RGB_IMAGES/cam/lossless.mp4"]


def test_depth_requests_lossless_only_and_never_falls_back_to_lossy() -> None:
    # Depth's lossy.mp4 is a visualisation, never a valid source, so a missing
    # depth lossless must fail loudly rather than silently decode the proxy.
    requested: list[str] = []

    def fake_get(url, params=None, headers=None):
        requested.append(params["filepath"])
        return _response(404)

    session = MagicMock()
    session.get.side_effect = fake_get

    with pytest.raises(requests.HTTPError):
        _call_get_video_url(session, DataType.DEPTH_IMAGES)
    assert requested == ["DEPTH_IMAGES/cam/lossless.mp4"]


def test_non_404_on_lossy_fallback_propagates() -> None:
    # The lossless 404 falls back to lossy; a real error (here 500) on the lossy
    # leg must propagate, not be swallowed as "no artefact found".
    def fake_get(url, params=None, headers=None):
        if params["filepath"].endswith("lossless.mp4"):
            return _response(404)
        return _response(500)

    session = MagicMock()
    session.get.side_effect = fake_get

    with pytest.raises(requests.HTTPError):
        _call_get_video_url(session)
