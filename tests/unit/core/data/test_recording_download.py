"""Tests for Recording.download."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from neuracore_types import RecordingMetadata

from neuracore.core.data import recording as recording_module
from neuracore.core.data.recording import Recording

SIGNED_URL = "https://storage.googleapis.com/bucket/obj?X-Goog-Signature=sig"
AUTH_HEADERS = {"Authorization": "Bearer api-key"}


def _recording() -> Recording:
    return Recording(
        dataset=SimpleNamespace(org_id="org"),
        recording_id="rec",
        total_bytes=0,
        robot_id="robot",
        instance=0,
        start_time=0.0,
        end_time=0.0,
        metadata=RecordingMetadata(),
    )


def test_download_fetches_signed_url_through_download_helper():
    api_session = MagicMock()
    api_session.get.return_value.json.return_value = {"url": SIGNED_URL}
    auth = SimpleNamespace(get_headers=lambda: AUTH_HEADERS)

    with (
        patch.object(recording_module, "API_URL", "https://api.test"),
        patch.object(recording_module, "get_auth", return_value=auth),
        patch.object(
            recording_module, "thread_local_session", return_value=api_session
        ),
        patch.object(
            recording_module, "download_bytes", return_value=b"payload"
        ) as download_bytes,
    ):
        assert _recording().download("JOINT_POSITIONS/trace.json") == b"payload"

    api_session.get.assert_called_once()
    assert api_session.get.call_args.args == (
        "https://api.test/org/org/recording/rec/download_url",
    )
    assert api_session.get.call_args.kwargs["headers"] == AUTH_HEADERS
    download_bytes.assert_called_once_with(SIGNED_URL)
