from __future__ import annotations

import logging
import threading

import pytest
import requests
from neuracore_types import (
    IMPORT_SESSION_RESERVED_FLOOR,
    ImportSession,
    ImportSessionStatus,
)

from neuracore.core.const import API_URL
from neuracore.importer.core import import_session as import_session_module
from neuracore.importer.core.import_session import (
    ImportSessionError,
    ImportSessionHeartbeat,
    close_import_session,
    create_import_session,
    heartbeat_import_session,
)

ORG_ID = "test-org-id"
SESSIONS_URL = f"{API_URL}/org/{ORG_ID}/import-sessions"


def _session_json(status: str = "active") -> dict:
    return {
        "id": "session-1",
        "org_id": ORG_ID,
        "robot_id": "robot-1",
        "dataset_id": "dataset-1",
        "created_by": "user-1",
        "range_index": 0,
        "instance_start": IMPORT_SESSION_RESERVED_FLOOR,
        "worker_count": 2,
        "status": status,
        "created_at": "2026-01-01T00:00:00Z",
        "last_heartbeat_at": "2026-01-01T00:00:00Z",
        "closed_at": None if status == "active" else "2026-01-01T00:00:00Z",
        "heartbeat_timeout_s": 300.0,
    }


@pytest.mark.usefixtures("mock_login", "reset_neuracore")
class TestCreateImportSession:
    def test_posts_the_robot_dataset_and_worker_count(self, mock_auth_requests):
        mock_auth_requests.post(SESSIONS_URL, json=_session_json())

        session = create_import_session("robot-1", "dataset-1", 2)

        assert session.instance_start == IMPORT_SESSION_RESERVED_FLOOR
        assert mock_auth_requests.request_history[-1].json() == {
            "robot_id": "robot-1",
            "dataset_id": "dataset-1",
            "worker_count": 2,
        }

    @pytest.mark.parametrize(
        ("status_code", "body", "message"),
        [
            (503, {}, "HTTP 503"),
            (
                403,
                {"detail": {"error_type": "UNAUTHORIZED", "message": "No access"}},
                "HTTP 403 UNAUTHORIZED",
            ),
            (404, {"detail": "Not Found"}, "does not support import"),
        ],
        ids=["server_error", "backend_error_type", "backend_without_sessions"],
    )
    def test_a_rejected_create_is_reported_and_not_retried(
        self, mock_auth_requests, status_code, body, message
    ):
        mock_auth_requests.post(SESSIONS_URL, status_code=status_code, json=body)

        with pytest.raises(ImportSessionError, match=message):
            create_import_session("robot-1", "dataset-1", 2)

        posts = [r for r in mock_auth_requests.request_history if r.url == SESSIONS_URL]
        assert len(posts) == 1


@pytest.mark.usefixtures("mock_login", "reset_neuracore")
class TestHeartbeatAndClose:
    def test_heartbeat_posts_to_the_session(self, mock_auth_requests):
        mock_auth_requests.post(
            f"{SESSIONS_URL}/session-1/heartbeat", json=_session_json()
        )

        assert heartbeat_import_session("session-1").id == "session-1"

    def test_close_sends_the_terminal_status(self, mock_auth_requests):
        mock_auth_requests.post(
            f"{SESSIONS_URL}/session-1/close", json=_session_json("cancelled")
        )

        session = close_import_session("session-1", ImportSessionStatus.CANCELLED)

        assert session.status == ImportSessionStatus.CANCELLED
        assert mock_auth_requests.request_history[-1].json() == {"status": "cancelled"}

    def test_close_rejection_raises(self, mock_auth_requests):
        mock_auth_requests.post(
            f"{SESSIONS_URL}/session-1/close",
            status_code=409,
            json={
                "detail": {
                    "error_type": "IMPORT_SESSION_CLOSE_CONFLICT",
                    "message": "Already completed",
                }
            },
        )

        with pytest.raises(ImportSessionError, match="IMPORT_SESSION_CLOSE_CONFLICT"):
            close_import_session("session-1", ImportSessionStatus.CANCELLED)


def _http_error(status_code: int) -> ImportSessionError:
    response = requests.Response()
    response.status_code = status_code
    error = ImportSessionError(f"HTTP {status_code}")
    error.__cause__ = requests.exceptions.HTTPError(response=response)
    return error


def _heartbeat(monkeypatch, outcomes: list) -> tuple[ImportSessionHeartbeat, list]:
    """A heartbeat that beats immediately and plays back ``outcomes``."""
    calls: list[str] = []
    done = threading.Event()

    def beat(session_id):
        calls.append(session_id)
        outcome = outcomes[len(calls) - 1] if len(calls) <= len(outcomes) else None
        if len(calls) >= len(outcomes):
            done.set()
        if isinstance(outcome, Exception):
            raise outcome

    monkeypatch.setattr(import_session_module, "heartbeat_import_session", beat)
    heartbeat = ImportSessionHeartbeat(_session_model())
    heartbeat._interval_s = 0.001
    heartbeat._done = done
    return heartbeat, calls


def _session_model() -> ImportSession:
    return ImportSession.model_validate(_session_json())


def test_heartbeat_interval_is_a_tenth_of_the_timeout():
    heartbeat = ImportSessionHeartbeat(_session_model())

    assert heartbeat._interval_s == 30.0


def test_heartbeat_keeps_going_after_a_transient_failure(monkeypatch):
    heartbeat, calls = _heartbeat(monkeypatch, [_http_error(503), None, None])

    heartbeat.start()
    assert heartbeat._done.wait(5)
    heartbeat.stop()

    assert len(calls) >= 3


@pytest.mark.parametrize("status_code", [404, 409])
def test_heartbeat_stops_once_the_session_is_unusable(monkeypatch, caplog, status_code):
    heartbeat, calls = _heartbeat(monkeypatch, [_http_error(status_code)])

    with caplog.at_level(logging.ERROR):
        heartbeat.start()
        heartbeat._thread.join(5)

    assert not heartbeat._thread.is_alive()
    assert calls == ["session-1"]
    assert "no longer usable" in caplog.text
    heartbeat.stop()


def test_heartbeat_stop_ends_the_thread(monkeypatch):
    heartbeat, _ = _heartbeat(monkeypatch, [])
    heartbeat._interval_s = 60

    heartbeat.start()
    heartbeat.stop()

    assert not heartbeat._thread.is_alive()
