from __future__ import annotations

import pytest
from neuracore_types import ImportSessionItemState

from neuracore.core.const import API_URL
from neuracore.importer.core.exceptions import ImporterError
from neuracore.importer.core.import_session import (
    ImportSessionError,
    ImportSessionManifest,
    create_import_session,
    get_import_session,
    list_import_session_item_keys,
    load_import_session_manifest,
    save_import_session_manifest,
)

ORG_ID = "test-org-id"
SESSIONS_URL = f"{API_URL}/org/{ORG_ID}/import-sessions"
ITEM_KEYS_URL = f"{SESSIONS_URL}/session-1/item-keys"


def _session_json(recording_count: int = 0) -> dict:
    return {
        "id": "session-1",
        "org_id": ORG_ID,
        "robot_id": "robot-1",
        "dataset_id": "dataset-1",
        "created_by": "user-1",
        "created_at": "2026-01-01T00:00:00Z",
        "expected_recording_count": 2,
        "recording_count": recording_count,
        "is_complete": recording_count == 2,
    }


@pytest.mark.usefixtures("mock_login", "reset_neuracore")
class TestCreateImportSession:
    def test_posts_the_robot_dataset_and_expected_count(self, mock_auth_requests):
        mock_auth_requests.post(SESSIONS_URL, json=_session_json())

        session = create_import_session("robot-1", "dataset-1", 2)

        assert session.expected_recording_count == 2
        assert mock_auth_requests.request_history[-1].json() == {
            "robot_id": "robot-1",
            "dataset_id": "dataset-1",
            "expected_recording_count": 2,
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
class TestReadImportSession:
    def test_get_returns_the_saved_count(self, mock_auth_requests):
        mock_auth_requests.get(f"{SESSIONS_URL}/session-1", json=_session_json(2))

        session = get_import_session("session-1")

        assert (session.recording_count, session.is_complete) == (2, True)

    def test_get_of_a_missing_session_raises(self, mock_auth_requests):
        mock_auth_requests.get(
            f"{SESSIONS_URL}/session-1",
            status_code=404,
            json={
                "detail": {
                    "error_type": "IMPORT_SESSION_NOT_FOUND",
                    "message": "Import session not found",
                }
            },
        )

        with pytest.raises(ImportSessionError, match="IMPORT_SESSION_NOT_FOUND"):
            get_import_session("session-1")

    def test_item_keys_follow_every_page(self, mock_auth_requests):
        mock_auth_requests.get(
            ITEM_KEYS_URL,
            [
                {"json": {"item_keys": ["a", "b"], "next_cursor": "b"}},
                {"json": {"item_keys": ["c"], "next_cursor": None}},
            ],
        )

        item_keys = list_import_session_item_keys(
            "session-1", ImportSessionItemState.IN_FLIGHT
        )

        requests = [
            r for r in mock_auth_requests.request_history if r.path.endswith("keys")
        ]
        assert item_keys == {"a", "b", "c"}
        assert [r.qs for r in requests] == [
            {"state": ["in_flight"]},
            {"state": ["in_flight"], "cursor": ["b"]},
        ]


class TestManifest:
    def test_round_trips_without_leaving_a_temporary_file(self, tmp_path):
        path = tmp_path / "state" / "session.json"
        manifest = ImportSessionManifest(
            import_session_id="session-1",
            robot_id="robot-1",
            dataset_id="dataset-1",
            item_keys=["b", "a"],
        )

        save_import_session_manifest(path, manifest)

        assert load_import_session_manifest(path) == manifest
        assert [p.name for p in path.parent.iterdir()] == ["session.json"]

    def test_a_missing_manifest_means_a_new_import(self, tmp_path):
        assert load_import_session_manifest(tmp_path / "session.json") is None

    def test_a_corrupt_manifest_is_reported(self, tmp_path):
        path = tmp_path / "session.json"
        path.write_text("{", encoding="utf-8")

        with pytest.raises(ImporterError, match="corrupt. Delete it"):
            load_import_session_manifest(path)
