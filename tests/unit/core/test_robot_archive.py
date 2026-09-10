from __future__ import annotations

import pytest

from neuracore.core.const import API_URL
from neuracore.core.exceptions import RobotError
from neuracore.core.robot import Robot

ORG_ID = "org-1"
ROBOT_ID = "robot-id-1"
ROBOT_NAME = "robot"


def _register_robot_archive(mock_auth_requests, status_code=200, json=None):
    return mock_auth_requests.put(
        f"{API_URL}/org/{ORG_ID}/robots/{ROBOT_ID}/archive",
        status_code=status_code,
        json=json,
    )


def _make_robot(shared=False):
    robot = Robot(ROBOT_NAME, instance=0, org_id=ORG_ID, shared=shared)
    robot.id = ROBOT_ID
    robot.archived = False
    return robot


@pytest.mark.usefixtures("mock_login")
class TestRobotArchive:
    """Tests for set_archived updating the robot archive flag."""

    def test_set_archived_calls_api(self, mock_auth_requests):
        """set_archived issues the PUT request to the archive endpoint."""
        _register_robot_archive(mock_auth_requests)
        robot = _make_robot()

        robot.set_archived(True)

        last = mock_auth_requests.request_history[-1]
        assert last.method == "PUT"
        assert last.path.endswith(f"/org/{ORG_ID}/robots/{ROBOT_ID}/archive")
        assert last.json() == {"archived": True}
        assert robot.archived is True

    def test_set_archived_sends_is_shared_query(self, mock_auth_requests):
        """set_archived includes the is_shared query parameter."""
        _register_robot_archive(mock_auth_requests)
        robot = _make_robot(shared=True)

        robot.set_archived(False)

        last = mock_auth_requests.request_history[-1]
        assert "is_shared=true" in last.query
        assert last.json() == {"archived": False}
        assert robot.archived is False

    def test_set_archived_uninitialized_raises(self):
        """set_archived raises when the robot has no id."""
        robot = Robot(ROBOT_NAME, instance=0, org_id=ORG_ID)
        with pytest.raises(RobotError, match="Robot not initialized"):
            robot.set_archived(True)

    def test_set_archived_propagates_api_error(self, mock_auth_requests):
        """set_archived raises RobotError when the server rejects the request."""
        _register_robot_archive(
            mock_auth_requests,
            status_code=404,
            json={"detail": {"error": "Unable to archive robot"}},
        )
        robot = _make_robot()

        with pytest.raises(RobotError, match="Unable to archive robot"):
            robot.set_archived(True)
