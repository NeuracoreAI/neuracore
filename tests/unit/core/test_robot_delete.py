from __future__ import annotations

import gc
import sys

import pytest
from neuracore_types import RobotInstanceIdentifier

from neuracore.core import robot as robot_module
from neuracore.core.const import API_URL
from neuracore.core.exceptions import RobotError
from neuracore.core.robot import Robot

ORG_ID = "org-1"
ROBOT_ID = "robot-id-1"
ROBOT_NAME = "robot"


def _register_robot_delete(mock_auth_requests, status_code=200, json=None):
    return mock_auth_requests.delete(
        f"{API_URL}/org/{ORG_ID}/robots/{ROBOT_ID}",
        status_code=status_code,
        json=json,
    )


def _make_robot(shared=False):
    robot = Robot(ROBOT_NAME, instance=0, org_id=ORG_ID, shared=shared)
    robot.id = ROBOT_ID
    return robot


@pytest.fixture
def clean_registry():
    """Restore the global robot registries after each test."""
    robots = dict(robot_module._robots)
    name_mapping = dict(robot_module._robot_name_id_mapping)
    yield
    robot_module._robots.clear()
    robot_module._robots.update(robots)
    robot_module._robot_name_id_mapping.clear()
    robot_module._robot_name_id_mapping.update(name_mapping)


@pytest.fixture
def deleted_robot(mock_auth_requests, clean_registry):
    """A Robot that has already been deleted."""
    _register_robot_delete(mock_auth_requests)
    robot = _make_robot()
    robot.delete()
    return robot


@pytest.mark.usefixtures("mock_login")
class TestRobotDeletion:
    """Tests for delete removing the robot and invalidating the in-memory object."""

    def test_deleted_false_before_delete(self):
        """A live robot reports deleted as False."""
        assert _make_robot().deleted is False

    def test_delete_calls_api(self, mock_auth_requests, clean_registry):
        """delete issues the DELETE request to the robots endpoint."""
        _register_robot_delete(mock_auth_requests)
        robot = _make_robot()

        robot.delete()

        last = mock_auth_requests.request_history[-1]
        assert last.method == "DELETE"
        assert last.path.endswith(f"/org/{ORG_ID}/robots/{ROBOT_ID}")

    def test_delete_sends_no_query_string(self, mock_auth_requests, clean_registry):
        """delete does not send an is_shared query parameter."""
        _register_robot_delete(mock_auth_requests)
        robot = _make_robot()

        robot.delete()

        assert mock_auth_requests.request_history[-1].query == ""

    def test_delete_shared_robot_raises(self, mock_auth_requests, clean_registry):
        """Deleting a shared robot raises before any request is issued."""
        robot = _make_robot(shared=True)
        request_count = len(mock_auth_requests.request_history)

        with pytest.raises(RobotError, match="shared"):
            robot.delete()

        assert len(mock_auth_requests.request_history) == request_count

    def test_delete_uninitialized_robot_raises(
        self, mock_auth_requests, clean_registry
    ):
        """Deleting a robot without an id raises before any request is issued."""
        robot = Robot(ROBOT_NAME, instance=0, org_id=ORG_ID)
        request_count = len(mock_auth_requests.request_history)

        with pytest.raises(RobotError, match="not initialized"):
            robot.delete()

        assert len(mock_auth_requests.request_history) == request_count

    def test_deleted_true_after_delete(self, deleted_robot):
        """After delete, deleted reports True."""
        assert deleted_robot.deleted is True

    def test_deleted_robot_is_still_a_robot(self, deleted_robot):
        """A deleted robot keeps its Robot type for isinstance checks."""
        assert isinstance(deleted_robot, Robot)

    def test_attribute_access_after_delete_raises(self, deleted_robot):
        """Reading any attribute of a deleted robot raises RobotError."""
        with pytest.raises(RobotError, match="deleted"):
            _ = deleted_robot.name

    def test_method_call_after_delete_raises(self, deleted_robot):
        """Calling any method of a deleted robot raises RobotError."""
        with pytest.raises(RobotError, match="deleted"):
            deleted_robot.close()

    def test_delete_evicts_robot_from_registry(
        self, mock_auth_requests, clean_registry
    ):
        """delete drops the robot from both global registries."""
        _register_robot_delete(mock_auth_requests)
        robot = _make_robot()
        key = RobotInstanceIdentifier(robot_id=ROBOT_ID, robot_instance=0)
        robot_module._robots[key] = robot
        robot_module._robot_name_id_mapping[ROBOT_NAME] = ROBOT_ID

        robot.delete()

        assert key not in robot_module._robots
        assert ROBOT_NAME not in robot_module._robot_name_id_mapping

    def test_delete_surfaces_backend_error_detail(
        self, mock_auth_requests, clean_registry
    ):
        """A rejected delete raises RobotError carrying the backend error summary."""
        _register_robot_delete(
            mock_auth_requests,
            status_code=500,
            json={"detail": {"error": "Unable to delete robots"}},
        )
        robot = _make_robot()

        with pytest.raises(RobotError, match="Unable to delete robots"):
            robot.delete()

    def test_garbage_collecting_deleted_robot_is_silent(
        self, mock_auth_requests, clean_registry
    ):
        """Finalizing a deleted robot raises nothing through the unraisable hook."""
        _register_robot_delete(mock_auth_requests)
        robot = _make_robot()
        robot.delete()

        unraisable = []
        previous_hook = sys.unraisablehook
        sys.unraisablehook = unraisable.append
        try:
            del robot
            gc.collect()
        finally:
            sys.unraisablehook = previous_hook

        assert unraisable == []
