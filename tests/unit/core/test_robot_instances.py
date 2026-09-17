from __future__ import annotations

import pytest
from neuracore_types import RobotInstanceIdentifier

from neuracore.core import robot as robot_module
from neuracore.core.const import API_URL
from neuracore.core.exceptions import RobotError
from neuracore.core.robot import Robot

ORG_ID = "org-1"
ROBOT_ID = "robot-id-1"
ROBOT_NAME = "robot"


def _make_robot(shared: bool = False, instance: int = 0) -> Robot:
    robot = Robot(ROBOT_NAME, instance=instance, org_id=ORG_ID, shared=shared)
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


@pytest.mark.usefixtures("mock_login")
class TestRobotRemoveInstance:
    """Tests for Robot.remove_instance."""

    def test_remove_instance_calls_api(self, mock_auth_requests, clean_registry):
        mock_auth_requests.delete(
            f"{API_URL}/org/{ORG_ID}/robots/{ROBOT_ID}/instances/7",
            status_code=200,
        )
        robot = _make_robot()

        robot.remove_instance(7)

        last = mock_auth_requests.request_history[-1]
        assert last.method == "DELETE"
        assert last.path.endswith(f"/org/{ORG_ID}/robots/{ROBOT_ID}/instances/7")
        assert "is_shared=false" in last.query

    def test_remove_instance_sends_is_shared_query(
        self, mock_auth_requests, clean_registry
    ):
        mock_auth_requests.delete(
            f"{API_URL}/org/{ORG_ID}/robots/{ROBOT_ID}/instances/3",
            status_code=200,
        )
        robot = _make_robot(shared=True)

        robot.remove_instance(3)

        assert "is_shared=true" in mock_auth_requests.request_history[-1].query

    def test_remove_instance_evicts_registry_entry(
        self, mock_auth_requests, clean_registry
    ):
        mock_auth_requests.delete(
            f"{API_URL}/org/{ORG_ID}/robots/{ROBOT_ID}/instances/5",
            status_code=200,
        )
        parent = _make_robot(instance=0)
        worker = _make_robot(instance=5)
        robot_module._robots[
            RobotInstanceIdentifier(robot_id=ROBOT_ID, robot_instance=0)
        ] = parent
        robot_module._robots[
            RobotInstanceIdentifier(robot_id=ROBOT_ID, robot_instance=5)
        ] = worker

        parent.remove_instance(5)

        assert (
            RobotInstanceIdentifier(robot_id=ROBOT_ID, robot_instance=5)
            not in robot_module._robots
        )
        assert (
            RobotInstanceIdentifier(robot_id=ROBOT_ID, robot_instance=0)
            in robot_module._robots
        )
        assert parent.deleted is False

    def test_remove_instance_uninitialized_raises(self):
        robot = Robot(ROBOT_NAME, instance=0, org_id=ORG_ID)
        with pytest.raises(RobotError, match="Robot not initialized"):
            robot.remove_instance(1)

    def test_remove_instance_propagates_api_error(
        self, mock_auth_requests, clean_registry
    ):
        mock_auth_requests.delete(
            f"{API_URL}/org/{ORG_ID}/robots/{ROBOT_ID}/instances/1",
            status_code=404,
            json={"detail": {"error": "Robot or robot instance not found"}},
        )
        robot = _make_robot()

        with pytest.raises(RobotError, match="Robot or robot instance not found"):
            robot.remove_instance(1)


@pytest.mark.usefixtures("mock_login")
class TestRobotListInstances:
    """Tests for listing robot instance IDs."""

    def test_list_instance_ids(self, mock_auth_requests):
        mock_auth_requests.get(
            f"{API_URL}/org/{ORG_ID}/robots/{ROBOT_ID}",
            status_code=200,
            json={
                "id": ROBOT_ID,
                "name": ROBOT_NAME,
                "instances": [
                    {"robot_instance": 0},
                    {"robot_instance": 3},
                    {"robot_instance": 150000},
                ],
            },
        )
        robot = _make_robot()

        assert robot.list_instance_ids() == [0, 3, 150000]
        assert robot.max_instance_id() == 150000
        assert "is_shared=false" in mock_auth_requests.request_history[-1].query

    def test_max_instance_id_empty(self, mock_auth_requests):
        mock_auth_requests.get(
            f"{API_URL}/org/{ORG_ID}/robots/{ROBOT_ID}",
            status_code=200,
            json={"id": ROBOT_ID, "name": ROBOT_NAME, "instances": []},
        )
        robot = _make_robot(shared=True)

        assert robot.max_instance_id() == 0
        assert "is_shared=true" in mock_auth_requests.request_history[-1].query

    def test_list_instance_ids_uninitialized_raises(self):
        robot = Robot(ROBOT_NAME, instance=0, org_id=ORG_ID)
        with pytest.raises(RobotError, match="Robot not initialized"):
            robot.list_instance_ids()
