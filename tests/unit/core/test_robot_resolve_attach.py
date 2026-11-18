from __future__ import annotations

import pytest
from neuracore_types import RobotInstanceIdentifier

from neuracore.core import robot as robot_module
from neuracore.core.const import API_URL
from neuracore.core.exceptions import RobotArchivedError

ORG_ID = "test-org-id"
ROBOT_ID = "robot-id-1"
ROBOT_NAME = "robot"
ROBOTS_URL = f"{API_URL}/org/{ORG_ID}/robots"


@pytest.fixture
def clean_registry():
    robots = dict(robot_module._robots)
    name_mapping = dict(robot_module._robot_name_id_mapping)
    yield
    robot_module._robots.clear()
    robot_module._robots.update(robots)
    robot_module._robot_name_id_mapping.clear()
    robot_module._robot_name_id_mapping.update(name_mapping)


@pytest.mark.usefixtures("mock_login", "clean_registry")
class TestResolve:
    def test_resolves_the_robot_without_registering_an_instance(
        self, mock_auth_requests
    ):
        mock_auth_requests.post(
            f"{ROBOTS_URL}/resolve",
            json={"robot_id": ROBOT_ID, "has_urdf": False, "archived": False},
        )
        instance_registration = mock_auth_requests.post(ROBOTS_URL, json={})
        registry_before = dict(robot_module._robots)

        robot = robot_module.resolve(ROBOT_NAME, shared=True)

        last = mock_auth_requests.request_history[-1]
        assert last.path.endswith(f"/org/{ORG_ID}/robots/resolve")
        assert "is_shared=true" in last.query
        assert last.json() == {"name": ROBOT_NAME}
        assert not instance_registration.called
        assert robot.id == ROBOT_ID
        assert robot_module._robots == registry_before

    def test_an_archived_robot_raises(self, mock_auth_requests):
        mock_auth_requests.post(
            f"{ROBOTS_URL}/resolve",
            status_code=409,
            json={"detail": {"error_type": "ROBOT_ARCHIVED", "message": "archived"}},
        )

        with pytest.raises(RobotArchivedError):
            robot_module.resolve(ROBOT_NAME)


@pytest.mark.usefixtures("mock_login", "clean_registry")
class TestOrdinaryRegistration:
    def test_init_still_registers_the_instance(self, mock_auth_requests):
        mock_auth_requests.post(
            ROBOTS_URL,
            json={"robot_id": ROBOT_ID, "has_urdf": False, "archived": False},
        )

        robot_module.init(ROBOT_NAME, instance=3)

        last = mock_auth_requests.request_history[-1]
        assert last.path.endswith(f"/org/{ORG_ID}/robots")
        assert last.json() == {"name": ROBOT_NAME, "instance": 3}


@pytest.mark.usefixtures("clean_registry")
class TestAttach:
    def test_registers_a_local_handle_without_contacting_the_server(
        self, requests_mock
    ):
        robot = robot_module.attach(ROBOT_NAME, ROBOT_ID, 2**40 + 1, shared=True)

        assert requests_mock.request_history == []
        assert robot.id == ROBOT_ID
        assert robot.shared is True
        assert robot_module.get_robot(ROBOT_NAME, 2**40 + 1) is robot
        assert (
            robot_module._robots[
                RobotInstanceIdentifier(robot_id=ROBOT_ID, robot_instance=2**40 + 1)
            ]
            is robot
        )

    @pytest.mark.parametrize(
        ("robot_name", "robot_id"), [("", ROBOT_ID), (ROBOT_NAME, "")]
    )
    def test_requires_a_name_and_an_id(self, robot_name, robot_id):
        with pytest.raises(ValueError):
            robot_module.attach(robot_name, robot_id, 1, shared=False)


def test_resolve_requires_a_name():
    with pytest.raises(ValueError):
        robot_module.resolve("")
