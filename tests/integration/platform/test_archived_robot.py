"""Integration test for connecting to an archived robot.

An archived robot must not be usable: the connect fails with an error that
tells the user how to fix it, and no daemon starts for it.
"""

import subprocess
import sys
import uuid

import pytest

import neuracore as nc
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.exceptions import RobotError
from neuracore.core.robot import Robot, list_organization_robots
from neuracore.data_daemon.daemon_control import read_pid_from_file
from neuracore.data_daemon.helpers import get_daemon_pid_path

ALL_ROBOTS_LIST_MODE = "mixed"
DEFAULT_INSTANCE = 0


def _stop_daemon() -> None:
    subprocess.run([sys.executable, "-m", "neuracore.data_daemon", "stop"], check=True)


def _delete_robots_by_name(robot_names: set[str]) -> None:
    """Delete the robots with these names, archived or not, if the server has them."""
    for listed_robot in list_organization_robots(
        get_current_org(), mode=ALL_ROBOTS_LIST_MODE
    ):
        if listed_robot["name"] in robot_names:
            robot = Robot(listed_robot["name"], instance=DEFAULT_INSTANCE)
            robot.id = listed_robot["id"]
            robot.delete()


def test_connect_to_archived_robot_is_refused_until_unarchived() -> None:
    """Verify an archived robot gets an actionable error, no daemon and a rename."""
    robot_name = f"testing_archived_robot_{uuid.uuid4().hex[:6]}"
    new_robot_name = f"{robot_name}_renamed"
    expected_error = (
        f"Robot '{robot_name}' is archived. Unarchive it on the Archived Robots "
        "page of the Neuracore dashboard, then connect again."
    )

    nc.login()
    try:
        robot = nc.connect_robot(robot_name)
        robot.set_archived(True)
        _stop_daemon()

        with pytest.raises(RobotError) as raised:
            nc.connect_robot(robot_name)

        assert str(raised.value) == expected_error
        assert read_pid_from_file(get_daemon_pid_path()) is None

        assert nc.update_robot_name(robot.id, new_robot_name) == robot.id

        robot.set_archived(False)
        assert nc.connect_robot(new_robot_name).id == robot.id
    finally:
        try:
            _delete_robots_by_name({robot_name, new_robot_name})
        finally:
            _stop_daemon()
