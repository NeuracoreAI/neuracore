from __future__ import annotations

import io
import zipfile

import pytest

from neuracore.core.const import API_URL
from neuracore.core.exceptions import RobotError
from neuracore.core.robot import Robot

ORG_ID = "org-1"
ROBOT_ID = "robot-id-1"
ROBOT_NAME = "robot"
URDF_CONTENT = '<robot name="r"><link name="base"/></robot>'


def _make_package() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("robot.urdf", URDF_CONTENT)
        zf.writestr("meshes/base.stl", b"mesh")
    return buffer.getvalue()


def _register_robot_package(mock_auth_requests, status_code=200, content=b""):
    return mock_auth_requests.get(
        f"{API_URL}/org/{ORG_ID}/robots/{ROBOT_ID}/package",
        status_code=status_code,
        content=content,
    )


def _make_robot():
    robot = Robot(ROBOT_NAME, instance=0, org_id=ORG_ID)
    robot.id = ROBOT_ID
    return robot


@pytest.mark.usefixtures("mock_login")
class TestRobotDownloadUrdf:
    """Tests for download_urdf fetching and extracting the URDF package."""

    def test_download_urdf_extracts_package(self, mock_auth_requests, tmp_path):
        """download_urdf extracts robot.urdf and meshes into output_dir."""
        _register_robot_package(mock_auth_requests, content=_make_package())
        robot = _make_robot()

        urdf_path = robot.download_urdf(tmp_path / "out")

        assert urdf_path == tmp_path / "out" / "robot.urdf"
        assert urdf_path.read_text() == URDF_CONTENT
        assert (tmp_path / "out" / "meshes" / "base.stl").read_bytes() == b"mesh"

    def test_download_urdf_uninitialized_raises(self, tmp_path):
        """download_urdf raises when the robot has no id."""
        robot = Robot(ROBOT_NAME, instance=0, org_id=ORG_ID)
        with pytest.raises(RobotError, match="Robot not initialized"):
            robot.download_urdf(tmp_path)

    def test_download_urdf_missing_package_raises(self, mock_auth_requests, tmp_path):
        """download_urdf raises when the server has no URDF for the robot."""
        _register_robot_package(mock_auth_requests, status_code=404)
        robot = _make_robot()

        with pytest.raises(RobotError, match="No URDF available"):
            robot.download_urdf(tmp_path)

    def test_download_urdf_invalid_package_raises(self, mock_auth_requests, tmp_path):
        """download_urdf raises when the server returns a non-ZIP payload."""
        _register_robot_package(mock_auth_requests, content=b"not a zip")
        robot = _make_robot()

        with pytest.raises(RobotError, match="Invalid URDF package"):
            robot.download_urdf(tmp_path)
