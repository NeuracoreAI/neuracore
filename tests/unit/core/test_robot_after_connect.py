"""What a robot does right after it connects, before its server id is known."""

import tempfile
from pathlib import Path

import pytest

import neuracore as nc
from neuracore.api import core as api_core
from neuracore.core.const import API_URL
from neuracore.core.exceptions import RobotError
from neuracore.core.robot import Robot, get_robot


def test_a_converted_urdf_is_kept_until_the_robot_is_closed(
    temp_config_dir, mock_auth_requests, reset_neuracore, monkeypatch, tmp_path
):
    nc.login("test_api_key")
    robot = Robot("arm", 0)
    # As an MJCF robot holds the URDF it converted the model to.
    robot._temp_dir = tempfile.TemporaryDirectory(dir=tmp_path)
    robot.urdf_path = f"{robot._temp_dir.name}/model.urdf"
    open(robot.urdf_path, "w").close()
    monkeypatch.setattr(
        robot, "_package_urdf", lambda: {"robot_package": ("p.zip", b"zip")}
    )
    monkeypatch.setattr(
        "neuracore.core.robot.save_robot_package", lambda package: tmp_path / "p.zip"
    )

    robot.init()

    assert Path(robot.urdf_path).is_file()
    robot.close()
    assert not Path(robot.urdf_path).exists()


def test_a_robot_just_connected_is_archived_and_renamed_by_its_server_id(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id, monkeypatch
):
    # Nothing reads the id in the background.
    monkeypatch.setattr(
        api_core, "_start_live_data_once_registered", lambda robot: None
    )
    nc.login("test_api_key")
    robot = nc.connect_robot("arm", wait=False)
    assert not robot.is_registered
    archive = mock_auth_requests.put(
        f"{API_URL}/org/{mocked_org_id}/robots/cloud-robot-arm/archive", json={}
    )
    rename = mock_auth_requests.put(
        f"{API_URL}/org/{mocked_org_id}/robots/cloud-robot-arm/name", json={}
    )

    robot.set_archived(True)
    robot._id = None
    nc.update_robot_name("arm", "arm-2")

    assert archive.called
    assert rename.called


def test_a_deleted_robot_is_no_longer_connected(
    temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
):
    nc.login("test_api_key")
    robot = nc.connect_robot("arm")
    nc.connect_robot("gripper")
    mock_auth_requests.delete(
        f"{API_URL}/org/{mocked_org_id}/robots/cloud-robot-arm", json={}
    )
    rename = mock_auth_requests.put(
        f"{API_URL}/org/{mocked_org_id}/robots/cloud-robot-gripper/name", json={}
    )

    robot.delete()
    nc.update_robot_name("gripper", "gripper-2")

    assert rename.called
    with pytest.raises(RobotError):
        get_robot("arm", 0)


@pytest.mark.parametrize("overwrite", [False, True])
def test_a_package_that_cannot_be_built_is_left_out_unless_overwriting(
    temp_config_dir,
    mock_auth_requests,
    reset_neuracore,
    fake_registry,
    monkeypatch,
    tmp_path,
    overwrite,
):
    urdf = tmp_path / "arm.urdf"
    urdf.write_text("<robot name='arm'/>")

    def missing_mesh(self):
        raise RobotError("Mesh file not found: base.stl")

    monkeypatch.setattr(Robot, "_package_urdf", missing_mesh)
    monkeypatch.setattr(Robot, "_get_joint_info", lambda self: {})
    nc.login("test_api_key")

    if overwrite:
        with pytest.raises(RobotError, match="Mesh file not found"):
            nc.connect_robot("arm", urdf_path=str(urdf), overwrite=True)
    else:
        nc.connect_robot("arm", urdf_path=str(urdf))
        assert fake_registry.robot_specs[-1]["package_path"] is None
