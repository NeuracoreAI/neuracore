"""Robots connected after a login that could not reach the server."""

import io
import zipfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import requests_mock

import neuracore as nc
from neuracore.api import core as api_core
from neuracore.api import logging as api_logging
from neuracore.api.globals import GlobalSingleton
from neuracore.core import robot as core_robot
from neuracore.core.auth import get_auth
from neuracore.core.data.dataset import Dataset
from neuracore.core.exceptions import RobotError
from neuracore.core.robot import LOCAL_ROBOT_ID_PREFIX, local_robot_id
from neuracore.data_daemon import bridge
from neuracore.data_daemon.offline_robots import offline_robot_dir

ORG_ID = "11111111-1111-1111-1111-111111111111"

# The offline warnings are checked in their own tests; elsewhere they are noise.
ignore_offline_warnings = pytest.mark.filterwarnings(
    "ignore:(Robot|Dataset) .* will be (registered|found):UserWarning"
)


@pytest.fixture
def offline(monkeypatch, reset_neuracore, tmp_path) -> Iterator[requests_mock.Mocker]:
    """Log in offline; any HTTP request made during the test fails it."""
    monkeypatch.setenv("NEURACORE_ORG_ID", ORG_ID)
    monkeypatch.setenv("NEURACORE_DAEMON_RECORDINGS_ROOT", str(tmp_path / "rec"))
    auth = get_auth()
    monkeypatch.setattr(auth, "_access_token", None)
    monkeypatch.setattr(auth, "_offline", True)
    with requests_mock.Mocker() as no_network:
        yield no_network
    assert not no_network.called


@pytest.fixture
def live_data(monkeypatch) -> Mock:
    """Stand in for the live-data orchestrator, which needs the server."""
    orchestrator = Mock()
    monkeypatch.setattr(api_core, "StreamManagerOrchestrator", orchestrator)
    return orchestrator


def test_connect_robot_offline_uses_a_local_id(offline, live_data, monkeypatch):
    ensure_daemon_running = Mock(return_value=0)
    monkeypatch.setattr(api_core, "ensure_daemon_running", ensure_daemon_running)

    with pytest.warns(UserWarning, match="will be registered") as warned:
        robot = nc.connect_robot("arm", instance=2)

    assert warned[0].filename == __file__
    assert robot.is_local
    assert robot.id == local_robot_id("arm", shared=False)
    assert robot.id.startswith(LOCAL_ROBOT_ID_PREFIX)
    assert robot.instance == 2
    assert robot.org_id == ORG_ID
    assert GlobalSingleton()._active_robot is robot
    assert core_robot.get_robot("arm", 2) is robot
    ensure_daemon_running.assert_called_once_with()
    live_data.assert_not_called()


@ignore_offline_warnings
def test_every_process_gets_the_same_local_robot_id(offline, live_data):
    first = nc.connect_robot("arm", instance=0)
    second_instance = nc.connect_robot("arm", instance=1)

    assert first.id == second_instance.id
    assert first.id == local_robot_id("arm", shared=False)


def test_local_robot_id_depends_on_name_and_namespace():
    ids = {
        local_robot_id("arm", shared=False),
        local_robot_id("arm", shared=True),
        local_robot_id("gripper", shared=False),
    }
    assert len(ids) == 3


def test_local_robot_id_never_changes():
    # The data daemon stores these ids, and processes on different SDK
    # versions must agree on them.
    assert local_robot_id("arm", shared=False) == (
        "local-robot-f0394689-5dcd-5ce1-b678-e70de237e4b6"
    )
    assert local_robot_id("arm", shared=True) == (
        "local-robot-dc06d564-6b87-5da0-8907-7b77a450a08c"
    )


def _zip_contents(package: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _files(directory: Path) -> set[str]:
    return {path.name for path in directory.iterdir()}


def test_connect_robot_offline_saves_the_urdf_package(offline, live_data, mock_urdf):
    with pytest.warns(UserWarning, match="description uploaded") as warned:
        robot = nc.connect_robot("arm", instance=1, urdf_path=mock_urdf, overwrite=True)

    assert {w.filename for w in warned} == {__file__}
    directory = offline_robot_dir(robot.id)
    assert _files(directory) == {"package.zip"}
    # Each build stamps the time, so compare what is in the zips.
    assert _zip_contents((directory / "package.zip").read_bytes()) == _zip_contents(
        robot._package_urdf()["robot_package"][1]
    )


def test_connect_robot_offline_without_a_urdf_saves_nothing(offline, live_data):
    with pytest.warns(UserWarning, match="will be registered once"):
        robot = nc.connect_robot("arm")

    assert not offline_robot_dir(robot.id).exists()


@ignore_offline_warnings
def test_instances_of_a_robot_share_its_package(offline, live_data, mock_urdf):
    first = nc.connect_robot("arm", instance=0, urdf_path=mock_urdf)
    nc.connect_robot("arm", instance=1)

    assert _files(offline_robot_dir(first.id)) == {"package.zip"}


def test_connect_robot_offline_with_a_broken_urdf_raises(offline, live_data, mock_urdf):
    (Path(mock_urdf).parent / "meshes" / "base.stl").unlink()

    with pytest.raises(RobotError, match="Mesh file not found"):
        nc.connect_robot("arm", urdf_path=mock_urdf)

    assert not offline_robot_dir(local_robot_id("arm", shared=False)).exists()


def test_connect_robot_offline_that_cannot_save_raises_robot_error(
    offline, live_data, monkeypatch, mock_urdf
):
    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(core_robot, "save_offline_robot", disk_full)

    with pytest.raises(RobotError, match="No space left on device"):
        nc.connect_robot("arm", urdf_path=mock_urdf)


def test_connect_robot_offline_with_no_org_selected(
    offline, live_data, monkeypatch, temp_config_dir
):
    monkeypatch.delenv("NEURACORE_ORG_ID")

    with pytest.warns(UserWarning, match="will be registered"):
        robot = nc.connect_robot("arm")

    assert robot.is_local
    assert robot.org_id is None


@ignore_offline_warnings
def test_logging_offline_with_no_org_selected_streams_nothing(
    offline, live_data, monkeypatch, temp_config_dir
):
    # Live data needs the org and a backend robot id; a robot connected
    # offline has neither, so its data only goes to the daemon.
    monkeypatch.delenv("NEURACORE_ORG_ID")
    provide_enabled = Mock()
    provide_enabled.is_disabled.return_value = False
    monkeypatch.setattr(
        api_logging, "get_provide_live_data_enabled_manager", lambda: provide_enabled
    )
    monkeypatch.setattr(
        api_logging,
        "StreamManagerOrchestrator",
        Mock(side_effect=AssertionError("must not stream live data")),
    )
    native = Mock()
    monkeypatch.setattr(bridge, "_load_native", lambda: native)
    nc.connect_robot("arm")

    nc.log_joint_positions(positions={"waist": 0.5})
    nc.log_rgb("front_camera", np.zeros((8, 8, 3), dtype=np.uint8))

    native.log_joints.assert_called_once()
    native.log_frame.assert_called_once()


def test_a_local_robot_has_no_remote_nodes(offline, live_data, monkeypatch):
    from neuracore.core import get_latest_sync_point as sync_point_module

    consume_enabled = Mock()
    consume_enabled.is_disabled.return_value = False
    monkeypatch.setattr(
        sync_point_module,
        "get_consume_live_data_enabled_manager",
        lambda: consume_enabled,
    )
    org_nodes = Mock(side_effect=AssertionError("must not reach the server"))
    monkeypatch.setattr(sync_point_module, "get_org_nodes_manager", org_nodes)
    with pytest.warns(UserWarning):
        robot = nc.connect_robot("arm")

    assert not sync_point_module.check_remote_nodes_connected(robot, 1)
    point = sync_point_module.get_latest_sync_point(robot, include_remote=True)
    assert point.data == {}
    org_nodes.assert_not_called()


@pytest.fixture
def native(monkeypatch) -> Mock:
    """Stand in for the compiled bridge, to see what reaches the daemon."""
    native_module = Mock()
    monkeypatch.setattr(bridge, "_load_native", lambda: native_module)
    monkeypatch.setattr(bridge, "ensure_daemon_running", lambda: None)
    return native_module


@ignore_offline_warnings
def test_start_recording_offline_sends_the_robot_and_dataset_details(
    offline, live_data, native
):
    robot = nc.connect_robot("arm", instance=3)
    dataset = nc.create_dataset("warehouse", description="d", tags=["t"])

    nc.start_recording()

    native.start_recording.assert_not_called()
    native.start_local_recording.assert_called_once()
    args = native.start_local_recording.call_args
    assert args.args[:3] == (robot.id, 3, dataset.id)
    assert args.kwargs["local_robot_name"] == "arm"
    assert args.kwargs["local_robot_shared"] is False
    assert args.kwargs["local_robot_overwrite_urdf"] is False
    assert args.kwargs["local_dataset_name"] == "warehouse"
    assert args.kwargs["local_dataset_shared"] is False
    assert args.kwargs["local_dataset_description"] == "d"
    assert args.kwargs["local_dataset_tags"] == ["t"]


def test_local_robot_with_a_cloud_dataset_sends_only_the_robot(
    offline, live_data, native
):
    with pytest.warns(UserWarning):
        robot = nc.connect_robot("arm", overwrite=True)
    cloud_dataset = Dataset(
        id="cloud-dataset-1",
        org_id=ORG_ID,
        name="warehouse",
        size_bytes=0,
        tags=[],
        data_types=[],
        is_shared=False,
    )
    GlobalSingleton()._active_dataset_id = cloud_dataset.id
    GlobalSingleton()._active_dataset = cloud_dataset

    robot.start_recording(cloud_dataset.id)

    args = native.start_local_recording.call_args
    assert args.args[:3] == (robot.id, 0, "cloud-dataset-1")
    assert args.kwargs["local_robot_name"] == "arm"
    assert args.kwargs["local_robot_overwrite_urdf"] is True
    assert args.kwargs["local_dataset_name"] is None
