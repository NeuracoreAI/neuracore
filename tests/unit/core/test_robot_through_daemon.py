"""Robots are connected through the data daemon and record by their row."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import neuracore as nc
from neuracore.api import core as api_core
from neuracore.api.globals import GlobalSingleton
from neuracore.core.auth import Auth
from neuracore.core.exceptions import RobotError, ServerUnreachableError
from neuracore.core.robot import get_robot


@pytest.fixture
def offline(monkeypatch):
    """Make the last login one that could not reach the server."""
    monkeypatch.setattr(Auth, "is_offline", property(lambda self: True))


def test_live_data_starts_once_the_server_registers_the_robot(
    temp_config_dir, mock_auth_requests, reset_neuracore, monkeypatch, fake_registry
):
    orchestrator = Mock()
    monkeypatch.setattr(api_core, "StreamManagerOrchestrator", lambda: orchestrator)
    monkeypatch.setattr(api_core, "LIVE_DATA_POLL_S", 0.02)
    nc.login("test_api_key")
    fake_registry.set_robot_state("arm", "pending")

    robot = nc.connect_robot("arm")
    orchestrator.get_provider_manager.assert_not_called()
    fake_registry._robot_states.pop("arm")

    deadline = time.monotonic() + 5
    while not orchestrator.get_provider_manager.called:
        assert time.monotonic() < deadline, "live data never started"
        time.sleep(0.05)
    orchestrator.get_provider_manager.assert_called_once_with(
        "cloud-robot-arm", robot.instance
    )


def test_live_data_waits_through_a_hold_the_user_fixes(
    temp_config_dir, mock_auth_requests, reset_neuracore, monkeypatch, fake_registry
):
    orchestrator = Mock()
    monkeypatch.setattr(api_core, "StreamManagerOrchestrator", lambda: orchestrator)
    monkeypatch.setattr(api_core, "LIVE_DATA_POLL_S", 0.02)
    nc.login("test_api_key")
    fake_registry.set_robot_state("arm", "pending", "robot_archived")

    nc.connect_robot("arm")
    time.sleep(0.2)
    orchestrator.get_provider_manager.assert_not_called()
    # Unarchived on the dashboard: the daemon registers it.
    fake_registry._robot_states.pop("arm")

    deadline = time.monotonic() + 5
    while not orchestrator.get_provider_manager.called:
        assert time.monotonic() < deadline, "live data never started"
        time.sleep(0.05)


def test_live_data_is_given_up_for_a_robot_refused_for_good(
    temp_config_dir, mock_auth_requests, reset_neuracore, monkeypatch, fake_registry
):
    orchestrator = Mock()
    monkeypatch.setattr(api_core, "StreamManagerOrchestrator", lambda: orchestrator)
    monkeypatch.setattr(api_core, "LIVE_DATA_POLL_S", 0.02)
    nc.login("test_api_key")
    fake_registry.set_robot_state("arm", "failed", "rejected")

    nc.connect_robot("arm")
    time.sleep(0.2)
    fake_registry._robot_states.pop("arm")
    time.sleep(0.2)

    orchestrator.get_provider_manager.assert_not_called()


def test_a_robot_is_found_by_name_before_its_id_is_known(
    temp_config_dir, mock_auth_requests, reset_neuracore, fake_registry
):
    nc.login("test_api_key")
    fake_registry.set_robot_state("arm", "pending")

    robot = nc.connect_robot("arm", instance=3)

    assert get_robot("arm", 3) is robot


def test_offline_a_robot_connects_but_waiting_for_the_server_raises(
    temp_config_dir, mock_auth_requests, reset_neuracore, fake_registry, offline
):
    robot = nc.connect_robot("arm")

    assert robot.source_key == str(robot._robot_index)
    assert fake_registry.robot_specs[-1]["org_id"] == "test-org-id"
    with pytest.raises(ServerUnreachableError):
        nc.connect_robot("arm", wait=True)
    with pytest.raises(ServerUnreachableError):
        robot.id


def test_recording_refuses_a_robot_and_dataset_in_different_organizations(
    temp_config_dir, mock_auth_requests, reset_neuracore, fake_registry, monkeypatch
):
    nc.login("test_api_key")
    nc.connect_robot("arm")
    monkeypatch.setattr(
        GlobalSingleton(),
        "_active_dataset",
        SimpleNamespace(
            name="warehouse",
            is_shared=False,
            kind_known=True,
            org_id="another-org",
            dataset_index=4,
        ),
    )

    with pytest.raises(RobotError, match="different organizations"):
        nc.start_recording()


def test_a_dataset_read_from_the_server_is_selected_through_the_daemon(
    temp_config_dir, mock_auth_requests, reset_neuracore, fake_registry, monkeypatch
):
    nc.login("test_api_key")
    robot = nc.connect_robot("arm")
    start_recording = Mock()
    monkeypatch.setattr(robot, "start_recording", start_recording)
    monkeypatch.setattr(
        GlobalSingleton(),
        "_active_dataset",
        SimpleNamespace(
            id="cloud-dataset-9",
            name="warehouse",
            is_shared=False,
            kind_known=True,
            org_id="test-org-id",
            dataset_index=None,
        ),
    )

    nc.start_recording()

    # By its server id, never looked up again by name.
    assert fake_registry.dataset_specs == []
    dataset = GlobalSingleton()._active_dataset
    assert dataset.dataset_index == fake_registry.dataset_row_for_cloud_id(
        org_id="test-org-id", cloud_dataset_id="cloud-dataset-9"
    )
    start_recording.assert_called_once_with(dataset.dataset_index, timestamp=None)
