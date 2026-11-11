"""Datasets selected after a login that could not reach the server."""

from collections.abc import Iterator
from unittest.mock import Mock

import pytest
import requests_mock

import neuracore as nc
from neuracore.api.globals import GlobalSingleton
from neuracore.core.auth import get_auth
from neuracore.core.data.dataset import (
    LOCAL_DATASET_ID_PREFIX,
    Dataset,
    local_dataset_id,
)
from neuracore.core.exceptions import ServerUnreachableError
from neuracore.core.robot import Robot
from neuracore.data_daemon import bridge

ORG_ID = "11111111-1111-1111-1111-111111111111"

# The offline warnings are checked in their own tests; elsewhere they are noise.
ignore_offline_warnings = pytest.mark.filterwarnings(
    "ignore:Dataset .* will be found:UserWarning"
)


@pytest.fixture
def offline(monkeypatch, reset_neuracore) -> Iterator[requests_mock.Mocker]:
    """Log in offline; any HTTP request made during the test fails it."""
    monkeypatch.setenv("NEURACORE_ORG_ID", ORG_ID)
    auth = get_auth()
    monkeypatch.setattr(auth, "_access_token", None)
    monkeypatch.setattr(auth, "_offline", True)
    with requests_mock.Mocker() as no_network:
        yield no_network
    assert not no_network.called


def test_create_dataset_offline_returns_a_local_dataset(offline):
    with pytest.warns(UserWarning, match="will be found, or created"):
        dataset = nc.create_dataset("warehouse", description="d", tags=["t"])

    assert dataset.is_local
    assert dataset.id.startswith(LOCAL_DATASET_ID_PREFIX)
    assert dataset.name == "warehouse"
    assert dataset.org_id == ORG_ID
    assert not dataset.is_shared
    assert dataset.description == "d"
    assert dataset.tags == ["t"]
    assert GlobalSingleton()._active_dataset_id == dataset.id
    assert GlobalSingleton()._active_dataset is dataset


@ignore_offline_warnings
def test_get_dataset_offline_agrees_with_create_dataset(offline):
    created = nc.create_dataset("warehouse")
    selected = nc.get_dataset("warehouse")

    assert selected.is_local
    assert selected.id == created.id
    assert GlobalSingleton()._active_dataset_id == created.id


@ignore_offline_warnings
def test_local_dataset_id_depends_on_name_and_namespace(offline):
    private = Dataset.local("warehouse")
    shared = Dataset.local("warehouse", shared=True)
    other_name = Dataset.local("factory")

    assert len({private.id, shared.id, other_name.id}) == 3
    assert shared.is_shared


def test_local_dataset_id_never_changes():
    # The data daemon stores these ids, and processes on different SDK
    # versions must agree on them.
    assert local_dataset_id("warehouse", shared=False) == (
        "local-dataset-efe17dc6-d333-55a7-aa83-20f7d18f85f8"
    )
    assert local_dataset_id("warehouse", shared=True) == (
        "local-dataset-3f8e4edc-0392-5413-b499-985412b1023e"
    )


@ignore_offline_warnings
def test_local_dataset_id_is_the_same_whatever_the_org(offline, monkeypatch):
    first = Dataset.local("warehouse")
    monkeypatch.setenv("NEURACORE_ORG_ID", "22222222-2222-2222-2222-222222222222")
    second = Dataset.local("warehouse")

    assert first.id == second.id
    assert second.org_id == "22222222-2222-2222-2222-222222222222"


def test_create_dataset_offline_with_no_org_selected(
    offline, monkeypatch, temp_config_dir
):
    monkeypatch.delenv("NEURACORE_ORG_ID")

    with pytest.warns(UserWarning, match="will be found, or created"):
        dataset = nc.create_dataset("warehouse")

    assert dataset.is_local
    assert dataset.org_id is None
    assert GlobalSingleton()._active_dataset_id == dataset.id


def test_offline_dataset_warnings_point_at_the_caller(offline, temp_config_dir):
    with pytest.warns(UserWarning) as from_create_dataset:
        nc.create_dataset("warehouse")
    with pytest.warns(UserWarning) as from_dataset_create:
        Dataset.create("warehouse")
    with pytest.warns(UserWarning) as from_get_dataset:
        nc.get_dataset("warehouse")

    warnings = [*from_create_dataset, *from_dataset_create, *from_get_dataset]
    assert len(warnings) == 3
    assert {w.filename for w in warnings} == {__file__}


def test_get_dataset_by_id_offline_says_to_use_the_name(offline):
    with pytest.raises(ServerUnreachableError, match="by name"):
        nc.get_dataset(id="0b4c3a7e-0000-0000-0000-000000000000")

    assert GlobalSingleton()._active_dataset_id is None


def test_a_recording_into_a_dataset_selected_offline_sends_its_details(
    offline, monkeypatch
):
    # A robot connected while online, recording into a dataset that only
    # exists on this machine so far.
    native = Mock()
    monkeypatch.setattr(bridge, "_load_native", lambda: native)
    monkeypatch.setattr(bridge, "ensure_daemon_running", lambda: None)
    robot = Robot("arm", 0)
    robot.id = "cloud-robot-1"
    GlobalSingleton()._active_robot = robot
    with pytest.warns(UserWarning, match="will be found"):
        dataset = nc.create_dataset("warehouse", description="d", tags=["t"])

    nc.start_recording()

    native.start_recording.assert_not_called()
    args = native.start_local_recording.call_args
    assert args.args[:3] == ("cloud-robot-1", 0, dataset.id)
    assert args.kwargs["local_robot_name"] is None
    assert args.kwargs["local_dataset_name"] == "warehouse"
    assert args.kwargs["local_dataset_shared"] is False
    assert args.kwargs["local_dataset_description"] == "d"
    assert args.kwargs["local_dataset_tags"] == ["t"]

    # Avoid Robot.__del__ consulting the process-global recording manager.
    robot.id = None
