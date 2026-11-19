from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from neuracore.data_daemon import bridge


class _NativeUnavailable(RuntimeError):
    pass


class _NativeRefused(RuntimeError):
    pass


@pytest.fixture
def native(monkeypatch) -> MagicMock:
    native = MagicMock()
    native.RegistryUnavailableError = _NativeUnavailable
    native.RegistryRefusedError = _NativeRefused
    monkeypatch.setattr(bridge, "_load_native", lambda: native)
    monkeypatch.setattr(bridge, "ensure_daemon_running", lambda: None)
    return native


def test_connecting_a_robot_returns_its_row_index(native) -> None:
    native.connect_robot.return_value = 12

    index = bridge.connect_robot_row(
        org_id="org-1",
        name="arm",
        shared=False,
        instance=7,
        overwrite_urdf=True,
        package_path="/saved/package.zip",
    )

    assert index == 12
    native.connect_robot.assert_called_once_with(
        org_id="org-1",
        name="arm",
        shared=False,
        instance=7,
        overwrite_urdf=True,
        package_path="/saved/package.zip",
        timeout_s=bridge.REGISTRY_TIMEOUT_S,
    )


def test_selecting_a_dataset_returns_its_row_index(native) -> None:
    native.select_dataset.return_value = 4

    index = bridge.select_dataset_row(
        org_id="org-1",
        name="warehouse",
        shared=False,
        description="pick and place",
        tags=["shelf-a"],
    )

    assert index == 4
    native.select_dataset.assert_called_once_with(
        org_id="org-1",
        name="warehouse",
        shared=False,
        description="pick and place",
        tags=["shelf-a"],
        timeout_s=bridge.REGISTRY_TIMEOUT_S,
    )


def test_a_row_state_is_read_back(native) -> None:
    native.robot_state.return_value = SimpleNamespace(
        status="pending", cloud_id=None, reason="robot_archived"
    )
    native.dataset_state.return_value = None

    assert bridge.robot_row_state(12) == bridge.RowState(
        status="pending", cloud_id=None, reason="robot_archived"
    )
    native.robot_state.assert_called_once_with(
        robot_index=12, timeout_s=bridge.REGISTRY_TIMEOUT_S
    )
    assert bridge.dataset_row_state(4) is None


def test_native_errors_become_sdk_errors(native) -> None:
    native.connect_robot.side_effect = _NativeRefused("no organization is selected")
    native.select_dataset.side_effect = _NativeUnavailable("no answer")

    with pytest.raises(bridge.RegistryRefusedError, match="no organization"):
        bridge.connect_robot_row(
            org_id="",
            name="arm",
            shared=False,
            instance=0,
            overwrite_urdf=False,
            package_path=None,
        )
    with pytest.raises(bridge.RegistryUnavailableError):
        bridge.select_dataset_row(
            org_id="org-1",
            name="warehouse",
            shared=False,
            description=None,
            tags=[],
        )


def test_reading_a_row_state_never_starts_the_daemon(native, monkeypatch) -> None:
    started = []
    monkeypatch.setattr(bridge, "ensure_daemon_running", lambda: started.append(1))
    native.robot_state.return_value = None
    native.dataset_state.return_value = None

    bridge.robot_row_state(1)
    bridge.dataset_row_state(1)

    assert started == []
