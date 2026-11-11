import pytest

from neuracore.core.exceptions import DatasetError, RobotArchivedError, RobotError
from neuracore.importer.core.exceptions import DatasetOperationError
from neuracore.importer.importer import (
    _connect_import_robot,
    _create_dataset_with_overwrite_guard,
    _robot_exists,
)


def test_create_dataset_with_overwrite_guard_retries_if_old_id_returned(monkeypatch):
    class _Dataset:
        def __init__(self, dataset_id: str):
            self.id = dataset_id

    responses = [_Dataset("old-id"), _Dataset("old-id"), _Dataset("new-id")]

    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "neuracore.importer.importer.nc.create_dataset",
        lambda **_kwargs: responses.pop(0),
    )
    monkeypatch.setattr(
        "neuracore.importer.importer.time.sleep",
        lambda interval: sleep_calls.append(interval),
    )

    result = _create_dataset_with_overwrite_guard(
        dataset_name="my-dataset",
        description="desc",
        tags=["tag"],
        shared=False,
        deleted_dataset_id="old-id",
        timeout_s=10.0,
        poll_interval_s=0.5,
    )

    assert result.id == "new-id"
    assert sleep_calls == [0.5, 0.5]


def test_create_dataset_with_overwrite_guard_raises_on_timeout(monkeypatch):
    class _Dataset:
        def __init__(self, dataset_id: str):
            self.id = dataset_id

    monkeypatch.setattr(
        "neuracore.importer.importer.nc.create_dataset",
        lambda **_kwargs: _Dataset("old-id"),
    )
    monkeypatch.setattr("neuracore.importer.importer.time.sleep", lambda *_args: None)

    monotonic_values = iter([0.0, 0.5, 1.1])
    monkeypatch.setattr(
        "neuracore.importer.importer.time.monotonic",
        lambda: next(monotonic_values),
    )

    with pytest.raises(
        DatasetOperationError,
        match="Timed out waiting to create replacement dataset 'my-dataset'.",
    ):
        _create_dataset_with_overwrite_guard(
            dataset_name="my-dataset",
            description=None,
            tags=None,
            shared=False,
            deleted_dataset_id="old-id",
            timeout_s=1.0,
            poll_interval_s=0.1,
        )


def test_create_dataset_with_overwrite_guard_retries_on_transient_create_error(
    monkeypatch,
):
    class _Dataset:
        def __init__(self, dataset_id: str):
            self.id = dataset_id

    responses = iter([
        DatasetError("Failed to create dataset: Dataset already exists"),
        _Dataset("new-id"),
    ])

    def fake_create_dataset(**_kwargs):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "neuracore.importer.importer.nc.create_dataset",
        fake_create_dataset,
    )
    monkeypatch.setattr(
        "neuracore.importer.importer.time.sleep",
        lambda interval: sleep_calls.append(interval),
    )

    result = _create_dataset_with_overwrite_guard(
        dataset_name="my-dataset",
        description=None,
        tags=None,
        shared=False,
        deleted_dataset_id="old-id",
        timeout_s=10.0,
        poll_interval_s=0.5,
    )

    assert result.id == "new-id"
    assert sleep_calls == [0.5]


def test_robot_exists_checks_private_and_shared_mixed_lists(monkeypatch):
    monkeypatch.setattr(
        "neuracore.importer.importer.get_current_org",
        lambda: "org-1",
    )

    def fake_list(_org_id, is_shared=False, mode="current"):
        assert mode == "mixed"
        if is_shared:
            return [{"name": "shared-bot"}]
        return [{"name": "private-bot"}]

    monkeypatch.setattr(
        "neuracore.importer.importer.list_organization_robots",
        fake_list,
    )
    assert _robot_exists("private-bot") is True
    assert _robot_exists("shared-bot") is True
    assert _robot_exists("missing") is False


class _FakeConnect:
    """nc.connect_robot that refuses the first calls as archived."""

    def __init__(self, archived_refusals: int) -> None:
        self.archived_refusals = archived_refusals
        self.calls: list[dict] = []
        self.robot = object()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.archived_refusals:
            raise RobotArchivedError("Robot 'bot' is archived.")
        return self.robot


def _patch_robot_api(monkeypatch, connect, unarchive_error=None):
    unarchived = []

    class FakeRobot:
        def __init__(self, robot_name, instance, shared=False):
            self.robot_name = robot_name
            self.shared = shared

        def set_archived(self, archived=True):
            if unarchive_error is not None:
                raise unarchive_error
            unarchived.append((self.robot_name, self.shared, archived))

    monkeypatch.setattr("neuracore.importer.importer.nc.connect_robot", connect)
    monkeypatch.setattr("neuracore.importer.importer.Robot", FakeRobot)
    return unarchived


def test_connect_import_robot_connects_once_when_not_archived(monkeypatch):
    connect = _FakeConnect(archived_refusals=0)
    unarchived = _patch_robot_api(monkeypatch, connect)

    robot, was_unarchived = _connect_import_robot("bot", None, None, False, False)

    assert robot is connect.robot
    assert was_unarchived is False
    assert len(connect.calls) == 1
    assert unarchived == []


def test_connect_import_robot_unarchives_then_connects_again(monkeypatch):
    connect = _FakeConnect(archived_refusals=1)
    unarchived = _patch_robot_api(monkeypatch, connect)

    robot, was_unarchived = _connect_import_robot("bot", "robot.urdf", None, True, True)

    assert robot is connect.robot
    assert was_unarchived is True
    assert unarchived == [("bot", True, False)]
    assert len(connect.calls) == 2
    assert connect.calls[0] == connect.calls[1]
    assert connect.calls[1]["urdf_path"] == "robot.urdf"


def test_connect_import_robot_raises_when_the_robot_cannot_be_unarchived(
    monkeypatch,
):
    connect = _FakeConnect(archived_refusals=1)
    _patch_robot_api(
        monkeypatch, connect, unarchive_error=RobotError("No archived robot")
    )

    with pytest.raises(RobotError, match="No archived robot"):
        _connect_import_robot("bot", None, None, False, False)

    assert len(connect.calls) == 1
