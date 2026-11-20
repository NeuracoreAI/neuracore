from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neuracore.core.exceptions import RobotError
from neuracore.core.robot import Robot
from neuracore.data_daemon.bridge import (
    ImportItemSource,
    LiveRecording,
    RecordingStateUnavailableError,
    RobotInstanceSource,
)


def test_is_recording_answers_from_the_daemon() -> None:
    """The daemon owns recording state, so it answers even for a recording
    this process never started — which is what makes web-driven and
    cross-process recordings visible here."""
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"

    live = LiveRecording(
        recording_index=7, cloud_recording_id="cloud-1", start_timestamp_us=1_000
    )
    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        return_value=live,
    ):
        assert robot.is_recording() is True
        assert robot.get_cloud_recording_id(timeout_s=0.0) == "cloud-1"

    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        return_value=None,
    ):
        assert robot.is_recording() is False
        assert robot.get_cloud_recording_id(timeout_s=0.0) is None

    robot.id = None


def test_an_unanswered_query_raises_rather_than_reading_as_not_recording() -> None:
    """A silent daemon must not read as "not recording".

    `nc.stop_recording` gives up when `is_recording` is False, so answering
    False on silence would leave a recording running with its stop never
    published. The caller is told the state is unknown instead.
    """
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"
    robot._daemon_recording_context = MagicMock()

    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        side_effect=RecordingStateUnavailableError("no answer"),
    ):
        with pytest.raises(RecordingStateUnavailableError):
            robot.is_recording()
        with pytest.raises(RecordingStateUnavailableError):
            robot.get_cloud_recording_id(timeout_s=0.0)

    robot.id = None


def test_the_cloud_id_wait_is_handed_to_the_bridge() -> None:
    """The daemon mints the cloud id after the recording opens, so a caller has
    to wait for it. The wait belongs in the bridge, which drops the GIL while it
    polls, rather than in a Python loop across the boundary."""
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"
    minted = LiveRecording(
        recording_index=7, cloud_recording_id="cloud-1", start_timestamp_us=1_000
    )

    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        return_value=minted,
    ) as query_recording_state:
        assert robot.get_cloud_recording_id(timeout_s=1.0) == "cloud-1"

    query_recording_state.assert_called_once_with(
        RobotInstanceSource("robot-id-1", 0), timeout_s=1.0
    )
    robot.id = None


def test_a_cloud_id_that_never_arrives_gives_up() -> None:
    """An offline recording never gets an id, so this must end, not hang."""
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"
    pending = LiveRecording(
        recording_index=7, cloud_recording_id=None, start_timestamp_us=1_000
    )

    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        return_value=pending,
    ):
        assert robot.get_cloud_recording_id(timeout_s=1.0) is None

    robot.id = None


@pytest.fixture
def native(monkeypatch) -> MagicMock:
    from neuracore.data_daemon import bridge

    native_module = MagicMock()
    monkeypatch.setattr(bridge, "_load_native", lambda: native_module)
    monkeypatch.setattr(bridge, "ensure_daemon_running", lambda: None)
    return native_module


def test_an_ordinary_start_records_under_the_robot_instance(native) -> None:
    robot = Robot("robot", instance=2, org_id="org-1")
    robot.id = "robot-id-1"

    robot.start_recording("dataset-1", timestamp=12.5)

    native.RecordingSource.robot_instance.assert_called_once_with("robot-id-1", 2)
    native.RecordingSource.import_item.assert_not_called()
    native.start_recording.assert_called_once_with(
        native.RecordingSource.robot_instance.return_value,
        "robot",
        "dataset-1",
        None,
        12_500_000,
    )
    robot.id = None


def test_an_import_start_records_and_answers_under_the_item(native) -> None:
    robot = Robot("robot", instance=None, org_id="org-1")
    robot.id = "robot-id-1"

    robot.start_recording(
        "dataset-1", timestamp=12.5, import_session_id="session-1", item_key="ep-3"
    )

    native.RecordingSource.robot_instance.assert_not_called()
    native.RecordingSource.import_item.assert_called_once_with(
        "robot-id-1", "session-1", "ep-3"
    )
    assert native.start_recording.call_args.args[0] is (
        native.RecordingSource.import_item.return_value
    )
    with patch(
        "neuracore.core.robot.recording_context.query_recording_state",
        return_value=None,
    ) as query_recording_state:
        robot.is_recording()
    query_recording_state.assert_called_once_with(
        ImportItemSource("robot-id-1", "session-1", "ep-3")
    )
    robot.id = None


@pytest.mark.parametrize(
    "import_args", [{"import_session_id": "session-1"}, {"item_key": "ep-3"}]
)
def test_an_import_start_needs_both_identifiers(native, import_args) -> None:
    robot = Robot("robot", instance=0, org_id="org-1")
    robot.id = "robot-id-1"

    with pytest.raises(ValueError, match="given together"):
        robot.start_recording("dataset-1", **import_args)

    native.start_recording.assert_not_called()
    robot.id = None


def test_a_handle_without_an_instance_only_records_import_items(native) -> None:
    robot = Robot("robot", instance=None, org_id="org-1")
    robot.id = "robot-id-1"

    with pytest.raises(RobotError, match="only records import items"):
        robot.start_recording("dataset-1")

    native.start_recording.assert_not_called()
    native.RecordingSource.robot_instance.assert_not_called()
    robot.id = None


def test_a_handle_without_an_instance_has_no_source_before_its_first_item(
    native,
) -> None:
    """Nothing it does before an import start may reach a robot instance."""
    robot = Robot("robot", instance=None, org_id="org-1")
    robot.id = "robot-id-1"

    with patch(
        "neuracore.core.robot.recording_context.query_recording_state"
    ) as query_recording_state:
        assert robot.is_recording() is False
        assert robot.get_cloud_recording_id(timeout_s=0.0) is None
    query_recording_state.assert_not_called()

    with pytest.raises(RuntimeError):
        robot._get_daemon_recording_context().log_joints(
            "JOINT_POSITIONS", 1_000, "j0", [0.0]
        )
    robot.stop_recording()
    robot.cancel_recording()

    native.stop_recording.assert_not_called()
    native.cancel_recording.assert_not_called()
    native.log_joints.assert_not_called()
    native.RecordingSource.robot_instance.assert_not_called()
    robot.id = None


def test_each_import_item_records_under_its_own_source(native) -> None:
    robot = Robot("robot", instance=None, org_id="org-1")
    robot.id = "robot-id-1"
    native.RecordingSource.import_item.side_effect = lambda *args: args

    for item_key in ("ep-1", "ep-2"):
        robot.start_recording(
            "dataset-1", import_session_id="session-1", item_key=item_key
        )
        robot.stop_recording()

    started = [call.args[0] for call in native.start_recording.call_args_list]
    stopped = [call.args[0] for call in native.stop_recording.call_args_list]
    assert (
        started
        == stopped
        == [
            ("robot-id-1", "session-1", "ep-1"),
            ("robot-id-1", "session-1", "ep-2"),
        ]
    )
    native.RecordingSource.robot_instance.assert_not_called()
    robot.id = None
