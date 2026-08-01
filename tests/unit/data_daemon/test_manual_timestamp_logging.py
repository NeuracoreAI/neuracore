from __future__ import annotations

from collections import defaultdict

import pytest

from tests.integration.platform.data_daemon.shared.test_case import (
    build_test_case_context as context_module,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DETAIL_FLAT,
    PACING_BURST_ALL,
)


def _fail_sleep(*args: object, **kwargs: object) -> None:
    del args, kwargs
    pytest.fail("manual timestamp logging must not sleep for wall-clock pacing")


def test_synchronous_manual_logging_uses_precomputed_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps: dict[str, list[float]] = defaultdict(list)

    monkeypatch.setattr(context_module.time, "sleep", _fail_sleep)
    for name in (
        "log_joint_positions",
        "log_joint_velocities",
        "log_joint_torques",
    ):
        monkeypatch.setattr(
            context_module.nc,
            name,
            lambda *args, _name=name, **kwargs: timestamps[_name].append(
                kwargs["timestamp"]
            ),
        )
    monkeypatch.setattr(
        context_module.nc,
        "log_custom_1d",
        lambda *args, **kwargs: timestamps["custom"].append(kwargs["timestamp"]),
    )
    monkeypatch.setattr(
        context_module.nc,
        "log_rgb",
        lambda *args, **kwargs: timestamps["rgb"].append(kwargs["timestamp"]),
    )

    context_module.log_synchronous_frames(
        robot_name="robot",
        joint_frame_count=3,
        video_frame_count=2,
        recording_index=0,
        timestamp_start_s=10.0,
        joint_names=["joint"],
        camera_name_list=["camera"],
        image_width=4,
        image_height=4,
        joint_fps=2,
        video_fps=1,
        marker_name="marker",
        context_index=0,
        video_detail=DETAIL_FLAT,
        pacing=PACING_BURST_ALL,
    )

    expected_joint_timestamps = [10.0, 10.5, 11.0]
    assert timestamps["log_joint_positions"] == expected_joint_timestamps
    assert timestamps["log_joint_velocities"] == expected_joint_timestamps
    assert timestamps["log_joint_torques"] == expected_joint_timestamps
    assert timestamps["custom"] == expected_joint_timestamps
    assert timestamps["rgb"] == [10.0, 11.0]


def test_threaded_manual_logging_uses_precomputed_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps: dict[str, list[float]] = defaultdict(list)

    monkeypatch.setattr(context_module.time, "sleep", _fail_sleep)
    monkeypatch.setattr(context_module, "set_thread_policy_for_macos", lambda: None)
    for name in (
        "log_joint_positions",
        "log_joint_velocities",
        "log_joint_torques",
    ):
        monkeypatch.setattr(
            context_module.nc,
            name,
            lambda *args, _name=name, **kwargs: timestamps[_name].append(
                kwargs["timestamp"]
            ),
        )
    monkeypatch.setattr(
        context_module.nc,
        "log_custom_1d",
        lambda marker_name, *args, **kwargs: timestamps[f"custom:{marker_name}"].append(
            kwargs["timestamp"]
        ),
    )
    monkeypatch.setattr(
        context_module.nc,
        "log_rgb",
        lambda *args, **kwargs: timestamps["rgb"].append(kwargs["timestamp"]),
    )

    context_module.run_threaded_logging(
        robot_name="robot",
        joint_frame_count=3,
        video_frame_count=2,
        recording_index=0,
        timestamp_start_s=20.0,
        joint_fps=4,
        video_fps=2,
        context_index=0,
        joint_names=["joint"],
        camera_name_list=["camera"],
        image_width=4,
        image_height=4,
        video_detail=DETAIL_FLAT,
        pacing=PACING_BURST_ALL,
    )

    expected_joint_timestamps = [20.0, 20.25, 20.5]
    assert timestamps["log_joint_positions"] == expected_joint_timestamps
    assert timestamps["log_joint_velocities"] == expected_joint_timestamps
    assert timestamps["log_joint_torques"] == expected_joint_timestamps
    assert timestamps["custom:marker_joint_positions"] == expected_joint_timestamps
    assert timestamps["custom:marker_joint_velocities"] == expected_joint_timestamps
    assert timestamps["custom:marker_joint_torques"] == expected_joint_timestamps
    assert timestamps["rgb"] == [20.0, 20.5]
    assert timestamps["custom:marker_camera"] == [20.0, 20.5]
