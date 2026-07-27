from __future__ import annotations

from collections import defaultdict

import pytest

from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
    SYNCHRONOUS_PLAN_KEY,
    ContextCaseSpec,
    StochasticReplayScheduler,
    build_stochastic_timestamp_plan,
    log_synchronous_frames,
    run_threaded_logging,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    PRODUCER_PER_THREAD,
    PRODUCER_SYNCHRONOUS,
    SCHEDULER_TOLERANCE_S,
    STOP_RECORDING_NO_WAIT_SLA_S,
    TIMESTAMP_MODE_MANUAL,
    TIMESTAMP_MODE_STOCHASTIC,
    stochastic_jitter_window,
)


def _fail_on_sleep(_seconds: float) -> None:
    pytest.fail("stochastic replay must not sleep")


class _ImmediateStochasticScheduler:
    """Record scheduled timestamps without making unit tests wait in real time."""

    def __init__(self) -> None:
        self.started = False
        self.cancelled = False
        self.waited_timestamps: list[float] = []
        self.producer_keys: list[str] = []
        self.deadline_checks: list[tuple[bool, str]] = []

    def start(self) -> None:
        self.started = True

    def wait_until(
        self,
        logical_timestamp_s: float,
        *,
        producer_key: str = "default",
        assert_deadline: bool = False,
        label: str = "stochastic frame",
    ) -> bool:
        assert self.started
        self.waited_timestamps.append(logical_timestamp_s)
        self.producer_keys.append(producer_key)
        self.deadline_checks.append((assert_deadline, label))
        return not self.cancelled

    def cancel(self) -> None:
        self.cancelled = True


def _context_case_spec(*, timestamp_mode: str) -> ContextCaseSpec:
    return ContextCaseSpec(
        duration_sec=60,
        joint_count=10,
        producer_channels=PRODUCER_SYNCHRONOUS,
        video_count=1,
        image_width=1920,
        image_height=1080,
        joint_fps=15,
        video_fps=15,
        wait=False,
        timestamp_mode=timestamp_mode,
    )


@pytest.mark.parametrize(
    "timestamp_mode", [TIMESTAMP_MODE_MANUAL, TIMESTAMP_MODE_STOCHASTIC]
)
def test_no_wait_stop_sla_remains_strict(timestamp_mode: str) -> None:
    case = _context_case_spec(timestamp_mode=timestamp_mode)

    assert case.stop_recording_sla_s == STOP_RECORDING_NO_WAIT_SLA_S


def test_stochastic_scheduler_uses_interruptible_monotonic_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.integration.platform.data_daemon.shared.test_case import (
        build_test_case_context as context,
    )

    timeouts: list[float] = []

    def recording_select(
        _readers: list[object],
        _writers: list[object],
        _errors: list[object],
        timeout: float,
    ) -> tuple[list[object], list[object], list[object]]:
        timeouts.append(timeout)
        return [], [], []

    monotonic_ticks = iter((100.0, 100.25, 102.0))
    monkeypatch.setattr(context.select, "select", recording_select)
    monkeypatch.setattr(context.time, "monotonic", lambda: next(monotonic_ticks))

    scheduler = StochasticReplayScheduler(logical_start_s=10.0)
    scheduler.start()

    assert scheduler.wait_until(12.0) is True
    assert timeouts == [pytest.approx(1.75)]

    scheduler.cancel()
    assert scheduler.wait_until(13.0) is False


def test_stochastic_scheduler_cancel_wakes_select_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import select
    import threading

    from tests.integration.platform.data_daemon.shared.test_case import (
        build_test_case_context as context,
    )

    select_entered = threading.Event()
    original_select = select.select

    def notifying_select(
        readers: list[object],
        writers: list[object],
        errors: list[object],
        timeout: float,
    ) -> tuple[list[object], list[object], list[object]]:
        select_entered.set()
        return original_select(readers, writers, errors, timeout)

    monkeypatch.setattr(context.select, "select", notifying_select)
    scheduler = StochasticReplayScheduler(logical_start_s=10.0)
    scheduler.start()
    results: list[bool] = []
    waiter = threading.Thread(target=lambda: results.append(scheduler.wait_until(20.0)))

    waiter.start()
    assert select_entered.wait(timeout=1.0)
    scheduler.cancel()
    waiter.join(timeout=1.0)

    assert not waiter.is_alive()
    assert results == [False]


@pytest.mark.parametrize(
    ("actual_offset_s", "should_raise"),
    [
        (SCHEDULER_TOLERANCE_S - 0.001, False),
        (SCHEDULER_TOLERANCE_S + 0.001, True),
    ],
)
def test_stochastic_scheduler_validates_producer_lateness(
    monkeypatch: pytest.MonkeyPatch,
    actual_offset_s: float,
    should_raise: bool,
) -> None:
    from tests.integration.platform.data_daemon.shared.test_case import (
        build_test_case_context as context,
    )

    monotonic_ticks = iter((100.0, 100.0 + actual_offset_s, 100.0 + actual_offset_s))
    monkeypatch.setattr(context.time, "monotonic", lambda: next(monotonic_ticks))

    scheduler = StochasticReplayScheduler(logical_start_s=10.0)
    scheduler.start()

    if should_raise:
        with pytest.raises(
            AssertionError, match="joint frame producer reached scheduler too late"
        ):
            scheduler.wait_until(10.0, assert_deadline=True, label="joint frame")
    else:
        assert scheduler.wait_until(10.0, assert_deadline=True, label="joint frame")


def test_stochastic_scheduler_excludes_kernel_wake_delay_from_producer_sla(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from tests.integration.platform.data_daemon.shared.test_case import (
        build_test_case_context as context,
    )

    monotonic_ticks = iter((100.0, 100.0, 100.2))
    monkeypatch.setattr(
        context.select,
        "select",
        lambda _readers, _writers, _errors, _timeout: ([], [], []),
    )
    monkeypatch.setattr(context.time, "monotonic", lambda: next(monotonic_ticks))

    scheduler = StochasticReplayScheduler(logical_start_s=10.0)
    scheduler.start()

    with caplog.at_level("WARNING"):
        assert scheduler.wait_until(
            10.1, assert_deadline=True, label="synchronous tick 199"
        )

    assert "dispatch_lateness=+0.100s" in caplog.text
    assert "excluded from producer SLA" in caplog.text


def test_stochastic_scheduler_rebases_after_small_lateness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.integration.platform.data_daemon.shared.test_case import (
        build_test_case_context as context,
    )

    clock = [100.0]

    timeouts: list[float] = []

    def advancing_select(
        _readers: list[object],
        _writers: list[object],
        _errors: list[object],
        timeout: float,
    ) -> tuple[list[object], list[object], list[object]]:
        timeouts.append(timeout)
        clock[0] += timeout
        return [], [], []

    monkeypatch.setattr(context.select, "select", advancing_select)
    monkeypatch.setattr(context.time, "monotonic", lambda: clock[0])

    scheduler = StochasticReplayScheduler(logical_start_s=10.0)
    scheduler.start()

    clock[0] += 0.040
    assert scheduler.wait_until(10.0, assert_deadline=True)
    assert scheduler.wait_until(10.1, assert_deadline=True)

    assert timeouts == [pytest.approx(0.100)]
    assert clock[0] == pytest.approx(100.14)


def test_stochastic_scheduler_reports_when_producer_enters_late(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.integration.platform.data_daemon.shared.test_case import (
        build_test_case_context as context,
    )

    clock = [100.0]
    monkeypatch.setattr(context.time, "monotonic", lambda: clock[0])

    scheduler = StochasticReplayScheduler(logical_start_s=10.0)
    scheduler.start()
    clock[0] += 0.072

    with pytest.raises(AssertionError, match=r"producer_lateness=\+0.072s"):
        scheduler.wait_until(10.0, assert_deadline=True, label="joint frame 417")


def test_stochastic_plan_is_deterministic_and_bounded() -> None:
    arguments = {
        "timestamp_start_s": 10.0,
        "joint_frame_count": 20,
        "video_frame_count": 20,
        "joint_fps": 20,
        "video_fps": 20,
        "producer_channels": PRODUCER_SYNCHRONOUS,
        "joint_names": ["joint"],
        "camera_name_list": ["camera"],
        "context_index": 2,
        "recording_index": 3,
    }

    first = build_stochastic_timestamp_plan(**arguments)
    second = build_stochastic_timestamp_plan(**arguments)

    assert first == second
    assert set(first.by_producer) == {SYNCHRONOUS_PLAN_KEY}
    timestamps = first.timestamps_for(SYNCHRONOUS_PLAN_KEY)
    window = stochastic_jitter_window(20)
    for frame_index, timestamp in enumerate(timestamps):
        nominal = 10.0 + (frame_index / 20)
        assert abs(timestamp - nominal) <= window
    assert list(timestamps) == sorted(timestamps)


def test_synchronous_stochastic_plan_rejects_independent_rates() -> None:
    with pytest.raises(ValueError, match="requires matching frame counts and fps"):
        build_stochastic_timestamp_plan(
            timestamp_start_s=10.0,
            joint_frame_count=20,
            video_frame_count=10,
            joint_fps=20,
            video_fps=10,
            producer_channels=PRODUCER_SYNCHRONOUS,
            joint_names=["joint"],
            camera_name_list=["camera"],
            context_index=0,
            recording_index=0,
        )


def test_stochastic_scheduler_rebases_each_producer_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.integration.platform.data_daemon.shared.test_case import (
        build_test_case_context as context,
    )

    clock = [100.0]
    timeouts: list[float] = []

    def advancing_select(
        _readers: list[object],
        _writers: list[object],
        _errors: list[object],
        timeout: float,
    ) -> tuple[list[object], list[object], list[object]]:
        timeouts.append(timeout)
        clock[0] += timeout
        return [], [], []

    monkeypatch.setattr(context.select, "select", advancing_select)
    monkeypatch.setattr(context.time, "monotonic", lambda: clock[0])

    scheduler = StochasticReplayScheduler(logical_start_s=10.0)
    scheduler.start()

    clock[0] += 0.010
    assert scheduler.wait_until(10.0, producer_key="joint")
    clock[0] += 0.020
    assert scheduler.wait_until(10.0, producer_key="video")
    assert scheduler.wait_until(10.1, producer_key="joint")
    assert scheduler.wait_until(10.1, producer_key="video")

    assert timeouts == [pytest.approx(0.080), pytest.approx(0.020)]
    assert clock[0] == pytest.approx(100.13)


def test_synchronous_stochastic_replay_uses_plan_without_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.integration.platform.data_daemon.shared.test_case import (
        build_test_case_context as context,
    )

    timestamps: dict[str, list[float]] = defaultdict(list)
    monkeypatch.setattr(context.time, "sleep", _fail_on_sleep)
    monkeypatch.setattr(
        context.nc,
        "log_joint_positions",
        lambda _values, *, robot_name, timestamp: timestamps["joint"].append(timestamp),
    )
    monkeypatch.setattr(
        context.nc,
        "log_joint_velocities",
        lambda _values, *, robot_name, timestamp: None,
    )
    monkeypatch.setattr(
        context.nc,
        "log_joint_torques",
        lambda _values, *, robot_name, timestamp: None,
    )
    monkeypatch.setattr(
        context.nc,
        "log_custom_1d",
        lambda _name, _values, *, robot_name, timestamp: None,
    )
    monkeypatch.setattr(
        context.nc,
        "log_rgb",
        lambda _name, _image, *, robot_name, timestamp: timestamps["video"].append(
            timestamp
        ),
    )

    plan = build_stochastic_timestamp_plan(
        timestamp_start_s=5.0,
        joint_frame_count=4,
        video_frame_count=4,
        joint_fps=4,
        video_fps=4,
        producer_channels=PRODUCER_SYNCHRONOUS,
        joint_names=["joint"],
        camera_name_list=["camera"],
        context_index=0,
        recording_index=0,
    )
    scheduler = _ImmediateStochasticScheduler()
    log_synchronous_frames(
        robot_name="robot",
        joint_frame_count=4,
        video_frame_count=4,
        recording_index=0,
        timestamp_start_s=5.0,
        joint_names=["joint"],
        camera_name_list=["camera"],
        image_width=4,
        image_height=4,
        joint_fps=4,
        video_fps=4,
        marker_name="marker",
        context_index=0,
        use_stochastic_timestamps=True,
        stochastic_plan=plan,
        stochastic_scheduler=scheduler,
        assert_deadline=True,
    )

    shared_timestamps = list(plan.timestamps_for(SYNCHRONOUS_PLAN_KEY))
    assert timestamps["joint"] == shared_timestamps
    assert timestamps["video"] == shared_timestamps
    assert scheduler.waited_timestamps == shared_timestamps
    assert scheduler.producer_keys == [SYNCHRONOUS_PLAN_KEY] * 4
    assert scheduler.cancelled is False
    assert scheduler.deadline_checks == [
        (True, f"synchronous tick {frame_index}") for frame_index in range(4)
    ]


def test_threaded_stochastic_replay_uses_per_role_plan_without_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.integration.platform.data_daemon.shared.test_case import (
        build_test_case_context as context,
    )

    timestamps: dict[str, list[float]] = defaultdict(list)
    monkeypatch.setattr(context.time, "sleep", _fail_on_sleep)
    monkeypatch.setattr(
        context.nc,
        "log_joint_positions",
        lambda _values, *, robot_name, timestamp: timestamps[
            "marker_joint_positions"
        ].append(timestamp),
    )
    monkeypatch.setattr(
        context.nc,
        "log_joint_velocities",
        lambda _values, *, robot_name, timestamp: timestamps[
            "marker_joint_velocities"
        ].append(timestamp),
    )
    monkeypatch.setattr(
        context.nc,
        "log_joint_torques",
        lambda _values, *, robot_name, timestamp: timestamps[
            "marker_joint_torques"
        ].append(timestamp),
    )
    monkeypatch.setattr(
        context.nc,
        "log_rgb",
        lambda _name, _image, *, robot_name, timestamp: timestamps[
            "marker_camera"
        ].append(timestamp),
    )
    monkeypatch.setattr(
        context.nc,
        "log_custom_1d",
        lambda _name, _values, *, robot_name, timestamp: None,
    )

    plan = build_stochastic_timestamp_plan(
        timestamp_start_s=7.0,
        joint_frame_count=4,
        video_frame_count=2,
        joint_fps=4,
        video_fps=2,
        producer_channels=PRODUCER_PER_THREAD,
        joint_names=["joint"],
        camera_name_list=["camera"],
        context_index=1,
        recording_index=2,
    )
    scheduler = _ImmediateStochasticScheduler()
    run_threaded_logging(
        robot_name="robot",
        joint_frame_count=4,
        video_frame_count=2,
        recording_index=2,
        timestamp_start_s=7.0,
        joint_fps=4,
        video_fps=2,
        context_index=1,
        joint_names=["joint"],
        camera_name_list=["camera"],
        image_width=4,
        image_height=4,
        use_stochastic_timestamps=True,
        stochastic_plan=plan,
        stochastic_scheduler=scheduler,
        assert_deadline=True,
    )

    for producer_key, actual in timestamps.items():
        assert actual == list(plan.timestamps_for(producer_key))
    assert sorted(scheduler.waited_timestamps) == sorted(
        timestamp
        for producer_timestamps in plan.by_producer.values()
        for timestamp in producer_timestamps
    )
    assert all(check_enabled for check_enabled, _label in scheduler.deadline_checks)
    assert set(scheduler.producer_keys) == set(plan.by_producer)
    assert {label.rsplit(" ", 1)[0] for _, label in scheduler.deadline_checks} == {
        "joint_positions frame",
        "joint_velocities frame",
        "joint_torques frame",
        "rgb frame",
    }
