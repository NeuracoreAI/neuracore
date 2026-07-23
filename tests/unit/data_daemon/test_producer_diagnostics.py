"""Focused tests for performance-test producer diagnostics."""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from tests.integration.platform.data_daemon.shared import producer_diagnostics
from tests.integration.platform.data_daemon.shared.process_control import (
    assert_on_schedule,
)
from tests.integration.platform.data_daemon.shared.producer_diagnostics import (
    ProducerDiagnosticHistory,
    ProducerHeartbeatRegistry,
)


def make_history(**kwargs: object) -> ProducerDiagnosticHistory:
    collect_statistics = kwargs.pop("collect_statistics", False)
    return ProducerDiagnosticHistory(
        context_index=2,
        recording_index=3,
        collect_statistics=bool(collect_statistics),
        **kwargs,
    )


def test_bounded_history_drops_old_events() -> None:
    history = make_history(max_events=3)

    for frame_index in range(5):
        history.record("operation", role_name="rgb", frame_index=frame_index)

    assert [event.frame_index for event in history.events] == [2, 3, 4]


def test_sleep_diagnostics_calculate_oversleep(monkeypatch: pytest.MonkeyPatch) -> None:
    history = make_history()
    samples = iter((1_000_000_000, 1_017_500_000))
    monkeypatch.setattr(
        producer_diagnostics.time, "perf_counter_ns", lambda: next(samples)
    )
    monkeypatch.setattr(producer_diagnostics.time, "sleep", lambda _: None)

    history.sleep(0.010, role_name="rgb", frame_index=4, deadline=100.0)

    event = history.events[-1]
    assert event.operation == "time.sleep"
    assert event.duration_ms == pytest.approx(17.5)
    details = event.details
    assert details.get("requested_ms") == pytest.approx(10.0)
    assert details.get("actual_ms") == pytest.approx(17.5)


def test_wait_delay_warnings_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    history = make_history(collect_statistics=True)
    warning = Mock()
    monkeypatch.setattr(producer_diagnostics.logger, "warning", warning)
    monkeypatch.setattr(producer_diagnostics, "WAIT_DELAY_LOG_LIMIT", 2)
    perf_samples = iter((
        0,
        20_000_000,
        30_000_000,
        50_000_000,
        60_000_000,
        80_000_000,
    ))
    monkeypatch.setattr(
        producer_diagnostics.time, "perf_counter_ns", lambda: next(perf_samples)
    )
    monkeypatch.setattr(producer_diagnostics.time, "sleep", lambda _: None)

    for frame_index in range(3):
        history.sleep(
            0.010,
            role_name="rgb",
            frame_index=frame_index,
            deadline=100.0,
        )

    assert warning.call_count == 3
    assert "log limit reached" in warning.call_args.args[0]


def test_failed_schedule_assertion_includes_recent_events() -> None:
    history = make_history()
    with history.measure(
        "nc.log_rgb",
        role_name="rgb",
        frame_index=18,
        deadline=10.0,
        details={"camera_name": "camera_0"},
    ):
        pass

    with pytest.raises(AssertionError) as caught:
        assert_on_schedule(
            10.0,
            0.05,
            "rgb frame",
            observed_at=10.060,
            diagnostic_history=history,
            role_name="rgb",
            frame_index=19,
        )

    message = str(caught.value)
    assert message.startswith(
        "rgb frame fired at wrong moment: lateness=+0.060s, tolerance=±0.050s"
    )
    assert "Recent producer diagnostics:" in message
    assert "nc.log_rgb" in message
    assert "camera_name=camera_0" in message


def test_diagnostic_failure_cannot_suppress_original_assertion(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    history = make_history()

    def fail_format(**_: object) -> str:
        raise RuntimeError("formatter broke")

    monkeypatch.setattr(history, "format_failure", fail_format)
    with caplog.at_level(logging.ERROR), pytest.raises(AssertionError) as caught:
        assert_on_schedule(
            10.0,
            0.05,
            "joint frame",
            observed_at=10.1,
            diagnostic_history=history,
        )

    assert str(caught.value) == (
        "joint frame fired at wrong moment: lateness=+0.100s, tolerance=±0.050s"
    )
    assert "Failed to format producer scheduling diagnostics" in caplog.text


def test_failure_shows_heartbeats_from_multiple_roles() -> None:
    heartbeats = ProducerHeartbeatRegistry()
    rgb = make_history(heartbeat_registry=heartbeats)
    joints = make_history(heartbeat_registry=heartbeats)
    rgb.record("nc.log_rgb", role_name="rgb:camera_0", frame_index=7)
    joints.record("nc.log_joint_positions", role_name="joint_positions", frame_index=8)

    output = rgb.format_failure(
        role_name="rgb:camera_0",
        frame_index=8,
        deadline=10.0,
        observed_at=10.1,
    )

    assert "Producer heartbeats:" in output
    assert "rgb:camera_0" in output
    assert "joint_positions" in output


def test_disabled_history_is_inert_and_schedule_result_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = make_history(enabled=False)
    slept: list[float] = []
    monkeypatch.setattr(producer_diagnostics.time, "sleep", slept.append)

    history.record("operation", role_name="rgb", frame_index=1)
    history.sleep(0.002, role_name="rgb", frame_index=1, deadline=10.0)

    assert history.events == ()
    assert slept == [0.002]
    assert assert_on_schedule(
        10.0, 0.05, "rgb frame", observed_at=10.04
    ) == pytest.approx(0.04)
    with pytest.raises(AssertionError):
        assert_on_schedule(10.0, 0.05, "rgb frame", observed_at=10.06)
