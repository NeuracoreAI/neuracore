"""Tests for low-contention sleep latency records."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from tests.integration.platform.data_daemon.shared import diagnostic_sleep


def test_fallback_sleep_records_monotonic_overshoot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = iter((1_000_000_000, 1_017_000_000))
    recorder = Mock()
    slept: list[float] = []
    monkeypatch.setattr(diagnostic_sleep.time, "monotonic_ns", lambda: next(samples))
    monkeypatch.setattr(diagnostic_sleep.time, "sleep", slept.append)
    monkeypatch.setattr(diagnostic_sleep, "_native_sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(diagnostic_sleep, "_get_recorder", lambda: recorder)

    record = diagnostic_sleep.diagnostic_sleep(
        0.010,
        "rgb frame",
        anomaly_threshold_ms=5.0,
        correlation_id="ctx=1/frame=2",
    )

    assert slept == [0.010]
    assert record.actual_ns == 17_000_000
    assert record.overshoot_ns == 7_000_000
    assert record.expected_deadline_ns == 1_010_000_000
    assert record.native_helper is False
    recorder.submit.assert_called_once_with(record, emit=True)


def test_native_boundaries_separate_wait_from_gil_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = iter((2_000_000_000, 2_018_000_000))
    recorder = Mock()
    bridge = Mock()
    monkeypatch.setattr(diagnostic_sleep.time, "monotonic_ns", lambda: next(samples))
    monkeypatch.setattr(
        diagnostic_sleep,
        "_native_sleep",
        lambda *args, **kwargs: (
            2_001_000_000,
            2_012_000_000,
            2_017_000_000,
            0,
            None,
            None,
            21,
            0,
        ),
    )
    monkeypatch.setattr(diagnostic_sleep, "_data_bridge", bridge)
    monkeypatch.setattr(diagnostic_sleep, "_get_recorder", lambda: recorder)

    record = diagnostic_sleep.diagnostic_sleep(
        0.010,
        "joint frame",
        anomaly_threshold_ms=5.0,
        correlation_id="ctx=0/frame=3",
    )

    assert record.native_wait_ns == 11_000_000
    assert record.gil_reacquire_ns == 5_000_000
    assert record.python_post_native_ns == 1_000_000
    assert record.overshoot_ns == 8_000_000
    assert record.native_helper is True
    bridge.diagnostic_signpost_anomaly.assert_called_once()
    recorder.submit.assert_called_once_with(record, emit=True)
