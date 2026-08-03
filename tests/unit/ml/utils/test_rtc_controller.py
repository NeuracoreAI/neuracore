"""Tests for the asynchronous real-time chunking controller."""

import threading
import time

import numpy as np
import pytest
from neuracore_types import DataType, SynchronizedPoint

from neuracore.ml.utils.real_time_chunking import RTCConfig
from neuracore.ml.utils.rtc_controller import RealTimeChunker, RTCInferenceError

HORIZON = 16
EXECUTION_HORIZON = 8
ACTION_DIM = 3
# Ticks are much faster than a real control loop so the tests stay quick; the
# controller only ever reasons in ticks, never in wall-clock.
CONTROL_HZ = 200.0
TICK = 1.0 / CONTROL_HZ


class FakePolicyInference:
    """Stands in for PolicyInference with a controllable inference latency."""

    def __init__(self, latency: float, fail_after: int | None = None) -> None:
        self.latency = latency
        self.fail_after = fail_after
        self.prediction_horizon = HORIZON
        self.model = object()
        self.calls = 0
        self.guided_calls = 0
        self.delays_requested: list[int] = []

    @property
    def supports_real_time_chunking(self) -> bool:
        return True

    def output_action_names(self) -> list[tuple[DataType, str | None]]:
        return [
            (DataType.JOINT_TARGET_POSITIONS, f"joint_{i}") for i in range(ACTION_DIM)
        ]

    def predict_action_chunk(self, sync_point, prev_chunk=None, rtc_config=None):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated inference failure")
        if prev_chunk is not None:
            self.guided_calls += 1
            assert rtc_config is not None, "guided calls must carry a config"
            assert prev_chunk.shape == (HORIZON, ACTION_DIM)
            self.delays_requested.append(rtc_config.inference_delay)
        time.sleep(self.latency)
        return np.full((HORIZON, ACTION_DIM), float(self.calls), dtype=np.float32)


class UnsupportedPolicyInference(FakePolicyInference):
    @property
    def supports_real_time_chunking(self) -> bool:
        return False


def _observation() -> SynchronizedPoint:
    return SynchronizedPoint(timestamp=time.time(), data={})


def _make_chunker(policy, delay=2, adapt=False, execution_horizon=EXECUTION_HORIZON):
    return RealTimeChunker(
        policy,
        _observation,
        RTCConfig(inference_delay=delay, execution_horizon=execution_horizon),
        control_hz=CONTROL_HZ,
        adapt_inference_delay=adapt,
    )


def _drive(chunker: RealTimeChunker, ticks: int) -> list[np.ndarray | None]:
    """Consume `ticks` actions at the control rate."""
    actions = []
    for _ in range(ticks):
        actions.append(chunker.get_action())
        time.sleep(TICK)
    return actions


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_rejects_policy_without_rtc_support():
    with pytest.raises(ValueError, match="diffusion policy"):
        _make_chunker(UnsupportedPolicyInference(latency=0.0))


@pytest.mark.parametrize(
    "delay,execution_horizon",
    [
        (HORIZON - EXECUTION_HORIZON + 1, EXECUTION_HORIZON),  # d > H - s
        (0, 0),  # zero execution horizon
        (0, HORIZON + 1),  # execution horizon beyond the chunk
    ],
)
def test_rejects_infeasible_horizons(delay, execution_horizon):
    with pytest.raises(ValueError):
        _make_chunker(
            FakePolicyInference(latency=0.0),
            delay=delay,
            execution_horizon=execution_horizon,
        )


def test_get_action_returns_none_before_the_first_chunk():
    chunker = _make_chunker(FakePolicyInference(latency=0.0))
    assert chunker.get_action() is None


# --------------------------------------------------------------------------- #
# Steady state
# --------------------------------------------------------------------------- #


def test_actions_stream_without_gaps_when_inference_keeps_up():
    policy = FakePolicyInference(latency=TICK * 2)
    chunker = _make_chunker(policy, delay=3)
    chunker.start()
    try:
        assert chunker.wait_for_first_chunk(timeout=10.0)
        actions = _drive(chunker, 120)
        stats = chunker.stats()
    finally:
        chunker.stop()

    assert all(action is not None for action in actions), "gap in the action stream"
    assert all(action.shape == (ACTION_DIM,) for action in actions)
    assert stats.chunks > 1, "the chunker never replanned"
    assert stats.stalled_ticks == 0
    assert stats.deadline_misses == 0


def test_replans_are_guided_by_the_previous_chunk():
    policy = FakePolicyInference(latency=TICK * 2)
    chunker = _make_chunker(policy, delay=3)
    chunker.start()
    try:
        chunker.wait_for_first_chunk(timeout=10.0)
        _drive(chunker, 120)
    finally:
        chunker.stop()

    assert policy.calls > 1
    assert (
        policy.guided_calls == policy.calls - 1
    ), "only the very first chunk may be unguided"
    assert set(policy.delays_requested) == {3}


def test_chunk_swaps_land_exactly_at_the_inference_delay():
    """The cursor must resume at d, so the frozen prefix covers what executed."""
    policy = FakePolicyInference(latency=TICK * 2)
    delay = 3
    chunker = _make_chunker(policy, delay=delay)
    chunker.start()
    try:
        chunker.wait_for_first_chunk(timeout=10.0)
        seen_indices = []
        previous_chunk_id = None
        for _ in range(200):
            peeked = chunker.peek_chunk()
            if peeked is not None:
                chunk, index = peeked
                chunk_id = float(chunk[0, 0])
                if previous_chunk_id is not None and chunk_id != previous_chunk_id:
                    seen_indices.append(index)
                previous_chunk_id = chunk_id
            chunker.get_action()
            time.sleep(TICK)
    finally:
        chunker.stop()

    assert seen_indices, "no chunk swap was observed"
    # Sampling races the swap by at most one tick, so allow d or d + 1.
    assert all(delay <= index <= delay + 1 for index in seen_indices), seen_indices


def test_peek_chunk_returns_a_copy():
    policy = FakePolicyInference(latency=TICK)
    chunker = _make_chunker(policy)
    chunker.start()
    try:
        chunker.wait_for_first_chunk(timeout=10.0)
        chunk, _ = chunker.peek_chunk()
        chunk[:] = 999.0
        again, _ = chunker.peek_chunk()
        assert not np.any(again == 999.0)
    finally:
        chunker.stop()


def test_action_names_match_the_action_width():
    policy = FakePolicyInference(latency=0.0)
    chunker = _make_chunker(policy)
    assert len(chunker.action_names) == ACTION_DIM
    assert chunker.prediction_horizon == HORIZON


# --------------------------------------------------------------------------- #
# Degraded operation
# --------------------------------------------------------------------------- #


def test_slow_inference_records_misses_but_still_commands_the_robot():
    policy = FakePolicyInference(latency=TICK * 20)
    chunker = _make_chunker(policy, delay=2)
    chunker.start()
    try:
        assert chunker.wait_for_first_chunk(timeout=10.0)
        actions = _drive(chunker, 120)
        stats = chunker.stats()
    finally:
        chunker.stop()

    assert stats.deadline_misses > 0, "a late chunk should be reported"
    assert stats.stalled_ticks > 0, "the chunk should have been exhausted"
    assert all(
        action is not None for action in actions
    ), "the controller must keep issuing actions even when it falls behind"


def test_inference_delay_adapts_upward_to_measured_latency():
    policy = FakePolicyInference(latency=TICK * 6)
    chunker = _make_chunker(policy, delay=1, adapt=True)
    chunker.start()
    try:
        chunker.wait_for_first_chunk(timeout=10.0)
        _drive(chunker, 200)
        stats = chunker.stats()
    finally:
        chunker.stop()

    assert stats.inference_delay > 1, "d did not grow to cover the latency"
    assert (
        stats.inference_delay <= HORIZON - EXECUTION_HORIZON
    ), "d must stay within the real-time constraint"


def test_pinned_delay_does_not_adapt():
    policy = FakePolicyInference(latency=TICK * 6)
    chunker = _make_chunker(policy, delay=2, adapt=False)
    chunker.start()
    try:
        chunker.wait_for_first_chunk(timeout=10.0)
        _drive(chunker, 150)
        stats = chunker.stats()
    finally:
        chunker.stop()

    assert stats.inference_delay == 2


def test_inference_failure_surfaces_through_get_action():
    policy = FakePolicyInference(latency=TICK, fail_after=1)
    chunker = _make_chunker(policy)
    chunker.start()
    try:
        chunker.wait_for_first_chunk(timeout=10.0)
        with pytest.raises(RTCInferenceError, match="simulated inference failure"):
            for _ in range(300):
                chunker.get_action()
                time.sleep(TICK)
    finally:
        chunker.stop()


def test_first_chunk_failure_surfaces_from_wait():
    policy = FakePolicyInference(latency=TICK, fail_after=0)
    chunker = _make_chunker(policy)
    chunker.start()
    try:
        with pytest.raises(RTCInferenceError):
            chunker.wait_for_first_chunk(timeout=10.0)
    finally:
        chunker.stop()


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_stop_joins_the_inference_thread():
    policy = FakePolicyInference(latency=TICK * 2)
    chunker = _make_chunker(policy)
    chunker.start()
    chunker.wait_for_first_chunk(timeout=10.0)
    _drive(chunker, 20)
    chunker.stop()

    assert not any(
        thread.name == "rtc-inference" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_stop_is_idempotent_and_start_is_not_reentrant():
    policy = FakePolicyInference(latency=TICK)
    chunker = _make_chunker(policy)
    chunker.start()
    chunker.start()  # second call must be a no-op
    chunker.wait_for_first_chunk(timeout=10.0)
    chunker.stop()
    chunker.stop()


def test_wait_for_first_chunk_times_out_without_raising():
    policy = FakePolicyInference(latency=5.0)
    chunker = _make_chunker(policy)
    chunker.start()
    try:
        assert chunker.wait_for_first_chunk(timeout=0.1) is False
    finally:
        chunker.stop(timeout=10.0)
