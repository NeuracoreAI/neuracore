"""Tests for the asynchronous chunking controller (RTC + temporal ensemble)."""

import threading
import time

import numpy as np
import pytest
from neuracore_types import DataType, SynchronizedPoint

from neuracore.ml.utils.real_time_chunking import RTCConfig
from neuracore.ml.utils.rtc_controller import (
    ChunkingController,
    ChunkingInferenceError,
    RealTimeChunker,
    RTCInferenceError,
    TemporalEnsembleReplanner,
)
from neuracore.ml.utils.temporal_ensemble import TemporalEnsembleConfig

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
        self.execution_horizons_requested: list[int] = []

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
            self.execution_horizons_requested.append(rtc_config.execution_horizon)
        elif rtc_config is not None:
            self.delays_requested.append(rtc_config.inference_delay)
            self.execution_horizons_requested.append(rtc_config.execution_horizon)
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
        config=RTCConfig(inference_delay=delay, execution_horizon=execution_horizon),
        control_hz=CONTROL_HZ,
        adapt_inference_delay=adapt,
    )


def _seed(chunker: ChunkingController) -> None:
    """Push an observation so the inference thread can produce the first chunk."""
    chunker.get_action(_observation())


def _drive(chunker: ChunkingController, ticks: int) -> list[np.ndarray | None]:
    """Consume `ticks` actions at the control rate."""
    actions = []
    for _ in range(ticks):
        actions.append(chunker.get_action(_observation()))
        time.sleep(TICK)
    return actions


def _start_ready(chunker: ChunkingController, timeout: float = 10.0) -> None:
    chunker.start()
    _seed(chunker)
    assert chunker.wait_for_first_chunk(timeout=timeout)


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
        (EXECUTION_HORIZON + 1, EXECUTION_HORIZON),  # d > s
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
    assert chunker.get_action(_observation()) is None


# --------------------------------------------------------------------------- #
# Steady state
# --------------------------------------------------------------------------- #


def test_actions_stream_without_gaps_when_inference_keeps_up():
    policy = FakePolicyInference(latency=TICK * 2)
    chunker = _make_chunker(policy, delay=3)
    try:
        _start_ready(chunker)
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
    try:
        _start_ready(chunker)
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
    try:
        _start_ready(chunker)
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
            chunker.get_action(_observation())
            time.sleep(TICK)
    finally:
        chunker.stop()

    assert seen_indices, "no chunk swap was observed"
    # Sampling races the swap by at most one tick, so allow d or d + 1.
    assert all(delay <= index <= delay + 1 for index in seen_indices), seen_indices


def test_peek_chunk_returns_a_copy():
    policy = FakePolicyInference(latency=TICK)
    chunker = _make_chunker(policy)
    try:
        _start_ready(chunker)
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
    try:
        _start_ready(chunker)
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
    try:
        _start_ready(chunker)
        _drive(chunker, 200)
        stats = chunker.stats()
    finally:
        chunker.stop()

    assert stats.inference_delay > 1, "d did not grow to cover the latency"
    assert stats.inference_delay <= stats.execution_horizon, "d must stay ≤ s"
    assert (
        stats.inference_delay <= HORIZON - stats.execution_horizon
    ), "d must stay within H - s"
    assert stats.execution_horizon >= EXECUTION_HORIZON


def test_execution_horizon_grows_with_delay_when_d_exceeds_s_min():
    """Paper: s = max(d, s_min). A small s_min must rise once d catches up."""
    s_min = 3
    policy = FakePolicyInference(latency=TICK * 10)
    chunker = _make_chunker(policy, delay=1, adapt=True, execution_horizon=s_min)
    try:
        _start_ready(chunker)
        _drive(chunker, 250)
        stats = chunker.stats()
    finally:
        chunker.stop()

    assert stats.inference_delay >= s_min
    assert stats.execution_horizon == max(s_min, stats.inference_delay)
    assert stats.inference_delay <= HORIZON // 2


def test_late_replan_passes_consumed_as_execution_horizon():
    """When a chunk lands late, the next mask must use s = consumed, not s_min."""
    policy = FakePolicyInference(latency=TICK * 20)
    chunker = _make_chunker(policy, delay=2, adapt=False)
    try:
        _start_ready(chunker)
        _drive(chunker, 120)
    finally:
        chunker.stop()

    assert policy.guided_calls >= 2, "need at least two guided replans"
    assert all(s >= EXECUTION_HORIZON for s in policy.execution_horizons_requested)
    assert (
        max(policy.execution_horizons_requested) > EXECUTION_HORIZON
    ), policy.execution_horizons_requested


def test_pinned_delay_does_not_adapt():
    policy = FakePolicyInference(latency=TICK * 6)
    chunker = _make_chunker(policy, delay=2, adapt=False)
    try:
        _start_ready(chunker)
        _drive(chunker, 150)
        stats = chunker.stats()
    finally:
        chunker.stop()

    assert stats.inference_delay == 2
    assert stats.execution_horizon == EXECUTION_HORIZON


def test_inference_failure_surfaces_through_get_action():
    policy = FakePolicyInference(latency=TICK, fail_after=1)
    chunker = _make_chunker(policy)
    try:
        _start_ready(chunker)
        with pytest.raises(RTCInferenceError, match="simulated inference failure"):
            for _ in range(300):
                chunker.get_action(_observation())
                time.sleep(TICK)
    finally:
        chunker.stop()


def test_first_chunk_failure_surfaces_from_wait():
    policy = FakePolicyInference(latency=TICK, fail_after=0)
    chunker = _make_chunker(policy)
    chunker.start()
    try:
        _seed(chunker)
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
    _start_ready(chunker)
    _drive(chunker, 20)
    chunker.stop()

    assert not any(
        thread.name == "chunking-inference" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_request_stop_returns_without_joining():
    """A control loop cannot afford to block on an in-flight inference."""
    policy = FakePolicyInference(latency=0.5)
    chunker = _make_chunker(policy)
    try:
        _start_ready(chunker)
        started = time.monotonic()
        chunker.request_stop()
        elapsed = time.monotonic() - started
        assert elapsed < 0.05, f"request_stop blocked for {elapsed * 1e3:.0f} ms"
    finally:
        chunker.stop(timeout=10.0)


def test_start_reaps_a_thread_left_by_request_stop():
    policy = FakePolicyInference(latency=TICK)
    chunker = _make_chunker(policy)
    _start_ready(chunker)
    chunker.request_stop()

    chunker.start()
    try:
        _seed(chunker)
        assert chunker.wait_for_first_chunk(timeout=10.0), "restart after request_stop"
        assert chunker.get_action(_observation()) is not None
    finally:
        chunker.stop()

    assert not any(
        thread.name == "chunking-inference" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_error_property_reports_failure_and_has_no_side_effects():
    """`.error` must let a control loop poll for failure without side effects."""
    policy = FakePolicyInference(latency=TICK, fail_after=1)
    chunker = _make_chunker(policy)
    try:
        _start_ready(chunker)
        assert chunker.error is None

        deadline = time.monotonic() + 10.0
        while chunker.error is None and time.monotonic() < deadline:
            try:
                chunker.get_action(_observation())
            except ChunkingInferenceError:
                break
            time.sleep(TICK)
        assert chunker.error is not None, "failure never surfaced on .error"

        _, index_before = chunker.peek_chunk()
        for _ in range(20):
            assert chunker.error is not None
        _, index_after = chunker.peek_chunk()
        assert index_after == index_before, "polling .error advanced the cursor"
    finally:
        chunker.stop()


def test_no_replan_is_attempted_while_actions_are_not_consumed():
    """The cursor only advances on get_action, so an idle chunker must not replan."""
    policy = FakePolicyInference(latency=TICK)
    chunker = _make_chunker(policy)
    try:
        _start_ready(chunker)
        calls_after_first = policy.calls
        time.sleep(TICK * 40)
        assert policy.calls == calls_after_first, "chunker replanned while idle"
        assert chunker.peek_chunk()[1] == 0
    finally:
        chunker.stop()


def test_error_is_none_while_healthy():
    policy = FakePolicyInference(latency=TICK)
    chunker = _make_chunker(policy)
    try:
        _start_ready(chunker)
        _drive(chunker, 30)
        assert chunker.error is None
    finally:
        chunker.stop()


def test_restart_discards_the_stale_chunk_and_replans():
    """A restart must plan from a fresh observation, not resume the old chunk."""
    policy = FakePolicyInference(latency=TICK)
    chunker = _make_chunker(policy)

    _start_ready(chunker)
    _drive(chunker, EXECUTION_HORIZON + 4)
    first_chunk, first_index = chunker.peek_chunk()
    calls_before = policy.calls
    chunker.stop()

    chunker.start()
    try:
        _seed(chunker)
        assert chunker.wait_for_first_chunk(timeout=10.0)
        second_chunk, second_index = chunker.peek_chunk()
        stats = chunker.stats()
    finally:
        chunker.stop()

    assert policy.calls > calls_before, "restart did not replan"
    assert second_index == 0, "restart must rewind the cursor"
    assert not np.array_equal(second_chunk, first_chunk), "stale chunk was reused"
    assert stats.chunks == 1, "restart must reset the chunk counter"
    assert stats.deadline_misses == 0 and stats.stalled_ticks == 0


def test_restart_clears_a_previous_failure():
    """A failed run must not poison the next start."""
    policy = FakePolicyInference(latency=TICK, fail_after=0)
    chunker = _make_chunker(policy)
    chunker.start()
    _seed(chunker)
    with pytest.raises(RTCInferenceError):
        chunker.wait_for_first_chunk(timeout=10.0)
    chunker.stop()

    policy.fail_after = None
    policy.calls = 0
    chunker.start()
    try:
        _seed(chunker)
        assert chunker.wait_for_first_chunk(timeout=10.0), "restart still errored"
        assert chunker.get_action(_observation()) is not None
    finally:
        chunker.stop()


def test_stop_is_idempotent_and_start_is_not_reentrant():
    policy = FakePolicyInference(latency=TICK)
    chunker = _make_chunker(policy)
    chunker.start()
    chunker.start()  # second call must be a no-op
    _seed(chunker)
    chunker.wait_for_first_chunk(timeout=10.0)
    chunker.stop()
    chunker.stop()


def test_wait_for_first_chunk_times_out_without_raising():
    policy = FakePolicyInference(latency=5.0)
    chunker = _make_chunker(policy)
    chunker.start()
    try:
        _seed(chunker)
        assert chunker.wait_for_first_chunk(timeout=0.1) is False
    finally:
        chunker.stop(timeout=10.0)


# --------------------------------------------------------------------------- #
# Temporal ensemble
# --------------------------------------------------------------------------- #


def test_temporal_ensemble_controller_streams_and_merges():
    policy = FakePolicyInference(latency=TICK * 2)
    config = TemporalEnsembleConfig(execution_horizon=EXECUTION_HORIZON, m=0.1)
    chunker = ChunkingController(
        policy,
        TemporalEnsembleReplanner(policy, config),
        execution_horizon=config.execution_horizon,
        inference_delay=0,
        control_hz=CONTROL_HZ,
        adapt_inference_delay=False,
        enforce_rtc_invariant=False,
    )
    try:
        _start_ready(chunker)
        actions = _drive(chunker, 80)
        stats = chunker.stats()
    finally:
        chunker.stop()

    assert all(action is not None for action in actions)
    assert stats.chunks > 1
    assert policy.guided_calls == 0, "TE must not use RTC-guided predict"
    assert policy.calls > 1


def test_temporal_ensemble_s1_instant_matches_act_ensembler():
    """With s=1 and zero latency, async TE must match ACTTemporalEnsembler."""
    from neuracore.ml.utils.temporal_ensemble import ACTTemporalEnsembler

    horizon = 4
    m = 0.01
    chunks = [
        np.full((horizon, ACTION_DIM), float(i + 1), dtype=np.float32)
        for i in range(12)
    ]

    class ScriptedPolicy:
        prediction_horizon = horizon
        model = object()
        guided_calls = 0
        calls = 0

        def output_action_names(self):
            return [
                (DataType.JOINT_TARGET_POSITIONS, f"joint_{i}")
                for i in range(ACTION_DIM)
            ]

        def predict_action_chunk(self, sync_point, prev_chunk=None, rtc_config=None):
            del sync_point, prev_chunk, rtc_config
            idx = min(self.calls, len(chunks) - 1)
            self.calls += 1
            return chunks[idx].copy()

    policy = ScriptedPolicy()
    config = TemporalEnsembleConfig(execution_horizon=1, m=m, blend_steps=0)
    chunker = ChunkingController(
        policy,
        TemporalEnsembleReplanner(policy, config),
        execution_horizon=1,
        inference_delay=0,
        control_hz=CONTROL_HZ,
        adapt_inference_delay=False,
        enforce_rtc_invariant=False,
    )

    ens = ACTTemporalEnsembler(m, horizon)
    expected = [ens.update(chunk.copy()) for chunk in chunks]

    try:
        chunker.start()
        _seed(chunker)
        assert chunker.wait_for_first_chunk(timeout=5.0)
        # First get_action only seeded; consume one action per scripted chunk.
        got: list[np.ndarray] = []
        for _ in range(len(chunks)):
            action = None
            for _retry in range(50):
                action = chunker.get_action(_observation())
                if action is not None:
                    break
                time.sleep(TICK)
            assert action is not None
            got.append(action)
            time.sleep(TICK)
    finally:
        chunker.stop()

    assert len(got) == len(expected)
    for i, (actual, want) in enumerate(zip(got, expected)):
        np.testing.assert_allclose(
            actual, want, rtol=1e-5, atol=1e-5, err_msg=f"step {i}"
        )
