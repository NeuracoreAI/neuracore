"""Asynchronous action-chunk controller for realtime policy execution.

Shared control/inference split used by both real-time chunking (RTC) and
temporal-ensemble prefetch: a background thread generates the next chunk while
the caller keeps consuming the current one.

The caller drives it with one :meth:`ChunkingController.get_action` per control
tick. That call is non-blocking and does no inference, so it is safe from inside
a hard real-time loop. Pass an observation explicitly (sim / tests) or omit it
to fall back to :func:`~neuracore.core.get_latest_sync_point.get_latest_sync_point`.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from neuracore_types import DataType, SynchronizedPoint

from neuracore.ml.utils.policy_inference import PolicyInference
from neuracore.ml.utils.real_time_chunking import (
    RTCConfig,
    align_previous_chunk,
    max_feasible_inference_delay,
)
from neuracore.ml.utils.temporal_ensemble import TemporalEnsembleConfig

logger = logging.getLogger(__name__)

DEFAULT_DELAY_BUFFER_SIZE = 8
# Extra ticks added to the measured latency when adapting the inference delay,
# so a chunk that lands a hair late does not immediately trip a deadline miss.
DELAY_HEADROOM_TICKS = 1


class ChunkingInferenceError(RuntimeError):
    """Raised from ``get_action`` when the inference thread has failed."""


# Backward-compatible alias.
RTCInferenceError = ChunkingInferenceError


@dataclass(frozen=True)
class ChunkingStats:
    """Snapshot of async chunking health.

    Attributes:
        chunks: Chunks generated since ``start``.
        inference_delay: Current ``d``, in control ticks.
        execution_horizon: Current ``s``, in control ticks.
        prediction_horizon: ``H``, the chunk length.
        last_latency_s: Wall-clock duration of the most recent inference.
        median_latency_s: Median inference duration observed so far.
        max_latency_s: Slowest inference observed so far.
        deadline_misses: Chunks that landed later than ``d`` ticks.
        stalled_ticks: Ticks served by repeating the final action because no
            fresh chunk had arrived. Any value above zero means the robot ran
            open-loop past the end of a chunk.
    """

    chunks: int
    inference_delay: int
    execution_horizon: int
    prediction_horizon: int
    last_latency_s: float
    median_latency_s: float
    max_latency_s: float
    deadline_misses: int
    stalled_ticks: int


# Backward-compatible alias.
RTCStats = ChunkingStats


class Replanner(Protocol):
    """Strategy that produces the next action chunk for the controller."""

    def replan(
        self,
        observation: SynchronizedPoint,
        prev_chunk: np.ndarray | None,
        *,
        ticks_consumed: int,
        inference_delay: int,
        execution_horizon: int,
    ) -> np.ndarray:
        """Sample / merge one replacement chunk.

        Args:
            observation: Sync point to condition on.
            prev_chunk: Full previous chunk, or ``None`` for the first chunk.
            ticks_consumed: Actions already executed from ``prev_chunk``.
            inference_delay: Current ``d`` in control ticks.
            execution_horizon: Effective ``s`` for this replan.

        Returns:
            np.ndarray: New chunk with shape ``(H, A)``.
        """
        ...


class RTCReplanner:
    """Guided RTC sampling via :meth:`PolicyInference.predict_action_chunk`."""

    def __init__(self, policy: PolicyInference, config: RTCConfig) -> None:
        """Initialize the RTC replanner.

        Args:
            policy: In-process policy with RTC hooks.
            config: Base RTC configuration (horizons overridden per replan).
        """
        if not policy.supports_real_time_chunking:
            raise ValueError(
                "Real-time chunking requires a diffusion policy loaded in "
                f"process; got {type(policy.model).__name__}."
            )
        self._policy = policy
        self._config = config

    def replan(
        self,
        observation: SynchronizedPoint,
        prev_chunk: np.ndarray | None,
        *,
        ticks_consumed: int,
        inference_delay: int,
        execution_horizon: int,
    ) -> np.ndarray:
        """Run guided (or bootstrap) RTC sampling."""
        aligned: np.ndarray | None = None
        if prev_chunk is not None:
            aligned = align_previous_chunk(
                torch.from_numpy(prev_chunk).unsqueeze(0),
                ticks_consumed,
                self._policy.prediction_horizon,
            )[0].numpy()
        return self._policy.predict_action_chunk(
            observation,
            prev_chunk=aligned,
            rtc_config=self._config.with_horizons(inference_delay, execution_horizon),
        )


class TemporalEnsembleReplanner:
    """Unguided predict fused with :class:`ACTTemporalEnsembler` (ACT Algo 2).

    Each replan runs a full action-chunk prediction and feeds it to an online
    ACT ensembler that favors older predictions. The returned array is the
    current ensembled action plus the remaining buffer (length ``H``) so the
    :class:`ChunkingController` can stream it. With ``execution_horizon=1``
    this matches predict-every-step ACT; larger ``s`` only updates the
    ensembler every ``s`` ticks (approximate).
    """

    def __init__(self, policy: PolicyInference, config: TemporalEnsembleConfig) -> None:
        """Initialize the temporal-ensemble replanner.

        Args:
            policy: Policy that can return a raw action chunk.
            config: Temporal-ensemble configuration.
        """
        from neuracore.ml.utils.temporal_ensemble import ACTTemporalEnsembler

        self._policy = policy
        self._config = config
        self._ensembler: ACTTemporalEnsembler | None = None

    def replan(
        self,
        observation: SynchronizedPoint,
        prev_chunk: np.ndarray | None,
        *,
        ticks_consumed: int,
        inference_delay: int,
        execution_horizon: int,
    ) -> np.ndarray:
        """Predict a chunk, update the ACT ensembler, return the H-row buffer."""
        del inference_delay, execution_horizon  # TE swaps as soon as ready
        from neuracore.ml.utils.temporal_ensemble import continuity_blend

        new_chunk = self._policy.predict_action_chunk(observation)
        horizon = int(new_chunk.shape[0])
        if self._ensembler is None or self._ensembler.chunk_size != horizon:
            from neuracore.ml.utils.temporal_ensemble import ACTTemporalEnsembler

            self._ensembler = ACTTemporalEnsembler(self._config.m, horizon)
        if prev_chunk is None:
            self._ensembler.reset()

        action = self._ensembler.update(new_chunk)
        assert self._ensembler.ensembled_actions is not None
        # Controller expects an H-row chunk; index 0 is the action just popped.
        merged = np.concatenate(
            [action.reshape(1, -1), self._ensembler.ensembled_actions],
            axis=0,
        ).astype(np.float32, copy=False)
        if self._config.blend_steps > 0 and prev_chunk is not None and len(prev_chunk):
            head_idx = min(max(ticks_consumed, 0), len(prev_chunk) - 1)
            merged = continuity_blend(
                merged, prev_chunk[head_idx], self._config.blend_steps
            )
        return merged


class ChunkingController:
    """Runs chunk inference in the background while actions are consumed.

    The shared state is the current chunk and a cursor into it. ``get_action``
    advances the cursor and stores the latest observation; the inference thread
    waits until ``execution_horizon`` actions have been consumed, snapshots the
    observation and previous chunk under the lock, and asks the :class:`Replanner`
    for a replacement. For RTC the replacement is held until ``d`` ticks have
    elapsed; for temporal ensemble ``d`` is typically 0 so it swaps immediately.
    """

    def __init__(
        self,
        policy_inference: PolicyInference,
        replanner: Replanner,
        *,
        execution_horizon: int,
        inference_delay: int = 0,
        control_hz: float,
        delay_buffer_size: int = DEFAULT_DELAY_BUFFER_SIZE,
        adapt_inference_delay: bool = False,
        enforce_rtc_invariant: bool = False,
    ) -> None:
        """Initialise the controller.

        Args:
            policy_inference: Loaded policy providing action-chunk prediction.
            replanner: Strategy used to build each replacement chunk.
            execution_horizon: Minimum ``s``; may grow with ``d`` under adaptation.
            inference_delay: Initial ``d`` in control ticks.
            control_hz: Rate at which ``get_action`` will be called.
            delay_buffer_size: How many recent latencies feed adaptive delay.
            adapt_inference_delay: Grow or shrink ``d`` (and ``s`` with it).
            enforce_rtc_invariant: If True, require ``d <= s <= H - d`` at
                construction (RTC). Temporal ensemble leaves this False.

        Raises:
            ValueError: If the horizons are inconsistent.
        """
        self._policy = policy_inference
        self._replanner = replanner
        self._control_hz = control_hz
        self._tick_period = 1.0 / control_hz
        self._adapt = adapt_inference_delay
        self._enforce_rtc = enforce_rtc_invariant

        self._horizon = policy_inference.prediction_horizon
        self._min_execution_horizon = execution_horizon
        self._execution_horizon = execution_horizon
        self._initial_inference_delay = inference_delay
        if (
            self._min_execution_horizon < 1
            or self._min_execution_horizon > self._horizon
        ):
            raise ValueError(
                f"execution_horizon must be in [1, {self._horizon}], "
                f"got {self._min_execution_horizon}."
            )
        if inference_delay < 0:
            raise ValueError(f"inference_delay must be >= 0, got {inference_delay}")
        if enforce_rtc_invariant:
            if inference_delay > self._execution_horizon:
                raise ValueError(
                    f"Real-time constraint violated: inference_delay "
                    f"d={inference_delay} exceeds execution_horizon "
                    f"s={self._execution_horizon}. Raise s, or lower d."
                )
            if inference_delay > self._horizon - self._execution_horizon:
                raise ValueError(
                    f"Real-time constraint violated: inference_delay "
                    f"d={inference_delay} exceeds H - s = "
                    f"{self._horizon - self._execution_horizon} "
                    f"(H={self._horizon}, s={self._execution_horizon})."
                )

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._chunk: np.ndarray | None = None
        self._latest_obs: SynchronizedPoint | None = None
        self._index = 0
        self._tick = 0
        self._running = False
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

        self._cuda = torch.cuda.is_available()
        self._inference_delay = inference_delay
        self._delays: deque[int] = deque(maxlen=delay_buffer_size)
        self._latencies: deque[float] = deque(maxlen=256)
        self._chunks = 0
        self._deadline_misses = 0
        self._stalled_ticks = 0

    @property
    def action_names(self) -> list[tuple[DataType, str | None]]:
        """Column layout of the arrays returned by ``get_action``."""
        return self._policy.output_action_names()

    @property
    def prediction_horizon(self) -> int:
        """``H``, the number of actions in a chunk."""
        return self._horizon

    @property
    def error(self) -> BaseException | None:
        """The failure that stopped the inference thread, if any."""
        with self._lock:
            return self._error

    def start(self) -> None:
        """Spawn the inference thread, discarding any state from a prior run.

        Returns immediately; the thread waits for the first observation via
        :meth:`get_action` before producing a chunk. Call
        :meth:`wait_for_first_chunk` before relying on actions.
        """
        if self._thread is not None:
            if self._running and self._thread.is_alive():
                return
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._cond:
            self._chunk = None
            self._latest_obs = None
            self._index = 0
            self._tick = 0
            self._error = None
            self._delays.clear()
            self._latencies.clear()
            self._chunks = 0
            self._deadline_misses = 0
            self._stalled_ticks = 0
            self._inference_delay = self._initial_inference_delay
            self._execution_horizon = self._min_execution_horizon
        self._running = True
        self._thread = threading.Thread(
            target=self._inference_loop, name="chunking-inference", daemon=True
        )
        self._thread.start()

    def request_stop(self) -> None:
        """Ask the inference thread to exit without waiting for it."""
        with self._cond:
            self._running = False
            self._cond.notify_all()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the inference thread and wait for it to exit.

        Args:
            timeout: Seconds to wait for the thread to join.
        """
        self.request_stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def wait_for_first_chunk(self, timeout: float = 30.0) -> bool:
        """Block until the first chunk is available.

        Args:
            timeout: Seconds to wait.

        Returns:
            bool: True if a chunk is ready, False if the wait timed out.

        Raises:
            ChunkingInferenceError: If inference failed while waiting.
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._chunk is None and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            self._raise_if_failed()
            return self._chunk is not None

    def get_action(
        self, observation: SynchronizedPoint | None = None
    ) -> np.ndarray | None:
        """Return the action for this control tick and advance the cursor.

        Stores ``observation`` (or :func:`get_latest_sync_point` when omitted)
        for the next replan. Non-blocking apart from a short lock acquisition,
        and never runs inference. Call exactly once per control tick.

        Args:
            observation: Explicit sync point. When ``None``, falls back to
                ``get_latest_sync_point()`` (requires an active Neuracore robot).

        Returns:
            np.ndarray | None: Action with shape ``(action_dim,)``, or ``None``
            if the first chunk has not arrived yet.

        Raises:
            ChunkingInferenceError: If the inference thread has failed.
        """
        obs = observation if observation is not None else _resolve_latest_sync_point()
        with self._cond:
            self._raise_if_failed()
            self._latest_obs = obs
            self._cond.notify_all()
            if self._chunk is None:
                return None

            if self._index >= self._horizon:
                self._stalled_ticks += 1
                if self._stalled_ticks == 1:
                    logger.warning(
                        "Async chunking fell behind: chunk exhausted before "
                        "its replacement arrived. Holding the final action."
                    )
                index = self._horizon - 1
            else:
                index = self._index

            action = self._chunk[index].copy()
            self._index += 1
            self._tick += 1
            self._cond.notify_all()
            return action

    def peek_chunk(self) -> tuple[np.ndarray, int] | None:
        """Return a copy of the current chunk and cursor, for visualisation."""
        with self._lock:
            if self._chunk is None:
                return None
            return self._chunk.copy(), self._index

    def stats(self) -> ChunkingStats:
        """Return a snapshot of chunking health."""
        with self._lock:
            latencies = list(self._latencies)
            return ChunkingStats(
                chunks=self._chunks,
                inference_delay=self._inference_delay,
                execution_horizon=self._execution_horizon,
                prediction_horizon=self._horizon,
                last_latency_s=latencies[-1] if latencies else 0.0,
                median_latency_s=statistics.median(latencies) if latencies else 0.0,
                max_latency_s=max(latencies) if latencies else 0.0,
                deadline_misses=self._deadline_misses,
                stalled_ticks=self._stalled_ticks,
            )

    def _raise_if_failed(self) -> None:
        """Re-raise a failure captured by the inference thread."""
        if self._error is not None:
            raise ChunkingInferenceError(
                f"Async chunking inference failed: {self._error}"
            ) from self._error

    def _timed_replan(
        self,
        observation: SynchronizedPoint,
        prev_chunk: np.ndarray | None,
        delay: int,
        execution_horizon: int,
        ticks_consumed: int,
    ) -> np.ndarray:
        """Run one replan, timing it."""
        started = time.monotonic()
        chunk = self._replanner.replan(
            observation,
            prev_chunk,
            ticks_consumed=ticks_consumed,
            inference_delay=delay,
            execution_horizon=execution_horizon,
        )
        if self._cuda:
            torch.cuda.synchronize()
        with self._lock:
            self._latencies.append(time.monotonic() - started)
        return chunk

    def _inference_loop(self) -> None:
        """Generate chunks until stopped, capturing any failure for the caller."""
        try:
            with self._cond:
                self._cond.wait_for(
                    lambda: not self._running or self._latest_obs is not None
                )
                if not self._running:
                    return
                observation = self._latest_obs
                assert observation is not None
                delay = self._inference_delay
                execution_horizon = self._execution_horizon
            first = self._timed_replan(
                observation, None, delay, execution_horizon, ticks_consumed=0
            )
            with self._cond:
                self._chunk = first
                self._index = 0
                self._chunks += 1
                self._cond.notify_all()

            while True:
                with self._cond:
                    self._cond.wait_for(
                        lambda: not self._running
                        or self._index >= self._execution_horizon
                    )
                    if not self._running:
                        return
                    consumed = self._index
                    start_tick = self._tick
                    delay = self._inference_delay
                    prev = self._chunk
                    assert prev is not None
                    observation = self._latest_obs
                    assert observation is not None

                s_eff = min(max(consumed, 1), self._horizon)
                d_eff = (
                    min(delay, self._horizon - s_eff) if self._enforce_rtc else delay
                )
                d_eff = max(d_eff, 0)

                new_chunk = self._timed_replan(
                    observation, prev, d_eff, s_eff, ticks_consumed=consumed
                )

                with self._cond:
                    if not self._running:
                        return
                    self._record_delay(self._tick - start_tick)
                    self._cond.wait_for(
                        lambda: not self._running or self._tick - start_tick >= d_eff
                    )
                    if not self._running:
                        return
                    elapsed = self._tick - start_tick
                    if delay > 0 and elapsed > delay:
                        self._deadline_misses += 1
                        logger.warning(
                            "Chunk landed %d ticks late (d=%d); its frozen "
                            "prefix no longer covers everything that executed.",
                            elapsed - delay,
                            delay,
                        )
                    self._chunk = new_chunk
                    self._index = min(elapsed, self._horizon)
                    self._chunks += 1
                    self._cond.notify_all()
        except BaseException as exc:  # surfaced to the caller via get_action
            logger.exception("Async chunking inference thread failed")
            with self._cond:
                self._error = exc
                self._cond.notify_all()

    def _record_delay(self, observed_ticks: int) -> None:
        """Update the adaptive inference delay (and execution horizon).

        Must be called with the lock held.
        """
        self._delays.append(observed_ticks)
        if not self._adapt:
            return
        ceiling = max_feasible_inference_delay(
            self._horizon, self._min_execution_horizon
        )
        target_d = min(max(self._delays) + DELAY_HEADROOM_TICKS, ceiling)
        target_d = max(target_d, 1)
        target_s = max(self._min_execution_horizon, target_d)
        if target_d == self._inference_delay and target_s == self._execution_horizon:
            return
        logger.info(
            "Adapting chunking horizons: d %d -> %d, s %d -> %d "
            "(%.0f ms delay at %.0f Hz)",
            self._inference_delay,
            target_d,
            self._execution_horizon,
            target_s,
            target_d * self._tick_period * 1e3,
            self._control_hz,
        )
        self._inference_delay = target_d
        self._execution_horizon = target_s


def _resolve_latest_sync_point() -> SynchronizedPoint:
    """Pull the latest robot sync point, with a clear error if unavailable."""
    try:
        from neuracore.core.get_latest_sync_point import get_latest_sync_point

        return get_latest_sync_point()
    except Exception as exc:
        raise ChunkingInferenceError(
            "get_action() was called without an observation and "
            "get_latest_sync_point() failed. Pass observation= explicitly "
            "(required for sim / offline), or connect a Neuracore robot with "
            "active streams."
        ) from exc


class RealTimeChunker(ChunkingController):
    """Deprecated alias for RTC-backed :class:`ChunkingController`.

    Prefer constructing via :meth:`neuracore.core.endpoint.RealTimePolicy.make_chunker`
    or :meth:`RealTimePolicy.start`. Kept for older call sites that passed an
    ``observation_fn`` and :class:`RTCConfig` directly.
    """

    def __init__(
        self,
        policy_inference: PolicyInference,
        observation_fn: Callable[[], SynchronizedPoint] | None = None,
        config: RTCConfig | None = None,
        *,
        control_hz: float,
        delay_buffer_size: int = DEFAULT_DELAY_BUFFER_SIZE,
        adapt_inference_delay: bool = True,
        replanner: Replanner | None = None,
        execution_horizon: int | None = None,
        inference_delay: int | None = None,
        enforce_rtc_invariant: bool = True,
    ) -> None:
        """Initialise an RTC chunker.

        The legacy ``observation_fn`` is accepted but ignored: observations are
        pushed via :meth:`get_action`. Prefer the new keyword-only construction
        used by :class:`RealTimePolicy`.
        """
        del observation_fn  # observations come from get_action
        if config is None and replanner is None:
            raise ValueError("Either config or replanner must be provided.")
        resolved: Replanner
        if config is not None:
            resolved = RTCReplanner(policy_inference, config)
            exec_h = config.execution_horizon
            delay = config.inference_delay
            enforce = True
        else:
            assert replanner is not None
            resolved = replanner
            if execution_horizon is None:
                raise ValueError("execution_horizon is required without RTCConfig.")
            exec_h = execution_horizon
            delay = 0 if inference_delay is None else inference_delay
            enforce = enforce_rtc_invariant
        super().__init__(
            policy_inference,
            resolved,
            execution_horizon=exec_h,
            inference_delay=delay,
            control_hz=control_hz,
            delay_buffer_size=delay_buffer_size,
            adapt_inference_delay=adapt_inference_delay,
            enforce_rtc_invariant=enforce,
        )
