"""Asynchronous action-chunk controller for real-time chunking.

Implements the control/inference split from "Real-Time Execution of Action
Chunking Flow Policies" (Black, Galliker, Levine; arXiv:2506.07339): a
background thread generates the next chunk while the caller keeps consuming the
current one, so the robot never pauses for inference.

The caller drives it with one :meth:`RealTimeChunker.get_action` per control
tick. That call is non-blocking and does no inference, so it is safe from inside
a hard real-time loop.
"""

import logging
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from neuracore_types import DataType, SynchronizedPoint

from neuracore.ml.utils.policy_inference import PolicyInference
from neuracore.ml.utils.real_time_chunking import RTCConfig, align_previous_chunk

logger = logging.getLogger(__name__)

DEFAULT_DELAY_BUFFER_SIZE = 8
# Extra ticks added to the measured latency when adapting the inference delay,
# so a chunk that lands a hair late does not immediately trip a deadline miss.
DELAY_HEADROOM_TICKS = 1


class RTCInferenceError(RuntimeError):
    """Raised from ``get_action`` when the inference thread has failed."""


@dataclass(frozen=True)
class RTCStats:
    """Snapshot of real-time chunking health.

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


class RealTimeChunker:
    """Runs chunk inference in the background while actions are consumed.

    The shared state is the current chunk and a cursor into it. ``get_action``
    advances the cursor; the inference thread waits until ``execution_horizon``
    actions have been consumed, snapshots the observation and the unexecuted
    remainder of the chunk, and generates a replacement guided by that
    remainder. The replacement is swapped in only once exactly ``d`` ticks have
    elapsed since the snapshot, which is what makes its frozen prefix line up
    with the actions that really executed.
    """

    def __init__(
        self,
        policy_inference: PolicyInference,
        observation_fn: Callable[[], SynchronizedPoint],
        config: RTCConfig,
        *,
        control_hz: float,
        delay_buffer_size: int = DEFAULT_DELAY_BUFFER_SIZE,
        adapt_inference_delay: bool = True,
    ) -> None:
        """Initialise the chunker.

        Args:
            policy_inference: Loaded policy providing ``predict_action_chunk``.
            observation_fn: Builds a sync point from the caller's latest sensor
                snapshot. Called from the inference thread while the chunker's
                lock is held, so it may read state the control loop writes.
            config: Real-time chunking configuration. ``execution_horizon`` and
                the initial ``inference_delay`` come from here.
            control_hz: Rate at which ``get_action`` will be called. Used to
                report latency in ticks.
            delay_buffer_size: How many recent latencies feed the adaptive
                inference delay.
            adapt_inference_delay: Grow or shrink ``d`` to track measured
                latency. Disable to pin ``d`` to the configured value.

        Raises:
            ValueError: If the policy does not support real-time chunking, or
                the horizons violate ``d <= H - s``.
        """
        if not policy_inference.supports_real_time_chunking:
            raise ValueError(
                "Real-time chunking requires a diffusion policy loaded in "
                f"process; got {type(policy_inference.model).__name__}."
            )

        self._policy = policy_inference
        self._observation_fn = observation_fn
        self._config = config
        self._control_hz = control_hz
        self._tick_period = 1.0 / control_hz
        self._adapt = adapt_inference_delay

        self._horizon = policy_inference.prediction_horizon
        self._execution_horizon = config.execution_horizon
        if self._execution_horizon < 1 or self._execution_horizon > self._horizon:
            raise ValueError(
                f"execution_horizon must be in [1, {self._horizon}], "
                f"got {self._execution_horizon}."
            )
        if config.inference_delay > self._horizon - self._execution_horizon:
            raise ValueError(
                f"Real-time constraint violated: inference_delay "
                f"d={config.inference_delay} exceeds H - s = "
                f"{self._horizon - self._execution_horizon} "
                f"(H={self._horizon}, s={self._execution_horizon})."
            )

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._chunk: np.ndarray | None = None
        self._index = 0
        self._tick = 0
        self._running = False
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

        self._cuda = torch.cuda.is_available()
        self._inference_delay = config.inference_delay
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
        """The failure that stopped the inference thread, if any.

        Lets a caller poll for failure without calling :meth:`get_action`, which
        would consume an action and advance the cursor.
        """
        with self._lock:
            return self._error

    def start(self) -> None:
        """Spawn the inference thread, discarding any state from a prior run.

        Safe to call again after :meth:`stop`. The chunk held from a previous run
        is dropped so the first chunk of the new run comes from a fresh
        observation - a stale chunk would command the robot toward wherever it
        was when that chunk was planned.

        Returns immediately; call :meth:`wait_for_first_chunk` before the first
        :meth:`get_action` if the caller needs an action straight away.
        """
        if self._thread is not None:
            if self._running and self._thread.is_alive():
                return
            # Reap a thread left behind by request_stop before starting again.
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._cond:
            self._chunk = None
            self._index = 0
            self._tick = 0
            self._error = None
            self._delays.clear()
            self._latencies.clear()
            self._chunks = 0
            self._deadline_misses = 0
            self._stalled_ticks = 0
            self._inference_delay = self._config.inference_delay
        self._running = True
        self._thread = threading.Thread(
            target=self._inference_loop, name="rtc-inference", daemon=True
        )
        self._thread.start()

    def request_stop(self) -> None:
        """Ask the inference thread to exit without waiting for it.

        Use this from a real-time control loop. :meth:`stop` has to wait for an
        in-flight inference to finish, which can take long enough to starve a
        robot watchdog; this returns immediately and the thread winds down on its
        own. :meth:`start` reaps it, and :meth:`stop` still joins at shutdown.
        """
        with self._cond:
            self._running = False
            self._cond.notify_all()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the inference thread and wait for it to exit.

        Blocks for up to an in-flight inference. Prefer :meth:`request_stop` when
        called from a loop that must keep commanding hardware.

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
            RTCInferenceError: If inference failed while waiting.
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

    def get_action(self) -> np.ndarray | None:
        """Return the action for this control tick and advance the cursor.

        Non-blocking apart from a short lock acquisition, and never runs
        inference. Call exactly once per control tick.

        Returns:
            np.ndarray | None: Action with shape ``(action_dim,)``, or ``None``
            if the first chunk has not arrived yet.

        Raises:
            RTCInferenceError: If the inference thread has failed.
        """
        with self._cond:
            self._raise_if_failed()
            if self._chunk is None:
                return None

            if self._index >= self._horizon:
                # No fresh chunk arrived in time; hold the final action rather
                # than leaving the robot without a command.
                self._stalled_ticks += 1
                if self._stalled_ticks == 1:
                    logger.warning(
                        "Real-time chunking fell behind: chunk exhausted before "
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
        """Return a copy of the current chunk and cursor, for visualisation.

        Returns:
            tuple[np.ndarray, int] | None: The chunk with shape ``(H, A)`` and
            the index of the next action, or ``None`` before the first chunk.
        """
        with self._lock:
            if self._chunk is None:
                return None
            return self._chunk.copy(), self._index

    def stats(self) -> RTCStats:
        """Return a snapshot of chunking health.

        Returns:
            RTCStats: Current counters and latency summary.
        """
        with self._lock:
            latencies = list(self._latencies)
            return RTCStats(
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
        """Re-raise a failure captured by the inference thread.

        Raises:
            RTCInferenceError: If the inference thread has recorded an error.
        """
        if self._error is not None:
            raise RTCInferenceError(
                f"Real-time chunking inference failed: {self._error}"
            ) from self._error

    def _predict(
        self,
        observation: SynchronizedPoint,
        prev_chunk: np.ndarray | None,
        delay: int,
    ) -> np.ndarray:
        """Run one chunk prediction, timing it.

        Args:
            observation: Sync point to condition on.
            prev_chunk: Aligned previous chunk, or ``None`` for the first chunk.
            delay: The ``d`` the chunk is being generated for.

        Returns:
            np.ndarray: New chunk with shape ``(H, A)``.
        """
        started = time.monotonic()
        # The config is passed even for the unguided bootstrap chunk so it picks
        # up the step-count and scheduler overrides; otherwise the first chunk
        # runs at the model's offline default and skews the latency stats.
        chunk = self._policy.predict_action_chunk(
            observation,
            prev_chunk=prev_chunk,
            rtc_config=self._config.with_inference_delay(delay),
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
                observation = self._observation_fn()
            first = self._predict(observation, None, self._inference_delay)
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
                    # The observation, the cursor and the tick must come from
                    # one critical section or the alignment below is off by
                    # however many ticks slipped between them.
                    consumed = self._index
                    start_tick = self._tick
                    delay = self._inference_delay
                    prev = self._chunk
                    assert prev is not None
                    observation = self._observation_fn()

                aligned = align_previous_chunk(
                    torch.from_numpy(prev).unsqueeze(0), consumed, self._horizon
                )[0].numpy()

                new_chunk = self._predict(observation, aligned, delay)

                with self._cond:
                    if not self._running:
                        return
                    # Measured before waiting, so the adaptive delay tracks the
                    # true latency rather than the delay we chose to enforce.
                    self._record_delay(self._tick - start_tick)
                    # Hold the new chunk back until exactly `delay` ticks have
                    # passed, so its frozen prefix covers what really executed.
                    self._cond.wait_for(
                        lambda: not self._running or self._tick - start_tick >= delay
                    )
                    if not self._running:
                        return
                    elapsed = self._tick - start_tick
                    if elapsed > delay:
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
            logger.exception("Real-time chunking inference thread failed")
            with self._cond:
                self._error = exc
                self._cond.notify_all()

    def _record_delay(self, observed_ticks: int) -> None:
        """Update the adaptive inference delay from a measured latency.

        Must be called with the lock held.

        Args:
            observed_ticks: Ticks that elapsed during the last inference.
        """
        self._delays.append(observed_ticks)
        if not self._adapt:
            return
        ceiling = self._horizon - self._execution_horizon
        target = min(max(self._delays) + DELAY_HEADROOM_TICKS, ceiling)
        target = max(target, 1)
        if target != self._inference_delay:
            logger.info(
                "Adapting real-time chunking inference delay d: %d -> %d ticks "
                "(%.0f ms at %.0f Hz)",
                self._inference_delay,
                target,
                target * self._tick_period * 1e3,
                self._control_hz,
            )
            self._inference_delay = target
