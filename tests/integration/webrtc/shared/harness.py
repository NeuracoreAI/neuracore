"""In-process signaling relay and event pump for two native WebRTC peers.

No separate signaling server: a background pump repeatedly drains both peers'
event queues and relays the signaling-out events between them, so the producer
offers and the consumer answers entirely in-process.

Relay rules (the producer is the sole offerer; the consumer never offers and
never adds tracks):

  * producer offer  ``on_local_description`` -> ``consumer.set_remote_offer``
  * consumer answer ``on_local_description`` -> ``producer.set_remote_answer``
  * either peer's ``on_local_candidate``     -> the other peer's
    ``add_remote_candidate`` (trickle, both directions)

Every drained event is also recorded (in arrival order, with a perf-clock
receive timestamp) into a per-peer log that tests assert against via the
``wait_*`` / ``*_events`` helpers.

Against the PR0 stubs no signaling-out events are emitted, so the pump records
only the initial ``on_state: "new"`` and never invokes a stubbed
``set_remote_*`` / ``add_remote_candidate`` — the harness itself runs clean; only
the test bodies, which call the stubbed producer methods directly, go red.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from tests.integration.webrtc.shared import constants
from tests.integration.webrtc.shared.frames import (
    decode_frame,
    encode_frame,
    parse_video_frame_event,
)

EventPredicate = Callable[[dict], bool]

# Key the relay stamps onto every recorded event with its perf-clock arrival
# time (same clock as submit timestamps, so glass-to-glass subtracts cleanly).
RECV_TS_KEY = "_recv_perf"


def _kind(event: dict) -> str | None:
    return event.get("kind")


def recv_time(event: dict) -> float | None:
    """Perf-clock time at which the relay recorded ``event`` (or None)."""
    return event.get(RECV_TS_KEY)


class Relay:
    """One producer + one consumer joined by an in-process pump thread."""

    POLL_INTERVAL_S = 0.002

    def __init__(self, producer: object, consumer: object, *, name: str = "relay"):
        self.producer = producer
        self.consumer = consumer
        self.name = name
        self._lock = threading.RLock()
        self._events: dict[str, list[dict]] = {"producer": [], "consumer": []}
        # Exceptions raised while *relaying* signaling (not test failures). In a
        # red run this stays empty (no signaling-out events fire).
        self.dispatch_errors: list[BaseException] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- lifecycle -----------------------------------------------------------
    def start(self) -> Relay:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name=self.name, daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def close(self) -> None:
        self.stop()
        for peer in (self.producer, self.consumer):
            try:
                peer.close()
            except Exception:  # noqa: BLE001 - teardown best effort
                pass

    # --- pump ----------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.pump_once()
            except Exception as exc:  # noqa: BLE001 - never let the pump die
                self.dispatch_errors.append(exc)
            time.sleep(self.POLL_INTERVAL_S)

    def pump_once(self) -> None:
        """Drain both peers once and relay their signaling-out events."""
        for event in self.producer.drain_events():
            self._record("producer", event)
            self._relay(event, dst=self.consumer)
        for event in self.consumer.drain_events():
            self._record("consumer", event)
            self._relay(event, dst=self.producer)

    def _record(self, which: str, event: dict) -> None:
        event[RECV_TS_KEY] = time.perf_counter()
        with self._lock:
            self._events[which].append(event)

    def _relay(self, event: dict, dst: object) -> None:
        kind = _kind(event)
        try:
            if kind == "on_local_description":
                if event.get("sdp_type") == "offer":
                    dst.set_remote_offer(event["sdp"])
                else:
                    dst.set_remote_answer(event["sdp"])
            elif kind == "on_local_candidate":
                dst.add_remote_candidate(event["candidate"], event.get("mid"))
        except Exception as exc:  # noqa: BLE001 - surfaced via dispatch_errors
            self.dispatch_errors.append(exc)

    # --- observation ---------------------------------------------------------
    def events(self, which: str) -> list[dict]:
        with self._lock:
            return list(self._events[which])

    def producer_events(self) -> list[dict]:
        return self.events("producer")

    def consumer_events(self) -> list[dict]:
        return self.events("consumer")

    def state_sequence(self, which: str) -> list[str]:
        """Ordered ``on_state`` values seen for a peer (for PC-reset checks)."""
        return [e["state"] for e in self.events(which) if _kind(e) == "on_state"]

    def wait_for(
        self, which: str, predicate: EventPredicate, timeout: float
    ) -> dict | None:
        """Return the first recorded event matching ``predicate``, or None."""
        deadline = time.monotonic() + timeout
        while True:
            for event in self.events(which):
                if predicate(event):
                    return event
            if time.monotonic() >= deadline:
                return None
            time.sleep(self.POLL_INTERVAL_S)

    def wait_consumer(self, predicate: EventPredicate, timeout: float) -> dict | None:
        return self.wait_for("consumer", predicate, timeout)

    def wait_producer(self, predicate: EventPredicate, timeout: float) -> dict | None:
        return self.wait_for("producer", predicate, timeout)

    def wait_connected(self, timeout: float = constants.CONNECT_TIMEOUT_S) -> bool:
        """True once *both* peers have reported on_state "connected"."""

        def connected(events: list[dict]) -> bool:
            return any(
                _kind(e) == "on_state" and e.get("state") == "connected" for e in events
            )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if connected(self.producer_events()) and connected(self.consumer_events()):
                return True
            time.sleep(self.POLL_INTERVAL_S)
        return False

    # --- data-channel message collection -------------------------------------
    def messages(self, which: str, label: str) -> list[bytes]:
        """Ordered payloads of ``on_message`` events for ``label``."""
        return [
            e["data"]
            for e in self.events(which)
            if _kind(e) == "on_message" and e.get("label") == label
        ]

    def wait_messages(
        self, which: str, label: str, count: int, timeout: float
    ) -> list[bytes]:
        """Wait until ``count`` messages for ``label`` arrive; return them."""
        deadline = time.monotonic() + timeout
        while True:
            got = self.messages(which, label)
            if len(got) >= count or time.monotonic() >= deadline:
                return got
            time.sleep(self.POLL_INTERVAL_S)

    # --- video frame collection ----------------------------------------------
    def video_frames(self, which: str, track_id: str | None = None) -> list[dict]:
        """Ordered ``on_frame`` events, optionally filtered by track_id."""
        return [
            e
            for e in self.events(which)
            if _kind(e) == "on_frame"
            and (track_id is None or e.get("track_id") == track_id)
        ]


class BroadcastRelay:
    """One :class:`Broadcaster` joined to N answer-only consumers by a pump thread.

    The broadcaster is the sole offerer to each consumer; its signaling-out events
    are tagged with a ``consumer_id`` so the pump routes each to the right consumer
    peer, and each consumer's answer/candidates are routed back to the broadcaster
    with that id. This is the multi-consumer analogue of :class:`Relay`: one shared
    encode per source fans out to every consumer.

    Relay rules:
      * broadcaster ``on_local_description{consumer_id, offer}`` ->
        ``consumers[consumer_id].set_remote_offer``
      * broadcaster ``on_local_candidate{consumer_id, ...}`` ->
        ``consumers[consumer_id].add_remote_candidate``
      * consumer answer ``on_local_description`` ->
        ``broadcaster.set_remote_answer(consumer_id, sdp)``
      * consumer ``on_local_candidate`` ->
        ``broadcaster.add_remote_candidate(consumer_id, ...)``
    """

    POLL_INTERVAL_S = 0.002

    def __init__(self, broadcaster: object, *, name: str = "broadcast"):
        self.broadcaster = broadcaster
        self.name = name
        self.consumers: dict[str, object] = {}
        self._lock = threading.RLock()
        self._events: dict[str, list[dict]] = {"broadcaster": []}
        self.dispatch_errors: list[BaseException] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- lifecycle -----------------------------------------------------------
    def add_consumer(self, consumer_id: str, consumer: object) -> None:
        """Register a consumer peer and add it to the broadcaster (triggers its
        offer). The pump routes its signaling from then on."""
        with self._lock:
            self.consumers[consumer_id] = consumer
            self._events.setdefault(consumer_id, [])
        self.broadcaster.add_consumer(consumer_id)

    def remove_consumer(self, consumer_id: str) -> None:
        self.broadcaster.remove_consumer(consumer_id)
        consumer = self.consumers.pop(consumer_id, None)
        if consumer is not None:
            try:
                consumer.close()
            except Exception:  # noqa: BLE001 - teardown best effort
                pass

    def start(self) -> BroadcastRelay:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name=self.name, daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def close(self) -> None:
        self.stop()
        for peer in (self.broadcaster, *self.consumers.values()):
            try:
                peer.close()
            except Exception:  # noqa: BLE001 - teardown best effort
                pass

    # --- pump ----------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.pump_once()
            except Exception as exc:  # noqa: BLE001 - never let the pump die
                self.dispatch_errors.append(exc)
            time.sleep(self.POLL_INTERVAL_S)

    def pump_once(self) -> None:
        for event in self.broadcaster.drain_events():
            consumer_id = event.get("consumer_id")
            self._record("broadcaster", event)
            if consumer_id is not None:
                self._relay_to_consumer(event, consumer_id)
        # Snapshot the consumer set so a mid-pump add/remove does not error here.
        with self._lock:
            current = dict(self.consumers)
        for consumer_id, consumer in current.items():
            for event in consumer.drain_events():
                self._record(consumer_id, event)
                self._relay_to_broadcaster(event, consumer_id)

    def _record(self, which: str, event: dict) -> None:
        event[RECV_TS_KEY] = time.perf_counter()
        with self._lock:
            self._events.setdefault(which, []).append(event)

    def _relay_to_consumer(self, event: dict, consumer_id: str) -> None:
        consumer = self.consumers.get(consumer_id)
        if consumer is None:
            return
        kind = _kind(event)
        try:
            if kind == "on_local_description" and event.get("sdp_type") == "offer":
                consumer.set_remote_offer(event["sdp"])
            elif kind == "on_local_candidate":
                consumer.add_remote_candidate(event["candidate"], event.get("mid"))
        except Exception as exc:  # noqa: BLE001
            self.dispatch_errors.append(exc)

    def _relay_to_broadcaster(self, event: dict, consumer_id: str) -> None:
        kind = _kind(event)
        try:
            if kind == "on_local_description" and event.get("sdp_type") == "answer":
                self.broadcaster.set_remote_answer(consumer_id, event["sdp"])
            elif kind == "on_local_candidate":
                self.broadcaster.add_remote_candidate(
                    consumer_id, event["candidate"], event.get("mid")
                )
        except Exception as exc:  # noqa: BLE001
            self.dispatch_errors.append(exc)

    # --- observation ---------------------------------------------------------
    def events(self, which: str) -> list[dict]:
        with self._lock:
            return list(self._events.get(which, []))

    def _has_connected(self, which: str) -> bool:
        return any(
            _kind(e) == "on_state" and e.get("state") == "connected"
            for e in self.events(which)
        )

    def wait_for(
        self, which: str, predicate: EventPredicate, timeout: float
    ) -> dict | None:
        """Return the first recorded event for ``which`` matching ``predicate``."""
        deadline = time.monotonic() + timeout
        while True:
            for event in self.events(which):
                if predicate(event):
                    return event
            if time.monotonic() >= deadline:
                return None
            time.sleep(self.POLL_INTERVAL_S)

    def wait_consumer_connected(
        self, consumer_id: str, timeout: float = constants.CONNECT_TIMEOUT_S
    ) -> bool:
        """True once both the broadcaster (for this consumer) and the consumer
        peer report ``on_state: connected``."""

        def broadcaster_connected() -> bool:
            return any(
                e.get("consumer_id") == consumer_id
                and _kind(e) == "on_state"
                and e.get("state") == "connected"
                for e in self.events("broadcaster")
            )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if broadcaster_connected() and self._has_connected(consumer_id):
                return True
            time.sleep(self.POLL_INTERVAL_S)
        return False

    def video_frames(self, consumer_id: str, track_id: str | None = None) -> list[dict]:
        return [
            e
            for e in self.events(consumer_id)
            if _kind(e) == "on_frame"
            and (track_id is None or e.get("track_id") == track_id)
        ]

    # --- data-channel message collection -------------------------------------
    def messages(self, consumer_id: str, label: str) -> list[bytes]:
        """Ordered payloads of ``on_message`` events for ``label`` at a consumer."""
        return [
            e["data"]
            for e in self.events(consumer_id)
            if _kind(e) == "on_message" and e.get("label") == label
        ]

    def wait_messages(
        self, consumer_id: str, label: str, count: int, timeout: float
    ) -> list[bytes]:
        """Wait until ``count`` messages for ``label`` arrive at ``consumer_id``."""
        deadline = time.monotonic() + timeout
        while True:
            got = self.messages(consumer_id, label)
            if len(got) >= count or time.monotonic() >= deadline:
                return got
            time.sleep(self.POLL_INTERVAL_S)

    def collect_video_frames(
        self,
        consumer_id: str,
        track_id: str,
        *,
        settle: float = constants.FRAME_SETTLE_TIMEOUT_S,
        quiet: float = 0.5,
    ) -> list[dict]:
        """Drain until the decoded stream for ``consumer_id``/``track_id`` is quiet."""
        last_change = time.monotonic()
        deadline = last_change + settle
        seen = len(self.video_frames(consumer_id, track_id))
        while time.monotonic() < deadline:
            current = len(self.video_frames(consumer_id, track_id))
            if current != seen:
                seen = current
                last_change = time.monotonic()
                deadline = last_change + settle
            elif seen > 0 and time.monotonic() - last_change >= quiet:
                break
            time.sleep(0.01)
        return self.video_frames(consumer_id, track_id)

    def submit_at_rate(
        self, track_id: str, *, fps: float, seconds: float, start_counter: int = 0
    ) -> int:
        """Submit encoded frames to the broadcaster at ``fps`` for ``seconds`` —
        one shared encode, fanned to every consumer."""
        period = 1.0 / fps
        total = int(round(fps * seconds))
        origin = time.perf_counter()
        for index in range(total):
            target = origin + index * period
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)
            self.broadcaster.submit_frame(track_id, encode_frame(index + start_counter))
        return total


def bootstrap_connection(
    relay: Relay,
    *,
    control_label: str = "control",
    timeout: float = constants.CONNECT_TIMEOUT_S,
) -> None:
    """Open a control data channel (triggering the offer) and wait connected.

    Raises if the connection does not establish in ``timeout``. Against the PR0
    stubs ``add_data_channel`` raises ``NotImplementedError`` here, which is the
    expected red path.
    """
    relay.producer.add_data_channel(control_label, "reliable")
    if not relay.wait_connected(timeout):
        raise TimeoutError(f"connection did not establish within {timeout}s")


def submit_at_rate(
    relay: Relay,
    track_id: str,
    *,
    fps: float,
    seconds: float,
    start_counter: int = 0,
) -> tuple[int, dict[int, float]]:
    """Submit encoded frames at ``fps`` for ``seconds``.

    Returns ``(submitted_count, submit_times)`` where ``submit_times`` maps each
    counter to its perf-clock submit time (for glass-to-glass measurement).
    """
    period = 1.0 / fps
    total = int(round(fps * seconds))
    submit_times: dict[int, float] = {}
    origin = time.perf_counter()
    for index in range(total):
        target = origin + index * period
        now = time.perf_counter()
        if target > now:
            time.sleep(target - now)
        counter = start_counter + index
        submit_times[counter] = time.perf_counter()
        relay.producer.submit_frame(track_id, encode_frame(counter))
    return total, submit_times


def collect_video_frames(
    relay: Relay,
    track_id: str,
    *,
    settle: float = constants.FRAME_SETTLE_TIMEOUT_S,
    quiet: float = 0.5,
) -> list[dict]:
    """Drain until the decoded video stream for ``track_id`` goes quiet.

    Returns the recorded ``on_frame`` events once no new frame has arrived for
    ``quiet`` seconds, or after ``settle`` seconds total.
    """
    last_change = time.monotonic()
    deadline = last_change + settle
    seen = len(relay.video_frames("consumer", track_id))
    while time.monotonic() < deadline:
        current = len(relay.video_frames("consumer", track_id))
        if current != seen:
            seen = current
            last_change = time.monotonic()
            deadline = last_change + settle
        elif seen > 0 and time.monotonic() - last_change >= quiet:
            break
        time.sleep(0.01)
    return relay.video_frames("consumer", track_id)


def decoded_counters(frames: list[dict]) -> tuple[list[int], list[int]]:
    """Decode ``on_frame`` events into ``(counters, corrupted_counters)``.

    ``counters`` preserves arrival order; ``corrupted_counters`` holds any whose
    embedded checksum failed.
    """
    counters: list[int] = []
    corrupted: list[int] = []
    for event in frames:
        _, _, array = parse_video_frame_event(event)
        counter, ok = decode_frame(array)
        counters.append(counter)
        if not ok:
            corrupted.append(counter)
    return counters, corrupted
