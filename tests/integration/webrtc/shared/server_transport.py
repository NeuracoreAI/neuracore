"""Server-backed signaling transport for the WebRTC integration suite.

The default harness joins two native peers with an in-process relay (no signaling
server). This module adds an additive, env-gated alternative that drives the same
native ``Producer`` / ``Broadcaster`` / ``Consumer`` peers through the real
running backend's signaling: the REST submit endpoint plus the SSE notification
stream the deprecated aiortc path and the PR8 web path both use. It validates the
real transport end to end without a browser, as the intermediate gate between the
in-process loopback and the browser Playwright run.

Design: subclass the relays and swap only the cross-peer hop. The parent
drain-and-record pump is unchanged, so every test assertion and ``wait_*`` helper
keeps working; the only difference is that a peer's drained
``on_local_description`` / ``on_local_candidate`` is mapped to a
``HandshakeMessage`` and POSTed to the backend instead of handed to the other
peer in-process, and inbound messages arrive on per-stream SSE reader threads.

The producer-side mappings are reused from PR8
(:func:`outbound_signal` / :func:`inbound_candidate` / :func:`needs_reconnect`),
not duplicated. The consumer-side mirror lives here because the production
consumer is a browser, so it is test-only.

Wire contract (confirmed against the live backend):
  * submit:    POST /api/org/{org}/signalling/message/submit  (HandshakeMessage)
  * subscribe: GET  /api/org/{org}/signalling/notifications/{stream_id}  (SSE)
  * keepalive: POST /api/org/{org}/signalling/alive/{stream_id}  (body "pong")
  * auth:      Authorization: Bearer <token>
A submit whose ``to_id`` is not yet a registered SSE subscriber is silently
dropped with no backend queue, so a peer's SSE stream is opened and confirmed
before any offer is POSTed to it. The 25s inactivity reaper requires answering
SSE heartbeats by POSTing the alive endpoint. Both peers run in this one process,
so only signaling crosses the backend; media is local loopback P2P (no TURN).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from uuid import uuid4

import requests
from neuracore_types import HandshakeMessage, MessageType

from neuracore.core.streaming.p2p.provider.native_broadcast_provider import (
    inbound_candidate,
    needs_reconnect,
    outbound_signal,
)
from tests.integration.webrtc.shared import constants
from tests.integration.webrtc.shared.harness import BroadcastRelay, Relay

logger = logging.getLogger(__name__)

# Env that selects and configures the server-backed transport. When any is unset
# the suite falls back to the in-process relay and the server-backed tests skip.
URL_ENV = "NEURACORE_WEBRTC_SIGNALING_URL"
ORG_ENV = "NEURACORE_WEBRTC_SIGNALING_ORG"
TOKEN_ENV = "NEURACORE_WEBRTC_SIGNALING_TOKEN"

# Placeholder consumer id injected so the 1:1 producer (which emits no
# ``consumer_id``) and the consumer side can reuse the PR8 ``outbound_signal``
# candidate formatting without duplicating it. The value is never sent on the
# wire; addressing is by stream id.
_PLACEHOLDER_ID = "_"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SignalingConfig:
    """Operator-supplied backend coordinates for the server-backed transport."""

    base_url: str  # includes the /api prefix, e.g. http://host:8000/api
    org: str
    token: str


def signaling_config_from_env() -> SignalingConfig | None:
    """Build a config from env, or None when the transport is not selected."""
    base_url = os.environ.get(URL_ENV)
    org = os.environ.get(ORG_ENV)
    token = os.environ.get(TOKEN_ENV)
    if not (base_url and org and token):
        return None
    return SignalingConfig(base_url.rstrip("/"), org, token)


# --------------------------------------------------------------------------- #
# Pure mappings (peer-free, unit-tested)
# --------------------------------------------------------------------------- #
def consumer_outbound_signal(event: dict) -> tuple[MessageType, str] | None:
    """Map one drained consumer event to a signaling message.

    The consumer is answer-only, so it emits SDP **answers** and ICE candidates.
    Candidate formatting is delegated to the PR8 producer mapper (with a
    placeholder ``consumer_id``) so the JSON shape is not duplicated.

    Args:
        event: a single dict from ``Consumer.drain_events()``.

    Returns:
        ``(message_type, data)`` to submit, or ``None`` if not deliverable.
    """
    kind = event.get("kind")
    if kind == "on_local_description" and event.get("sdp_type") == "answer":
        return MessageType.SDP_ANSWER, event["sdp"]
    if kind == "on_local_candidate":
        signal = outbound_signal({**event, "consumer_id": _PLACEHOLDER_ID})
        if signal is not None:
            return signal.message_type, signal.data
    return None


def parse_sse_lines(lines: Iterable[bytes | str]) -> Iterator[tuple[str, str]]:
    """Parse a stream of SSE lines into ``(event_type, data)`` frames.

    Handles the backend's ``event:data`` / ``event:heartbeat`` / ``event:end``
    framing (optional leading space after the colon, comment lines, multi-line
    data joined by newline, and the blank-line frame boundary).
    """
    event = "message"
    data_parts: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if line == "":
            if data_parts:
                yield event, "\n".join(data_parts)
            event = "message"
            data_parts = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data_parts.append(value)
    if data_parts:
        yield event, "\n".join(data_parts)


# --------------------------------------------------------------------------- #
# Inbound application with trickle buffering
# --------------------------------------------------------------------------- #
class _Inbound:
    """Applies inbound signaling to one peer, buffering early candidates.

    Candidates that arrive before the remote description is set are buffered and
    flushed immediately after it, mirroring the web path and the aiortc
    ``received_offer_event`` / ``received_answer_event`` gate. The peer setters
    are injected so this is unit-testable with a fake peer.
    """

    def __init__(
        self,
        description: Callable[[str], None],
        add_candidate: Callable[[str, str | None], None],
    ) -> None:
        self._description = description
        self._add_candidate = add_candidate
        self._have_description = False
        self._pending: list[tuple[str, str | None]] = []
        self._lock = threading.Lock()

    def description(self, sdp: str) -> None:
        """Apply the remote offer/answer, then flush buffered candidates."""
        self._description(sdp)
        with self._lock:
            self._have_description = True
            pending = self._pending
            self._pending = []
        for candidate, mid in pending:
            self._add_candidate(candidate, mid)

    def candidate(self, candidate: str, mid: str | None) -> None:
        """Apply a candidate, or buffer it until the description is set."""
        with self._lock:
            if not self._have_description:
                self._pending.append((candidate, mid))
                return
        self._add_candidate(candidate, mid)


# --------------------------------------------------------------------------- #
# HTTP client (POST submit/alive + SSE subscribe)
# --------------------------------------------------------------------------- #
class _Subscription:
    """One SSE reader thread for a stream id, with reconnect and heartbeat."""

    def __init__(self, client: SignalingClient, stream_id: str) -> None:
        self._client = client
        self._stream_id = stream_id
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._response: requests.Response | None = None
        self._thread = threading.Thread(
            target=self._run, name=f"sse-{stream_id[:8]}", daemon=True
        )
        self._thread.start()

    def wait_connected(self, timeout: float) -> bool:
        """True once the SSE GET has established (the stream is registered)."""
        return self._connected.wait(timeout)

    def stop(self) -> None:
        self._stop.set()
        if self._response is not None:
            try:
                self._response.close()  # unblock a blocked iter_lines
            except Exception:  # noqa: BLE001 - teardown best effort
                pass
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        backoff = 0.05
        while not self._stop.is_set():
            try:
                response = self._client.open_stream(self._stream_id)
                self._response = response
                self._connected.set()
                backoff = 0.05
                for event, data in parse_sse_lines(
                    response.iter_lines(decode_unicode=True)
                ):
                    if self._stop.is_set():
                        break
                    if event == "data":
                        self._dispatch(data)
                    elif event == "heartbeat":
                        self._client.mark_alive(self._stream_id)
                    elif event == "end":
                        break
            except Exception as exc:  # noqa: BLE001 - reconnect on any error
                if not self._stop.is_set():
                    logger.warning("sse stream %s error: %s", self._stream_id, exc)
            if self._stop.is_set():
                break
            time.sleep(backoff)
            backoff = min(5.0, backoff * 2)

    def _dispatch(self, data: str) -> None:
        try:
            message = HandshakeMessage.model_validate_json(data)
        except Exception:  # noqa: BLE001 - ignore malformed frames
            logger.warning("sse stream %s dropped malformed message", self._stream_id)
            return
        self._client.deliver(self._stream_id, message)


class SignalingClient:
    """Thin HTTP client over the backend signaling contract.

    POSTs handshake messages, opens SSE subscriptions, answers heartbeats, and
    fans inbound messages to the handler registered for each stream id.
    """

    CONNECT_TIMEOUT_S = 10.0
    # Longer than the 20s server heartbeat so a healthy stream never times out
    # but a dead one eventually errors and the subscription reconnects.
    READ_TIMEOUT_S = 40.0

    def __init__(self, config: SignalingConfig) -> None:
        self._config = config
        self._session = requests.Session()
        self._handlers: dict[str, Callable[[HandshakeMessage], None]] = {}
        self._lock = threading.Lock()

    # --- url + headers -------------------------------------------------------
    def _signalling_base(self) -> str:
        return f"{self._config.base_url}/org/{self._config.org}/signalling"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._config.token}"}

    # --- outbound ------------------------------------------------------------
    def submit(
        self,
        from_id: str,
        to_id: str,
        connection_id: str,
        message_type: MessageType,
        data: str,
    ) -> None:
        """POST one handshake message; a fresh id avoids backend LRU de-duplication."""
        message = HandshakeMessage(
            from_id=from_id,
            to_id=to_id,
            connection_id=connection_id,
            type=message_type,
            data=data,
        )
        response = self._session.post(
            f"{self._signalling_base()}/message/submit",
            headers=self._headers(),
            json=message.model_dump(mode="json"),
            timeout=self.CONNECT_TIMEOUT_S,
        )
        response.raise_for_status()

    def mark_alive(self, stream_id: str) -> None:
        """Answer a heartbeat so the inactivity reaper keeps the stream."""
        try:
            self._session.post(
                f"{self._signalling_base()}/alive/{stream_id}",
                headers=self._headers(),
                data="pong",
                timeout=self.CONNECT_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001 - keepalive is best effort
            logger.warning("alive ping for %s failed: %s", stream_id, exc)

    # --- inbound -------------------------------------------------------------
    def subscribe(
        self, stream_id: str, on_message: Callable[[HandshakeMessage], None]
    ) -> _Subscription:
        """Register a handler and start an SSE reader thread for ``stream_id``."""
        with self._lock:
            self._handlers[stream_id] = on_message
        return _Subscription(self, stream_id)

    def open_stream(self, stream_id: str) -> requests.Response:
        """Open the SSE GET (used by :class:`_Subscription`)."""
        response = self._session.get(
            f"{self._signalling_base()}/notifications/{stream_id}",
            headers=self._headers(),
            stream=True,
            timeout=(self.CONNECT_TIMEOUT_S, self.READ_TIMEOUT_S),
        )
        response.raise_for_status()
        return response

    def deliver(self, stream_id: str, message: HandshakeMessage) -> None:
        """Route one inbound message to its stream's handler."""
        with self._lock:
            handler = self._handlers.get(stream_id)
        if handler is not None:
            handler(message)

    def close(self) -> None:
        with self._lock:
            self._handlers.clear()
        try:
            self._session.close()
        except Exception:  # noqa: BLE001 - teardown best effort
            pass


# --------------------------------------------------------------------------- #
# 1:1 server-backed relay
# --------------------------------------------------------------------------- #
class ServerRelay(Relay):
    """A :class:`Relay` that signals over the real backend instead of in-process.

    The parent pump still drains and records both peers; this subclass only
    rewrites the cross-peer hop (``_relay``) to POST and adds two SSE reader
    threads that apply inbound messages to the local peers.
    """

    def __init__(
        self,
        producer: object,
        consumer: object,
        *,
        config: SignalingConfig,
        name: str = "server-relay",
        connect_timeout: float = constants.CONNECT_TIMEOUT_S,
    ) -> None:
        super().__init__(producer, consumer, name=name)
        self._client = SignalingClient(config)
        self._connect_timeout = connect_timeout
        self._producer_stream_id = uuid4().hex
        self._consumer_stream_id = uuid4().hex
        self._connection_id = uuid4().hex
        self._producer_inbound = _Inbound(
            producer.set_remote_answer, producer.add_remote_candidate
        )
        self._consumer_inbound = _Inbound(
            consumer.set_remote_offer, consumer.add_remote_candidate
        )
        self._subs: list[_Subscription] = []

    def start(self) -> ServerRelay:
        # Open and confirm both SSE streams before the pump can POST any offer,
        # so neither peer is an unregistered (silently dropped) recipient.
        self._subs.append(
            self._client.subscribe(self._producer_stream_id, self._on_producer_message)
        )
        self._subs.append(
            self._client.subscribe(self._consumer_stream_id, self._on_consumer_message)
        )
        for sub in self._subs:
            sub.wait_connected(self._connect_timeout)
        super().start()
        return self

    def close(self) -> None:
        for sub in self._subs:
            sub.stop()
        self._client.close()
        super().close()

    # --- outbound (override the in-process hop) ------------------------------
    def _relay(self, event: dict, dst: object) -> None:
        try:
            if dst is self.consumer:
                # producer -> consumer: 1:1 producer emits no consumer_id, so
                # inject the placeholder to reuse the PR8 producer mapping.
                signal = outbound_signal({**event, "consumer_id": _PLACEHOLDER_ID})
                if signal is None:
                    return
                self._client.submit(
                    self._producer_stream_id,
                    self._consumer_stream_id,
                    self._connection_id,
                    signal.message_type,
                    signal.data,
                )
            else:
                mapped = consumer_outbound_signal(event)
                if mapped is None:
                    return
                message_type, data = mapped
                self._client.submit(
                    self._consumer_stream_id,
                    self._producer_stream_id,
                    self._connection_id,
                    message_type,
                    data,
                )
        except Exception as exc:  # noqa: BLE001 - surfaced via dispatch_errors
            self.dispatch_errors.append(exc)

    # --- inbound -------------------------------------------------------------
    def _on_producer_message(self, message: HandshakeMessage) -> None:
        if message.type == MessageType.SDP_ANSWER:
            self._producer_inbound.description(message.data)
        elif message.type == MessageType.ICE_CANDIDATE:
            candidate, mid = inbound_candidate(message.data)
            self._producer_inbound.candidate(candidate, mid)

    def _on_consumer_message(self, message: HandshakeMessage) -> None:
        if message.type == MessageType.SDP_OFFER:
            self._consumer_inbound.description(message.data)
        elif message.type == MessageType.ICE_CANDIDATE:
            candidate, mid = inbound_candidate(message.data)
            self._consumer_inbound.candidate(candidate, mid)


# --------------------------------------------------------------------------- #
# Multi-consumer server-backed relay
# --------------------------------------------------------------------------- #
class ServerBroadcastRelay(BroadcastRelay):
    """A :class:`BroadcastRelay` that signals each consumer over the backend.

    Per-consumer routing maps the broadcaster's internal ``consumer_id`` to a
    server ``connection_id`` and stream id so offers, answers, and candidates
    reach the right peer. A join opens the consumer's SSE stream before
    ``add_consumer`` (so the broadcaster's offer is not dropped); a leave tears
    its subscription down; a PR7 ``on_error{where:"connection"}`` is a remove +
    re-add over the server with a fresh ``connection_id`` (no ICE restart).
    """

    def __init__(
        self,
        broadcaster: object,
        *,
        config: SignalingConfig,
        name: str = "server-broadcast",
        connect_timeout: float = constants.CONNECT_TIMEOUT_S,
    ) -> None:
        super().__init__(broadcaster, name=name)
        self._client = SignalingClient(config)
        self._connect_timeout = connect_timeout
        self._broadcaster_stream_id = uuid4().hex
        self._consumer_stream: dict[str, str] = {}
        self._consumer_conn: dict[str, str] = {}
        self._conn_to_consumer: dict[str, str] = {}
        self._broadcaster_inbound: dict[str, _Inbound] = {}
        self._consumer_subs: dict[str, _Subscription] = {}
        self._broadcaster_sub: _Subscription | None = None

    def start(self) -> ServerBroadcastRelay:
        self._broadcaster_sub = self._client.subscribe(
            self._broadcaster_stream_id, self._on_broadcaster_message
        )
        self._broadcaster_sub.wait_connected(self._connect_timeout)
        super().start()
        return self

    def close(self) -> None:
        for sub in self._consumer_subs.values():
            sub.stop()
        if self._broadcaster_sub is not None:
            self._broadcaster_sub.stop()
        self._client.close()
        super().close()

    # --- consumer lifecycle --------------------------------------------------
    def add_consumer(self, consumer_id: str, consumer: object) -> None:
        stream_id = uuid4().hex
        connection_id = uuid4().hex
        with self._lock:
            self.consumers[consumer_id] = consumer
            self._events.setdefault(consumer_id, [])
            self._consumer_stream[consumer_id] = stream_id
            self._consumer_conn[consumer_id] = connection_id
            self._conn_to_consumer[connection_id] = consumer_id
            self._broadcaster_inbound[consumer_id] = self._make_broadcaster_inbound(
                consumer_id
            )
        sub = self._client.subscribe(
            stream_id, self._make_consumer_handler(consumer_id, consumer)
        )
        self._consumer_subs[consumer_id] = sub
        # Confirm the consumer is registered before the broadcaster offers to it.
        sub.wait_connected(self._connect_timeout)
        self.broadcaster.add_consumer(consumer_id)

    def remove_consumer(self, consumer_id: str) -> None:
        self.broadcaster.remove_consumer(consumer_id)
        sub = self._consumer_subs.pop(consumer_id, None)
        if sub is not None:
            sub.stop()
        consumer = self.consumers.pop(consumer_id, None)
        with self._lock:
            self._consumer_stream.pop(consumer_id, None)
            conn = self._consumer_conn.pop(consumer_id, None)
            if conn is not None:
                self._conn_to_consumer.pop(conn, None)
            self._broadcaster_inbound.pop(consumer_id, None)
        if consumer is not None:
            try:
                consumer.close()
            except Exception:  # noqa: BLE001 - teardown best effort
                pass

    def _make_broadcaster_inbound(self, consumer_id: str) -> _Inbound:
        return _Inbound(
            lambda sdp: self.broadcaster.set_remote_answer(consumer_id, sdp),
            lambda candidate, mid: self.broadcaster.add_remote_candidate(
                consumer_id, candidate, mid
            ),
        )

    def _make_consumer_handler(
        self, consumer_id: str, consumer: object
    ) -> Callable[[HandshakeMessage], None]:
        inbound = _Inbound(consumer.set_remote_offer, consumer.add_remote_candidate)

        def handle(message: HandshakeMessage) -> None:
            if message.type == MessageType.SDP_OFFER:
                inbound.description(message.data)
            elif message.type == MessageType.ICE_CANDIDATE:
                candidate, mid = inbound_candidate(message.data)
                inbound.candidate(candidate, mid)

        return handle

    # --- pump (add PR7 reconnect handling to the parent routing) -------------
    def pump_once(self) -> None:
        for event in self.broadcaster.drain_events():
            consumer_id = event.get("consumer_id")
            self._record("broadcaster", event)
            reconnect_id = needs_reconnect(event)
            if reconnect_id is not None:
                logger.warning(
                    "webrtc consumer %s needs reconnect: %s",
                    reconnect_id,
                    event.get("detail"),
                )
                self._reconnect(reconnect_id)
                continue
            if consumer_id is not None:
                self._relay_to_consumer(event, consumer_id)
        with self._lock:
            current = dict(self.consumers)
        for consumer_id, consumer in current.items():
            for event in consumer.drain_events():
                self._record(consumer_id, event)
                self._relay_to_broadcaster(event, consumer_id)

    def _reconnect(self, consumer_id: str) -> None:
        """Remove + re-add one consumer over the server with a fresh id."""
        if consumer_id not in self.consumers:
            return
        new_conn = uuid4().hex
        with self._lock:
            old_conn = self._consumer_conn.get(consumer_id)
            if old_conn is not None:
                self._conn_to_consumer.pop(old_conn, None)
            self._consumer_conn[consumer_id] = new_conn
            self._conn_to_consumer[new_conn] = consumer_id
            self._broadcaster_inbound[consumer_id] = self._make_broadcaster_inbound(
                consumer_id
            )
        self.broadcaster.remove_consumer(consumer_id)
        self.broadcaster.add_consumer(consumer_id)

    # --- outbound (override the in-process hops) -----------------------------
    def _relay_to_consumer(self, event: dict, consumer_id: str) -> None:
        try:
            signal = outbound_signal(event)  # event is already consumer-tagged
            if signal is None:
                return
            with self._lock:
                to_id = self._consumer_stream.get(consumer_id)
                connection_id = self._consumer_conn.get(consumer_id)
            if to_id is None or connection_id is None:
                return
            self._client.submit(
                self._broadcaster_stream_id,
                to_id,
                connection_id,
                signal.message_type,
                signal.data,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via dispatch_errors
            self.dispatch_errors.append(exc)

    def _relay_to_broadcaster(self, event: dict, consumer_id: str) -> None:
        try:
            mapped = consumer_outbound_signal(event)
            if mapped is None:
                return
            message_type, data = mapped
            with self._lock:
                from_id = self._consumer_stream.get(consumer_id)
                connection_id = self._consumer_conn.get(consumer_id)
            if from_id is None or connection_id is None:
                return
            self._client.submit(
                from_id,
                self._broadcaster_stream_id,
                connection_id,
                message_type,
                data,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced via dispatch_errors
            self.dispatch_errors.append(exc)

    # --- inbound -------------------------------------------------------------
    def _on_broadcaster_message(self, message: HandshakeMessage) -> None:
        with self._lock:
            consumer_id = self._conn_to_consumer.get(message.connection_id)
            inbound = (
                self._broadcaster_inbound.get(consumer_id)
                if consumer_id is not None
                else None
            )
        if inbound is None:
            return  # unknown connection_id: ignore, do not raise
        if message.type == MessageType.SDP_ANSWER:
            inbound.description(message.data)
        elif message.type == MessageType.ICE_CANDIDATE:
            candidate, mid = inbound_candidate(message.data)
            inbound.candidate(candidate, mid)
