"""Peer-free unit tests for the server-backed signaling transport.

These need no native peers and no backend, so they run in the sandbox: they
cover the consumer-side signaling mapping, connection_id / consumer_id routing,
and SSE frame dispatch by type with a fake feed.
"""

from __future__ import annotations

import json

from neuracore_types import HandshakeMessage, MessageType

from tests.integration.webrtc.shared.server_transport import (
    ServerBroadcastRelay,
    ServerRelay,
    SignalingConfig,
    _Inbound,
    consumer_outbound_signal,
    inbound_candidate,
    parse_sse_lines,
    signaling_config_from_env,
)

DUMMY_CONFIG = SignalingConfig(base_url="http://backend/api", org="org1", token="tok")


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeConsumer:
    """Records the consumer-side inbound setter calls."""

    def __init__(self) -> None:
        self.offers: list[str] = []
        self.candidates: list[tuple[str, str | None]] = []

    def set_remote_offer(self, sdp: str) -> None:
        self.offers.append(sdp)

    def add_remote_candidate(self, candidate: str, mid: str | None) -> None:
        self.candidates.append((candidate, mid))

    def set_remote_answer(self, sdp: str) -> None:  # producer-side use
        self.offers.append(sdp)


class FakeBroadcaster:
    """Records the broadcaster-side per-consumer inbound setter calls."""

    def __init__(self) -> None:
        self.answers: list[tuple[str, str]] = []
        self.candidates: list[tuple[str, str, str | None]] = []
        self.added: list[str] = []
        self.removed: list[str] = []

    def set_remote_answer(self, consumer_id: str, sdp: str) -> None:
        self.answers.append((consumer_id, sdp))

    def add_remote_candidate(
        self, consumer_id: str, candidate: str, mid: str | None
    ) -> None:
        self.candidates.append((consumer_id, candidate, mid))

    def add_consumer(self, consumer_id: str) -> None:
        self.added.append(consumer_id)

    def remove_consumer(self, consumer_id: str) -> None:
        self.removed.append(consumer_id)

    def drain_events(self) -> list[dict]:
        return []

    def close(self) -> None:
        pass


def _message(
    message_type: MessageType, data: str, *, connection_id: str = "conn"
) -> HandshakeMessage:
    return HandshakeMessage(
        from_id="from",
        to_id="to",
        connection_id=connection_id,
        type=message_type,
        data=data,
    )


# --------------------------------------------------------------------------- #
# Consumer-side mapping
# --------------------------------------------------------------------------- #
def test_consumer_answer_maps_to_sdp_answer() -> None:
    event = {"kind": "on_local_description", "sdp_type": "answer", "sdp": "the-sdp"}
    assert consumer_outbound_signal(event) == (MessageType.SDP_ANSWER, "the-sdp")


def test_consumer_offer_is_not_emitted() -> None:
    # The consumer is answer-only: it never offers, so an offer maps to None.
    event = {"kind": "on_local_description", "sdp_type": "offer", "sdp": "x"}
    assert consumer_outbound_signal(event) is None


def test_consumer_non_signaling_event_maps_to_none() -> None:
    assert consumer_outbound_signal({"kind": "on_state", "state": "connected"}) is None


def test_consumer_candidate_formats_and_round_trips() -> None:
    event = {"kind": "on_local_candidate", "candidate": "candidate:1 udp", "mid": "0"}
    mapped = consumer_outbound_signal(event)
    assert mapped is not None
    message_type, data = mapped
    assert message_type == MessageType.ICE_CANDIDATE
    payload = json.loads(data)
    assert payload["candidate"] == "candidate:1 udp"
    assert payload["sdpMid"] == "0"
    # The same parser the producer uses recovers (candidate, mid).
    assert inbound_candidate(data) == ("candidate:1 udp", "0")


def test_inbound_sdp_offer_drives_set_remote_offer() -> None:
    relay = ServerBroadcastRelay(FakeBroadcaster(), config=DUMMY_CONFIG)
    consumer = FakeConsumer()
    handle = relay._make_consumer_handler("c0", consumer)
    handle(_message(MessageType.SDP_OFFER, "offer-sdp"))
    handle(
        _message(
            MessageType.ICE_CANDIDATE,
            json.dumps({"candidate": "cand", "sdpMid": "1", "sdpMLineIndex": "1"}),
        )
    )
    assert consumer.offers == ["offer-sdp"]
    assert consumer.candidates == [("cand", "1")]


# --------------------------------------------------------------------------- #
# Trickle buffering
# --------------------------------------------------------------------------- #
def test_inbound_buffers_candidates_until_description() -> None:
    applied: list[str] = []
    candidates: list[tuple[str, str | None]] = []
    inbound = _Inbound(applied.append, lambda c, m: candidates.append((c, m)))

    inbound.candidate("early-1", "0")
    inbound.candidate("early-2", "0")
    assert candidates == []  # buffered, not applied before the description

    inbound.description("sdp")
    assert applied == ["sdp"]
    assert candidates == [("early-1", "0"), ("early-2", "0")]  # flushed in order

    inbound.candidate("late", "0")
    assert candidates[-1] == ("late", "0")  # applied immediately afterwards


# --------------------------------------------------------------------------- #
# Routing by connection_id / consumer_id
# --------------------------------------------------------------------------- #
def test_broadcaster_inbound_routes_answer_to_correct_consumer() -> None:
    broadcaster = FakeBroadcaster()
    relay = ServerBroadcastRelay(broadcaster, config=DUMMY_CONFIG)
    relay._conn_to_consumer["conn-a"] = "c-a"
    relay._conn_to_consumer["conn-b"] = "c-b"
    relay._broadcaster_inbound["c-a"] = relay._make_broadcaster_inbound("c-a")
    relay._broadcaster_inbound["c-b"] = relay._make_broadcaster_inbound("c-b")

    relay._on_broadcaster_message(
        _message(MessageType.SDP_ANSWER, "ans-b", connection_id="conn-b")
    )
    relay._on_broadcaster_message(
        _message(
            MessageType.ICE_CANDIDATE,
            json.dumps({"candidate": "cb", "sdpMid": "0"}),
            connection_id="conn-b",
        )
    )
    assert broadcaster.answers == [("c-b", "ans-b")]
    assert broadcaster.candidates == [("c-b", "cb", "0")]


def test_broadcaster_inbound_ignores_unknown_connection_id() -> None:
    broadcaster = FakeBroadcaster()
    relay = ServerBroadcastRelay(broadcaster, config=DUMMY_CONFIG)
    # No route registered: an unknown connection_id is ignored, not raised.
    relay._on_broadcaster_message(
        _message(MessageType.SDP_ANSWER, "x", connection_id="ghost")
    )
    assert broadcaster.answers == []


def test_relay_inbound_offer_and_answer_dispatch() -> None:
    producer = FakeConsumer()  # exposes set_remote_answer + add_remote_candidate
    consumer = FakeConsumer()
    relay = ServerRelay(producer, consumer, config=DUMMY_CONFIG)

    relay._on_consumer_message(_message(MessageType.SDP_OFFER, "offer"))
    relay._on_producer_message(_message(MessageType.SDP_ANSWER, "answer"))
    assert consumer.offers == ["offer"]
    assert producer.offers == ["answer"]  # FakeConsumer.set_remote_answer records here


# --------------------------------------------------------------------------- #
# SSE dispatch by type (fake feed)
# --------------------------------------------------------------------------- #
def test_parse_sse_lines_classifies_by_event_type() -> None:
    message = _message(MessageType.SDP_OFFER, "sdp-body").model_dump_json()
    feed = [
        "event:data",
        f"data:{message}",
        "",
        "event:heartbeat",
        "data:ping",
        "",
        "event:end",
        "data:",
        "",
    ]
    frames = list(parse_sse_lines(feed))
    assert [event for event, _ in frames] == ["data", "heartbeat", "end"]

    data_event, data_body = frames[0]
    assert data_event == "data"
    # The data frame round-trips back into a HandshakeMessage.
    parsed = HandshakeMessage.model_validate_json(data_body)
    assert parsed.type == MessageType.SDP_OFFER
    assert parsed.data == "sdp-body"


def test_parse_sse_lines_handles_comments_and_leading_space() -> None:
    feed = [
        ":comment-keepalive",
        "event: data",
        "data: hello",
        "",
    ]
    assert list(parse_sse_lines(feed)) == [("data", "hello")]


def test_parse_sse_lines_accepts_bytes() -> None:
    feed = [b"event:heartbeat", b"data:ping", b""]
    assert list(parse_sse_lines(feed)) == [("heartbeat", "ping")]


# --------------------------------------------------------------------------- #
# Env gating
# --------------------------------------------------------------------------- #
def test_config_from_env_none_when_unset(monkeypatch) -> None:
    for var in (
        "NEURACORE_WEBRTC_SIGNALING_URL",
        "NEURACORE_WEBRTC_SIGNALING_ORG",
        "NEURACORE_WEBRTC_SIGNALING_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    assert signaling_config_from_env() is None


def test_config_from_env_built_when_set(monkeypatch) -> None:
    monkeypatch.setenv("NEURACORE_WEBRTC_SIGNALING_URL", "http://backend/api/")
    monkeypatch.setenv("NEURACORE_WEBRTC_SIGNALING_ORG", "org42")
    monkeypatch.setenv("NEURACORE_WEBRTC_SIGNALING_TOKEN", "secret")
    config = signaling_config_from_env()
    assert config == SignalingConfig("http://backend/api", "org42", "secret")
