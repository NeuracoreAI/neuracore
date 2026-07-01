"""Peer-free unit tests for the web producer wiring (PR8).

These exercise :mod:`native_broadcast_provider` against a fake, peer-free
producer (a stand-in for the native ``Broadcaster``) so the event mapping,
add/remove/reconnect consumer lifecycle and the Chrome cname munge can be
checked without a real WebRTC connection.
"""

from __future__ import annotations

import json
import os

import pytest
from neuracore_types import MessageType

from neuracore.core.streaming.p2p.provider.native_broadcast_provider import (
    CHROME_SDP_ENV,
    NativeBroadcastProvider,
    inbound_candidate,
    needs_reconnect,
    outbound_signal,
)


class FakeProducer:
    """Records calls and replays a scripted ``drain_events`` queue."""

    def __init__(self) -> None:
        self.added: list[str] = []
        self.removed: list[str] = []
        self.answers: list[tuple[str, str]] = []
        self.candidates: list[tuple[str, str, str | None]] = []
        self.tracks: list[str] = []
        self.frames: list[tuple[str, object]] = []
        self.closed = False
        self._pending: list[dict] = []
        # Reference model of the native Broadcaster's data-channel fan-out: which
        # consumers are live, the registry of channels every consumer gets, and
        # each consumer's received messages keyed by label. This faithfully mirrors
        # the Rust contract (open on add, bootstrap for late joiners, tear down on
        # leave) so the peer-free tests exercise the real semantics.
        self._consumers: set[str] = set()
        self._registry: list[tuple[str, str]] = []
        self.consumer_channels: dict[str, dict[str, list[str]]] = {}

    def queue(self, *events: dict) -> None:
        self._pending.extend(events)

    # native Broadcaster API ------------------------------------------------
    def add_consumer(self, consumer_id: str) -> None:
        self.added.append(consumer_id)
        self._consumers.add(consumer_id)
        # A late joiner gets every already-registered channel at bootstrap.
        self.consumer_channels[consumer_id] = {
            label: [] for label, _kind in self._registry
        }

    def remove_consumer(self, consumer_id: str) -> None:
        self.removed.append(consumer_id)
        self._consumers.discard(consumer_id)
        # A leaving consumer's channels tear down with it (no registry leak).
        self.consumer_channels.pop(consumer_id, None)

    def set_remote_answer(self, consumer_id: str, sdp: str) -> None:
        self.answers.append((consumer_id, sdp))

    def add_remote_candidate(
        self, consumer_id: str, candidate: str, mid: str | None
    ) -> None:
        self.candidates.append((consumer_id, candidate, mid))

    def add_video_track(self, track_id: str) -> None:
        self.tracks.append(track_id)

    def remove_video_track(self, track_id: str) -> None:
        self.tracks.remove(track_id)

    def submit_frame(self, track_id: str, frame: object) -> None:
        self.frames.append((track_id, frame))

    def add_data_channel(self, label: str, kind: str) -> None:
        self._registry.append((label, kind))
        # Opened on every current consumer; future consumers get it at bootstrap.
        for channels in self.consumer_channels.values():
            channels.setdefault(label, [])

    def send_json(self, label: str, payload: str) -> None:
        # Fans to every consumer that carries the label.
        for channels in self.consumer_channels.values():
            if label in channels:
                channels[label].append(payload)

    def drain_events(self) -> list[dict]:
        drained, self._pending = self._pending, []
        return drained

    def close(self) -> None:
        self.closed = True


def make_provider(*, browser_facing: bool = True):
    producer = FakeProducer()
    sent: list[tuple] = []
    provider = NativeBroadcastProvider(
        producer,
        lambda cid, rid, mt, data: sent.append((cid, rid, mt, data)),
        browser_facing=browser_facing,
    )
    return provider, producer, sent


# --- pure helpers ----------------------------------------------------------


def test_outbound_signal_maps_offer_and_candidate():
    offer = outbound_signal({
        "kind": "on_local_description",
        "sdp_type": "offer",
        "sdp": "v=0...",
        "consumer_id": "c1",
    })
    assert offer.consumer_id == "c1"
    assert offer.message_type == MessageType.SDP_OFFER
    assert offer.data == "v=0..."

    cand = outbound_signal({
        "kind": "on_local_candidate",
        "candidate": "candidate:1 ...",
        "mid": "0",
        "consumer_id": "c1",
    })
    assert cand.message_type == MessageType.ICE_CANDIDATE
    payload = json.loads(cand.data)
    assert payload["candidate"] == "candidate:1 ..."
    assert payload["sdpMid"] == "0"


def test_outbound_signal_ignores_answers_and_untagged_events():
    # The producer is the offerer; an answer sdp_type is never sent outbound.
    assert (
        outbound_signal({
            "kind": "on_local_description",
            "sdp_type": "answer",
            "sdp": "x",
            "consumer_id": "c1",
        })
        is None
    )
    # A shared-encode event without a consumer_id is not deliverable.
    assert outbound_signal({"kind": "on_local_candidate", "candidate": "x"}) is None
    assert outbound_signal({"kind": "on_state", "state": "connected"}) is None


def test_inbound_candidate_parses_browser_payload():
    data = json.dumps({
        "candidate": "candidate:2 ...",
        "sdpMid": "1",
        "sdpMLineIndex": 1,
    })
    assert inbound_candidate(data) == ("candidate:2 ...", "1")


def test_needs_reconnect_only_for_connection_errors():
    assert (
        needs_reconnect({
            "kind": "on_error",
            "where": "connection",
            "consumer_id": "c1",
        })
        == "c1"
    )
    assert (
        needs_reconnect({"kind": "on_error", "where": "encode", "consumer_id": "c1"})
        is None
    )
    assert needs_reconnect({"kind": "on_state", "state": "failed"}) is None


# --- cname munge -----------------------------------------------------------


def test_browser_session_enables_cname_munge(monkeypatch):
    monkeypatch.delenv(CHROME_SDP_ENV, raising=False)
    make_provider(browser_facing=True)
    assert os.environ.get(CHROME_SDP_ENV) == "1"


def test_loopback_session_leaves_cname_munge_off(monkeypatch):
    monkeypatch.delenv(CHROME_SDP_ENV, raising=False)
    make_provider(browser_facing=False)
    assert CHROME_SDP_ENV not in os.environ


# --- consumer lifecycle ----------------------------------------------------


def test_add_and_remove_consumer():
    provider, producer, _ = make_provider()
    provider.add_consumer("c1", "stream-1")
    provider.add_consumer("c1", "stream-1")  # idempotent
    assert producer.added == ["c1"]

    provider.remove_consumer("c1")
    provider.remove_consumer("c1")  # idempotent
    assert producer.removed == ["c1"]


def test_reconnect_is_remove_then_readd():
    provider, producer, _ = make_provider()
    provider.add_consumer("c1", "stream-1")
    provider.reconnect_consumer("c1")
    assert producer.removed == ["c1"]
    assert producer.added == ["c1", "c1"]


def test_reconnect_unknown_consumer_is_noop():
    provider, producer, _ = make_provider()
    provider.reconnect_consumer("ghost")
    assert producer.removed == []


# --- inbound signaling -----------------------------------------------------


def test_on_answer_and_candidate_feed_the_producer():
    provider, producer, _ = make_provider()
    provider.add_consumer("c1", "stream-1")
    provider.on_answer("c1", "answer-sdp")
    provider.on_ice_candidate(
        "c1", json.dumps({"candidate": "candidate:9 ...", "sdpMid": "2"})
    )
    assert producer.answers == [("c1", "answer-sdp")]
    assert producer.candidates == [("c1", "candidate:9 ...", "2")]


def test_inbound_for_unknown_consumer_is_ignored():
    provider, producer, _ = make_provider()
    provider.on_answer("ghost", "sdp")
    provider.on_ice_candidate("ghost", json.dumps({"candidate": "x", "sdpMid": "0"}))
    assert producer.answers == []
    assert producer.candidates == []


# --- the pump --------------------------------------------------------------


def test_pump_routes_offer_to_the_right_browser_transport():
    provider, producer, sent = make_provider()
    provider.add_consumer("c1", "stream-1")
    producer.queue(
        {
            "kind": "on_local_description",
            "sdp_type": "offer",
            "sdp": "offer-sdp",
            "consumer_id": "c1",
        },
        {
            "kind": "on_local_candidate",
            "candidate": "candidate:1 ...",
            "mid": "0",
            "consumer_id": "c1",
        },
    )
    provider.pump_once()

    assert sent[0] == ("c1", "stream-1", MessageType.SDP_OFFER, "offer-sdp")
    assert sent[1][0:3] == ("c1", "stream-1", MessageType.ICE_CANDIDATE)


def test_pump_handles_reconnect_error_by_readding_consumer():
    provider, producer, sent = make_provider()
    provider.add_consumer("c1", "stream-1")
    producer.queue({
        "kind": "on_error",
        "where": "connection",
        "consumer_id": "c1",
        "detail": "dropped",
    })
    provider.pump_once()

    assert producer.removed == ["c1"]
    assert producer.added == ["c1", "c1"]
    assert sent == []  # an error is not forwarded as a signaling message


def test_pump_drops_signal_for_an_unknown_consumer():
    provider, producer, sent = make_provider()
    producer.queue({
        "kind": "on_local_description",
        "sdp_type": "offer",
        "sdp": "x",
        "consumer_id": "ghost",
    })
    provider.pump_once()
    assert sent == []


# --- media + close ---------------------------------------------------------


def test_video_track_registration_and_close():
    provider, producer, _ = make_provider()
    provider.add_video_track("cam0")
    provider.add_video_track("cam0")  # idempotent
    provider.submit_frame("cam0", object())
    assert producer.tracks == ["cam0"]
    assert len(producer.frames) == 1

    provider.close()
    assert producer.closed is True


# --- data-channel fan-out --------------------------------------------------


def test_add_data_channel_fans_to_all_current_consumers():
    provider, producer, _ = make_provider()
    provider.add_consumer("a", "stream-a")
    provider.add_consumer("b", "stream-b")

    provider.add_data_channel("joints")

    # Opened on every current consumer.
    assert "joints" in producer.consumer_channels["a"]
    assert "joints" in producer.consumer_channels["b"]


def test_add_data_channel_is_idempotent_per_label():
    provider, producer, _ = make_provider()
    provider.add_consumer("a", "stream-a")
    provider.add_data_channel("joints")
    provider.add_data_channel("joints")  # second add is a no-op
    assert producer._registry == [("joints", "reliable")]


def test_send_json_fans_to_every_consumer():
    provider, producer, _ = make_provider()
    provider.add_consumer("a", "stream-a")
    provider.add_consumer("b", "stream-b")
    provider.add_data_channel("json")

    provider.send_json("json", '{"i": 1}')

    assert producer.consumer_channels["a"]["json"] == ['{"i": 1}']
    assert producer.consumer_channels["b"]["json"] == ['{"i": 1}']


def test_a_consumer_added_after_add_data_channel_still_gets_the_channel():
    provider, producer, _ = make_provider()
    provider.add_data_channel("json")
    # The browser joins only afterwards.
    provider.add_consumer("late", "stream-late")

    provider.send_json("json", '{"i": 7}')

    # The late joiner got the channel at bootstrap and receives the message.
    assert producer.consumer_channels["late"]["json"] == ['{"i": 7}']


def test_a_leaving_consumer_tears_down_its_channels_no_registry_leak():
    provider, producer, _ = make_provider()
    provider.add_consumer("a", "stream-a")
    provider.add_consumer("b", "stream-b")
    provider.add_data_channel("json")

    provider.remove_consumer("a")
    provider.send_json("json", '{"i": 2}')

    # 'a' is gone with its channels; only 'b' receives.
    assert "a" not in producer.consumer_channels
    assert producer.consumer_channels["b"]["json"] == ['{"i": 2}']


def test_recording_bridge_routes_log_calls_to_broadcaster_send_json():
    """The RecordingContext bridge drives ``send_json`` over the Broadcaster."""
    from neuracore.core.streaming.p2p.recording_bridge import WebrtcRecordingBridge

    provider, producer, _ = make_provider()
    provider.add_consumer("a", "stream-a")
    bridge = WebrtcRecordingBridge(provider)

    bridge.log_joints("joint_positions", timestamp=1.0, items=[("j0", 0.5)])
    bridge.log_json("rgb", "cam", b"{}", timestamp=2.0)

    # Both log entry points opened their channel and reached the consumer.
    assert json.loads(producer.consumer_channels["a"]["joint_positions"][0]) == {
        "type": "joints",
        "data_type": "joint_positions",
        "timestamp": 1.0,
        "values": {"j0": 0.5},
    }
    assert json.loads(producer.consumer_channels["a"]["rgb/cam"][0])["type"] == "json"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
