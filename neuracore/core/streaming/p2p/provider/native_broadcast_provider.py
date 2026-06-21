"""Native (Rust) broadcast provider for the web streaming path.

This is the producer-side wiring of the new ``neuracore_webrtc`` stack for
browser consumers, gated by ``NCD_RUST_WEBRTC`` alongside the legacy aiortc
:class:`PeerToPeerProviderConnection`. The producer is the **sole offerer** and
every browser is an **answer-only** consumer, so there is no glare.

One :class:`NativeBroadcastProvider` owns a single native ``Broadcaster`` (one
shared encode per source fanned out to N browsers). It maps the broadcaster's
drained, per-consumer signaling events onto the web signaling transport
(``send_handshake_message``) and feeds the browser's answer / candidates back in
via ``set_remote_answer(consumer_id, …)`` / ``add_remote_candidate(consumer_id, …)``.

Lifecycle: a browser connecting is ``add_consumer``; disconnecting is
``remove_consumer``; a PR7 reconnect-needed ``on_error{where:"connection"}`` is a
remove + re-add for that consumer (the binding cannot ICE-restart — libjuice is
single-shot, upstream #130).

The Chrome ``a=ssrc … cname`` munge is turned ON for these browser-facing
sessions (``NCD_WEBRTC_CHROME_SDP``); it is left off only for the
libdatachannel-to-libdatachannel loopback tests that assert byte-identical SDP.

The module-level helpers (:func:`outbound_signal`, :func:`inbound_candidate`)
are pure so they can be unit-tested peer-free with a fake producer.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from neuracore_types import MessageType

logger = logging.getLogger(__name__)


class BroadcasterProducer(Protocol):
    """The subset of the native ``Broadcaster`` API this adapter drives."""

    def add_consumer(self, consumer_id: str) -> None:
        """Stand up an answer-only consumer peer connection."""

    def remove_consumer(self, consumer_id: str) -> None:
        """Tear down one consumer peer connection."""

    def set_remote_answer(self, consumer_id: str, sdp: str) -> None:
        """Apply a consumer's SDP answer."""

    def add_remote_candidate(
        self, consumer_id: str, candidate: str, mid: str | None
    ) -> None:
        """Apply a consumer's trickled ICE candidate."""

    def add_video_track(self, track_id: str) -> None:
        """Add a shared video source visible to all consumers."""

    def remove_video_track(self, track_id: str) -> None:
        """Remove a shared video source."""

    def submit_frame(self, track_id: str, frame: object) -> None:
        """Submit one frame for the shared encode."""

    def add_data_channel(self, label: str, kind: str) -> None:
        """Open a reliable data channel with ``label`` on every consumer."""

    def send_json(self, label: str, payload: str) -> None:
        """Send a JSON payload over ``label`` to every consumer that has it."""

    def drain_events(self) -> list[dict]:
        """Drain pending per-consumer events."""

    def close(self) -> None:
        """Tear down the broadcaster."""


# Env flag that turns on the producer's Chrome-only SDP munge (bare ``a=ssrc`` ->
# ``a=ssrc … cname``). Browser sessions need it; the byte-identical loopback
# tests deliberately leave it unset.
CHROME_SDP_ENV = "NCD_WEBRTC_CHROME_SDP"


@dataclass(frozen=True)
class OutboundSignal:
    """A signaling message the producer must deliver to one browser consumer."""

    consumer_id: str
    message_type: MessageType
    data: str


def outbound_signal(event: dict) -> OutboundSignal | None:
    """Map one drained broadcaster event to a web signaling message.

    The producer is the sole offerer, so it emits SDP **offers** and ICE
    candidates; the browser replies with an answer (handled inbound). Events
    without a ``consumer_id`` (a shared-encode error) or that are not signaling
    map to ``None``.

    Args:
        event: a single dict from ``Broadcaster.drain_events()``.

    Returns:
        The message to send to that consumer, or ``None`` if not deliverable.
    """
    consumer_id = event.get("consumer_id")
    if consumer_id is None:
        return None
    kind = event.get("kind")
    if kind == "on_local_description" and event.get("sdp_type") == "offer":
        return OutboundSignal(consumer_id, MessageType.SDP_OFFER, event["sdp"])
    if kind == "on_local_candidate":
        payload = json.dumps({
            "candidate": event["candidate"],
            "sdpMid": event.get("mid"),
            "sdpMLineIndex": event.get("mid"),
        })
        return OutboundSignal(consumer_id, MessageType.ICE_CANDIDATE, payload)
    return None


def inbound_candidate(data: str) -> tuple[str, str | None]:
    """Parse a browser ICE candidate (``RTCIceCandidate.toJSON()``) for intake.

    Args:
        data: JSON string the browser sent as the handshake payload.

    Returns:
        ``(candidate, mid)`` for ``add_remote_candidate(consumer_id, …)``.
    """
    content = json.loads(data)
    return content["candidate"], content.get("sdpMid")


def needs_reconnect(event: dict) -> str | None:
    """Return the consumer id that needs a reconnect, or ``None``.

    PR7 surfaces a dropped connection as ``on_error{where:"connection"}`` with
    the originating ``consumer_id``; the producer recovers it by remove + re-add.
    """
    if event.get("kind") == "on_error" and event.get("where") == "connection":
        return event.get("consumer_id")
    return None


@dataclass
class _Consumer:
    """Bookkeeping for one browser consumer of the broadcast."""

    connection_id: str
    remote_stream_id: str


# (connection_id, remote_stream_id, message_type, data) -> None
SendMessage = Callable[[str, str, MessageType, str], None]


class NativeBroadcastProvider:
    """Owns the native ``Broadcaster`` and bridges it to the web transport."""

    def __init__(
        self,
        producer: BroadcasterProducer,
        send_message: SendMessage,
        *,
        browser_facing: bool = True,
    ) -> None:
        """Initialize the provider.

        Args:
            producer: a native ``Broadcaster`` instance (injected so tests can
                pass a fake, peer-free producer).
            send_message: delivers an outbound signaling message to a browser.
            browser_facing: when True (the web path) the Chrome ``a=ssrc cname``
                munge is enabled for the process.
        """
        self.producer = producer
        self.send_message = send_message
        self._consumers: dict[str, _Consumer] = {}
        self._video_tracks: set[str] = set()
        self._data_channels: set[str] = set()
        if browser_facing:
            self._enable_chrome_sdp()

    @staticmethod
    def _enable_chrome_sdp() -> None:
        """Turn the Chrome SDP munge on for browser sessions (idempotent)."""
        os.environ.setdefault(CHROME_SDP_ENV, "1")

    # --- consumer lifecycle --------------------------------------------------

    def add_consumer(self, connection_id: str, remote_stream_id: str) -> None:
        """A browser connected: stand up an answer-only consumer for it."""
        if connection_id in self._consumers:
            return
        self._consumers[connection_id] = _Consumer(connection_id, remote_stream_id)
        self.producer.add_consumer(connection_id)

    def remove_consumer(self, connection_id: str) -> None:
        """A browser disconnected: tear down only its consumer."""
        if self._consumers.pop(connection_id, None) is None:
            return
        self.producer.remove_consumer(connection_id)

    def reconnect_consumer(self, connection_id: str) -> None:
        """PR7 recovery: remove + re-add one consumer (no ICE restart)."""
        consumer = self._consumers.get(connection_id)
        if consumer is None:
            return
        self.producer.remove_consumer(connection_id)
        self.producer.add_consumer(connection_id)

    # --- media / data sources ------------------------------------------------

    def add_video_track(self, track_id: str) -> None:
        """Register a video source visible to every (current + future) browser."""
        if track_id in self._video_tracks:
            return
        self._video_tracks.add(track_id)
        self.producer.add_video_track(track_id)

    def remove_video_track(self, track_id: str) -> None:
        """Drop a video source from the broadcast."""
        if track_id not in self._video_tracks:
            return
        self._video_tracks.discard(track_id)
        self.producer.remove_video_track(track_id)

    def submit_frame(self, track_id: str, frame: object) -> None:
        """Submit one frame for the shared encode (fanned out to all browsers)."""
        self.producer.submit_frame(track_id, frame)

    def add_data_channel(self, label: str, kind: str = "reliable") -> None:
        """Open a reliable data channel ``label`` on every (current + future) browser.

        Mirrors :meth:`add_video_track`: the channel is opened on each consumer by
        the broadcaster (over the existing SCTP association for a live consumer, at
        bootstrap for a future one), so json/joints reach every browser.
        """
        if label in self._data_channels:
            return
        self._data_channels.add(label)
        self.producer.add_data_channel(label, kind)

    def send_json(self, label: str, payload: str) -> None:
        """Fan a serialised JSON payload to every browser's ``label`` channel."""
        self.producer.send_json(label, payload)

    # --- inbound signaling ---------------------------------------------------

    def on_answer(self, connection_id: str, sdp: str) -> None:
        """Feed a browser's SDP answer back into its consumer."""
        if connection_id not in self._consumers:
            return
        self.producer.set_remote_answer(connection_id, sdp)

    def on_ice_candidate(self, connection_id: str, data: str) -> None:
        """Feed a browser's trickled ICE candidate back into its consumer."""
        if connection_id not in self._consumers:
            return
        candidate, mid = inbound_candidate(data)
        self.producer.add_remote_candidate(connection_id, candidate, mid)

    # --- outbound signaling pump ---------------------------------------------

    def pump_once(self) -> None:
        """Drain the broadcaster and dispatch every pending event.

        Offers/candidates go to the browser via the web transport; a
        reconnect-needed error triggers a remove + re-add for that consumer.
        """
        for event in self.producer.drain_events():
            reconnect_id = needs_reconnect(event)
            if reconnect_id is not None:
                logger.warning(
                    "webrtc consumer %s needs reconnect: %s",
                    reconnect_id,
                    event.get("detail"),
                )
                self.reconnect_consumer(reconnect_id)
                continue
            if event.get("kind") == "on_error":
                logger.warning(
                    "webrtc error where=%s consumer=%s detail=%s",
                    event.get("where"),
                    event.get("consumer_id"),
                    event.get("detail"),
                )
            signal = outbound_signal(event)
            if signal is None:
                continue
            consumer = self._consumers.get(signal.consumer_id)
            if consumer is None:
                continue
            self.send_message(
                consumer.connection_id,
                consumer.remote_stream_id,
                signal.message_type,
                signal.data,
            )

    def close(self) -> None:
        """Tear down every consumer and the shared encode."""
        self._consumers.clear()
        self._video_tracks.clear()
        self._data_channels.clear()
        self.producer.close()
