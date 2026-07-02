"""Bridge the data daemon's recording entry points onto the WebRTC data plane.

Both native send peers expose the same single channel send path,
``add_data_channel(label, kind)`` + ``send_json(label, payload)``: the 1:1
``Producer`` (one consumer) and the ``Broadcaster`` (one shared producer fanned
out to N answer-only browsers). PR1's integration suite drives that path
directly; the live recording pipeline reaches it through the same ``log_json`` /
``log_joints`` entry points it already uses for disk recording (see
``neuracore/data_daemon/.../recording_context.py``). This adapter mirrors those
two signatures and forwards each call to ``send_json`` over a reliable-ordered
data channel, so the disk path and the streaming path converge on one send path.

It is deliberately duck-typed against that send path (anything exposing
``add_data_channel(label, kind)`` and ``send_json(label, payload)``), so it is
**parameterised over the ``Producer`` or the ``Broadcaster`` rather than
duplicated**: over a ``Broadcaster`` each ``send_json`` fans to every browser's
channel for that label. It pulls in no heavy dependencies, so it can be wired in
at the cutover without disturbing the existing aiortc provider. One reliable
channel is opened lazily per stream; the reserved ``"control"`` label is never
used for application data.
"""

from __future__ import annotations

import json
from typing import Protocol


class _SendPath(Protocol):
    """The slice of the native ``Producer`` / ``Broadcaster`` this bridge needs."""

    def add_data_channel(self, label: str, kind: str) -> None: ...

    def send_json(self, label: str, payload: str) -> None: ...


class WebrtcRecordingBridge:
    """Forward ``log_json`` / ``log_joints`` recording calls to ``send_json``.

    Each distinct stream maps to one reliable-ordered data channel, opened on
    first use (which drives the send peer's negotiation just like any other
    ``add_data_channel``). The wire payload is a small JSON envelope carrying the
    capture timestamp and the sample, so a consumer can route by data type. The
    same bridge serves the 1:1 ``Producer`` and the fan-out ``Broadcaster``
    interchangeably (see the module docstring).
    """

    #: Reserved by the transport for the manifest; never an application stream.
    CONTROL_LABEL = "control"

    def __init__(self, producer: _SendPath) -> None:
        """Bridge recording calls onto ``producer``'s ``send_json`` path.

        Args:
            producer: any send peer exposing ``add_data_channel`` /
                ``send_json`` — a native ``Producer`` or a ``Broadcaster``.
        """
        self._producer = producer
        self._open: set[str] = set()

    def _channel(self, label: str) -> str:
        """Open ``label`` as a reliable channel on first use; return it."""
        if label == self.CONTROL_LABEL:
            raise ValueError("'control' is reserved for the manifest transport")
        if label not in self._open:
            self._producer.add_data_channel(label, "reliable")
            self._open.add(label)
        return label

    def log_joints(
        self,
        data_type: str,
        timestamp: float,
        items: list[tuple[str, float]],
    ) -> None:
        """Stream a batch of ``(joint_name, value)`` samples for ``data_type``.

        Mirrors ``RecordingContext.log_joints``; routes to ``send_json`` over the
        ``data_type`` channel.
        """
        if not items:
            return
        label = self._channel(data_type)
        envelope = {
            "type": "joints",
            "data_type": data_type,
            "timestamp": timestamp,
            "values": {name: value for name, value in items},
        }
        self._producer.send_json(label, json.dumps(envelope))

    def log_json(
        self,
        data_type: str,
        name: str,
        payload: bytes,
        timestamp: float,
    ) -> None:
        """Stream one already-serialised JSON sample for the ``name`` stream.

        Mirrors ``RecordingContext.log_json``; routes to ``send_json`` over the
        ``data_type/name`` channel. ``payload`` is the serialised sample as
        produced by the recording path and is forwarded verbatim as text.
        """
        label = self._channel(f"{data_type}/{name}")
        envelope = {
            "type": "json",
            "data_type": data_type,
            "name": name,
            "timestamp": timestamp,
            "payload": payload.decode("utf-8"),
        }
        self._producer.send_json(label, json.dumps(envelope))
