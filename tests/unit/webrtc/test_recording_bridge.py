"""Unit tests for ``WebrtcRecordingBridge`` (PR2).

The bridge mirrors ``RecordingContext.log_json`` / ``log_joints`` and forwards
each call to ``Producer.send_json`` over a lazily-opened reliable data channel.
It is duck-typed against the native producer, so a tiny spy implementing
``add_data_channel`` / ``send_json`` stands in for it — no native module, no peer
connection. These tests pin the channel-label derivation, the lazy open, and the
JSON forwarded onto the one send path.
"""

from __future__ import annotations

import json

import pytest

from neuracore.core.streaming.p2p.recording_bridge import WebrtcRecordingBridge


class SpyProducer:
    """The slice of the native ``Producer`` the bridge touches, recorded."""

    def __init__(self) -> None:
        self.opened: list[tuple[str, str]] = []
        self.sent: list[tuple[str, str]] = []

    def add_data_channel(self, label: str, kind: str) -> None:
        self.opened.append((label, kind))

    def send_json(self, label: str, payload: str) -> None:
        self.sent.append((label, payload))


def test_log_json_derives_the_data_type_slash_name_channel() -> None:
    producer = SpyProducer()
    bridge = WebrtcRecordingBridge(producer)

    bridge.log_json("rgb", "wrist_cam", b'{"frame": 1}', timestamp=12.5)

    # Label is data_type/name; the channel is opened reliable on first use.
    assert producer.opened == [("rgb/wrist_cam", "reliable")]
    assert len(producer.sent) == 1
    label, payload = producer.sent[0]
    assert label == "rgb/wrist_cam"
    assert json.loads(payload) == {
        "type": "json",
        "data_type": "rgb",
        "name": "wrist_cam",
        "timestamp": 12.5,
        "payload": '{"frame": 1}',
    }


def test_log_joints_derives_the_data_type_channel() -> None:
    producer = SpyProducer()
    bridge = WebrtcRecordingBridge(producer)

    bridge.log_joints(
        "joint_positions", timestamp=3.0, items=[("j0", 1.0), ("j1", 2.5)]
    )

    assert producer.opened == [("joint_positions", "reliable")]
    label, payload = producer.sent[0]
    assert label == "joint_positions"
    assert json.loads(payload) == {
        "type": "joints",
        "data_type": "joint_positions",
        "timestamp": 3.0,
        "values": {"j0": 1.0, "j1": 2.5},
    }


def test_channels_open_lazily_and_exactly_once_per_label() -> None:
    producer = SpyProducer()
    bridge = WebrtcRecordingBridge(producer)

    bridge.log_json("rgb", "cam", b"{}", timestamp=0.0)
    bridge.log_json("rgb", "cam", b"{}", timestamp=1.0)
    bridge.log_joints("joints", timestamp=0.0, items=[("j0", 0.0)])
    bridge.log_joints("joints", timestamp=1.0, items=[("j0", 1.0)])

    # One open per distinct stream, despite four log calls.
    assert producer.opened == [("rgb/cam", "reliable"), ("joints", "reliable")]
    assert len(producer.sent) == 4


def test_empty_joint_batch_sends_nothing() -> None:
    producer = SpyProducer()
    bridge = WebrtcRecordingBridge(producer)

    bridge.log_joints("joints", timestamp=0.0, items=[])

    assert producer.opened == []
    assert producer.sent == []


def test_control_label_is_reserved_and_rejected() -> None:
    producer = SpyProducer()
    bridge = WebrtcRecordingBridge(producer)

    # "control" carries the manifest; the recording path must never use it as a
    # stream label. log_joints' label is the data_type, so this hits the guard.
    with pytest.raises(ValueError):
        bridge.log_joints("control", timestamp=0.0, items=[("j0", 0.0)])
    assert producer.opened == []
    assert producer.sent == []
