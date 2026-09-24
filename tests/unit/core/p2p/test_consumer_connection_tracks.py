"""Tests for pairing data channels with signalling tracks that arrive later."""

import time
from unittest.mock import MagicMock, patch

from neuracore_types import (
    DataType,
    JointData,
    OpenConnectionDetails,
    RobotStreamTrack,
    VideoFormat,
)

from neuracore.core.streaming.p2p.consumer.consumer_connection import (
    PeerToPeerConsumerConnection,
)
from neuracore.core.streaming.p2p.consumer.ice_models import IceConfig, IceServer


class _FakePeerConnection:
    """Records event handlers registered during connection setup."""

    def __init__(self, configuration: object | None = None) -> None:
        self.handlers: dict[str, object] = {}
        self.connectionState = "new"

    def on(self, event: str):
        def decorator(fn):
            self.handlers[event] = fn
            return fn

        return decorator


class _FakeDataChannel:
    """Data channel that stores message listeners and can emit them."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._message_handlers: list = []

    def on(self, event: str):
        def decorator(fn):
            if event == "message":
                self._message_handlers.append(fn)
            return fn

        return decorator

    def emit(self, message: str) -> None:
        for handler in self._message_handlers:
            handler(message)


def _track(mid: str, label: str) -> RobotStreamTrack:
    return RobotStreamTrack(
        robot_id="robot-1",
        robot_instance=0,
        stream_id="remote-1",
        data_type=DataType.JOINT_POSITIONS,
        label=label,
        mid=mid,
    )


def _joint_message(timestamp: float, value: float) -> str:
    return JointData(timestamp=timestamp, value=value).model_dump_json()


def _connection(
    expected_tracks: list[RobotStreamTrack] | None = None,
) -> tuple[PeerToPeerConsumerConnection, _FakePeerConnection]:
    peer = _FakePeerConnection()
    with (
        patch(
            "neuracore.core.streaming.p2p.consumer.consumer_connection.RTCPeerConnection",
            return_value=peer,
        ),
        patch(
            "neuracore.core.streaming.p2p.consumer.consumer_connection.get_current_org",
            return_value="org-1",
        ),
        patch(
            "neuracore.core.streaming.p2p.consumer.consumer_connection.get_auth",
            return_value=MagicMock(),
        ),
    ):
        connection = PeerToPeerConsumerConnection(
            connection_id="conn-1",
            local_stream_id="local-1",
            remote_stream_id="remote-1",
            ice_config=IceConfig(
                iceServers=[IceServer(urls="stun:stun.example.com:19302")]
            ),
            connection_details=OpenConnectionDetails(
                connection_token="token",
                robot_id="robot-1",
                robot_instance=0,
                video_format=VideoFormat.NEURACORE_CUSTOM,
            ),
            expected_tracks=expected_tracks or [],
            org_id="org-1",
            loop=MagicMock(),
            enabled_manager=MagicMock(),
        )
    return connection, peer


def _open_channel(peer: _FakePeerConnection, mid: str) -> _FakeDataChannel:
    channel = _FakeDataChannel(mid)
    peer.handlers["datachannel"](channel)
    return channel


def test_samples_arriving_before_the_track_are_kept() -> None:
    """A channel that opens before its description still contributes its samples."""
    connection, peer = _connection()
    channel = _open_channel(peer, "shoulder-mid")

    # Later than the connection's initial sync-point timestamp, so each sample is
    # newer than what is already stored and the second one is the one kept.
    logged_at = time.time()
    channel.emit(_joint_message(timestamp=logged_at, value=0.1))
    channel.emit(_joint_message(timestamp=logged_at + 1, value=0.4))

    assert DataType.JOINT_POSITIONS not in connection.get_latest_data().data

    connection.expected_tracks = [_track("shoulder-mid", "JOINT_POSITIONS:shoulder")]

    latest = connection.get_latest_data()
    shoulder = latest.data[DataType.JOINT_POSITIONS]["shoulder"]
    assert shoulder.value == 0.4
    assert shoulder.timestamp == logged_at + 1
    assert connection.fully_connected()


def test_samples_arriving_after_the_track_are_ingested_immediately() -> None:
    """A channel that opens once its description is known is parsed live."""
    connection, peer = _connection([_track("shoulder-mid", "JOINT_POSITIONS:shoulder")])
    channel = _open_channel(peer, "shoulder-mid")

    channel.emit(_joint_message(timestamp=10.0, value=0.25))

    shoulder = connection.get_latest_data().data[DataType.JOINT_POSITIONS]["shoulder"]
    assert shoulder.value == 0.25
    assert connection._pending_messages["shoulder-mid"] == []


def test_unmatched_channel_stays_buffered_when_other_tracks_arrive() -> None:
    """A description for a different channel does not drop held samples."""
    connection, peer = _connection()
    channel = _open_channel(peer, "camera-mid")
    channel.emit(_joint_message(timestamp=9.9, value=1.0))

    connection.expected_tracks = [_track("shoulder-mid", "JOINT_POSITIONS:shoulder")]

    assert connection.get_latest_data().data == {}
    assert len(connection._pending_messages["camera-mid"]) == 1
