"""Client streaming manager for real-time robot data streaming.

This module provides WebRTC-based peer-to-peer streaming capabilities for robot
sensor data including video feeds and JSON event streams. It handles signaling,
connection management, and automatic reconnection with exponential backoff.
"""

import asyncio
from concurrent.futures import Future
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from aiohttp import ClientSession, ClientTimeout
from aiohttp_sse_client import client as sse_client

from neuracore.core.auth import Auth, get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.streaming.client_stream.json_source import JSONSource
from neuracore.core.streaming.client_stream.models import (
    HandshakeMessage,
    MessageType,
    RobotStreamTrack,
)
from neuracore.core.streaming.client_stream.stream_enabled import EnabledManager
from neuracore.core.streaming.event_loop_utils import get_running_loop

from ...const import API_URL, LIVE_DATA_ENABLED
from .connection import PierToPierConnection
from .video_source import DepthVideoSource, VideoSource

# must be less than zero -> a reconnection delay of more
# than one second is considered dead
# TODO: resubmit tracks if connection is re-established
# after more than one second
MINIMUM_BACKOFF_LEVEL = -2


class ClientStreamingManager:
    """Manages WebRTC streaming connections for robot sensor data.

    Handles peer-to-peer connections, signaling, video tracks, and JSON data streams
    with automatic reconnection and proper cleanup.
    """

    def __init__(
        self,
        robot_id: str,
        robot_instance: int,
        client_session: ClientSession,
        loop: asyncio.AbstractEventLoop,
        auth: Optional[Auth] = None,
    ):
        """Initialize the client streaming manager.

        Args:
            robot_id: Unique identifier for the robot
            robot_instance: Instance number of the robot
            client_session: HTTP client session for API requests
            loop: Event loop for async operations
            auth: Authentication object. If not provided, uses default auth
        """
        self.org_id = get_current_org()
        self.robot_id = robot_id
        self.robot_instance = robot_instance
        self.loop = loop
        self.client_session = client_session
        self.auth = auth or get_auth()
        self.streaming = EnabledManager(LIVE_DATA_ENABLED, loop=self.loop)
        self.streaming.add_listener(EnabledManager.DISABLED, self.__close)
        self.connections: Dict[str, PierToPierConnection] = {}
        self.video_tracks_cache: dict[str, VideoSource] = {}
        self.event_source_cache: Dict[str, JSONSource] = {}
        self.track_lock = asyncio.Lock()
        self.tracks: List[VideoSource] = []
        self.local_stream_id = uuid4().hex
        self.signalling_stream_future = asyncio.run_coroutine_threadsafe(
            self.connect_signalling_stream(), self.loop
        )

    def get_video_source(
        self, sensor_name: str, kind: str, sensor_key: str
    ) -> VideoSource:
        """Get or create a video source for streaming camera data.

        Args:
            sensor_name: Name of the sensor/camera
            kind: Type of video data ("rgb", "depth", etc.)

        Returns:
            VideoSource: Video source for streaming frames
        """
        if sensor_key in self.video_tracks_cache:
            return self.video_tracks_cache[sensor_key]

        mid = str(len(self.tracks))
        asyncio.run_coroutine_threadsafe(
            self.submit_track(mid, kind, sensor_name), self.loop
        )

        video_source = (
            DepthVideoSource(mid=mid, stream_enabled=self.streaming)
            if kind == "depth"
            else VideoSource(mid=mid, stream_enabled=self.streaming)
        )
        self.video_tracks_cache[sensor_key] = video_source
        self.tracks.append(video_source)

        return video_source

    def get_json_source(
        self, sensor_name: str, kind: str, sensor_key: str
    ) -> JSONSource:
        """Get or create a JSON source for streaming structured data.

        Args:
            sensor_name: Name of the sensor
            kind: Type of data being streamed
            sensor_key: Optional custom key for caching. Defaults to (sensor_name, kind)

        Returns:
            JSONSource: JSON source for streaming structured data
        """
        if sensor_key in self.event_source_cache:
            return self.event_source_cache[sensor_key]

        mid = uuid4().hex

        asyncio.run_coroutine_threadsafe(
            self.submit_track(mid, kind, sensor_name), self.loop
        )
        source = JSONSource(mid=mid, stream_enabled=self.streaming, loop=self.loop)

        self.event_source_cache[sensor_key] = source
        return source

    async def submit_track(self, mid: str, kind: str, label: str) -> None:
        """Submit a new track to the signaling server.

        Args:
            mid: Media ID for the track
            kind: Type of media (e.g., "video", "audio", "application")
            label: Human-readable label for the track

        Raises:
            ConfigError: If there is an error trying to get the current org
        """
        if not self.streaming.is_enabled():
            return

        await self.client_session.post(
            f"{API_URL}/org/{self.org_id}/signalling/track",
            headers=self.auth.get_headers(),
            json=RobotStreamTrack(
                robot_id=self.robot_id,
                robot_instance=self.robot_instance,
                stream_id=self.local_stream_id,
                mid=mid,
                kind=kind,
                label=label,
            ).model_dump(mode="json"),
        )

    async def heartbeat_response(self) -> None:
        """Send heartbeat response to keep the signaling connection alive."""
        if not self.streaming.is_enabled():
            return

        await self.client_session.post(
            f"{API_URL}/org/{self.org_id}/signalling/alive/{self.local_stream_id}",
            headers=self.auth.get_headers(),
            data="pong",
        )

    async def create_new_connection(
        self, remote_stream_id: str, connection_id: str, connection_token: str
    ) -> PierToPierConnection:
        """Create a new peer-to-peer connection to a remote stream.

        Args:
            remote_stream_id: ID of the remote stream to connect to
            connection_id: Unique identifier for this connection
            connection_token: Authentication token for the connection

        Returns:
            PierToPierConnection: The newly created P2P connection
        """

        def on_close() -> None:
            self.connections.pop(remote_stream_id, None)

        connection = PierToPierConnection(
            local_stream_id=self.local_stream_id,
            remote_stream_id=remote_stream_id,
            id=connection_id,
            connection_token=connection_token,
            org_id=self.org_id,
            on_close=on_close,
            client_session=self.client_session,
            auth=self.auth,
            loop=self.loop,
        )

        connection.setup_connection()

        for video_track in self.tracks:
            connection.add_video_source(video_track)

        for data_channel in self.event_source_cache.values():
            connection.add_event_source(data_channel)

        self.connections[remote_stream_id] = connection
        await connection.send_offer()
        return connection

    async def connect_signalling_stream(self) -> None:
        """Connect to the signaling server and process incoming messages.

        Maintains a persistent SSE connection with exponential backoff retry logic.
        Handles heartbeats, connection tokens, SDP offers/answers, and ICE candidates.
        """
        backoff = MINIMUM_BACKOFF_LEVEL
        while self.streaming.is_enabled():
            try:
                async with sse_client.EventSource(
                    f"{API_URL}/org/{self.org_id}/signalling/notifications/{self.local_stream_id}",
                    session=self.client_session,
                    headers=self.auth.get_headers(),
                    reconnection_time=timedelta(seconds=0.1),
                ) as event_source:
                    async for event in event_source:
                        try:
                            backoff = max(MINIMUM_BACKOFF_LEVEL, backoff - 1)
                            if not self.streaming.is_enabled():
                                return
                            if event.type == "heartbeat":
                                await self.heartbeat_response()
                                continue

                            message = HandshakeMessage.model_validate_json(event.data)
                            if message.from_id == "system":
                                continue

                            connection = self.connections.get(message.from_id)

                            if message.type == MessageType.CONNECTION_TOKEN:
                                await self.create_new_connection(
                                    remote_stream_id=message.from_id,
                                    connection_id=message.connection_id,
                                    connection_token=message.data,
                                )
                                continue

                            if (
                                connection is None
                                or connection.id != message.connection_id
                            ):
                                continue

                            match message.type:
                                case MessageType.SDP_OFFER:
                                    await connection.on_offer(message.data)
                                case MessageType.ICE_CANDIDATE:
                                    await connection.on_ice(message.data)
                                case MessageType.SDP_ANSWER:
                                    await connection.on_answer(message.data)
                                case _:
                                    pass
                        except asyncio.TimeoutError:
                            await asyncio.sleep(2 ^ backoff)
                            backoff += 1
                            continue
                        except Exception as e:
                            print(f"Signaling message error: {e}")
                            await asyncio.sleep(2**backoff)
                            backoff += 1
            except Exception as e:
                print(f"Signaling connection error: {e}")
                await asyncio.sleep(2**backoff)
                backoff += 1

    async def close_connections(self) -> None:
        """Close all active peer-to-peer connections."""
        await asyncio.gather(
            *(connection.close() for connection in self.connections.values())
        )
        self.connections.clear()

    def __close(self) -> None:
        """Internal cleanup method called when streaming is disabled."""
        if self.signalling_stream_future.running():
            self.signalling_stream_future.cancel()

        asyncio.run_coroutine_threadsafe(self.close_connections(), self.loop)
        asyncio.run_coroutine_threadsafe(self.client_session.close(), self.loop)

    def close(self) -> None:
        """Close all connections and streams gracefully.

        Disables streaming, closes all P2P connections, stops video tracks,
        and cleans up resources.
        """
        self.streaming.disable()
        self.available_for_connections = False
        asyncio.run_coroutine_threadsafe(self.close_connections(), self.loop)

        for track in self.video_tracks_cache.values():
            if track is not None and hasattr(track, "stop"):
                assert hasattr(track, "stop")
                track.stop()  # type: ignore

        self.connections.clear()
        self.video_tracks_cache.clear()
        asyncio.run_coroutine_threadsafe(self.client_session.close(), self.loop)


_streaming_managers: Dict[Tuple[str, int], Future[ClientStreamingManager]] = {}


async def _create_client_streaming_manager(
    robot_id: str, instance: int
) -> ClientStreamingManager:
    """Create a new client streaming manager instance.

    Args:
        robot_id: Unique identifier for the robot
        instance: Instance number of the robot

    Returns:
        ClientStreamingManager: Configured streaming manager instance
    """
    # We want to keep the signalling connection alive for as long as possible
    timeout = ClientTimeout(sock_read=None, total=None)
    return ClientStreamingManager(
        robot_id=robot_id,
        robot_instance=instance,
        loop=asyncio.get_event_loop(),
        client_session=ClientSession(timeout=timeout),
    )


def get_robot_streaming_manager(robot_id: str, instance: int) -> ClientStreamingManager:
    """Get or create a streaming manager for a specific robot instance.

    Uses a singleton pattern to ensure only one streaming manager exists per
    robot instance. Thread-safe and handles event loop coordination.

    Args:
        robot_id: Unique identifier for the robot
        instance: Instance number of the robot

    Returns:
        ClientStreamingManager: Streaming manager for the specified robot instance
    """
    key = (robot_id, instance)
    if key not in _streaming_managers:
        # This needs to be run in the event loop thread
        # otherwise we will get a 'RuntimeError: no running event loop'
        manager = asyncio.run_coroutine_threadsafe(
            _create_client_streaming_manager(robot_id, instance), get_running_loop()
        )
        _streaming_managers[key] = manager
    return _streaming_managers[key].result()
