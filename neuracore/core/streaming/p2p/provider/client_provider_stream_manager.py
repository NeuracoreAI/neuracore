"""Client streaming manager for real-time robot data streaming.

This module provides WebRTC-based peer-to-peer streaming capabilities for robot
sensor data including video feeds and JSON event streams. It handles signaling,
connection management, and automatic reconnection with exponential backoff.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from uuid import uuid4

from aiohttp import ClientSession
from neuracore_types import (
    HandshakeMessage,
    MessageType,
    OpenConnectionDetails,
    RobotStreamTrack,
    TrackKind,
    VideoFormat,
)

from neuracore.core.auth import Auth, get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.streaming.p2p.base_p2p_connection_manager import (
    BaseP2PStreamManager,
)
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.streaming.p2p.provider.json_source import JSONSource
from neuracore.core.utils.background_coroutine_tracker import BackgroundCoroutineTracker

from .global_live_data_enabled import global_provide_live_data_manager
from .provider_connection import PierToPierProviderConnection
from .video_source import DepthVideoSource, VideoSource

logger = logging.getLogger(__name__)


class ClientProviderStreamManager(BaseP2PStreamManager):
    """Manages WebRTC streaming connections for robot sensor data.

    Handles peer-to-peer connections, signaling, video tracks, and JSON data streams
    with automatic reconnection and proper cleanup.
    """

    def __init__(
        self,
        robot_id: str,
        robot_instance: int,
        local_stream_id: str,
        client_session: ClientSession,
        loop: asyncio.AbstractEventLoop,
        org_id: Optional[str] = None,
        auth: Optional[Auth] = None,
    ):
        """Initialize the client streaming manager.

        Args:
            robot_id: Unique identifier for the robot
            robot_instance: Instance number of the robot
            local_stream_id: Unique identifier of this recipient.
            client_session: HTTP client session for API requests
            loop: Event loop for async operations
            org_id: The unique identifier for the organization. If not provided,
                defaults to the current org.
            auth: Authentication object. If not provided, uses default auth
        """
        self.robot_id = robot_id
        self.robot_instance = robot_instance
        self.local_stream_id = local_stream_id
        self.loop = loop
        self.client_session = client_session
        self.org_id = org_id or get_current_org()
        self.auth = auth or get_auth()
        self.background_tracker = BackgroundCoroutineTracker(loop=self.loop)

        self.streaming = EnabledManager.derived_manger(
            global_provide_live_data_manager, loop=loop
        )
        global_provide_live_data_manager.add_listener(
            EnabledManager.DISABLED, self._on_close
        )
        self.streaming.add_listener(EnabledManager.DISABLED, self._on_close)
        self.connections: Dict[str, PierToPierProviderConnection] = {}

        self.video_tracks_cache: Dict[str, VideoSource] = {}
        self.event_source_cache: Dict[str, JSONSource] = {}
        self.tracks: List[VideoSource] = []
        self.track_metadata: dict[str, RobotStreamTrack] = {}

    @property
    def enabled_manager(self) -> EnabledManager:
        """Get the enabled manager for this streaming manager.

        Returns:
            EnabledManager: determines wether this streaming manager is enabled
        """
        return self.streaming

    def get_video_source(
        self, sensor_name: str, kind: TrackKind, sensor_key: str
    ) -> VideoSource:
        """Get or create a video source for streaming camera data.

        Args:
            sensor_name: Name of the sensor/camera
            kind: Type of video data (TrackKind.RGB or TrackKind.DEPTH are supported)
            sensor_key: custom key for caching.

        Returns:
            VideoSource: Video source for streaming frames
        """
        if sensor_key in self.video_tracks_cache:
            return self.video_tracks_cache[sensor_key]

        mid = str(len(self.tracks))
        self.background_tracker.submit_background_coroutine(
            self.submit_track(mid, kind, sensor_name)
        )
        if kind == TrackKind.RGB:
            video_source = VideoSource(mid=mid, stream_enabled=self.streaming)
        elif kind == TrackKind.DEPTH:
            video_source = DepthVideoSource(mid=mid, stream_enabled=self.streaming)
        else:
            raise ValueError(f"Unsupported video kind {kind}")

        self.video_tracks_cache[sensor_key] = video_source
        self.tracks.append(video_source)

        for connection in self.connections.values():
            if (
                connection.connection_details.video_format
                == VideoFormat.WEB_RTC_NEGOTIATED
            ):
                connection.add_video_source(video_source)
            else:
                connection.add_event_source(video_source.get_neuracore_custom_track())

        return video_source

    def get_json_source(
        self, sensor_name: str, kind: TrackKind, sensor_key: str
    ) -> JSONSource:
        """Get or create a JSON source for streaming structured data.

        Args:
            sensor_name: Name of the sensor
            kind: Type of data being streamed
            sensor_key: custom key for caching.

        Returns:
            JSONSource: JSON source for streaming structured data
        """
        if sensor_key in self.event_source_cache:
            return self.event_source_cache[sensor_key]

        mid = uuid4().hex

        self.background_tracker.submit_background_coroutine(
            self.submit_track(mid, kind, sensor_name)
        )
        source = JSONSource(mid=mid, stream_enabled=self.streaming, loop=self.loop)

        self.event_source_cache[sensor_key] = source

        for connection in self.connections.values():
            connection.add_event_source(source)

        return source

    async def submit_track(self, mid: str, kind: TrackKind, label: str) -> None:
        """Submit a new track to the signaling server.

        Args:
            mid: Media ID for the track
            kind: Type of media (e.g., "video", "audio", "application")
            label: Human-readable label for the track

        Raises:
            ConfigError: If there is an error trying to get the current org
        """
        if self.streaming.is_disabled():
            return
        track = RobotStreamTrack(
            robot_id=self.robot_id,
            robot_instance=self.robot_instance,
            stream_id=self.local_stream_id,
            mid=mid,
            kind=kind,
            label=label,
        )
        self.track_metadata[track.id] = track

        await self.client_session.post(
            f"{API_URL}/org/{self.org_id}/signalling/track",
            headers=self.auth.get_headers(),
            json=track.model_dump(mode="json"),
        )

    async def on_stream_resurrected(self) -> None:
        """Resubmit tracks to the signaling server."""
        await asyncio.gather(*(
            self.client_session.post(
                f"{API_URL}/org/{self.org_id}/signalling/track",
                headers=self.auth.get_headers(),
                json=track.model_dump(mode="json"),
            )
            for track in self.track_metadata.values()
        ))

    async def create_new_connection(
        self,
        remote_stream_id: str,
        connection_id: str,
        connection_details: OpenConnectionDetails,
    ) -> EnabledManager:
        """Create a new peer-to-peer connection to a remote stream.

        Args:
            remote_stream_id: ID of the remote stream to connect to
            connection_id: Unique identifier for this connection
            connection_details: The describes the type of connection to establish.
        """
        connection = PierToPierProviderConnection(
            connection_id=connection_id,
            local_stream_id=self.local_stream_id,
            remote_stream_id=remote_stream_id,
            connection_details=connection_details,
            org_id=self.org_id,
            client_session=self.client_session,
            auth=self.auth,
            loop=self.loop,
        )

        @connection.enabled_manager.on(EnabledManager.DISABLED)
        def on_close() -> None:
            self.connections.pop(connection_id, None)

        for video_track in self.tracks:
            if connection_details.video_format == VideoFormat.WEB_RTC_NEGOTIATED:
                connection.add_video_source(video_track)
            else:
                connection.add_event_source(
                    video_track.get_neuracore_custom_track(loop=self.loop)
                )

        for data_channel in self.event_source_cache.values():
            connection.add_event_source(data_channel)

        self.connections[connection_id] = connection
        self.background_tracker.submit_background_coroutine(connection.send_offer())

        return connection.enabled_manager

    async def remove_connection(self, connection_id: str) -> None:
        """Remove a peer-to-peer connection.

        Args:
            connection_id: ID of the connection to end.
        """
        connection = self.connections.pop(connection_id, None)
        if connection is None:
            return

        connection.close()

    async def on_message(self, message: HandshakeMessage) -> None:
        """Handle a signalling message for one of the manager's connections.

        Args:
            message: The message to handle.
        """
        connection = self.connections.get(message.connection_id, None)
        if not connection:
            raise ValueError(f"Connection not found for id: {message.connection_id}")

        if message.type == MessageType.ICE_CANDIDATE:
            await connection.on_ice(message.data)
        elif message.type == MessageType.SDP_ANSWER:
            await connection.on_answer(message.data)
        else:
            logger.warning(f"Unsupported message type: {message.type}")

    def _on_close(self) -> None:
        """Internal cleanup method called when streaming is disabled."""
        for connection in self.connections.values():
            connection.close()

        self.connections.clear()
        self.tracks.clear()
        self.video_tracks_cache.clear()
        self.event_source_cache.clear()
        self.track_metadata.clear()
