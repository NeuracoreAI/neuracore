"""Client streaming manager for real-time robot data streaming.

This module provides WebRTC-based peer-to-peer streaming capabilities for robot
sensor data including video feeds and JSON event streams. It handles signaling,
connection management, and automatic reconnection with exponential backoff.
"""

import asyncio
import logging
from uuid import uuid4

import numpy as np
from aiohttp import ClientSession
from neuracore_types import (
    DataType,
    HandshakeMessage,
    MessageType,
    OpenConnectionDetails,
    RobotStreamTrack,
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
from neuracore.core.streaming.p2p.provider.native_broadcast_provider import (
    NativeBroadcastProvider,
)
from neuracore.core.streaming.p2p.webrtc_selection import (
    load_native,
    rust_webrtc_enabled,
)
from neuracore.core.utils.background_coroutine_tracker import BackgroundCoroutineTracker

from .global_live_data_enabled import get_provide_live_data_enabled_manager
from .provider_connection import PeerToPeerProviderConnection
from .video_source import STREAMING_FPS, DepthVideoSource, VideoSource

logger = logging.getLogger(__name__)

# How often the native pump drains signaling events and re-submits frames for
# the shared encode (matches the streaming frame rate).
NATIVE_PUMP_INTERVAL_S = 1 / STREAMING_FPS


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
        org_id: str | None = None,
        auth: Auth | None = None,
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
            get_provide_live_data_enabled_manager(), loop=loop
        )
        get_provide_live_data_enabled_manager().add_listener(
            EnabledManager.DISABLED, self._on_close
        )
        self.streaming.add_listener(EnabledManager.DISABLED, self._on_close)
        self.connections: dict[str, PeerToPeerProviderConnection] = {}

        self.video_tracks_cache: dict[str, VideoSource] = {}
        self.event_source_cache: dict[str, JSONSource] = {}
        self.tracks: list[VideoSource] = []
        self.track_metadata: dict[str, RobotStreamTrack] = {}

        # The new Rust stack: one Broadcaster fanned out to every browser, the
        # legacy aiortc per-connection path stays untouched when the flag is off.
        self._native: NativeBroadcastProvider | None = None
        self._native_enabled: dict[str, EnabledManager] = {}
        self._native_sources: dict[str, VideoSource] = {}
        if rust_webrtc_enabled():
            self._setup_native()

    def _setup_native(self) -> None:
        """Stand up the native Broadcaster and start its drain/feed pump."""
        broadcaster = load_native().Broadcaster()
        self._native = NativeBroadcastProvider(
            broadcaster, self._send_native_signal, browser_facing=True
        )
        self.background_tracker.submit_background_coroutine(self._native_pump_loop())

    def _send_native_signal(
        self,
        connection_id: str,
        remote_stream_id: str,
        message_type: MessageType,
        data: str,
    ) -> None:
        """Deliver a broadcaster signaling event over the web transport."""
        self.background_tracker.submit_background_coroutine(
            self.client_session.post(
                f"{API_URL}/org/{self.org_id}/signalling/message/submit",
                headers=self.auth.get_headers(),
                json=HandshakeMessage(
                    connection_id=connection_id,
                    from_id=self.local_stream_id,
                    to_id=remote_stream_id,
                    type=message_type,
                    data=data,
                ).model_dump(mode="json"),
            )
        )

    @staticmethod
    def _native_frame(source: VideoSource) -> np.ndarray:
        """Latest frame as a C-contiguous (H, W, 3) uint8 array for submit_frame."""
        return np.ascontiguousarray(source.get_last_frame().to_ndarray(format="rgb24"))

    def _native_send_json(self, label: str, message: str) -> None:
        """Fan one JSON state update to every browser, guarded against teardown."""
        if self._native is None:
            return
        self._native.send_json(label, message)

    async def _native_pump_loop(self) -> None:
        """Drain broadcaster events and re-feed the shared encode at the FPS."""
        assert self._native is not None
        while self.streaming.is_enabled():
            try:
                self._native.pump_once()
                for mid, source in list(self._native_sources.items()):
                    self._native.submit_frame(mid, self._native_frame(source))
            except Exception:
                logger.exception("native webrtc pump error")
            await asyncio.sleep(NATIVE_PUMP_INTERVAL_S)

    @property
    def enabled_manager(self) -> EnabledManager:
        """Get the enabled manager for this streaming manager.

        Returns:
            EnabledManager: determines wether this streaming manager is enabled
        """
        return self.streaming

    def get_video_source(
        self, sensor_name: str, data_type: DataType, sensor_key: str
    ) -> VideoSource:
        """Get or create a video source for streaming camera data.

        Args:
            sensor_name: Name of the sensor/camera
            data_type: Type of video data (DataType.RGB or DataType.DEPTH are supported)
            sensor_key: custom key for caching.

        Returns:
            VideoSource: Video source for streaming frames
        """
        if sensor_key in self.video_tracks_cache:
            return self.video_tracks_cache[sensor_key]

        # The Producer owns the mid: this one value is registered in available_robots
        # (submit_track) and handed to the native Broadcaster, which uses it verbatim
        # as the SDP m-line mid. So the SSE manifest the browser keys on and the
        # offer's a=mid always agree, and the browser's identityForMid succeeds for
        # both the initial connection and a mid-session add. The "v" prefix keeps the
        # mid out of the data channel's m-line namespace (the control/json channels
        # take a=mid:0); a bare index would collide with that, and libdatachannel
        # would silently drop the colliding video track. self.tracks only grows, so
        # the index is stable and never reused per source.
        mid = f"v{len(self.tracks)}"
        self.background_tracker.submit_background_coroutine(
            self.submit_track(mid, data_type, sensor_name)
        )
        if data_type == DataType.RGB_IMAGES:
            video_source = VideoSource(mid=mid, stream_enabled=self.streaming)
        elif data_type == DataType.DEPTH_IMAGES:
            video_source = DepthVideoSource(mid=mid, stream_enabled=self.streaming)
        else:
            raise ValueError(f"Unsupported video data type {data_type}")

        self.video_tracks_cache[sensor_key] = video_source
        self.tracks.append(video_source)

        if self._native is not None:
            # One shared encode per source, fanned out to every browser; the
            # pump loop feeds frames. No per-connection aiortc track.
            self._native.add_video_track(mid)
            self._native_sources[mid] = video_source
            return video_source

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
        self, sensor_name: str, data_type: DataType, sensor_key: str
    ) -> JSONSource:
        """Get or create a JSON source for streaming structured data.

        Args:
            sensor_name: Name of the sensor
            data_type: Type of data being streamed
            sensor_key: custom key for caching.

        Returns:
            JSONSource: JSON source for streaming structured data
        """
        if sensor_key in self.event_source_cache:
            return self.event_source_cache[sensor_key]

        mid = uuid4().hex

        self.background_tracker.submit_background_coroutine(
            self.submit_track(mid, data_type, sensor_name)
        )
        source = JSONSource(mid=mid, stream_enabled=self.streaming, loop=self.loop)

        self.event_source_cache[sensor_key] = source

        if self._native is not None:
            # One reliable data channel per source, fanned to every browser; each
            # state update is sent to all of them. The channel label is the source
            # mid, exactly as the aiortc path names it, so the SSE manifest the
            # browser keys by lines up. Future browsers get the channel at bootstrap.
            self._native.add_data_channel(mid)
            source.add_listener(
                source.STATE_UPDATED_EVENT,
                lambda message, label=mid: self._native_send_json(label, message),
            )
            return source

        for connection in self.connections.values():
            connection.add_event_source(source)

        return source

    async def submit_track(self, mid: str, data_type: DataType, label: str) -> None:
        """Submit a new track to the signaling server.

        Args:
            mid: Media ID for the track
            data_type: Type of media (e.g., "video", "audio", "application")
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
            data_type=data_type,
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
        if (
            self._native is not None
            and connection_details.video_format == VideoFormat.WEB_RTC_NEGOTIATED
        ):
            return self._create_native_connection(remote_stream_id, connection_id)

        connection = PeerToPeerProviderConnection(
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

    def _create_native_connection(
        self, remote_stream_id: str, connection_id: str
    ) -> EnabledManager:
        """Add one browser as an answer-only consumer of the shared broadcast.

        The producer is the sole offerer; the offer rides the native pump.
        """
        assert self._native is not None
        enabled = EnabledManager.derived_manger(self.streaming, loop=self.loop)
        self._native_enabled[connection_id] = enabled
        self._native.add_consumer(connection_id, remote_stream_id)

        @enabled.on(EnabledManager.DISABLED)
        def on_close() -> None:
            if self._native is not None:
                self._native.remove_consumer(connection_id)
            self._native_enabled.pop(connection_id, None)

        return enabled

    async def remove_connection(self, connection_id: str) -> None:
        """Remove a peer-to-peer connection.

        Args:
            connection_id: ID of the connection to end.
        """
        native_enabled = self._native_enabled.get(connection_id)
        if native_enabled is not None:
            native_enabled.disable()
            return

        connection = self.connections.pop(connection_id, None)
        if connection is None:
            return

        connection.close()

    async def on_message(self, message: HandshakeMessage) -> None:
        """Handle a signalling message for one of the manager's connections.

        Args:
            message: The message to handle.
        """
        if self._native is not None and message.connection_id in self._native_enabled:
            if message.type == MessageType.ICE_CANDIDATE:
                self._native.on_ice_candidate(message.connection_id, message.data)
            elif message.type == MessageType.SDP_ANSWER:
                self._native.on_answer(message.connection_id, message.data)
            else:
                logger.warning(f"Unsupported message type: {message.type}")
            return

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

        if self._native is not None:
            self._native.close()
            self._native_enabled.clear()
            self._native_sources.clear()

        self.connections.clear()
        self.tracks.clear()
        self.video_tracks_cache.clear()
        self.event_source_cache.clear()
        self.track_metadata.clear()
