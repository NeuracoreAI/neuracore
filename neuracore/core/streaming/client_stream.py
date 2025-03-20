from dataclasses import Field, dataclass
from enum import Enum
import uuid
import asyncio
import logging
from typing import Optional, Dict, Any

from av import VideoFrame
import numpy as np
from pydantic import BaseModel

from neuracore.core.auth import Auth, get_auth
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
    RTCIceCandidate,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc.contrib.media import MediaStreamOutput, MediaBlackhole
from aiohttp_sse_client import client as sse_client
from aiortc.sdp import candidate_from_sdp
from aiohttp import ClientSession
from ..const import API_URL

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    SDP_OFFER = "offer"
    SDP_ANSWER = "answer"
    ICE_CANDIDATE = "ice"
    HEARTBEAT = "heartbeat"
    STREAM_INFO = "stream_info"


class HandshakeMessage(BaseModel):
    from_id: str
    to_id: str
    data: str
    type: MessageType
    id: str


@dataclass(slots=True)
class PierToPierConnection:
    local_stream_id: str
    remote_stream_id: str
    client_session: ClientSession = Field(
        default_factory=lambda: ClientSession(API_URL)
    )
    auth: Auth = Field(default_factory=get_auth)
    connection: RTCPeerConnection = Field(
        default_factory=lambda: RTCPeerConnection(
            configuration=RTCConfiguration(
                iceServers=[
                    RTCIceServer(urls="stun:stun.l.google.com:19302"),
                    RTCIceServer(urls="stun:stun1.l.google.com:19302"),
                ]
            )
        )
    )
    tracks: Dict[str, MediaStreamTrack] = Field(default_factory=dict)
    _closed: bool = False

    async def setup_connection(self):
        """Set up event handlers for the connection"""

        @self.connection.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                # Convert candidate to SDP format and send it
                candidate_sdp = candidate.to_sdp()
                await self.send_message(MessageType.ICE_CANDIDATE, candidate_sdp)

        @self.connection.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(
                f"Connection state changed to: {self.connection.connectionState}"
            )
            if (
                self.connection.connectionState == "failed"
                or self.connection.connectionState == "closed"
            ):
                await self.close()

        @self.connection.on("track")
        async def on_track(track):
            logger.info(f"Track received: {track.kind}")
            # Record the track for potential processing
            self.tracks[track.kind] = track

            # throw away any incoming media
            blackhole = MediaBlackhole()
            blackhole.addTrack(track)
            await blackhole.start()

    async def add_track(self, track: MediaStreamTrack):
        """Add a track to the connection"""
        self.connection.addTrack(track)

    async def create_offer(self):
        """Create an offer and set it as local description"""
        offer = await self.connection.createOffer()
        await self.connection.setLocalDescription(offer)
        await self.send_message(
            MessageType.SDP_OFFER, self.connection.localDescription.sdp
        )

    async def send_message(self, message_type: MessageType, content: str):
        """Send a message to the remote peer through the signaling server"""
        await self.client_session.post(
            f"/signalling/submit/{str(message_type)}/from/{self.local_stream_id}/to/{self.remote_stream_id}",
            headers=self.auth.get_headers(),
            data=content,
        )

    async def on_ice(self, ice: str):
        """Handle received ICE candidate"""
        candidate = candidate_from_sdp(ice)
        await self.connection.addIceCandidate(candidate)

    async def on_offer(self, offer: str):
        """Handle received SDP offer"""
        offer = RTCSessionDescription(offer, type="offer")
        await self.connection.setRemoteDescription(offer)

        answer = await self.connection.createAnswer()
        await self.connection.setLocalDescription(answer)
        await self.send_message(MessageType.SDP_ANSWER, answer.sdp)

    async def on_answer(self, answer: str):
        """Handle received SDP answer"""
        answer = RTCSessionDescription(answer, type="answer")
        await self.connection.setRemoteDescription(answer)

    async def close(self):
        """Close the connection"""
        if not self._closed:
            self._closed = True
            for track in self.tracks.values():
                track.stop()
            await self.connection.close()


@dataclass(slots=True)
class VideoTrackSource:
    recording_id: str
    sensor_name: str
    _queue: asyncio.Queue[VideoFrame] = Field(default_factory=asyncio.Queue)
    _closed = False

    """A source for video track data"""

    def __init__(self):
        self._closed = False

    def add_frame(self, frame_data: np.ndarray):
        """Add a frame to the queue"""
        if not self._closed:
            self._queue.put_nowait(VideoFrame.from_ndarray(frame_data))

    async def get_frame(self) -> Optional[VideoFrame]:
        """Get the next frame from the queue"""
        if self._closed:
            return None

        try:
            return await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    def close(self):
        """Close the source"""
        self._closed = True


class VideoTrack(MediaStreamTrack):
    """A media stream track for video"""

    kind = "video"

    def __init__(self, source: VideoTrackSource):
        super().__init__()
        self.source = source
        self._ended = False

    async def recv(self):
        """Receive the next frame"""
        if self._ended:
            raise Exception("Track has ended")

        frame_data = await self.source.get_frame()
        if frame_data is None:
            self._ended = True
            raise Exception("Track has ended")

        return frame_data

    def stop(self):
        """Stop the track"""
        self._ended = True
        self.source.close()


@dataclass(slots=True)
class ClientStreamingManager:
    robot_id: str
    available_for_connections = True
    client_session: ClientSession = Field(
        default_factory=lambda: ClientSession(API_URL)
    )
    auth: Auth = Field(default_factory=get_auth)
    connections: Dict[str, PierToPierConnection] = Field(default_factory=dict)
    video_sources: Dict[str, VideoTrackSource] = Field(default_factory=dict)
    video_tracks: Dict[str, VideoTrack] = Field(default_factory=dict)

    def get_recording_video_stream(
        self, recording_id: str, sensor_name: str
    ) -> VideoTrackSource:
        """Start a new recording stream"""
        stream_key = f"{recording_id}_{sensor_type.value}_{sensor_name}"

        if stream_key in self.video_sources:
            return self.video_sources[stream_key]

        video_source = VideoTrackSource()
        self.video_sources[stream_key] = video_source

        video_track = VideoTrack(video_source)
        self.video_tracks[stream_key] = video_track

        # Add this track to all existing connections
        for connection in self.connections.values():
            connection.add_track(video_track)

        return video_source

    async def create_new_connection(
        self, remote_stream_id: str
    ) -> PierToPierConnection:
        """Create a new P2P connection to a remote stream"""
        connection = PierToPierConnection(
            local_stream_id=self.robot_id,
            remote_stream_id=remote_stream_id,
            client_session=self.client_session,
            auth=self.auth,
        )

        # Set up the connection
        await connection.setup_connection()

        # Add existing tracks to the connection
        for video_track in self.video_tracks.values():
            await connection.add_track(video_track)

        # Create and send an offer
        await connection.create_offer()

        self.connections[remote_stream_id] = connection
        return connection

    async def connect_signalling_stream(self):
        """Connect to the signaling server and process messages"""
        while self.available_for_connections:
            try:
                async with sse_client.EventSource(
                    f"{API_URL}/signalling/notifications/robot/{self.robot_id}",
                    headers=self.auth.get_headers(),
                ) as event_source:
                    async for event in event_source:
                        message = HandshakeMessage.model_validate_json(event.data)
                        if not self.available_for_connections:
                            return

                        # Skip system messages
                        if message.from_id == "system":
                            continue

                        # Get or create connection
                        connection = self.connections.get(message.from_id)
                        if connection is None:
                            connection = await self.create_new_connection(
                                message.from_id
                            )

                        # Process the message
                        match message.type:
                            case MessageType.SDP_OFFER:
                                await connection.on_offer(message.data)
                            case MessageType.SDP_ANSWER:
                                await connection.on_answer(message.data)
                            case MessageType.ICE_CANDIDATE:
                                await connection.on_ice(message.data)
                            case MessageType.HEARTBEAT:
                                pass
                            case MessageType.STREAM_INFO:
                                pass

            except ConnectionError as e:
                logger.error(f"Connection error: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting

    async def close(self):
        """Close all connections and streams"""
        self.available_for_connections = False

        for connection in self.connections.values():
            await connection.close()

        for stream in self.streams.values():
            await stream.close()

        for track in self.video_tracks.values():
            track.stop()

        self.connections.clear()
        self.streams.clear()
        self.video_sources.clear()
        self.video_tracks.clear()


_streaming_managers: Dict[str, ClientStreamingManager] = {}


def get_robot_streaming_manager(robot_id: str) -> "ClientStreamingManager":
    global _streaming_managers

    if robot_id in _streaming_managers:
        return _streaming_managers[robot_id]

    manager = ClientStreamingManager(robot_id=robot_id)
    asyncio.create_task(manager.connect_signalling_stream())

    _streaming_managers[robot_id] = manager
    return manager
