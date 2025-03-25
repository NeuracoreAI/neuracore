from dataclasses import field, dataclass
import asyncio
from typing import Dict
from neuracore.core.auth import Auth, get_auth
from aiohttp_sse_client import client as sse_client
from aiohttp import ClientSession

from neuracore.core.streaming.client_stream.models import HandshakeMessage, MessageType
from ...const import API_URL
from .video_source import VideoSource
from .connection import PierToPierConnection



@dataclass
class ClientStreamingManager:
    robot_id: str
    available_for_connections: bool = True
    client_session: ClientSession = field(default_factory=ClientSession)
    auth: Auth = field(default_factory=get_auth)
    connections: Dict[str, PierToPierConnection] = field(default_factory=dict)
    video_tracks: Dict[str, VideoSource] = field(default_factory=dict)

    def get_recording_video_stream(
        self, recording_id: str, sensor_name: str
    ) -> VideoSource:
        """Start a new recording stream"""
        stream_key = f"{recording_id}_{sensor_name}"

        if stream_key in self.video_tracks:
            return self.video_tracks[stream_key]

        video_track = VideoSource()
        self.video_tracks[stream_key] = video_track

        # Add this track to all existing connections
        for connection in self.connections.values():
            connection.add_video_source(video_track)

        return video_track

    async def create_new_connection(
        self, local_stream_id: str, remote_stream_id: str
    ) -> PierToPierConnection:
        """Create a new P2P connection to a remote stream"""

        def on_close():
            del self.connections[remote_stream_id]

        connection = PierToPierConnection(
            local_stream_id=local_stream_id,
            remote_stream_id=remote_stream_id,
            on_close=on_close,
            client_session=self.client_session,
            auth=self.auth,
        )

        connection.setup_connection()

        for video_track in self.video_tracks.values():
            await connection.add_video_source(video_track)

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
                        print(f"Message Received {message}")
                        if not self.available_for_connections:
                            return

                        # Skip system messages
                        if message.from_id == "system":
                            continue

                        # Get or create connection
                        connection = self.connections.get(message.from_id)
                        if connection is None:
                            connection = await self.create_new_connection(
                                message.to_id, message.from_id
                            )

                        # Process the message
                        match message.type:
                            case MessageType.SDP_OFFER:
                                await connection.on_offer(message.data)
                            case MessageType.ICE_CANDIDATE:
                                await connection.on_ice(message.data)
                            case _:
                                pass

            except ConnectionError as e:
                print(f"Connection error: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Unexpected error: {e}")
                await asyncio.sleep(5)

    async def close_connections(self):
        await asyncio.gather(
            *(connection.close() for connection in self.connections.values())
        )

    def close(self):
        """Close all connections and streams"""
        self.available_for_connections = False

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.close_connections())
        except RuntimeError:
            asyncio.run(self.close_connections())

        for track in self.video_tracks.values():
            track.stop()

        self.connections.clear()
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
