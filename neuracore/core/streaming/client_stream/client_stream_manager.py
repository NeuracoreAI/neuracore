from dataclasses import field, dataclass
import asyncio
from typing import Dict, List
from uuid import uuid4
from neuracore.core.auth import Auth, get_auth
from aiohttp_sse_client import client as sse_client
from aiohttp import ClientSession
import traceback
from neuracore.core.streaming.client_stream.models import (
    HandshakeMessage,
    MessageType,
    RobotStreamTrack,
)
from ...const import API_URL
from .video_source import DepthVideoSource, VideoSource
from .connection import PierToPierConnection


def get_loop():
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


@dataclass
class ClientStreamingManager:
    robot_id: str
    available_for_connections: bool = True
    client_session: ClientSession = field(default_factory=ClientSession)
    auth: Auth = field(default_factory=get_auth)
    connections: Dict[str, PierToPierConnection] = field(default_factory=dict)
    video_tracks_cache: Dict[str, VideoSource] = field(default_factory=dict)
    tracks: List[VideoSource] = field(default_factory=list)
    local_stream_id: str = field(default_factory=lambda: uuid4().hex)

    def get_recording_video_stream(self, sensor_name: str, kind: str) -> VideoSource:
        """Start a new recording stream"""
        sensor_key = (sensor_name, kind)
        if sensor_key in self.video_tracks_cache:
            return self.video_tracks_cache[sensor_key]

        video_track = DepthVideoSource(id=sensor_name) if kind == "depth" else VideoSource(id=sensor_name)
        self.video_tracks_cache[sensor_key] = video_track
        self.tracks.append(video_track)

        # Add this track to all existing connections
        for connection in self.connections.values():
            connection.add_video_source(video_track)

        get_loop().create_task(
            self.submit_track(str(len(self.tracks) - 1), kind, sensor_name)
        )

        return video_track

    async def submit_track(self, mid: str, kind: str, label: str):
        """Submit new track data"""
        print(f"Submit track {self.local_stream_id=} {mid=} {kind=} {label=}")
        await self.client_session.post(
            f"{API_URL}/signalling/track",
            headers=self.auth.get_headers(),
            json=RobotStreamTrack(
                robot_id=self.robot_id,
                stream_id=self.local_stream_id,
                mid=mid,
                kind=kind,
                label=label,
            ).model_dump(mode="json"),
        )

    async def heartbeat_response(self):
        """Submit new track data"""
        await self.client_session.post(
            f"{API_URL}/signalling/alive/{self.local_stream_id}",
            headers=self.auth.get_headers(),
            data="pong",
        )

    async def create_new_connection(
        self, remote_stream_id: str
    ) -> PierToPierConnection:
        """Create a new P2P connection to a remote stream"""

        def on_close():
            del self.connections[remote_stream_id]

        connection = PierToPierConnection(
            local_stream_id=self.local_stream_id,
            remote_stream_id=remote_stream_id,
            on_close=on_close,
            client_session=self.client_session,
            auth=self.auth,
        )

        connection.setup_connection()

        for video_track in self.tracks:
            await connection.add_video_source(video_track)

        self.connections[remote_stream_id] = connection
        return connection

    async def connect_signalling_stream(self):
        """Connect to the signaling server and process messages"""
        while self.available_for_connections:
            try:
                async with sse_client.EventSource(
                    f"{API_URL}/signalling/notifications/{self.local_stream_id}",
                    headers=self.auth.get_headers(),
                ) as event_source:
                    async for event in event_source:
                        print(f"received event {event.type=} {event.message=}")
                        if event.type == "heartbeat":
                            await self.heartbeat_response()
                            continue

                        message = HandshakeMessage.model_validate_json(event.data)
                        print(f"Message Received {message}")
                        if not self.available_for_connections:
                            return

                        if message.from_id == "system":
                            continue

                        connection = self.connections.get(message.from_id)
                        if connection is None:
                            connection = await self.create_new_connection(
                                message.from_id
                            )

                        match message.type:
                            case MessageType.SDP_OFFER:
                                await connection.on_offer(message.data)
                            case MessageType.ICE_CANDIDATE:
                                await connection.on_ice(message.data)
                            case MessageType.CONNECTION_TOKEN:
                                await connection.on_token(message.data)
                            case MessageType.SDP_ANSWER:
                                await connection.on_answer(message.data)
                            case _:
                                pass

            except ConnectionError as e:
                print(f"Connection error: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Unexpected error: {e}")
                print(traceback.format_exc())
                await asyncio.sleep(5)

    async def close_connections(self):
        await asyncio.gather(
            *(connection.close() for connection in self.connections.values())
        )

    def close(self):
        """Close all connections and streams"""
        self.available_for_connections = False

        get_loop().create_task(self.close_connections())

        for track in self.video_tracks_cache.values():
            track.stop()

        self.connections.clear()
        self.video_tracks_cache.clear()
        self.client_session.close()


_streaming_managers: Dict[str, ClientStreamingManager] = {}


def get_robot_streaming_manager(robot_id: str) -> "ClientStreamingManager":
    global _streaming_managers

    if robot_id in _streaming_managers:
        return _streaming_managers[robot_id]

    manager = ClientStreamingManager(robot_id=robot_id)
    asyncio.create_task(manager.connect_signalling_stream())

    _streaming_managers[robot_id] = manager
    return manager
