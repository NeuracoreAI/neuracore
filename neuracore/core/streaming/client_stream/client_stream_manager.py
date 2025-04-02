import asyncio
import logging
import threading
import traceback
from dataclasses import dataclass, field
from typing import Dict, List
from uuid import uuid4

from aiohttp import ClientSession
from aiohttp_sse_client import client as sse_client

from neuracore.api.globals import GlobalSingleton
from neuracore.core.auth import Auth, get_auth
from neuracore.core.streaming.client_stream.models import (
    HandshakeMessage,
    MessageType,
    RecordingNotification,
    RobotStreamTrack,
)

from ...const import API_URL
from .connection import PierToPierConnection
from .video_source import DepthVideoSource, VideoSource

logger = logging.getLogger(__name__)


def get_loop():
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        threading.Thread(target=lambda: loop.run_forever(), daemon=True).start()
        return loop


@dataclass
class ClientStreamingManager:
    robot_id: str
    loop: asyncio.AbstractEventLoop
    client_session: ClientSession
    available_for_connections: bool = True
    auth: Auth = field(default_factory=get_auth)
    connections: Dict[str, PierToPierConnection] = field(default_factory=dict)
    video_tracks_cache: Dict[str, VideoSource] = field(default_factory=dict)
    tracks: List[VideoSource] = field(default_factory=list)
    local_stream_id: str = field(default_factory=lambda: uuid4().hex)
    recording_connection_established: bool = False

    def get_recording_video_stream(self, sensor_name: str, kind: str) -> VideoSource:
        """Start a new recording stream"""
        sensor_key = (sensor_name, kind)
        if sensor_key in self.video_tracks_cache:
            return self.video_tracks_cache[sensor_key]

        mid = str(len(self.tracks))
        video_track = (
            DepthVideoSource(mid=mid) if kind == "depth" else VideoSource(mid=mid)
        )
        self.video_tracks_cache[sensor_key] = video_track
        self.tracks.append(video_track)

        # Add this track to all existing connections
        for connection in self.connections.values():
            connection.add_video_source(video_track)

        print(f"create stream {sensor_name=}, {self.loop.is_running()=}")
        asyncio.run_coroutine_threadsafe(
            self.submit_track(mid, kind, sensor_name), self.loop
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
            loop=self.loop,
            connection_token=self.loop.create_future(),
        )

        connection.setup_connection()

        for video_track in self.tracks:
            connection.add_video_source(video_track)

        self.connections[remote_stream_id] = connection
        return connection

    async def connect_recording_notification_stream(self, robot_id: str):
        while self.available_for_connections:
            try:
                async with sse_client.EventSource(
                    f"{API_URL}/signalling/recording_notifications/{self.local_stream_id}",
                    headers=self.auth.get_headers(),
                ) as event_source:
                    async for event in event_source:
                        message = RecordingNotification.model_validate_json(event.data)
                        if message.recording:
                            GlobalSingleton()._active_recording_ids[
                                robot_id
                            ] = message.recording_id
                        else:
                            GlobalSingleton()._active_recording_ids.pop(robot_id, None)
            except ConnectionError as e:
                print(f"Connection error: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Unexpected error: {e}")
                print(traceback.format_exc())
                await asyncio.sleep(5)

    async def connect_signalling_stream(self):
        """Connect to the signaling server and process messages"""
        while self.available_for_connections:
            try:
                async with sse_client.EventSource(
                    f"{API_URL}/signalling/notifications/{self.local_stream_id}",
                    headers=self.auth.get_headers(),
                ) as event_source:
                    self.recording_connection_established = True
                    async for event in event_source:
                        logger.info(f"received event {event.type=} {event.message=}")
                        if event.type == "heartbeat":
                            await self.heartbeat_response()
                            continue

                        message = HandshakeMessage.model_validate_json(event.data)
                        logger.info(f"Message Received {message}")
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

        asyncio.run_coroutine_threadsafe(self.close_connections(), self.loop)

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

    loop = get_loop()
    manager = ClientStreamingManager(
        robot_id=robot_id, loop=loop, client_session=ClientSession(loop=loop)
    )
    asyncio.run_coroutine_threadsafe(
        manager.connect_recording_notification_stream(robot_id), loop
    )
    asyncio.run_coroutine_threadsafe(manager.connect_signalling_stream(), loop)

    # Wait until recording_connection_established is True
    while not manager.recording_connection_established:
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0.5), loop)

    _streaming_managers[robot_id] = manager
    return manager
