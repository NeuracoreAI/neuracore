import asyncio
import logging
import threading
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from uuid import uuid4

from aiohttp import ClientSession, ClientTimeout
from aiohttp_sse_client import client as sse_client

from neuracore.api.globals import GlobalSingleton
from neuracore.core.auth import Auth, get_auth
from neuracore.core.streaming.client_stream.event_source import EventSource
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
    event_source_cache: dict[Tuple[str, str], EventSource] = field(default_factory=dict)
    track_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tracks: List[VideoSource] = field(default_factory=list)
    local_stream_id: str = field(default_factory=lambda: uuid4().hex)
    recording_notification_stream_id: str = field(default_factory=lambda: uuid4().hex)

    async def _create_video_source(self, sensor_name: str, kind: str) -> VideoSource:
        sensor_key = (sensor_name, kind)
        async with self.track_lock:
            if sensor_key in self.video_tracks_cache:
                return self.video_tracks_cache[sensor_key]

            print(f"creating stream {sensor_key=}")
            mid = str(len(self.tracks))
            video_source = (
                DepthVideoSource(mid=mid) if kind == "depth" else VideoSource(mid=mid)
            )
            self.video_tracks_cache[sensor_key] = video_source
            self.tracks.append(video_source)

            await self.submit_track(mid, kind, sensor_name)

            return video_source

    def get_video_source(self, sensor_name: str, kind: str) -> VideoSource:
        """Start a new recording stream"""

        return asyncio.run_coroutine_threadsafe(
            self._create_video_source(sensor_name, kind), self.loop
        ).result()

    def get_event_source(self, sensor_name: str, kind: str) -> EventSource:
        sensor_key = (sensor_name, kind)
        if sensor_key in self.event_source_cache:
            return self.event_source_cache[sensor_key]

        mid = uuid4().hex
        asyncio.run_coroutine_threadsafe(
            self.submit_track(mid, kind, sensor_name), self.loop
        )

        source = EventSource(mid=mid, loop=self.loop)
        self.event_source_cache[sensor_key] = source

        return source

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
        self,
        remote_stream_id: str,
        connection_id: str,
        connection_token: str,
    ) -> PierToPierConnection:
        """Create a new P2P connection to a remote stream"""

        def on_close():
            self.connections.pop(remote_stream_id, None)

        connection = PierToPierConnection(
            local_stream_id=self.local_stream_id,
            remote_stream_id=remote_stream_id,
            id=connection_id,
            connection_token=connection_token,
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

    async def connect_recording_notification_stream(self):
        backoff = 0
        while self.available_for_connections:
            try:
                sid = self.recording_notification_stream_id
                async with sse_client.EventSource(
                    f"{API_URL}/signalling/recording_notifications/{sid}",
                    session=self.client_session,
                    headers=self.auth.get_headers(),
                ) as event_source:
                    backoff = max(0, backoff - 1)
                    async for event in event_source:
                        if event.type == "heartbeat":
                            continue
                        message = RecordingNotification.model_validate_json(event.data)

                        if message.recording:
                            GlobalSingleton()._active_recording_ids[
                                self.robot_id
                            ] = message.recording_id
                        else:
                            GlobalSingleton()._active_recording_ids.pop(
                                self.robot_id, None
                            )

            except asyncio.TimeoutError:
                print("Timeout error: Retrying connection...")
                await asyncio.sleep(2^backoff)
                backoff += 1
                continue
            except ConnectionError as e:
                print(f"Connection error: {e}")
                await asyncio.sleep(2^backoff)
                backoff += 1
            except Exception as e:
                print(f"Unexpected error: {e}")
                print(traceback.format_exc())
                await asyncio.sleep(2^backoff)
                backoff += 1

    async def connect_signalling_stream(self):
        """Connect to the signaling server and process messages"""
        backoff = 0
        while self.available_for_connections:
            try:
                async with sse_client.EventSource(
                    f"{API_URL}/signalling/notifications/{self.local_stream_id}",
                    session=self.client_session,
                    headers=self.auth.get_headers(),
                ) as event_source:
                    async for event in event_source:
                        backoff = max(0, backoff - 1)
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

                        if message.type == MessageType.CONNECTION_TOKEN:
                            await self.create_new_connection(
                                remote_stream_id=message.from_id,
                                connection_id=message.connection_id,
                                connection_token=message.data,
                            )
                            continue

                        if connection is None or connection.id != message.connection_id:
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
                print("Timeout error: Retrying connection...")
                await asyncio.sleep(2^backoff)
                backoff += 1
                continue
            except ConnectionError as e:
                print(f"Connection error: {e}")
                await asyncio.sleep(2^backoff)
                backoff += 1
            except Exception as e:
                print(f"Unexpected error: {e}")
                print(traceback.format_exc())
                await asyncio.sleep(2^backoff)
                backoff += 1

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


async def create_client_streaming_manager(robot_id):
    # We want to keep the signalling connection alive for as long as possible
    timeout = ClientTimeout(sock_read=None, total=None)
    manager = ClientStreamingManager(
        robot_id=robot_id, loop=asyncio.get_event_loop(), client_session=ClientSession(timeout=timeout)
    )
    asyncio.create_task(manager.connect_signalling_stream())
    asyncio.create_task(manager.connect_recording_notification_stream())

    return manager


def get_robot_streaming_manager(robot_id: str) -> "ClientStreamingManager":
    global _streaming_managers

    if robot_id in _streaming_managers:
        return _streaming_managers[robot_id]

    loop = get_loop()

    manager = asyncio.run_coroutine_threadsafe(
        create_client_streaming_manager(robot_id), loop
    ).result()

    _streaming_managers[robot_id] = manager
    return manager
