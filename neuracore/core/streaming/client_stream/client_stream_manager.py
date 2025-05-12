import asyncio
import traceback
from datetime import timedelta
from typing import Dict, List, Tuple
from uuid import uuid4

from aiohttp import ClientSession, ClientTimeout
from aiohttp_sse_client import client as sse_client

from neuracore.api.globals import GlobalSingleton
from neuracore.core.auth import Auth, get_auth
from neuracore.core.streaming.client_stream.async_runtime import AsyncRuntime
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

# must be less than zero -> a reconnection delay of more
# than one second is considered dead
# TODO: resubmit tracks if connection is re-established
# after more than one second
MINIMUM_BACKOFF_LEVEL = -2


class ClientStreamingManager:
    def __init__(
        self,
        robot_id: str,
        robot_instance: int,
        auth: Auth = None,
    ):
        self.robot_id = robot_id
        self.robot_instance = robot_instance
        self.runtime = AsyncRuntime.get_instance()
        self.auth = auth or get_auth()
        self.available_for_connections = True

        self.connections: Dict[str, PierToPierConnection] = {}
        self.video_tracks_cache: Dict[str, VideoSource] = {}
        self.event_source_cache: Dict[Tuple[str, str], EventSource] = {}
        self.tracks: List[VideoSource] = []
        self.local_stream_id = uuid4().hex

        timeout = ClientTimeout(sock_read=None, total=None)
        self.client_session = self.runtime.run_coroutine(self._create_session(timeout))

        self.runtime.submit_coroutine(self.connect_signalling_stream())
        self.runtime.submit_coroutine(self.connect_recording_notification_stream())

    async def _create_session(self, timeout):
        return ClientSession(timeout=timeout)

    def get_video_source(self, sensor_name: str, kind: str) -> VideoSource:
        """Start a new recording stream"""
        sensor_key = (sensor_name, kind)
        if sensor_key in self.video_tracks_cache:
            return self.video_tracks_cache[sensor_key]

        mid = str(len(self.tracks))
        self.runtime.submit_coroutine(self.submit_track(mid, kind, sensor_name))

        video_source = (
            DepthVideoSource(mid=mid) if kind == "depth" else VideoSource(mid=mid)
        )
        self.video_tracks_cache[sensor_key] = video_source
        self.tracks.append(video_source)

        return video_source

    def get_event_source(
        self, sensor_name: str, kind: str, sensor_key: tuple | None = None
    ) -> EventSource:
        sensor_key = sensor_key or (sensor_name, kind)
        if sensor_key in self.event_source_cache:
            return self.event_source_cache[sensor_key]

        mid = uuid4().hex
        self.runtime.submit_coroutine(self.submit_track(mid, kind, sensor_name))
        source = EventSource(mid=mid, loop=self.runtime.loop)
        self.event_source_cache[sensor_key] = source
        return source

    async def submit_track(self, mid: str, kind: str, label: str):
        """Submit new track data"""
        await self.client_session.post(
            f"{API_URL}/signalling/track",
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

    async def heartbeat_response(self):
        """Submit new track data"""
        await self.client_session.post(
            f"{API_URL}/signalling/alive/{self.local_stream_id}",
            headers=self.auth.get_headers(),
            data="pong",
        )

    async def create_new_connection(
        self, remote_stream_id: str, connection_id: str, connection_token: str
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
            loop=self.runtime.loop,
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
        backoff = MINIMUM_BACKOFF_LEVEL
        while self.available_for_connections:
            try:
                async with sse_client.EventSource(
                    f"{API_URL}/signalling/recording_notifications/{self.local_stream_id}",
                    session=self.client_session,
                    headers=self.auth.get_headers(),
                    reconnection_time=timedelta(seconds=0.1),
                ) as event_source:
                    backoff = max(MINIMUM_BACKOFF_LEVEL, backoff - 1)
                    async for event in event_source:
                        if event.type == "data":
                            message = RecordingNotification.model_validate_json(
                                event.data
                            )
                            if message.recording:
                                GlobalSingleton()._active_recording_ids[
                                    message.robot_id
                                ] = message.recording_id
                            else:
                                rec_id = GlobalSingleton()._active_recording_ids.pop(
                                    message.robot_id, None
                                )
                                if rec_id is None:
                                    continue
                                for (
                                    sname,
                                    stream,
                                ) in GlobalSingleton()._data_streams.items():
                                    with stream.lock:
                                        if (
                                            sname.startswith(message.robot_id)
                                            and stream.is_recording()
                                        ):
                                            stream.stop_recording()
            except Exception as e:
                print(f"Recording stream error: {e}")
                print(traceback.format_exc())
                await asyncio.sleep(2**backoff)
                backoff += 1

    async def connect_signalling_stream(self):
        """Connect to the signaling server and process messages"""
        backoff = MINIMUM_BACKOFF_LEVEL
        while self.available_for_connections:
            try:
                async with sse_client.EventSource(
                    f"{API_URL}/signalling/notifications/{self.local_stream_id}",
                    session=self.client_session,
                    headers=self.auth.get_headers(),
                    reconnection_time=timedelta(seconds=0.1),
                ) as event_source:
                    async for event in event_source:
                        try:
                            backoff = max(MINIMUM_BACKOFF_LEVEL, backoff - 1)
                            if not self.available_for_connections:
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

    async def close_connections(self):
        await asyncio.gather(
            *(connection.close() for connection in self.connections.values())
        )

    def close(self):
        """Close all connections and streams"""
        self.available_for_connections = False
        self.runtime.submit_coroutine(self.close_connections())

        for track in self.video_tracks_cache.values():
            track.stop()

        self.connections.clear()
        self.video_tracks_cache.clear()
        self.runtime.submit_coroutine(self.client_session.close())


_streaming_managers: Dict[Tuple[str, int], ClientStreamingManager] = {}


def get_robot_streaming_manager(robot_id: str, instance: int) -> ClientStreamingManager:
    key = (robot_id, instance)
    if key not in _streaming_managers:
        _streaming_managers[key] = ClientStreamingManager(robot_id, instance)
    return _streaming_managers[key]
