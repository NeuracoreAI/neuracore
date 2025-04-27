import asyncio
import logging
import threading
import traceback
from concurrent.futures import Future
from datetime import timedelta
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
        loop: asyncio.AbstractEventLoop,
        client_session: ClientSession,
        auth: Auth = None,
    ):
        self.robot_id = robot_id
        self.robot_instance = robot_instance
        self.loop = loop
        self.client_session = client_session
        self.auth = auth if auth is not None else get_auth()
        self.available_for_connections = True

        self.connections: Dict[str, PierToPierConnection] = {}
        self.video_tracks_cache: Dict[str, VideoSource] = {}
        self.event_source_cache: Dict[Tuple[str, str], EventSource] = {}
        self.track_lock = asyncio.Lock()
        self.tracks: List[VideoSource] = []
        self.local_stream_id = uuid4().hex

        self.signalling_stream_future: Future = asyncio.run_coroutine_threadsafe(
            self.connect_signalling_stream(), self.loop
        )
        self.recording_stream_future: Future = asyncio.run_coroutine_threadsafe(
            self.connect_recording_notification_stream(), self.loop
        )

    async def _create_video_source(self, sensor_name: str, kind: str) -> VideoSource:
        sensor_key = (sensor_name, kind)
        async with self.track_lock:
            if sensor_key in self.video_tracks_cache:
                return self.video_tracks_cache[sensor_key]

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
            except ConnectionError as e:
                print(f"Connection error: {e}")
                await asyncio.sleep(2 ^ backoff)
                backoff += 1
            except Exception as e:
                print(f"Unexpected error: {e}")
                print(traceback.format_exc())
                await asyncio.sleep(2 ^ backoff)
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
                            print(f"Application error: {e}")
                            await asyncio.sleep(2 ^ backoff)
                            backoff += 1
            except Exception as e:
                print(f"Connection error: {e}")
                await asyncio.sleep(2 ^ backoff)
                backoff += 1

    async def close_connections(self):
        await asyncio.gather(
            *(connection.close() for connection in self.connections.values())
        )

    def close(self):
        """Close all connections and streams"""
        self.available_for_connections = False

        if self.signalling_stream_future.running():
            self.signalling_stream_future.cancel()
        if self.recording_stream_future.running():
            self.recording_stream_future.cancel()

        asyncio.run_coroutine_threadsafe(self.close_connections(), self.loop)

        for track in self.video_tracks_cache.values():
            track.stop()

        self.connections.clear()
        self.video_tracks_cache.clear()
        asyncio.run_coroutine_threadsafe(self.client_session.close(), self.loop)


_streaming_managers: Dict[tuple[str, int], Future[ClientStreamingManager]] = {}


async def create_client_streaming_manager(robot_id: str, instance: int):
    # We want to keep the signalling connection alive for as long as possible
    timeout = ClientTimeout(sock_read=None, total=None)
    manager = ClientStreamingManager(
        robot_id=robot_id,
        robot_instance=instance,
        loop=asyncio.get_event_loop(),
        client_session=ClientSession(timeout=timeout),
    )

    return manager


def get_robot_streaming_manager(
    robot_id: str, instance: int
) -> "ClientStreamingManager":
    global _streaming_managers
    key = (robot_id, instance)
    if key in _streaming_managers:
        return _streaming_managers[key].result()

    loop = get_loop()
    manager = asyncio.run_coroutine_threadsafe(
        create_client_streaming_manager(robot_id, instance), loop
    )
    _streaming_managers[key] = manager
    return manager.result()
