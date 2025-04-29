import asyncio
from concurrent.futures import Future
import traceback
from typing import Set
from aiohttp import ClientSession, ClientTimeout

from neuracore.api.globals import GlobalSingleton
from neuracore.core.auth import Auth
from neuracore.core.const import API_URL
from neuracore.core.streaming.client_stream.client_stream_manager import (
    MINIMUM_BACKOFF_LEVEL,
    get_loop,
)
from aiohttp_sse_client import client as sse_client
from neuracore.core.streaming.client_stream.models import RecordingNotification


class RecordingStateManager:

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        client_session: ClientSession,
        auth: Auth = None,
    ):
        self.loop = loop
        self.client_session = client_session
        self.auth = auth if auth is not None else get_auth()

        self.recording_stream_future: Future = asyncio.run_coroutine_threadsafe(
            self.connect_recording_notification_stream(), self.loop
        )

        self.recording_robot_instances: Set[tuple[str, int]] = Set()


    def is_recording(self, robot_id: str, instance: int) -> bool:
        instance_key = (robot_id, instance)
        return instance_key in self.recording_robot_instances

    async def recording_started(self, robot_id: str, instance: int):
        instance_key = (robot_id, instance)
        if instance_key in self.recording_robot_instances:
            return
        self.recording_robot_instances.add(instance_key)
        for recording in GlobalSingleton().list_all_streams(instance_key).values():
            recording.start_recording()

    async def recording_stopped(self, robot_id: str, instance: int):
        instance_key = (robot_id, instance)
        if instance_key not in self.recording_robot_instances:
            return
        self.recording_robot_instances.remove(instance_key)
        for recording in GlobalSingleton().list_all_streams(instance_key).values():
            recording.stop_recording()

    async def connect_recording_notification_stream(self):
        backoff = MINIMUM_BACKOFF_LEVEL
        while self.available_for_connections:
            try:
                async with sse_client.EventSource(
                    f"{API_URL}/signalling/recording_notifications/{self.local_stream_id}",
                    headers=self.auth.get_headers(),
                ) as event_source:
                    backoff = max(MINIMUM_BACKOFF_LEVEL, backoff - 1)
                    async for event in event_source:
                        if event.type != "data":
                            continue

                        message = RecordingNotification.model_validate_json(event.data)
                        instance_key = (message.robot_id, message.instance)

                        is_recording = instance_key in self.recording_robot_instances

                        if is_recording == message.recording:
                            # no change, nothing to do
                            continue

                        if message.recording:
                            asyncio.run_coroutine_threadsafe(
                                self.recording_started(
                                    message.robot_id, message.instance
                                ),
                                self.loop,
                            )
                        else:
                            asyncio.run_coroutine_threadsafe(
                                self.recording_stopped(
                                    message.robot_id, message.instance
                                ),
                                self.loop,
                            )

            except ConnectionError as e:
                print(f"Connection error: {e}")
                await asyncio.sleep(2 ^ backoff)
                backoff += 1
            except Exception as e:
                print(f"Unexpected error: {e}")
                print(traceback.format_exc())
                await asyncio.sleep(2 ^ backoff)
                backoff += 1

    def close(self):
        """Closes connection to server for updates"""

        if self.recording_stream_future.running():
            self.recording_stream_future.cancel()


_recording_manager: Future[RecordingStateManager] | None = None


async def create_recording_state_manager():
    # We want to keep the signalling connection alive for as long as possible
    timeout = ClientTimeout(sock_read=None, total=None)
    manager = RecordingStateManager(
        loop=asyncio.get_event_loop(),
        client_session=ClientSession(timeout=timeout),
    )
    return manager


def get_recording_state_manager() -> "RecordingStateManager":
    global _recording_manager
    if _recording_manager is not None:
        return _recording_manager.result()

    loop = get_loop()
    _recording_manager = asyncio.run_coroutine_threadsafe(
        create_recording_state_manager(), loop
    )
    return _recording_manager.result()
