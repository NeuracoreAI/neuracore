import asyncio
from concurrent.futures import Future
from datetime import timedelta
import traceback
from aiohttp import ClientSession, ClientTimeout

from neuracore.api.globals import GlobalSingleton
from neuracore.core.auth import Auth
from neuracore.core.const import API_URL, REMOTE_RECORDING_TRIGGER_ENABLED
from neuracore.core.streaming.client_stream.client_stream_manager import (
    MINIMUM_BACKOFF_LEVEL,
    get_loop,
)
from aiohttp_sse_client import client as sse_client
from neuracore.core.streaming.client_stream.models import RecordingNotification, RecordingNotificationType
from neuracore.core.streaming.client_stream.stream_enabled import EnabledManager


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

        self.remote_trigger_enabled = EnabledManager(
            REMOTE_RECORDING_TRIGGER_ENABLED, loop=self.loop
        )
        self.remote_trigger_enabled.add_listener(
            EnabledManager.DISABLED, self.__stop_remote_trigger
        )

        self.recording_stream_future: Future = asyncio.run_coroutine_threadsafe(
            self.connect_recording_notification_stream(), self.loop
        )

        self.recording_robot_instances: dict[tuple[str, int], str] = dict()

    def get_current_recording_id(self, robot_id: str, instance: int) -> str | None:
        instance_key = (robot_id, instance)
        return self.recording_robot_instances.get(instance_key, None)

    def is_recording(self, robot_id: str, instance: int) -> bool:
        instance_key = (robot_id, instance)
        return instance_key in self.recording_robot_instances

    async def recording_started(self, robot_id: str, instance: int, recording_id: str):
        instance_key = (robot_id, instance)
        previous_recording_id = self.recording_robot_instances.get(instance_key, None)

        if previous_recording_id == recording_id:
            return
        if previous_recording_id != recording_id:
            await self.recording_stopped(robot_id, instance, blocking=True)

        self.recording_robot_instances[instance_key] = recording_id
        for recording in GlobalSingleton().list_all_streams(instance_key).values():
            recording.start_recording(recording_id)

    def recording_started_sync(self, robot_id: str, instance: int, recording_id: str):
        asyncio.run_coroutine_threadsafe(
            self.recording_started(robot_id, instance, recording_id), self.loop
        ).result()

    async def recording_stopped(
        self, robot_id: str, instance: int, blocking: bool = False
    ):
        instance_key = (robot_id, instance)
        if instance_key not in self.recording_robot_instances:
            return
        self.recording_robot_instances.pop(instance_key, None)

        stopping_threads = [
            recording.stop_recording()
            for recording in GlobalSingleton().list_all_streams(instance_key).values()
        ]

        def join_threads():
            for thread in stopping_threads:
                thread.join()

        if blocking:
            await asyncio.to_thread(join_threads)

    def recording_stopped_sync(self, robot_id: str, instance: int,blocking: bool = False):
        asyncio.run_coroutine_threadsafe(
            self.recording_stopped(robot_id, instance, blocking), self.loop
        ).result()

    async def connect_recording_notification_stream(self):
        backoff = MINIMUM_BACKOFF_LEVEL
        while self.remote_trigger_enabled.is_enabled():
            try:
                async with sse_client.EventSource(
                    f"{API_URL}/recording/notifications/{self.local_stream_id}",
                    session=self.cliSent_session,
                    headers=self.auth.get_headers(),
                    reconnection_time=timedelta(seconds=0.1),
                ) as event_source:
                    backoff = max(MINIMUM_BACKOFF_LEVEL, backoff - 1)
                    async for event in event_source:
                        if event.type != "data":
                            continue

                        message = RecordingNotification.model_validate_json(event.data)
                        instance_key = (message.robot_id, message.instance)

                        was_recording = instance_key in self.recording_robot_instances
                        previous_recording_id = self.recording_robot_instances.get(
                            instance_key, None
                        )
                        # TODO: make this work with the new structure
                        if message.type is RecordingNotificationType.START:
                            pass
                        

                        if (
                            was_recording == message.type
                            and previous_recording_id == message.recording_id
                        ):
                            # no change, nothing to do
                            continue

                        if message.recording:

                            await self.recording_started(
                                message.robot_id,
                                message.instance,
                                message.recording_id,
                            )

                        else:

                            await self.recording_stopped(
                                message.robot_id, message.instance
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

    def __stop_remote_trigger(self):
        if self.recording_stream_future.running():
            self.recording_stream_future.cancel()

    def disable_remote_trigger(self):
        """Closes connection to server for updates"""
        self.remote_trigger_enabled.disable()


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
