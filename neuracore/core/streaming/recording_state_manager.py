import asyncio
import threading
from concurrent.futures import Future
from datetime import timedelta

from aiohttp import ClientSession, ClientTimeout
from aiohttp_sse_client import client as sse_client
from pyee.asyncio import AsyncIOEventEmitter

from neuracore.core.auth import Auth, get_auth
from neuracore.core.const import API_URL, REMOTE_RECORDING_TRIGGER_ENABLED
from neuracore.core.streaming.client_stream.client_stream_manager import (
    MINIMUM_BACKOFF_LEVEL,
)
from neuracore.core.streaming.client_stream.models import (
    BaseRecodingUpdatePayload,
    RecordingNotification,
    RecordingNotificationType,
)
from neuracore.core.streaming.client_stream.stream_enabled import EnabledManager


class RecordingStateManager(AsyncIOEventEmitter):
    RECORDING_STARTED = "RECORDING_STARTED"
    RECORDING_STOPPED = "RECORDING_STOPPED"
    RECORDING_SAVED = "RECORDING_SAVED"

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        client_session: ClientSession,
        auth: Auth = None,
    ):
        super().__init__(loop=loop)
        self.client_session = client_session
        self.auth = auth if auth is not None else get_auth()

        self.remote_trigger_enabled = EnabledManager(
            REMOTE_RECORDING_TRIGGER_ENABLED, loop=self._loop
        )
        self.remote_trigger_enabled.add_listener(
            EnabledManager.DISABLED, self.__stop_remote_trigger
        )

        self.recording_stream_future: Future = asyncio.run_coroutine_threadsafe(
            self.connect_recording_notification_stream(), self._loop
        )

        self.recording_robot_instances: dict[tuple[str, int], str] = dict()

    def get_current_recording_id(self, robot_id: str, instance: int) -> str | None:
        instance_key = (robot_id, instance)
        return self.recording_robot_instances.get(instance_key, None)

    def is_recording(self, robot_id: str, instance: int) -> bool:
        instance_key = (robot_id, instance)
        return instance_key in self.recording_robot_instances

    def recording_started(self, robot_id: str, instance: int, recording_id: str):
        instance_key = (robot_id, instance)
        previous_recording_id = self.recording_robot_instances.get(instance_key, None)

        if previous_recording_id == recording_id:
            return
        if previous_recording_id is not None:
            self.recording_stopped(robot_id, instance, previous_recording_id)

        self.recording_robot_instances[instance_key] = recording_id
        self.emit(
            self.RECORDING_STARTED,
            robot_id=robot_id,
            instance=instance,
            recording_id=recording_id,
        )

    def recording_stopped(self, robot_id: str, instance: int, recording_id: str):
        instance_key = (robot_id, instance)
        current_recording = self.recording_robot_instances.get(instance_key, None)
        if current_recording != recording_id:
            return
        self.recording_robot_instances.pop(instance_key, None)
        self.emit(
            self.RECORDING_STOPPED,
            robot_id=robot_id,
            instance=instance,
            recording_id=recording_id,
        )

    def updated_recording_state(
        self, is_recording: bool, details: BaseRecodingUpdatePayload
    ):
        robot_id = details.robot_id
        instance = details.instance
        recording_id = details.recording_id

        previous_recording_id = self.recording_robot_instances.get(
            (robot_id, instance), None
        )
        was_recording = previous_recording_id is not None

        if was_recording == is_recording and previous_recording_id == recording_id:
            # no change
            return

        if is_recording:
            self.recording_started(
                robot_id=robot_id,
                instance=instance,
                recording_id=recording_id,
            )
        else:
            self.recording_stopped(
                robot_id=robot_id,
                instance=instance,
                recording_id=recording_id,
            )

    async def connect_recording_notification_stream(self):
        backoff = MINIMUM_BACKOFF_LEVEL
        while self.remote_trigger_enabled.is_enabled():
            try:
                async with sse_client.EventSource(
                    f"{API_URL}/recording/notifications",
                    session=self.client_session,
                    headers=self.auth.get_headers(),
                    reconnection_time=timedelta(seconds=0.1),
                ) as event_source:
                    backoff = max(MINIMUM_BACKOFF_LEVEL, backoff - 1)
                    async for event in event_source:
                        if event.type != "data":
                            continue

                        message = RecordingNotification.model_validate_json(event.data)

                        match message.type:
                            case RecordingNotificationType.SAVED:
                                self.emit(
                                    self.RECORDING_SAVED, **message.payload.model_dump()
                                )
                            case (
                                RecordingNotificationType.START
                                | RecordingNotificationType.REQUESTED
                            ):
                                self.updated_recording_state(
                                    is_recording=True, details=message.payload
                                )

                            case (
                                RecordingNotificationType.STOP
                                | RecordingNotificationType.SAVED
                                | RecordingNotificationType.DISCARDED
                                | RecordingNotificationType.EXPIRED
                            ):
                                self.updated_recording_state(
                                    is_recording=False, details=message.payload
                                )

                            case RecordingNotificationType.INIT:
                                for recording in message.payload:
                                    self.updated_recording_state(
                                        is_recording=True, details=recording
                                    )

            except ConnectionError as e:
                print(f"Connection error: {e}")
                await asyncio.sleep(2 ^ backoff)
                backoff += 1
            except Exception as e:
                print(f"Unexpected error: {e}")
                await asyncio.sleep(2 ^ backoff)
                backoff += 1

    def __stop_remote_trigger(self):
        if self.recording_stream_future.running():
            self.recording_stream_future.cancel()

    def disable_remote_trigger(self):
        """Closes connection to server for updates"""
        self.remote_trigger_enabled.disable()


_recording_manager: Future[RecordingStateManager] | None = None


def _get_loop():
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        threading.Thread(target=lambda: loop.run_forever(), daemon=True).start()
        return loop


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
    _recording_manager = asyncio.run_coroutine_threadsafe(
        create_recording_state_manager(), _get_loop()
    )
    return _recording_manager.result()
