"""Recording state management for robot data capture sessions.

This module provides centralized management of recording state across robot
instances with real-time notifications via Server-Sent Events. Handles
recording lifecycle events and maintains synchronization between local
state and remote recording triggers.
"""

import asyncio
import logging
from concurrent.futures import Future
from datetime import timedelta
from typing import Optional

from aiohttp import ClientSession, ClientTimeout
from aiohttp_sse_client import client as sse_client
from pyee.asyncio import AsyncIOEventEmitter

from neuracore.core.auth import Auth, get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL, REMOTE_RECORDING_TRIGGER_ENABLED
from neuracore.core.streaming.client_stream.client_stream_manager import (
    MAXIMUM_BACKOFF_TIME_S,
    MINIMUM_BACKOFF_TIME_S,
)
from neuracore.core.streaming.client_stream.models import (
    BaseRecodingUpdatePayload,
    RecordingNotification,
    RecordingNotificationType,
)
from neuracore.core.streaming.client_stream.stream_enabled import EnabledManager
from neuracore.core.streaming.event_loop_utils import get_running_loop

logger = logging.getLogger(__name__)


class RecordingStateManager(AsyncIOEventEmitter):
    """Manages recording state across robot instances with real-time notifications.

    Provides centralized tracking of recording sessions for multiple robot instances,
    with automatic synchronization via Server-Sent Events and event emission for
    state changes.
    """

    RECORDING_STARTED = "RECORDING_STARTED"
    RECORDING_STOPPED = "RECORDING_STOPPED"
    RECORDING_SAVED = "RECORDING_SAVED"

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        client_session: ClientSession,
        auth: Optional[Auth] = None,
    ):
        """Initialize the recording state manager.

        Args:
            loop: Event loop for async operations
            client_session: HTTP client session for API communication
            auth: Authentication object. If not provided, uses default auth
        """
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

    def get_current_recording_id(self, robot_id: str, instance: int) -> Optional[str]:
        """Get the current recording ID for a robot instance.

        Args:
            robot_id: Unique identifier for the robot
            instance: Instance number of the robot

        Returns:
            str: Recording ID if currently recording, None otherwise
        """
        instance_key = (robot_id, instance)
        return self.recording_robot_instances.get(instance_key, None)

    def is_recording(self, robot_id: str, instance: int) -> bool:
        """Check if a robot instance is currently recording.

        Args:
            robot_id: Unique identifier for the robot
            instance: Instance number of the robot

        Returns:
            bool: True if currently recording, False otherwise
        """
        instance_key = (robot_id, instance)
        return instance_key in self.recording_robot_instances

    def recording_started(
        self, robot_id: str, instance: int, recording_id: str
    ) -> None:
        """Handle recording start for a robot instance.

        Updates internal state and emits RECORDING_STARTED event. If the robot
        was already recording with a different ID, stops the previous recording first.

        Args:
            robot_id: Unique identifier for the robot
            instance: Instance number of the robot
            recording_id: Unique identifier for the recording session
        """
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

    def recording_stopped(
        self, robot_id: str, instance: int, recording_id: str
    ) -> None:
        """Handle recording stop for a robot instance.

        Updates internal state and emits RECORDING_STOPPED event. Only processes
        the stop if the recording ID matches the current recording.

        Args:
            robot_id: Unique identifier for the robot
            instance: Instance number of the robot
            recording_id: Unique identifier for the recording session
        """
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
    ) -> None:
        """Update recording state based on remote notification.

        Processes recording state changes from remote notifications and calls
        appropriate start/stop methods if the state actually changed.

        Args:
            is_recording: Whether the robot should be recording
            details: Recording details including robot ID, instance, and recording ID
        """
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

    async def connect_recording_notification_stream(self) -> None:
        """Connect to recording notification stream via Server-Sent Events.

        Maintains a persistent connection to receive real-time recording state
        updates with exponential backoff retry logic. Processes different types
        of recording notifications and updates local state accordingly.
        """
        backoff_time = MINIMUM_BACKOFF_TIME_S

        while self.remote_trigger_enabled.is_enabled():
            try:
                await asyncio.sleep(backoff_time)
                backoff_time = min(MAXIMUM_BACKOFF_TIME_S, backoff_time * 2)

                org_id = get_current_org()
                async with sse_client.EventSource(
                    f"{API_URL}/org/{org_id}/recording/notifications",
                    session=self.client_session,
                    headers=self.auth.get_headers(),
                    reconnection_time=timedelta(seconds=0.1),
                ) as event_source:
                    backoff_time = max(MINIMUM_BACKOFF_TIME_S, backoff_time / 2)
                    async for event in event_source:
                        if event.type != "data":
                            continue

                        message = RecordingNotification.model_validate_json(event.data)
                        # Python 3.9 compatibility: replace match/case with if/elif
                        if message.type == RecordingNotificationType.SAVED:
                            self.emit(
                                self.RECORDING_SAVED, **message.payload.model_dump()
                            )
                        elif message.type in (
                            RecordingNotificationType.START,
                            RecordingNotificationType.REQUESTED,
                        ):
                            self.updated_recording_state(
                                is_recording=True, details=message.payload
                            )
                        elif message.type in (
                            RecordingNotificationType.STOP,
                            RecordingNotificationType.SAVED,
                            RecordingNotificationType.DISCARDED,
                            RecordingNotificationType.EXPIRED,
                        ):
                            self.updated_recording_state(
                                is_recording=False, details=message.payload
                            )
                        elif message.type == RecordingNotificationType.INIT:
                            for recording in message.payload:
                                self.updated_recording_state(
                                    is_recording=True, details=recording
                                )

            except Exception as e:
                logger.warning(f"Recording signalling error: {e}")

    def __stop_remote_trigger(self) -> None:
        """Internal method to stop the remote trigger connection."""
        if self.recording_stream_future.running():
            self.recording_stream_future.cancel()

    def disable_remote_trigger(self) -> None:
        """Disable remote recording triggers and close server connection.

        Stops listening for remote recording notifications and closes the
        persistent connection to the notification stream.
        """
        self.remote_trigger_enabled.disable()


_recording_manager: Optional[Future[RecordingStateManager]] = None


async def create_recording_state_manager() -> RecordingStateManager:
    """Create a new recording state manager instance.

    Returns:
        RecordingStateManager: Configured recording state
            manager with persistent connection
    """
    # We want to keep the signalling connection alive for as long as possible
    timeout = ClientTimeout(sock_read=None, total=None)
    manager = RecordingStateManager(
        loop=asyncio.get_event_loop(),
        client_session=ClientSession(timeout=timeout),
    )
    return manager


def get_recording_state_manager() -> "RecordingStateManager":
    """Get the global recording state manager instance.

    Uses a singleton pattern to ensure only one recording state manager
    exists globally. Thread-safe and handles event loop coordination.

    Returns:
        RecordingStateManager: The global recording state manager instance
    """
    global _recording_manager
    if _recording_manager is not None:
        return _recording_manager.result()
    _recording_manager = asyncio.run_coroutine_threadsafe(
        create_recording_state_manager(), get_running_loop()
    )
    return _recording_manager.result()
