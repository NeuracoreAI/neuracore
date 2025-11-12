"""Recording state management for robot data capture sessions.

This module provides centralized management of recording state across robot
instances with real-time notifications via Server-Sent Events. Handles
recording lifecycle events and maintains synchronization between local
state and remote recording triggers.
"""

import asyncio
import logging
from concurrent.futures import Future
from threading import RLock, Timer
from typing import Optional

from aiohttp import ClientSession
from neuracore_types import (
    BaseRecodingUpdatePayload,
    RecordingNotification,
    RecordingNotificationType,
    RobotInstanceIdentifier,
)
from pyee.asyncio import AsyncIOEventEmitter

from neuracore.core.auth import Auth, get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.streaming.base_sse_consumer import (
    BaseSSEConsumer,
    EventSourceConfig,
)
from neuracore.core.streaming.event_loop_utils import get_running_loop
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.utils.background_coroutine_tracker import BackgroundCoroutineTracker

logger = logging.getLogger(__name__)


class RecordingStateManager(BaseSSEConsumer, AsyncIOEventEmitter):
    """Manages recording state across robot instances with real-time notifications.

    Provides centralized tracking of recording sessions for multiple robot instances,
    with automatic synchronization via Server-Sent Events and event emission for
    state changes.
    """

    RECORDING_STARTED = "RECORDING_STARTED"
    RECORDING_STOPPED = "RECORDING_STOPPED"
    RECORDING_SAVED = "RECORDING_SAVED"

    # 5 minutes
    MAX_RECORDING_DURATION_S = 60 * 5

    def __init__(
        self,
        org_id: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        enabled_manager: Optional[EnabledManager] = None,
        background_coroutine_tracker: Optional[BackgroundCoroutineTracker] = None,
        client_session: Optional[ClientSession] = None,
        auth: Optional[Auth] = None,
    ):
        """Initialize the recording state manager.

        Args:
            org_id: the organization to receive signalling from. If not provided
                defaults to the current org.
            loop: the event loop to run on. Defaults to the running loop if not
                provided.
            enabled_manager: The enabled manager for whether this should be
                consuming. Defaults to a new enabled manger if not provided.
            background_coroutine_tracker: The storage for background tasks
                scheduled on receiving events. Defaults to a new tracker if not
                provided.
            client_session: The http session to use. Defaults to a new session
                if not provided.
            auth: The auth instance used to connect to the signalling server or
                defaults to the global auth provider if not provided.
        """
        super().__init__(
            loop=loop,
            enabled_manager=enabled_manager,
            background_coroutine_tracker=background_coroutine_tracker,
            client_session=client_session,
        )
        self.org_id = org_id or get_current_org()
        self.auth = auth if auth is not None else get_auth()

        self.recording_robot_instances: dict[RobotInstanceIdentifier, str] = dict()
        self._recording_expiry_timers: dict[str, Timer] = {}
        self._expired_recording_ids: set[str] = set()
        self._timers_lock = RLock()

    def get_current_recording_id(self, robot_id: str, instance: int) -> Optional[str]:
        """Get the current recording ID for a robot instance.

        Args:
            robot_id: Unique identifier for the robot
            instance: Instance number of the robot

        Returns:
            str: Recording ID if currently recording, None otherwise
        """
        instance_key = RobotInstanceIdentifier(
            robot_id=robot_id, robot_instance=instance
        )
        return self.recording_robot_instances.get(instance_key, None)

    def is_recording(self, robot_id: str, instance: int) -> bool:
        """Check if a robot instance is currently recording.

        Args:
            robot_id: Unique identifier for the robot
            instance: Instance number of the robot

        Returns:
            bool: True if currently recording, False otherwise
        """
        instance_key = RobotInstanceIdentifier(
            robot_id=robot_id, robot_instance=instance
        )
        return instance_key in self.recording_robot_instances

    def is_recording_expired(self, recording_id: str) -> bool:
        """Checks recording expired status.

        Args:
            recording_id: Unique identifier for the recording session

        Returns:
            bool: True if recording is expired, False otherwise
        """
        return recording_id in self._expired_recording_ids

    def mark_recording_expired(self, recording_id: str) -> None:
        """Mark the given recording id as expired and warn the user."""
        if not recording_id:
            return
        self._expired_recording_ids.add(recording_id)
        logger.warning(
            f"Your recording ({recording_id}) has reached the 5-minute limit"
            " and was stopped automatically. "
        )
        self._recording_expiry_timers.pop(recording_id, None)

    def cancel_expiry_timer(self, recording_id: str) -> None:
        """Cancel and remove the expiry timer for a given recording id."""
        with self._timers_lock:
            timer = self._recording_expiry_timers.pop(recording_id, None)
        if timer:
            timer.cancel()

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
        instance_key = RobotInstanceIdentifier(
            robot_id=robot_id, robot_instance=instance
        )
        previous_recording_id = self.recording_robot_instances.get(instance_key, None)

        if previous_recording_id == recording_id:
            return
        if previous_recording_id is not None:
            self.recording_stopped(robot_id, instance, previous_recording_id)

        self.recording_robot_instances[instance_key] = recording_id
        if recording_id not in self._recording_expiry_timers:
            recording_expiry_timer = Timer(
                self.MAX_RECORDING_DURATION_S,
                self.mark_recording_expired,
                args=(recording_id,),
            )
            recording_expiry_timer.daemon = True
            recording_expiry_timer.start()
            self._recording_expiry_timers[recording_id] = recording_expiry_timer

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
        instance_key = RobotInstanceIdentifier(
            robot_id=robot_id, robot_instance=instance
        )
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
            RobotInstanceIdentifier(robot_id=robot_id, robot_instance=instance), None
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

    def get_sse_client_config(self) -> EventSourceConfig:
        """Used to configure the event client to consume events from the server.

        Returns:
            the configuration to be used to connect to the client
        """
        return EventSourceConfig(
            url=f"{API_URL}/org/{self.org_id}/recording/notifications",
            request_options={
                "headers": self.auth.get_headers(),
            },
        )

    async def on_message(self, message_data: str) -> None:
        """The main handler for when the stream receives a message.

        Args:
            message_data: The raw string data of the message

        """
        message = RecordingNotification.model_validate_json(message_data)
        # Python 3.9 compatibility: replace match/case with if/elif
        if message.type == RecordingNotificationType.SAVED:
            self.cancel_expiry_timer(recording_id=message.payload.recording_id)
            self.emit(self.RECORDING_SAVED, **message.payload.model_dump())
        elif message.type in (
            RecordingNotificationType.START,
            RecordingNotificationType.REQUESTED,
        ):
            self.updated_recording_state(is_recording=True, details=message.payload)
        elif message.type in (
            RecordingNotificationType.STOP,
            RecordingNotificationType.SAVED,
            RecordingNotificationType.DISCARDED,
            RecordingNotificationType.EXPIRED,
        ):
            self.updated_recording_state(is_recording=False, details=message.payload)
        elif message.type == RecordingNotificationType.INIT:
            for recording in message.payload:
                self.updated_recording_state(is_recording=True, details=recording)


_recording_manager: Optional[Future[RecordingStateManager]] = None


async def create_recording_state_manager() -> RecordingStateManager:
    """Create a new recording state manager instance.

    Returns:
        RecordingStateManager: Configured recording state
            manager with persistent connection
    """
    return RecordingStateManager(loop=asyncio.get_event_loop())


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
