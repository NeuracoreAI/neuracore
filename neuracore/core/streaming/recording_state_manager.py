"""Recording state management for robot data capture sessions.

This module provides centralized management of recording state across robot
instances with real-time notifications via Server-Sent Events. Handles
recording lifecycle events and maintains synchronization between local
state and remote recording triggers.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future

from aiohttp import ClientSession
from neuracore_types import (
    BaseRecodingUpdatePayload,
    RecordingNotification,
    RecordingNotificationType,
    RecordingStartPayload,
    RobotInstanceIdentifier,
)

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
from neuracore.data_daemon.communications_management.shared_transport import (
    recording_context as _recording_context,
)
from neuracore.data_daemon.lifecycle.daemon_os_control import ensure_daemon_running
from neuracore.data_daemon.rust_selection import is_rust_daemon_enabled

logger = logging.getLogger(__name__)


def _notify_data_bridge_of_expiry(robot_id: str, instance: int) -> None:
    """Tell the Rust producer a source's recording has been locally auto-expired.

    Calls the native ``stop_recording`` for the source so the producer flushes
    any in-progress NUT chunk and publishes ``StopRecording``. The daemon is
    idempotent, so a later explicit ``nc.stop_recording`` that races this is a
    no-op. The stop boundary is wall-clock here (the expiry path has no access
    to the recording context's tracked data-clock timestamp) — acceptable for a
    forced 5-minute timeout. Failures are logged and swallowed: the local
    expiry must always succeed.
    """
    try:
        _recording_context._load_native().stop_recording(
            robot_id, instance, time.time_ns()
        )
    except Exception:
        logger.exception(
            "Failed to notify data bridge of recording expiry for %s:%s",
            robot_id,
            instance,
        )


class RecordingStateManager(BaseSSEConsumer):
    """Manages recording state across robot instances with real-time notifications.

    Provides centralized tracking of recording sessions for multiple robot instances,
    with automatic synchronization via Server-Sent Events.
    """

    RECORDING_EXPIRY_WARNING = 60 * 4.5  # 4.5 minutes
    MAX_RECORDING_DURATION_S = 60 * 5  # 5 minutes

    def __init__(
        self,
        org_id: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        enabled_manager: EnabledManager | None = None,
        background_coroutine_tracker: BackgroundCoroutineTracker | None = None,
        client_session: ClientSession | None = None,
        auth: Auth | None = None,
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

        self._connected_robot_id: str | None = None
        self.recording_robot_instances: dict[RobotInstanceIdentifier, str] = dict()
        self._expired_recording_ids: set[str] = set()
        self._recording_timers: dict[str, list[asyncio.TimerHandle]] = {}
        self.active_dataset_ids: dict[RobotInstanceIdentifier, str] = {}
        self._drain_callbacks: dict[RobotInstanceIdentifier, Callable[[str], None]] = {}

    def get_current_recording_id(self, robot_id: str, instance: int) -> str | None:
        """Get the current recording ID for a robot instance.

        Args:
            robot_id: Robot ID
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
            robot_id: Robot ID
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

    def recording_started(
        self, robot_id: str, instance: int, recording_id: str
    ) -> None:
        """Handle recording start for a robot instance.

        Updates internal state. If the robot was already recording under a
        different id (e.g. the local handle being replaced by the backend cloud
        id), the handle is replaced in place and the previous recording's timers
        are retired — the instance is never transiently cleared, so a concurrent
        ``log_*`` cannot observe a ``None`` recording id and drop a frame.

        Args:
            robot_id: Robot ID
            instance: Instance number of the robot
            recording_id: Unique identifier for the recording session
        """
        instance_key = RobotInstanceIdentifier(
            robot_id=robot_id, robot_instance=instance
        )
        previous_recording_id = self.recording_robot_instances.get(instance_key, None)

        if previous_recording_id == recording_id:
            return

        try:
            ensure_daemon_running()
        except Exception:
            logger.exception("Failed to ensure data daemon is running")
            return

        self.recording_robot_instances[instance_key] = recording_id
        if previous_recording_id is not None:
            self._cancel_recording_timers(previous_recording_id)
        self._schedule_recording_timers(
            robot_id=robot_id,
            instance=instance,
            recording_id=recording_id,
        )

    def _schedule_recording_timers(
        self,
        robot_id: str,
        instance: int,
        recording_id: str,
    ) -> None:
        """Schedule local warning and expiry timers for a recording."""
        # clear any previous timers for this recording ID just in case
        self._cancel_recording_timers(recording_id)

        def warn_if_still_active() -> None:
            current_recording_id = self.get_current_recording_id(robot_id, instance)
            if current_recording_id == recording_id:
                logger.warning(
                    f"Recording {recording_id} is about to reach the 5-minute limit. "
                    "Stop it now to avoid it being expired."
                )

        def expire_if_still_active() -> None:
            current_recording_id = self.get_current_recording_id(robot_id, instance)
            if current_recording_id == recording_id:
                logger.warning(
                    f"Your Recording {recording_id} "
                    "has reached the 5-minute limit and has been expired"
                )
                self._expired_recording_ids.add(recording_id)
                self.recording_stopped(robot_id, instance, recording_id)
                if is_rust_daemon_enabled():
                    _notify_data_bridge_of_expiry(robot_id, instance)

        loop = get_running_loop()

        def _schedule() -> None:
            warn_handle = loop.call_later(
                self.RECORDING_EXPIRY_WARNING,
                warn_if_still_active,
            )
            expiry_handle = loop.call_later(
                self.MAX_RECORDING_DURATION_S,
                expire_if_still_active,
            )
            self._recording_timers[recording_id] = [warn_handle, expiry_handle]

        loop.call_soon_threadsafe(_schedule)

    def _cancel_recording_timers(self, recording_id: str) -> None:
        """Cancel any scheduled timers for a recording."""
        loop = get_running_loop()

        def _cancel() -> None:
            handles = self._recording_timers.pop(recording_id, [])
            for handle in handles:
                handle.cancel()

        loop.call_soon_threadsafe(_cancel)

    def recording_stopped(
        self, robot_id: str, instance: int, recording_id: str | None
    ) -> None:
        """Handle recording stop for a robot instance.

        Updates internal state. Only processes the stop if the recording ID
        matches the current recording.

        Args:
            robot_id: Robot ID
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
        if recording_id is not None:
            self._cancel_recording_timers(recording_id)

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
            RobotInstanceIdentifier(robot_id=robot_id, robot_instance=instance),
            None,
        )
        was_recording = previous_recording_id is not None

        if was_recording == is_recording and previous_recording_id == recording_id:
            return

        if is_recording:
            assert isinstance(
                details, RecordingStartPayload
            ), "recording must be started by a start event"

            # Only react to the robot this client connected to, not other
            # robots in the org that may be recording concurrently.
            if robot_id != self._connected_robot_id:
                return

            assert (
                len(details.dataset_ids) == 1
            ), "Recording can only be started in one dataset"
            dataset_id = details.dataset_ids[0]
            instance_key = RobotInstanceIdentifier(
                robot_id=robot_id, robot_instance=instance
            )
            self.active_dataset_ids[instance_key] = dataset_id
            logger.info(
                "active_dataset_received_from_sse: dataset_id=%s recording_id=%s",
                dataset_id,
                recording_id,
            )
            self.recording_started(
                robot_id=robot_id,
                instance=instance,
                recording_id=recording_id,
            )
        else:
            instance_key = RobotInstanceIdentifier(
                robot_id=robot_id, robot_instance=instance
            )
            if previous_recording_id != recording_id:
                return
            callback = self._drain_callbacks.get(instance_key)
            if callback and was_recording:
                threading.Thread(
                    target=callback,
                    args=(recording_id,),
                    daemon=True,
                    name=f"remote-stop-{recording_id[:8]}",
                ).start()
            self.active_dataset_ids.pop(instance_key, None)
            self.recording_stopped(
                robot_id=robot_id,
                instance=instance,
                recording_id=recording_id,
            )

    def register_connected_robot(self, robot_id: str) -> None:
        """Register the robot that this client is connected to.

        Args:
            robot_id: The ID of the robot that was connected.
        """
        self._connected_robot_id = robot_id

    def register_remote_stop_handler(
        self, robot_id: str, instance: int, callback: Callable[[str], None]
    ) -> None:
        """Register a callback to drain streams when a web-initiated stop arrives."""
        key = RobotInstanceIdentifier(robot_id=robot_id, robot_instance=instance)
        self._drain_callbacks[key] = callback

    def deregister_remote_stop_handler(self, robot_id: str, instance: int) -> None:
        """Remove the drain callback for a robot instance."""
        key = RobotInstanceIdentifier(robot_id=robot_id, robot_instance=instance)
        self._drain_callbacks.pop(key, None)

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
        if message.type == RecordingNotificationType.START:
            self.updated_recording_state(is_recording=True, details=message.payload)

        elif message.type in (
            RecordingNotificationType.STOP,
            RecordingNotificationType.DISCARDED,
            RecordingNotificationType.EXPIRED,
        ):
            self.updated_recording_state(is_recording=False, details=message.payload)
        elif message.type == RecordingNotificationType.INIT:
            for recording in message.payload:
                self.updated_recording_state(is_recording=True, details=recording)


_recording_manager: Future[RecordingStateManager] | None = None


async def create_recording_state_manager() -> RecordingStateManager:
    """Create a new recording state manager instance.

    Returns:
        RecordingStateManager: Configured recording state
            manager with persistent connection
    """
    return RecordingStateManager(loop=get_running_loop())


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
