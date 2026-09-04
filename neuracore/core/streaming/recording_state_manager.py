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
from typing import NamedTuple

from neuracore_types import (
    BaseRecodingUpdatePayload,
    RecordingNotification,
    RecordingNotificationType,
    RecordingStartPayload,
    RobotInstanceIdentifier,
)

from neuracore.core.auth import Auth, get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import STREAM_API_URL
from neuracore.core.streaming.base_sse_consumer import (
    BaseSSEConsumer,
    EventSourceConfig,
)
from neuracore.core.streaming.event_loop_utils import get_running_loop
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.utils.background_coroutine_tracker import BackgroundCoroutineTracker
from neuracore.data_daemon import bridge as _recording_context
from neuracore.data_daemon.daemon_control import ensure_daemon_running

logger = logging.getLogger(__name__)

# The producer stamps a recording's start as `int(start_time * 1e9)`, so the
# value echoed back in a notification can sit a nanosecond below the one held
# locally. Staleness comparisons allow this much slack — far above that
# truncation, far below any real gap between two recordings of one instance.
STALE_START_TOLERANCE_S = 1e-3


class TrackedRecording(NamedTuple):
    """The recording currently tracked for one robot instance.

    Attributes:
        recording_id: The local correlation handle minted by
            ``Robot.start_recording``, replaced by the backend's cloud recording
            id once a START notification for the same recording arrives.
        start_time: The recording window's lower bound, on the same clock the
            backend reports in ``RecordingStartPayload.start_time``. Orders
            notifications against each other: one describing a recording that
            began earlier than this belongs to an already-finished recording.
        opened_locally: Whether the recording was started by the local client
    """

    recording_id: str
    start_time: float
    opened_locally: bool


def _notify_data_bridge_of_expiry(robot_id: str, instance: int) -> None:
    """Tell the producer a source's recording has been locally auto-expired.

    Publishes the stop, then runs ``flush_source``, which is not part of the
    stop and is the only thing that seals the in-progress chunk. A racing
    explicit ``nc.stop_recording`` is a no-op, the boundary is wall-clock here,
    and failures are swallowed: the local expiry must always succeed.
    """
    try:
        native = _recording_context._load_native()
        native.stop_recording(robot_id, instance, time.time_ns())
        native.flush_source(robot_id, instance)
    except Exception:
        logger.exception(
            "Failed to notify data bridge of recording expiry for %s:%s",
            robot_id,
            instance,
        )


def _notify_data_bridge_of_discard(recording_id: str) -> None:
    """Relay a backend ``DISCARDED`` to the daemon so it stops uploading.

    Flipping ``is_recording`` off stops capture; the uploads are what the server
    waits on before deleting the recording's bytes about a minute later.

    Relayed for any discarded recording, not just one this process cancelled —
    the notification is broadcast org-wide and the daemon ignores an id it does
    not hold. Runs on its own thread because publishing waits for the daemon to
    acknowledge, which must not stall the notification stream.
    """
    if not _recording_context.native_loaded():
        # A process that never recorded has no uploads to stop.
        return

    def relay() -> None:
        try:
            _recording_context._load_native().discard_recording(recording_id)
        except Exception:
            logger.exception(
                "Failed to notify data bridge that recording %s was discarded",
                recording_id,
            )

    threading.Thread(
        target=relay,
        daemon=True,
        name=f"discard-{recording_id[:8]}",
    ).start()


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
            auth: The auth instance used to connect to the signalling server or
                defaults to the global auth provider if not provided.
        """
        super().__init__(
            loop=loop,
            enabled_manager=enabled_manager,
            background_coroutine_tracker=background_coroutine_tracker,
        )
        self.org_id = org_id or get_current_org()
        self.auth = auth if auth is not None else get_auth()

        self._connected_robot_id: str | None = None
        # Guards the read-modify-write sequences over `recording_robot_instances`.
        # Local `nc.start_recording` / `nc.stop_recording` run on the caller's
        # thread while notifications are applied on the event loop thread, and
        # both decide what to do from the map's current contents. Reentrant
        # because `updated_recording_state` holds it across `recording_stopped`.
        #
        # Not taken by `get_current_recording_id` / `is_recording`: those sit on
        # the `log_*` hot path and a single dict lookup is already atomic.
        self._state_lock = threading.RLock()
        self.recording_robot_instances: dict[
            RobotInstanceIdentifier, TrackedRecording
        ] = dict()
        # Newest completed recording start time for each source. This lets us
        # recognize its delayed SSE START after the active entry has been removed.
        self._completed_start_time_watermarks: dict[RobotInstanceIdentifier, float] = {}
        self._expired_recording_ids: set[str] = set()
        self._recording_timers: dict[str, list[asyncio.TimerHandle]] = {}
        self.active_dataset_ids: dict[RobotInstanceIdentifier, str] = {}
        self._drain_callbacks: dict[RobotInstanceIdentifier, Callable[[str], None]] = {}
        self._start_callbacks: dict[
            RobotInstanceIdentifier, Callable[[str, str, float], None]
        ] = {}

        self._logout_listener = self._on_logout
        self.auth.once(Auth.LOGOUT_EVENT, self._logout_listener)

    def _on_logout(self) -> None:
        """Schedule teardown on the streaming event loop."""
        self._discard_cached_manager()
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.close)

    def _discard_cached_manager(self) -> None:
        """Remove this manager from the existing global cache."""
        global _recording_manager

        if _recording_manager is None or not _recording_manager.done():
            return
        try:
            cached_manager = _recording_manager.result()
        except Exception:
            return
        if cached_manager is self:
            _recording_manager = None

    def _on_close(self) -> None:
        """Close the SSE stream and clear locally owned recording state."""
        super()._on_close()
        self._remove_logout_listener()

        with self._state_lock:
            for handles in self._recording_timers.values():
                for handle in handles:
                    handle.cancel()
            self._recording_timers.clear()
            self.recording_robot_instances.clear()
            self._completed_start_time_watermarks.clear()
            self._expired_recording_ids.clear()
            self.active_dataset_ids.clear()
            self._drain_callbacks.clear()
            self._start_callbacks.clear()
            self._connected_robot_id = None

        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(
                self.loop.create_task, self.client_session.close()
            )

        self._discard_cached_manager()

    def on_authentication_error(self) -> None:
        """Stop consuming notifications, but keep locally owned recording state.

        A stream the platform rejects is a lost notification channel, not the
        end of the session. ``nc.stop_recording`` reads this state to decide
        there is anything to stop, so clearing it here would leave the stop
        unpublished and the daemon's recording window open until the reaper's
        max-duration bound. An explicit logout still goes through
        :meth:`close`, which clears everything — that is a statement that the
        session is over, which this is not.
        """
        logger.warning(
            "Recording notifications stopped: the stream could not authenticate. "
            "Locally started recordings can still be stopped; remote stops will "
            "not be observed until the next login."
        )

    def _remove_logout_listener(self) -> None:
        """Remove the logout listener when it has not already fired."""
        if self._logout_listener in self.auth.listeners(Auth.LOGOUT_EVENT):
            self.auth.remove_listener(Auth.LOGOUT_EVENT, self._logout_listener)

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
        tracked = self.recording_robot_instances.get(instance_key, None)
        return tracked.recording_id if tracked is not None else None

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
        self, robot_id: str, instance: int, recording_id: str, start_time: float
    ) -> None:
        """Handle recording start for a robot instance.

        Updates internal state. If the robot was already recording under a
        different id (e.g. the local handle being replaced by the backend cloud
        id), the handle is replaced in place and the previous recording's timers
        are retired — the instance is never transiently cleared, so a concurrent
        ``log_*`` cannot observe a ``None`` recording id and drop a frame.

        Publishes the tracked state under :attr:`_state_lock` before starting the
        daemon. Notifications are resolved against this map on another thread, so
        an instance missing from it can have a notification for a different
        recording adopted in its place.

        Args:
            robot_id: Robot ID
            instance: Instance number of the robot
            recording_id: Unique identifier for the recording session
            start_time: The recording's window lower bound — see
                :class:`TrackedRecording`. Callers reacting to a notification
                must pass the value the backend reported, so subsequent
                notifications can be ordered against it.
        """
        with self._state_lock:
            self._track_recording_started(
                robot_id=robot_id,
                instance=instance,
                recording_id=recording_id,
                start_time=start_time,
                opened_locally=True,
            )
        # After the state is visible, and outside the lock: this normally takes
        # a few milliseconds of filesystem work, which is long enough for a
        # notification to arrive and be resolved against the tracked recording.
        # Against a stale daemon it can block for a multiple of the daemon
        # startup budget before the failure is logged.
        self._ensure_daemon_for_recording()

    def _ensure_daemon_for_recording(self) -> None:
        """Start the data daemon if it is not already running.

        Never raises: the recording proceeds either way, since abandoning it here
        would silently turn every subsequent ``log_*`` into a no-op.
        """
        try:
            ensure_daemon_running()
        except Exception:
            logger.exception("Failed to ensure data daemon is running")

    def _track_recording_started(
        self,
        robot_id: str,
        instance: int,
        recording_id: str,
        start_time: float,
        opened_locally: bool,
    ) -> None:
        """Make ``recording_id`` this instance's tracked recording.

        Pure state transition plus timer bookkeeping — no blocking work, so it is
        safe to call with :attr:`_state_lock` held. The caller must hold it.
        """
        instance_key = RobotInstanceIdentifier(
            robot_id=robot_id, robot_instance=instance
        )
        previous = self.recording_robot_instances.get(instance_key, None)
        previous_recording_id = previous.recording_id if previous is not None else None

        if previous_recording_id == recording_id:
            return

        self.recording_robot_instances[instance_key] = TrackedRecording(
            recording_id=recording_id,
            start_time=start_time,
            opened_locally=opened_locally,
        )
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
        with self._state_lock:
            current = self.recording_robot_instances.get(instance_key, None)
            if current is None or current.recording_id != recording_id:
                return
            self.recording_robot_instances.pop(instance_key, None)
            # Never move the completed-recording boundary backward if stops
            # arrive out of order.
            watermark = self._completed_start_time_watermarks.get(instance_key)
            if watermark is None or current.start_time > watermark:
                self._completed_start_time_watermarks[instance_key] = current.start_time
        # Data-bridge stop is driven by the recording context or expiry timer —
        # not here — so the daemon gets exactly one StopRecording with the
        # correct data-clock boundary.
        if recording_id is not None:
            self._cancel_recording_timers(recording_id)

    def updated_recording_state(
        self, is_recording: bool, details: BaseRecodingUpdatePayload
    ) -> None:
        """Update recording state based on remote notification.

        Processes recording state changes from remote notifications and calls
        appropriate start/stop methods if the state actually changed.

        Runs under :attr:`_state_lock` so each decision and the state change it
        implies are atomic against a concurrent local start or stop. Daemon
        startup is performed after the lock is released.

        Args:
            is_recording: Whether the robot should be recording
            details: Recording details including robot ID, instance, and recording ID
        """
        instance_key = RobotInstanceIdentifier(
            robot_id=details.robot_id, robot_instance=details.instance
        )
        with self._state_lock:
            self._apply_recording_notification(is_recording, details)
            active = self.recording_robot_instances.get(instance_key)
            # Rejected stale or foreign STARTs do not become the active recording
            # and therefore must not auto-start a replacement daemon.
            should_ensure_daemon = (
                is_recording
                and details.robot_id == self._connected_robot_id
                and active is not None
                and active.recording_id == details.recording_id
            )
        if should_ensure_daemon:
            self._ensure_daemon_for_recording()

    def _apply_recording_notification(
        self, is_recording: bool, details: BaseRecodingUpdatePayload
    ) -> None:
        """Apply one notification. Caller must hold :attr:`_state_lock`."""
        robot_id = details.robot_id
        instance = details.instance
        recording_id = details.recording_id
        instance_key = RobotInstanceIdentifier(
            robot_id=robot_id, robot_instance=instance
        )

        previous = self.recording_robot_instances.get(instance_key, None)
        previous_recording_id = previous.recording_id if previous is not None else None
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

            completed_start_watermark = self._completed_start_time_watermarks.get(
                instance_key
            )
            # A local stop may beat its matching SSE START; timestamps identify
            # it even when the local and cloud recording IDs differ.
            if (
                completed_start_watermark is not None
                and details.start_time
                <= completed_start_watermark + STALE_START_TOLERANCE_S
            ):
                logger.debug(
                    "ignoring delayed recording start notification: "
                    "recording_id=%s start_time=%s is not newer than completed "
                    "start_time=%s",
                    recording_id,
                    details.start_time,
                    completed_start_watermark,
                )
                return

            # A START describing a recording that began before the tracked one
            # is a late notification for an already-finished recording: the
            # org-wide stream can lag a fast stop-then-start cycle by a whole
            # recording. Tracking it would hand this instance a finished
            # recording, whose STOP would then drain the live one.
            if (
                previous is not None
                and details.start_time < previous.start_time - STALE_START_TOLERANCE_S
            ):
                logger.debug(
                    "ignoring stale recording start notification: "
                    "recording_id=%s start_time=%s is older than tracked "
                    "recording_id=%s start_time=%s",
                    recording_id,
                    details.start_time,
                    previous.recording_id,
                    previous.start_time,
                )
                return

            assert (
                len(details.dataset_ids) == 1
            ), "Recording can only be started in one dataset"
            dataset_id = details.dataset_ids[0]
            self.active_dataset_ids[instance_key] = dataset_id
            logger.info(
                "active_dataset_received_from_sse: dataset_id=%s recording_id=%s",
                dataset_id,
                recording_id,
            )
            # Only open the window if no one else has already.
            opened_locally = previous is not None and previous.opened_locally
            if not opened_locally:
                start_callback = self._start_callbacks.get(instance_key)
                if start_callback is not None:
                    start_callback(recording_id, dataset_id, details.start_time)
                opened_locally = True

            self._track_recording_started(
                robot_id=robot_id,
                instance=instance,
                recording_id=recording_id,
                start_time=details.start_time,
                opened_locally=opened_locally,
            )
        else:
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

    def register_remote_start_handler(
        self,
        robot_id: str,
        instance: int,
        callback: Callable[[str, str, float], None],
    ) -> None:
        """Register a callback to open the daemon window for a web-initiated start."""
        key = RobotInstanceIdentifier(robot_id=robot_id, robot_instance=instance)
        self._start_callbacks[key] = callback

    def deregister_remote_start_handler(self, robot_id: str, instance: int) -> None:
        """Remove the remote-start callback for a robot instance."""
        key = RobotInstanceIdentifier(robot_id=robot_id, robot_instance=instance)
        self._start_callbacks.pop(key, None)

    def get_sse_client_config(self) -> EventSourceConfig:
        """Used to configure the event client to consume events from the server.

        Returns:
            the configuration to be used to connect to the client
        """
        return EventSourceConfig(
            url=f"{STREAM_API_URL}/org/{self.org_id}/recording/notifications",
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
            if message.type == RecordingNotificationType.DISCARDED:
                # Unconditional, unlike the state update above: this process
                # may be uploading a recording it never tracked.
                _notify_data_bridge_of_discard(message.payload.recording_id)
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
