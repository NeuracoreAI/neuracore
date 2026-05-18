"""High-level API for Neuracore robot management and data recording.

This module provides the main public interface for connecting to robots,
managing authentication, controlling data recording sessions, and handling
live data streaming. It maintains global state for active robots and
recording sessions.
"""

import logging
import time
from warnings import warn

from neuracore.core.config.config_manager import get_config_manager
from neuracore.core.organizations import list_my_orgs
from neuracore.core.streaming.p2p.provider.global_live_data_enabled import (
    get_provide_live_data_enabled_manager,
)
from neuracore.core.streaming.p2p.stream_manager_orchestrator import (
    StreamManagerOrchestrator,
)
from neuracore.core.streaming.recording_state_manager import get_recording_state_manager
from neuracore.core.utils import backend_utils

from ..core.auth import get_auth
from ..core.data.dataset import Dataset
from ..core.exceptions import DatasetError, RobotError
from ..core.robot import Robot, get_robot
from ..core.robot import init as _init_robot
from ..core.robot import update_robot_name as _update_robot_name
from .globals import GlobalSingleton

logger = logging.getLogger(__name__)


def _get_robot(robot_name: str | None, instance: int) -> Robot:
    """Get a robot by name and instance.

    Retrieves either the active robot from global state or looks up a specific
    robot by name and instance. Falls back to the active robot if no name
    is provided.

    Args:
        robot_name: Name of the robot to retrieve.
        instance: Instance number of the robot.

    Returns:
        The requested robot instance.

    Raises:
        RobotError: If no active robot exists and no robot_name is provided.
    """
    if robot_name is None:
        robot = GlobalSingleton()._active_robot
        if robot is None:
            raise RobotError(
                "No active robot. Call init() first or provide robot_name."
            )
    else:
        robot = get_robot(robot_name, instance)
    return robot


def validate_version() -> None:
    """Validate the Neuracore version compatibility.

    Checks if the current Neuracore client version is compatible with the
    server. This validation is performed once per session and cached.

    Raises:
        RobotError: If the Neuracore version is not compatible with the server.
    """
    if not GlobalSingleton()._has_validated_version:
        get_auth().validate_version()
        GlobalSingleton()._has_validated_version = True


def login(api_key: str | None = None) -> None:
    """Authenticate with the Neuracore server.

    Establishes authentication using an API key from the parameter, environment
    variable, or previously saved configuration. The authentication state is
    maintained for subsequent API calls.

    Args:
        api_key: API key for authentication. If not provided, will look for
            NEURACORE_API_KEY environment variable or previously saved configuration.

    Raises:
        AuthenticationError: If authentication fails due to invalid credentials
            or network issues.
        InputError: If there is an issue with the user's input.

    """
    get_auth().login(api_key)


def logout() -> None:
    """Clear authentication state and reset global session data.

    Logs out from the Neuracore server and resets all global state including
    active robots, recording IDs, dataset IDs, and version validation status.
    """
    get_auth().logout()
    GlobalSingleton()._active_robot = None
    GlobalSingleton()._active_dataset_id = None
    GlobalSingleton()._has_validated_version = False


def set_organization(id_or_name: str) -> None:
    """Set the current organization based upon its name or id.

    this value may be overridden by the `NEURACORE_ORG_ID` environment variable.

    Args:
        id_or_name: the uuid of the organization or its exact name

    Raises:
        AuthenticationError: If the user is not logged in
        OrganizationError: If there is an issue contacting the backend
        ValueError: If the id or name does not exist
    """
    orgs = list_my_orgs()

    org = next(
        (org for org in orgs if org.id == id_or_name or org.name == id_or_name), None
    )
    if not org:
        raise ValueError(f"No org found with id or name '{id_or_name}'")

    config_manager = get_config_manager()
    config_manager.config.current_org_id = org.id
    config_manager.save_config()


def connect_robot(
    robot_name: str,
    instance: int = 0,
    urdf_path: str | None = None,
    mjcf_path: str | None = None,
    overwrite: bool = False,
    shared: bool = False,
) -> Robot:
    """Initialize a robot connection and set it as the active robot.

    Creates or connects to a robot instance, validates version compatibility,
    and initializes streaming managers for live data and recording state updates.
    The robot becomes the active robot for subsequent operations.

    A robot description file (URDF or MJCF) is required for recordings to
    play back on the Neuracore dashboard. The SDK does not enforce this at
    `connect_robot` time — uploads will still succeed without one — but the
    dashboard refuses to render any recording whose robot has no description
    file ("urdf not found/provided"). Pass `urdf_path` or `mjcf_path` here,
    or upload one later, before relying on dashboard playback.

    Args:
        robot_name: Unique identifier for the robot.
        instance: Instance number of the robot for multi-instance deployments.
        urdf_path: Path to the robot's URDF file. Required for dashboard
            playback (see note above).
        mjcf_path: Path to the robot's MJCF file. This will be converted
            into URDF. Required for dashboard playback (see note above) if
            no `urdf_path` is provided.
        overwrite: Whether to overwrite an existing robot configuration
            with the same name.
        shared: Whether you want to register the robot as shared/open-source.
            Note that setting shared=True is only available to specific
            members allocated by the Neuracore team.

    Returns:
        The initialized and connected robot instance.
    """
    validate_version()
    robot = _init_robot(robot_name, instance, urdf_path, mjcf_path, overwrite, shared)
    GlobalSingleton()._active_robot = robot
    if robot.archived is True:
        warn(
            f"This robot '{robot.name}' is archived. Was this intentional?",
        )
    # Initialize push update managers
    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")
    StreamManagerOrchestrator().get_provider_manager(robot.id, robot.instance)
    get_recording_state_manager()
    return robot


def update_robot_name(
    robot_key: str,
    new_robot_name: str,
    instance: int = 0,
    shared: bool = False,
) -> str:
    """Update the robot name for a robot.

    Args:
        robot_key: Old robot name or ID of the robot to update.
        new_robot_name: New robot name to set for the robot.
        instance: Robot instance to associate with the update.
        shared: Whether the robot is shared/open-source.

    Returns:
        The resolved robot ID.
    """
    robot_id = _update_robot_name(
        robot_key,
        new_robot_name,
        instance=instance,
        shared=shared,
    )
    return robot_id


def get_training_job_logs(
    job_id: str, max_entries: int = 100, severity_filter: str | None = None
) -> dict:
    """Retrieve logs for a training job from neuracore backend.

    This convenience wrapper keeps the high-level API surface in `core.py`
    while delegating the backend call to the training API module.

    Args:
        job_id: The ID of the training job.
        max_entries: Maximum number of log entries to return.
        severity_filter: Optional log severity filter (for example: "ERROR").

    Returns:
        dict: Cloud compute logs payload.
    """
    from .training import get_training_job_logs as _get_training_job_logs

    return _get_training_job_logs(
        job_id=job_id,
        max_entries=max_entries,
        severity_filter=severity_filter,
    )


def is_recording(robot_name: str | None = None, instance: int = 0) -> bool:
    """Check if a robot is currently recording.

    Args:
        robot_name: Robot identifier. If not provided, uses the currently
            active robot from the global state.
        instance: Instance number of the robot for multi-instance scenarios.

    Returns:
        bool: True if the robot is recording, False otherwise.
    """
    robot = _get_robot(robot_name, instance)
    return robot.is_recording()


def start_recording(robot_name: str | None = None, instance: int = 0) -> None:
    """Start recording data for a specific robot.

    Begins a new recording session for the specified robot, capturing all
    configured data streams. Requires an active dataset to be set before
    starting the recording.

    Args:
        robot_name: Robot identifier. If not provided, uses the currently
            active robot from the global state.
        instance: Instance number of the robot for multi-instance scenarios.

    Raises:
        RobotError: If no robot is active and no robot_name is provided,
            if a recording is already in progress, or if no active dataset
            has been set.
    """
    robot = _get_robot(robot_name, instance)
    active_dataset_id = GlobalSingleton()._active_dataset_id
    if active_dataset_id is None:
        raise RobotError("No active dataset. Call create_dataset() first.")
    try:
        active_dataset = Dataset.get_by_id(active_dataset_id)
    except DatasetError:
        active_dataset = None
    if active_dataset is not None:
        if robot.shared and not active_dataset.is_shared:
            raise RobotError(
                "Shared robot cannot be used with a non-shared dataset. "
                "If you requested a shared dataset, creation may have failed "
                "because you are not authorized to upload shared datasets or "
                "an existing non-shared dataset with the same name was reused. "
                f"Active dataset: '{active_dataset.name}' ({active_dataset.id})."
            )
        if not robot.shared and active_dataset.is_shared:
            raise RobotError(
                "Non-shared robot cannot be used with a shared dataset. "
                "Shared datasets require shared robots. If you requested a "
                "shared robot, ensure connect_robot(shared=True) succeeded. "
                f"Active dataset: '{active_dataset.name}' ({active_dataset.id})."
            )
    robot.start_recording(active_dataset_id)


def stop_recording(
    robot_name: str | None = None,
    instance: int = 0,
    wait: bool = False,
    timeout: float | None = 300.0,
) -> None:
    """Stop recording data for a specific robot.

    Ends the current recording session for the specified robot. Optionally
    waits for all data streams to finish uploading before returning.

    Args:
        robot_name: Robot identifier. If not provided, uses the currently
            active robot from the global state.
        instance: Instance number of the robot for multi-instance scenarios.
        wait: Whether to block until all data streams have finished uploading
            to the backend storage.
        timeout: Maximum number of seconds to wait when `wait=True`. Pass
            ``None`` to wait indefinitely (matches pre-existing behavior).
            Defaults to 5 minutes — long enough for typical uploads but
            short enough that a recording that never registers traces
            surfaces a TimeoutError instead of hanging the calling script.

    Raises:
        RobotError: If no robot is active and no robot_name is provided.
        TimeoutError: If `wait=True` and traces do not register/drain within
            `timeout` seconds.
        backend_utils.RecordingNotFoundError: If `wait=True` and the backend
            has no record of the recording (org mismatch, deletion, or the
            recording was never registered server-side). Previously this was
            indistinguishable from a successful drain.
    """
    robot = _get_robot(robot_name, instance)
    if not robot.is_recording():
        warn(
            "No active recordings to stop. "
            "Your recording may have been stopped by another node."
        )
        return
    recording_id = robot.get_current_recording_id()
    if not recording_id:
        raise ValueError("Recording_id is None, no current recording")
    robot.stop_recording(recording_id, wait_for_producer_drain=wait)

    if not wait:
        return

    deadline = None if timeout is None else time.monotonic() + float(timeout)
    poll_interval_s = 0.2
    is_traces_registered = False

    while True:
        try:
            data_traces = backend_utils.get_active_data_traces(recording_id)
        except backend_utils.RecordingNotFoundError:
            # Distinguish "recording vanished" from "drained successfully".
            # Previously this collapsed into the same empty-list signal and
            # silently terminated the wait loop (or hung forever before any
            # traces registered, depending on ordering).
            raise

        if len(data_traces) > 0:
            is_traces_registered = True
        elif is_traces_registered:
            return

        if deadline is not None and time.monotonic() >= deadline:
            state = "registering" if not is_traces_registered else "draining"
            raise TimeoutError(
                f"stop_recording timed out after {timeout:.1f}s while "
                f"{state} traces for recording {recording_id!r}. The daemon "
                "may have failed to register or upload traces; see "
                "~/.neuracore/logs/daemon.log for details."
            )
        time.sleep(poll_interval_s)


def get_recording_processing_status(recording_id: str) -> dict:
    """Return the backend's view of a recording's processing state.

    Lets callers ask "is this recording actually viewable on the dashboard?"
    `stop_recording(wait=True)` only blocks until the upload pipeline reports
    `progress_reported='reported'`; the server then runs an additional
    processing/transcoding step before the video can play. Without this
    helper the only signal of post-upload failure is a perpetually-loading
    dashboard.

    Args:
        recording_id: Unique identifier for the recording.

    Returns:
        Backend response dict; commonly includes `status`, `processing_status`,
        `viewable`, and `urdf_present` fields.

    Raises:
        backend_utils.RecordingNotFoundError: If the backend has no record.
    """
    return backend_utils.get_recording_processing_status(recording_id)


def wait_until_recording_viewable(
    recording_id: str,
    timeout: float = 600.0,
    poll_interval: float = 2.0,
) -> dict:
    """Block until the backend reports the recording is fully processed.

    Polls `get_recording_processing_status` until the backend signals
    viewability. The first of the following conditions to evaluate truthy
    is treated as success:

    * `viewable` field is True
    * `processing_status` is one of {"processed", "ready", "complete"}
    * `status` is one of {"processed", "ready", "complete"}

    If the backend does not surface any of these fields the call falls back
    to polling for a stable response for two consecutive ticks, then returns
    the latest payload — better than the indistinguishable "is this done?"
    state the SDK had before.

    Args:
        recording_id: Unique identifier for the recording.
        timeout: Maximum seconds to wait.
        poll_interval: Seconds between polls.

    Returns:
        The final processing-status dict returned by the backend.

    Raises:
        TimeoutError: If the recording is not viewable within `timeout`.
        backend_utils.RecordingNotFoundError: If the recording is not found.
    """
    viewable_states = {"processed", "ready", "complete"}
    deadline = time.monotonic() + float(timeout)
    last_payload: dict = {}
    stable_ticks = 0

    while True:
        payload = backend_utils.get_recording_processing_status(recording_id)

        if payload.get("viewable") is True:
            return payload
        status_fields = (
            str(payload.get("processing_status") or "").lower(),
            str(payload.get("status") or "").lower(),
        )
        if any(value in viewable_states for value in status_fields if value):
            return payload

        if payload == last_payload:
            stable_ticks += 1
            if stable_ticks >= 2 and any(payload.get(k) for k in (
                "uploaded_trace_count", "trace_count", "viewable", "status"
            )):
                return payload
        else:
            stable_ticks = 0
            last_payload = payload

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Recording {recording_id!r} not viewable within "
                f"{timeout:.0f}s. Last status: {payload}"
            )
        time.sleep(poll_interval)


def stop_live_data(robot_name: str | None = None, instance: int = 0) -> None:
    """Stop sharing live data for active monitoring from the Neuracore platform.

    Terminates the live data streaming connection that allows real-time
    monitoring and visualization of robot data through the Neuracore platform.
    This does not affect data recording, only the live streaming capability.

    Args:
        robot_name: Robot identifier. If not provided disables streaming for all robots
        instance: Instance number of the robot for multi-instance scenarios.

    """
    if not robot_name:
        get_provide_live_data_enabled_manager().disable()
        return

    robot = _get_robot(robot_name, instance)
    if not robot.id:
        raise RobotError("Robot not initialized. Call init() first.")
    StreamManagerOrchestrator().remove_manager(robot.id, robot.instance)


def cancel_recording(robot_name: str | None = None, instance: int = 0) -> None:
    """Cancel the current recording for a specific robot without saving any data.

    Args:
        robot_name: Robot identifier.
        instance: Instance number of the robot for multi-instance scenarios.

    """
    robot = _get_robot(robot_name, instance)
    if not robot.is_recording():
        return
    recording_id = robot.get_current_recording_id()
    if not recording_id:
        return
    robot.cancel_recording(recording_id)
