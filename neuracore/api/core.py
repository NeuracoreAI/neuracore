"""High-level API for Neuracore robot management and data recording.

This module provides the main public interface for connecting to robots,
managing authentication, controlling data recording sessions, and handling
live data streaming. It maintains global state for active robots and
recording sessions.
"""

import logging
import threading
import time
from warnings import warn

from neuracore_types import normalize_user_tag_name

from neuracore.core.config.config_manager import get_config_manager
from neuracore.core.organizations import list_my_orgs
from neuracore.core.streaming.p2p.provider.global_live_data_enabled import (
    get_provide_live_data_enabled_manager,
)
from neuracore.core.streaming.p2p.stream_manager_orchestrator import (
    StreamManagerOrchestrator,
)
from neuracore.core.utils import backend_utils
from neuracore.data_daemon import bridge
from neuracore.data_daemon.daemon_control import ensure_daemon_running

from ..core.auth import get_auth
from ..core.exceptions import DatasetError, RobotError, ServerUnreachableError
from ..core.robot import Robot, get_robot
from ..core.robot import init as _init_robot
from ..core.robot import update_robot_name as _update_robot_name
from .globals import GlobalSingleton

logger = logging.getLogger(__name__)

_DEFAULT_STOP_RECORDING_WAIT_TIMEOUT_S = 60 * 15
_RECORDING_UPLOAD_POLL_INTERVAL_S = 0.2
# How long a connected robot waits for the server to register it before its
# live data streaming is given up, and how often it asks the daemon.
LIVE_DATA_WAIT_S = 10 * 60.0
LIVE_DATA_POLL_S = 1.0


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

    If the server cannot be reached, logs in offline with a warning instead,
    and authenticated requests to the server raise ``ServerUnreachableError``.

    Args:
        api_key: API key for authentication. If not provided, will look for
            NEURACORE_API_KEY environment variable or previously saved configuration.

    Raises:
        AuthenticationError: If authentication fails due to invalid credentials
            or server errors.
        InputError: If there is an issue with the user's input.

    """
    get_auth().login(api_key)


def logout() -> None:
    """Clear authentication state and reset global session data.

    Logs out from the Neuracore server and resets all global state including
    active robots, recording IDs, dataset IDs, and version validation status.
    """
    try:
        get_auth().logout()
    finally:
        GlobalSingleton()._active_robot = None
        GlobalSingleton()._active_dataset = None
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
    wait: bool = False,
) -> Robot:
    """Initialize a robot connection and set it as the active robot.

    Connects the robot instance through the data daemon, online or offline:
    the daemon has the server register it, and upload its description if the
    server needs it, in the background, so this returns straight away. The
    robot's server id is fetched when it is first read, which waits for the
    server; live data streaming starts once it is known. The robot becomes the
    active robot for subsequent operations.

    Upload of a robot description file (URDF or MJCF) is not required,
    but it is recommended for better visualization within Neuracore.

    Args:
        robot_name: Unique identifier for the robot.
        instance: Instance number of the robot for multi-instance deployments.
        urdf_path: Path to the robot's URDF file.
        mjcf_path: Path to the robot's MJCF file. This will be converted
            into URDF.
        overwrite: Whether to overwrite an existing robot configuration
            with the same name.
        shared: Whether you want to register the robot as shared/open-source.
            Note that setting shared=True is only available to specific
            members allocated by the Neuracore team.
        wait: Wait for the server to register the robot before returning, so
            a refusal raises here.

    Returns:
        The initialized and connected robot instance.

    Raises:
        RobotError: If ``wait`` and the server refused the robot, or the robot
            could not be initialized.
        RobotArchivedError: If ``wait`` and the robot is archived.
        ServerUnreachableError: If ``wait`` and the last login could not reach
            the server.
        ValidationError: If the model paths are not valid or if a URDF visual
            mesh has an unsupported format.
        OrganizationError: If offline and no organization is saved.
        DaemonLifecycleError: If the data daemon could not be started.
    """
    # Only the daemon hears a recording started from the web.
    ensure_daemon_running()
    robot = _init_robot(
        robot_name, instance, urdf_path, mjcf_path, overwrite, shared, wait=wait
    )
    GlobalSingleton()._active_robot = robot
    if robot.known_id is not None:
        StreamManagerOrchestrator().get_provider_manager(robot.known_id, robot.instance)
    else:
        threading.Thread(
            target=_start_live_data_once_registered,
            args=(robot,),
            name=f"neuracore-live-data-{robot_name}-{instance}",
            daemon=True,
        ).start()
    return robot


def _start_live_data_once_registered(robot: Robot) -> None:
    """Start the robot's live data streaming once the server registers it.

    Runs off the caller's thread and never raises. It waits up to
    :data:`LIVE_DATA_WAIT_S`, through holds the user can fix such as an archived
    robot, and gives up when the server refuses the robot for good, when this
    process's login was offline (it cannot stream), or when the time is up.
    """
    deadline = time.monotonic() + LIVE_DATA_WAIT_S
    try:
        while time.monotonic() < deadline:
            if get_auth().is_offline or robot._robot_index is None:
                return
            state = bridge.robot_row_state(robot._robot_index)
            if state is None or state.status == "failed":
                return
            if state.status == "done" and state.cloud_id:
                StreamManagerOrchestrator().get_provider_manager(
                    robot.id, robot.instance
                )
                return
            time.sleep(LIVE_DATA_POLL_S)
        logger.info(
            "No live data for robot %s: the server did not register it within "
            "%.0f minutes.",
            robot.name,
            LIVE_DATA_WAIT_S / 60,
        )
    except Exception as error:  # noqa: BLE001 - never raised to the user
        logger.debug("No live data for robot %s: %s", robot.name, error)


def update_robot_name(
    robot_key: str,
    new_robot_name: str,
    shared: bool = False,
) -> str:
    """Update the robot name for a robot.

    Args:
        robot_key: Robot ID, or name of a robot connected in this session.
        new_robot_name: New robot name to set for the robot.
        shared: Whether the robot is shared/open-source.

    Returns:
        The resolved robot ID.

    Raises:
        RobotError: If the robot is not found or the rename fails.
    """
    robot_id = _update_robot_name(
        robot_key,
        new_robot_name,
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


def start_recording(
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
) -> None:
    """Start recording data for a specific robot.

    Begins a new recording session for the specified robot, capturing all
    configured data streams. Requires an active dataset to be set before
    starting the recording.

    Args:
        robot_name: Robot identifier. If not provided, uses the currently
            active robot from the global state.
        instance: Instance number of the robot for multi-instance scenarios.
        timestamp: Optional capture time (Unix seconds) for the recording's
            start, matching the ``log_*`` methods. When omitted, the current
            time is captured.

    Raises:
        RobotError: If no robot is active and no robot_name is provided,
            if a recording is already in progress, if no active dataset
            has been set, or if the data daemon no longer keeps the robot.
        DatasetError: If the data daemon no longer keeps the dataset.
    """
    robot = _get_robot(robot_name, instance)
    active_dataset = GlobalSingleton()._active_dataset
    if active_dataset is None:
        raise RobotError("No active dataset. Call create_dataset() first.")
    if active_dataset.dataset_index is None:
        # Read from the server (by name online, by id, a merge or a clone):
        # recordings point at a daemon row, so the daemon is given the server
        # id it already has, for a row the server has confirmed.
        active_dataset.dataset_index = bridge.dataset_row_for_cloud_id(
            org_id=active_dataset.org_id, cloud_dataset_id=active_dataset.id
        )
    if active_dataset.org_id != robot.org_id:
        raise RobotError(
            f"Robot '{robot.name}' and dataset '{active_dataset.name}' are in "
            "different organizations. Connect the robot and select the dataset "
            "in the same organization."
        )
    if active_dataset.kind_known:
        if robot.shared and not active_dataset.is_shared:
            raise RobotError(
                "Shared robot cannot be used with a non-shared dataset. "
                "If you requested a shared dataset, creation may have failed "
                "because you are not authorized to upload shared datasets or "
                "an existing non-shared dataset with the same name was reused. "
                f"Active dataset: '{active_dataset.name}'."
            )
        if not robot.shared and active_dataset.is_shared:
            raise RobotError(
                "Non-shared robot cannot be used with a shared dataset. "
                "Shared datasets require shared robots. If you requested a "
                "shared robot, ensure connect_robot(shared=True) succeeded. "
                f"Active dataset: '{active_dataset.name}'."
            )
    assert active_dataset.dataset_index is not None
    _require_daemon_rows(robot, active_dataset.dataset_index, active_dataset.name)
    robot.start_recording(active_dataset.dataset_index, timestamp=timestamp)


def _require_daemon_rows(robot: Robot, dataset_index: int, dataset_name: str) -> None:
    """Raise if the data daemon no longer keeps the robot's or dataset's row.

    The daemon deletes a robot or dataset nothing has used for 7 days, and
    ``neuracore data-daemon reset`` deletes them all; a script that held on to
    one meanwhile must connect or select it again.
    """
    # The recording needs the daemon anyway; one that stopped is started
    # again first, so the check below is answered at once.
    ensure_daemon_running()
    try:
        robot_gone = (
            robot._robot_index is not None
            and bridge.robot_row_state(robot._robot_index) is None
        )
        dataset_gone = bridge.dataset_row_state(dataset_index) is None
    except (bridge.RegistryUnavailableError, bridge.RegistryRefusedError):
        # Only a missing row stops the recording: one the daemon could not
        # read is left to the recording itself.
        return
    if robot_gone:
        raise RobotError(
            f"The data daemon no longer keeps robot '{robot.name}' instance "
            f"{robot.instance}: it was not used for 7 days, or the daemon was "
            "reset. Connect it again with nc.connect_robot."
        )
    if dataset_gone:
        raise DatasetError(
            f"The data daemon no longer keeps dataset '{dataset_name}': it was "
            "not used for 7 days, or the daemon was reset. Select it again with "
            "nc.get_dataset or nc.create_dataset."
        )


def _normalize_stop_tags(tags: list[str] | None) -> list[str]:
    """Return the tag names normalized, each once, or raise ValueError."""
    if isinstance(tags, str):
        raise ValueError("tags must be a list of tag names, not a string")
    for name in tags or []:
        if not isinstance(name, str):
            raise ValueError(f"Tag name must be a string, not {type(name).__name__}")
    return list(dict.fromkeys(normalize_user_tag_name(name) for name in tags or []))


def stop_recording(
    robot_name: str | None = None,
    instance: int = 0,
    wait: bool = False,
    timestamp: float | None = None,
    wait_timeout_s: float = _DEFAULT_STOP_RECORDING_WAIT_TIMEOUT_S,
    tags: list[str] | None = None,
) -> None:
    """Stop recording data for a specific robot.

    Ends the current recording session for the specified robot. Optionally
    waits for all data streams to finish uploading before returning.

    Args:
        robot_name: Robot identifier. If not provided, uses the currently
            active robot from the global state.
        instance: Instance number of the robot for multi-instance scenarios.
        wait: Whether to block until all data streams have finished uploading
            to the backend storage. Offline, the recording is stopped and kept
            on this machine to upload later, and nothing is waited for.
        timestamp: Optional capture time (Unix seconds) for the recording's
            stop, matching the ``log_*`` methods. When omitted, the current
            time is captured.
        wait_timeout_s: Maximum number of seconds to wait for recording uploads
            to complete when ``wait=True``.
        tags: Optional user tag names to add to the recording. Names are
            normalized like ``recording.add_tag``, and a name given twice gives
            one tag. The tags reach the recording even when it was made offline.

    Raises:
        RobotError: If no robot is active and no robot_name is provided.
        ValueError: If a tag name is not valid. The recording is not stopped.
        ServerUnreachableError: If ``wait`` is set while offline, once the
            recording is stopped.
        TimeoutError: If the uploads do not complete within ``wait_timeout_s``.
    """
    stop_tags = _normalize_stop_tags(tags)
    robot = _get_robot(robot_name, instance)
    if not robot.is_recording():
        warn(
            "No active recordings to stop. "
            "Your recording may have been stopped by another node."
        )
        return
    if wait_timeout_s < 0:
        raise ValueError("wait_timeout_s must be non-negative")

    if wait and get_auth().is_offline:
        robot.stop_recording(timestamp=timestamp, tags=stop_tags)
        raise ServerUnreachableError(
            "The recording is stopped and stored on this machine. It uploads once "
            "the Neuracore server can be reached, so it cannot be waited for "
            "while offline."
        )
    if wait:
        if timestamp is None:
            # The recording stops when asked, not after the waits below.
            timestamp = time.time()
        try:
            # The server starts the recording only once it has confirmed the
            # robot and the dataset, so wait for both before waiting for the
            # recording's id. A refusal raises here.
            robot.id  # noqa: B018 - reading it waits for the server
            dataset = GlobalSingleton()._active_dataset
            if dataset is not None and dataset.dataset_index is not None:
                dataset.id  # noqa: B018 - reading it waits for the server
        except Exception:
            robot.stop_recording(timestamp=timestamp, tags=stop_tags)
            raise
    wait_deadline = time.monotonic() + wait_timeout_s
    cloud_recording_id = robot.get_cloud_recording_id() if wait else None
    robot.stop_recording(timestamp=timestamp, tags=stop_tags)
    if not wait or not cloud_recording_id:
        return
    recording_id = cloud_recording_id

    while time.monotonic() < wait_deadline:
        if backend_utils.is_recording_upload_complete(recording_id):
            return

        remaining_s = wait_deadline - time.monotonic()
        if remaining_s <= 0:
            break

        time.sleep(min(_RECORDING_UPLOAD_POLL_INTERVAL_S, remaining_s))

    raise TimeoutError(
        "Timed out waiting for recording uploads to complete "
        f"for recording '{recording_id}'"
    )


def get_cloud_recording_id(
    robot_name: str | None = None,
    instance: int = 0,
    timeout_s: float = 30.0,
) -> str | None:
    """Resolve the daemon-owned cloud recording id for a robot's open recording.

    The cloud recording id is assigned asynchronously by the daemon, and is
    resolvable only while the recording is open.

    Args:
        robot_name: Robot identifier. Defaults to the active robot.
        instance: Robot instance number.
        timeout_s: Maximum time to wait for the daemon to mint the id.

    Returns:
        The cloud recording id, or ``None`` when the robot is not recording or
        the id is not minted within ``timeout_s``.
    """
    robot = _get_robot(robot_name, instance)
    return robot.get_cloud_recording_id(timeout_s=timeout_s)


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
    if robot.known_id is None:
        # Never registered, so its live data never started.
        return
    StreamManagerOrchestrator().remove_manager(robot.known_id, robot.instance)


def cancel_recording(
    robot_name: str | None = None,
    instance: int = 0,
    timestamp: float | None = None,
) -> None:
    """Cancel the current recording for a specific robot without saving any data.

    Args:
        robot_name: Robot identifier.
        instance: Instance number of the robot for multi-instance scenarios.
        timestamp: Optional capture time (Unix seconds) for the cancel,
            mirroring ``stop_recording``. When omitted, the current time is
            captured.

    """
    robot = _get_robot(robot_name, instance)
    if not robot.is_recording():
        return
    robot.cancel_recording(timestamp=timestamp)
