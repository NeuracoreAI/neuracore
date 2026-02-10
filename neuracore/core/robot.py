"""Robot management and data stream coordination for Neuracore platform.

This module provides the Robot class for managing robot instances, their
kinematic models, data streams, and recording capabilities. It handles
URDF/MJCF model uploads, data stream management, and coordinates recording
state across multiple robot instances.
"""

import io
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from warnings import warn

import requests
from neuracore_types import RobotInstanceIdentifier

from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.streaming.data_stream import DataStream
from neuracore.core.streaming.recording_state_manager import get_recording_state_manager
from neuracore.core.utils.robot_mapping import RobotMapping
from neuracore.data_daemon.communications_management.producer import (
    RecordingContext as DaemonRecordingContext,
)
from neuracore.data_daemon.lifecycle.daemon_lifecycle import ensure_daemon_running

from .auth import Auth, get_auth
from .const import API_URL, MAX_DATA_STREAMS
from .exceptions import RobotError, ValidationError
from .utils.http_errors import extract_error_detail

logger = logging.getLogger(__name__)


@dataclass
class JointLimits:
    """Limit and capability values associated with a single joint."""

    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None


@dataclass
class JointInfo:
    """Kinematic metadata describing how a joint connects two links."""

    type: str | None = None
    parent: str | None = None
    child: str | None = None
    axis: list[float] | None = None
    limits: JointLimits = field(default_factory=JointLimits)


class Robot:
    """Represents a robot instance with kinematic model and data streaming capabilities.

    This class manages a robot's lifecycle including initialization, kinematic model
    upload, data stream management, and recording coordination. It supports both
    URDF and MJCF model formats and handles automatic conversion when needed.
    """

    def __init__(
        self,
        robot_name: str,
        instance: int,
        urdf_path: str | None = None,
        mjcf_path: str | None = None,
        overwrite: bool = False,
        shared: bool = False,
        org_id: str | None = None,
    ):
        """Initialize a Robot instance with configuration parameters.

        Args:
            robot_name: Unique identifier for the robot type.
            instance: Instance number for multi-robot deployments.
            urdf_path: Path to URDF kinematic model file.
                Mutually exclusive with mjcf_path.
            mjcf_path: Path to MJCF kinematic model file.
                Mutually exclusive with urdf_path.
            overwrite: Whether to overwrite existing robot configuration on server.
            shared: Whether the robot is shared/open-source.
                Note that setting shared=True is only available to specific
                members allocated by the Neuracore team.
            org_id: the organization to receive streaming information from. If n
                ot provided defaults to the current org.

        Raises:
            ValidationError: If both URDF and MJCF paths are provided,
                if files don't exist, or if file extensions are incorrect.
            ImportError: If MJCF conversion is requested but mujoco is not available.
        """
        self.name = robot_name
        self.instance = instance
        self.urdf_path = urdf_path
        self.mjcf_path = mjcf_path
        self.overwrite = overwrite
        self.shared = shared
        self.id: str | None = None
        self.archived: bool | None = None
        self._auth: Auth = get_auth()
        self._temp_dir = None
        self._data_streams: dict[str, DataStream] = dict()

        self.org_id = org_id or get_current_org()

        if urdf_path and mjcf_path:
            raise ValidationError(
                "Only one of urdf_path or mjcf_path should be provided."
            )
        if urdf_path:
            if not os.path.isfile(urdf_path):
                raise ValidationError(f"URDF file not found: {urdf_path}")
            if not urdf_path.lower().endswith(".urdf"):
                raise ValidationError("URDF file must have .urdf extension.")
        elif mjcf_path:
            mjcf_abs_path = Path(mjcf_path).expanduser().resolve()

            if mjcf_abs_path.suffix.lower() != ".xml":
                raise ValidationError(
                    "MJCF file must have a .xml extension.\n"
                    f"Provided path: {mjcf_abs_path}"
                )

            if not mjcf_abs_path.is_file():
                raise ValidationError(
                    "MJCF file not found.\n"
                    f"Expected path: {mjcf_abs_path}\n"
                    f"Working directory: {Path.cwd()}"
                )

            # Import conversion dependency with a helpful error
            try:
                from .mjcf_to_urdf import convert
            except ImportError as e:
                raise ImportError(
                    "MJCF to URDF conversion requires MuJoCo support.\n"
                    "Install the required extra/dependency (e.g., 'mujoco') and retry."
                ) from e
            self._temp_dir = tempfile.TemporaryDirectory(prefix="neuracore")
            self.urdf_path = os.path.join(self._temp_dir.name, "model.urdf")
            convert(mjcf_path, Path(self.urdf_path), asset_file_prefix="meshes/")

        self.joint_info: dict[str, JointInfo] = {}
        if self.urdf_path:
            self.joint_info = self._get_joint_info()

    def init(self) -> None:
        """Initialize the robot on the Neuracore server.

        Creates the robot instance on the server and uploads the kinematic model
        if provided. This must be called before using the robot for data streaming
        or recording.

        Raises:
            RobotError: If not authenticated or if server communication fails.
            ConfigError: If there is an error trying to get the current org
        """
        if not self._auth.is_authenticated:
            raise RobotError("Not authenticated. Please call nc.login() first.")

        if not self.org_id:
            raise RobotError(
                "Unauthorised: no organisation selected. "
                "Run `neuracore select-org` and try again."
            )

        try:
            response = requests.post(
                f"{API_URL}/org/{self.org_id}/robots?is_shared={self.shared}",
                json={
                    "name": self.name,
                    "instance": self.instance,
                },  # TODO: Add camera support
                headers=self._auth.get_headers(),
            )
            response.raise_for_status()
            response_body = response.json()
            self.id = response_body["robot_id"]
            self.archived = response_body.get("archived")
            has_urdf = response_body["has_urdf"]
            # Upload URDF and meshes if provided
            if self.urdf_path and (not has_urdf or self.overwrite):
                self._upload_urdf_and_meshes()
                if self._temp_dir:
                    self._temp_dir.cleanup()
        except requests.exceptions.ConnectionError:
            raise RobotError(
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            )
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                raise RobotError(
                    "Unauthorised: no organisation selected. "
                    "Run `neuracore select-org` and try again."
                )
            raise RobotError(f"Failed to initialize robot: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise RobotError(f"Failed to initialize robot: {str(e)}")

    def add_data_stream(self, stream_id: str, stream: DataStream) -> None:
        """Add a data stream to the robot for sensor data collection.

        Args:
            stream_id: Unique identifier for the data stream.
            stream: DataStream instance for handling the sensor data.

        Raises:
            RuntimeError: If the maximum number of data streams is exceeded.
            ValueError: If a stream with the same ID already exists.
        """
        if len(self._data_streams) >= MAX_DATA_STREAMS:
            raise RuntimeError("Excessive number of data streams")
        if stream_id in self._data_streams:
            raise ValueError("Stream already exists")
        self._data_streams[stream_id] = stream

    def get_data_stream(self, stream_id: str) -> DataStream | None:
        """Retrieve a data stream by its identifier.

        Args:
            stream_id: Unique identifier for the data stream.

        Returns:
            The DataStream instance if found, None otherwise.
        """
        return self._data_streams.get(stream_id, None)

    def list_all_streams(self) -> dict[str, DataStream]:
        """List all data streams registered with this robot.

        Returns:
            Dictionary mapping stream IDs to DataStream instances.
        """
        return self._data_streams

    def start_recording(self, dataset_id: str) -> str:
        """Start recording data from all active streams to a dataset.

        Initiates a recording session that will capture data from all registered
        data streams and associate it with the specified dataset.

        Args:
            dataset_id: Unique identifier of the dataset to record into.

        Returns:
            The unique recording ID for this recording session.

        Raises:
            RobotError: If the robot is not initialized or if
                the recording fails to start.
            ConfigError: If there is an error trying to get the current org
        """
        if not self.id:
            raise RobotError("Robot not initialized. Call init() first.")

        # Start daemon process
        ensure_daemon_running()

        try:

            response = requests.post(
                f"{API_URL}/org/{self.org_id}/recording/start",
                headers=self._auth.get_headers(),
                json={
                    "robot_id": self.id,
                    "instance": self.instance,
                    "dataset_id": dataset_id,
                    "upload_method": "data_daemon",
                },
            )
            response.raise_for_status()
            # Inform the state manager immediately to skip the round trip.

            recording_details = response.json()
            recording_id = recording_details["id"]
            assert isinstance(recording_id, str)

            if "start_time" in recording_details:
                warn("This recording had already been started!")

            get_recording_state_manager().recording_started(
                robot_id=self.id, instance=self.instance, recording_id=recording_id
            )
            return recording_id
        except requests.exceptions.ConnectionError:
            raise RobotError(
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            )
        except requests.exceptions.RequestException as e:
            detail = None
            if e.response is not None:
                detail = extract_error_detail(e.response)
            raise RobotError(f"Failed to start recording: {detail or str(e)}")

    def stop_recording(self, recording_id: str) -> None:
        """Stop an active recording session.

        Ends the specified recording session and stops data collection from
        all streams. The recorded data will be processed and stored in the
        associated dataset.

        Args:
            recording_id: Unique identifier of the recording session to stop.

        Raises:
            RobotError: If the robot is not initialized, if the recording cannot
                be stopped, or if storage limits are exceeded.
            ConfigError: If there is an error trying to get the current org
        """
        if not self.id:
            raise RobotError("Robot not initialized. Call init() first.")

        DaemonRecordingContext(recording_id=recording_id).stop_recording()
        self._stop_all_streams()

        try:
            response = requests.post(
                f"{API_URL}/org/{self.org_id}/recording/stop?recording_id={recording_id}",
                headers=self._auth.get_headers(),
            )

            response.raise_for_status()

            if response.json() == "WrongUser":
                raise RobotError("Cannot stop recording initiated by another user")

            if response.json() == "UsageLimitExceeded":
                raise RobotError("Storage limit exceeded. Please upgrade your plan.")

            get_recording_state_manager().recording_stopped(
                robot_id=self.id, instance=self.instance, recording_id=recording_id
            )
        except requests.exceptions.ConnectionError:
            raise RobotError(
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            )
        except requests.exceptions.RequestException as e:
            raise RobotError(f"Failed to stop recording: {str(e)}")

    def _stop_all_streams(self) -> None:
        """Stop recording on all data streams for this robot instance."""
        for stream_id, stream in self._data_streams.items():
            try:
                stream.stop_recording()
            except Exception:
                logger.exception("Failed to stop data stream %s", stream_id)

    def is_recording(self) -> bool:
        """Check if the robot is currently recording data.

        Returns:
            True if the robot is actively recording, False otherwise.
        """
        if not self.id:
            raise RobotError("Robot not initialized. Call init() first.")
        return get_recording_state_manager().is_recording(
            robot_id=self.id, instance=self.instance
        )

    def get_current_recording_id(self) -> str | None:
        """Get the ID of the current active recording session.

        Returns:
            The current recording ID if the robot is recording, None otherwise.
        """
        if not self.id:
            raise RobotError("Robot not initialized. Call init() first.")
        return get_recording_state_manager().get_current_recording_id(
            robot_id=self.id, instance=self.instance
        )

    def _package_urdf(self) -> dict:
        """Package URDF file and associated meshes into a ZIP archive.

        Creates a ZIP package containing the URDF file and all referenced mesh
        files, updating mesh paths to use a standardized directory structure.

        Returns:
            Dictionary containing the ZIP file data formatted for HTTP upload.

        Raises:
            ValidationError: If the URDF file is not found.
            RobotError: If mesh files cannot be located or if package creation fails.
        """
        if not self.urdf_path:
            raise ValueError("urdf path is None")
        if not os.path.exists(self.urdf_path):
            raise ValidationError(f"URDF file not found: {self.urdf_path}")

        # Read and parse URDF to find all mesh files
        with open(self.urdf_path) as f:
            urdf_content = f.read()

        root = ET.fromstring(urdf_content)
        urdf_dir = os.path.dirname(os.path.abspath(self.urdf_path))
        mesh_files: list[str] = []
        package_root_path = None

        # Collect all mesh files
        for mesh in root.findall(".//mesh"):
            filename = mesh.get("filename")
            if filename:
                mesh_path = None
                if filename.startswith("package://"):
                    # Handle package:// URLs
                    parts = filename.split("/")
                    package_name = parts[2]
                    relative_path = "/".join(parts[3:])

                    if package_root_path is None:
                        # Go up the tree until we find package dir
                        package_root_path = urdf_dir
                        while not os.path.exists(
                            os.path.join(package_root_path, package_name)
                        ):
                            parent = os.path.dirname(package_root_path)
                            if parent == package_root_path:  # Hit root directory
                                raise RobotError(
                                    f"Could not find package root for {package_name}"
                                )
                            package_root_path = parent

                    mesh_path = os.path.join(
                        package_root_path, package_name, relative_path
                    )
                    # Update the filename in the URDF to point to the new location
                    mesh.set(
                        "filename", os.path.join("meshes", os.path.basename(mesh_path))
                    )
                else:
                    # Handle relative paths
                    mesh_path = os.path.join(urdf_dir, filename)
                    if not os.path.exists(mesh_path):
                        # Go up one level and try again
                        mesh_path = os.path.join(urdf_dir, "..", filename)
                        if not os.path.exists(mesh_path):
                            raise RobotError(f"Mesh file not found: {mesh_path}")
                    # Update the filename to point to meshes folder
                    mesh.set(
                        "filename", os.path.join("meshes", os.path.basename(mesh_path))
                    )

                if mesh_path and mesh_path not in mesh_files:
                    if os.path.exists(mesh_path):
                        mesh_files.append(mesh_path)
                    else:
                        raise RobotError(f"Mesh file not found: {mesh_path}")

        # Get the modified URDF content
        updated_urdf_content = ET.tostring(root, encoding="unicode")

        # Create ZIP file in memory using BytesIO
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add URDF file with updated mesh paths
            zf.writestr("robot.urdf", updated_urdf_content)

            # Add mesh files in the meshes directory
            for mesh_path in mesh_files:
                zf.write(mesh_path, os.path.join("meshes", os.path.basename(mesh_path)))

        # Get the zip data
        zip_buffer.seek(0)
        zip_data = zip_buffer.getvalue()

        # Create the files dict with the ZIP data
        return {"robot_package": ("robot_package.zip", zip_data, "application/zip")}

    def _load_urdf(self) -> ET.Element:
        """Load the URDF contents either from disk or by downloading the package.

        Returns:
            Parsed URDF XML root element.

        Raises:
            RobotError: If no URDF is available or if fetching/parsing fails.
        """
        if self.urdf_path:
            try:
                return ET.parse(self.urdf_path).getroot()
            except FileNotFoundError:
                logger.debug(
                    "URDF path %s not found locally, attempting to fetch from server.",
                    self.urdf_path,
                )
            except ET.ParseError as e:
                raise RobotError(f"Failed to parse URDF: {str(e)}") from e

        if not self.id:
            raise RobotError(
                "Robot not initialized and no local URDF available. "
                "Provide urdf_path or call init() first."
            )
        if not self.org_id:
            raise RobotError(
                "Unauthorised: no organisation selected. "
                "Run `nc-select-org` and try again."
            )

        try:
            response = requests.get(
                f"{API_URL}/org/{self.org_id}/robots/{self.id}/package?is_shared={self.shared}",
                headers=self._auth.get_headers(),
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RobotError(
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            )
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise RobotError(
                    "No URDF available for this robot. "
                    "Upload a URDF package or provide urdf_path."
                ) from e
            raise RobotError(f"Failed to download URDF package: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise RobotError(f"Failed to download URDF package: {str(e)}")

        try:
            with zipfile.ZipFile(io.BytesIO(response.content), "r") as zf:
                if "robot.urdf" not in zf.namelist():
                    raise RobotError("URDF package does not contain robot.urdf")
                try:
                    return ET.fromstring(zf.read("robot.urdf"))
                except ET.ParseError as e:
                    raise RobotError(f"Failed to parse URDF: {str(e)}") from e
        except zipfile.BadZipFile as e:
            raise RobotError("Invalid URDF package received from server.") from e

    def _upload_urdf_and_meshes(self) -> None:
        """Upload URDF and associated mesh files as a ZIP package to the server.

        Packages the robot's kinematic model and visual assets into a ZIP archive
        and uploads it to the Neuracore platform for use in visualization and
        simulation.

        Raises:
            RobotError: If packaging or upload fails.
            ConfigError: If there is an error trying to get the current org
        """
        try:
            # Create the files dict with the ZIP data
            files = self._package_urdf()

            # Upload the package
            response = requests.put(
                f"{API_URL}/org/{self.org_id}/robots/{self.id}/package?is_shared={self.shared}",
                headers=self._auth.get_headers(),
                files=files,
            )

            # Log response for debugging
            logger.info(f"Upload response status: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Upload error response: {response.text}", exc_info=True)

            response.raise_for_status()

            logger.info(f"Successfully uploaded URDF package for robot {self.id}")
        except requests.exceptions.ConnectionError:
            raise RobotError(
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            )
        except requests.exceptions.RequestException as e:
            raise RobotError(f"Failed to upload URDF package: {str(e)}")
        except Exception as e:
            raise RobotError(f"Error preparing URDF package: {str(e)}")

    def _get_joint_info(self) -> dict[str, JointInfo]:
        """Return metadata for all joints defined in the robot URDF.

        Returns:
            Mapping of joint name to metadata including type, parent/child links,
            axis, and joint limits.

        Raises:
            RobotError: If no URDF is available or parsing fails.
        """
        root = self._load_urdf()

        joint_info: dict[str, JointInfo] = {}
        for joint in root.iter("joint"):
            name = joint.get("name")
            joint_type = joint.get("type")
            if not name or not joint_type:
                continue
            info = JointInfo(type=joint_type)
            limits = info.limits

            parent = joint.find("parent")
            if parent is not None:
                info.parent = parent.get("link")

            child = joint.find("child")
            if child is not None:
                info.child = child.get("link")

            axis = joint.find("axis")
            axis_xyz = axis.get("xyz") if axis is not None else None
            if axis_xyz:
                try:
                    info.axis = [float(val) for val in axis_xyz.split()]
                except ValueError:
                    logger.warning("Could not parse axis for joint %s", name)

            limit = joint.find("limit")
            if limit is not None:
                for key in ("lower", "upper", "effort", "velocity"):
                    if key in limit.attrib:
                        try:
                            setattr(limits, key, float(limit.attrib[key]))
                        except ValueError:
                            logger.warning(
                                "Could not parse limit '%s' for joint %s", key, name
                            )

            joint_info[name] = info

        return joint_info

    def cancel_recording(self, recording_id: str) -> None:
        """Cancel an active recording without saving any data.

        Args:
            recording_id: the ID of the recording to cancel.
        """
        if not self.id:
            raise RobotError("Robot not initialized. Call init() first.")

        DaemonRecordingContext(recording_id=recording_id).stop_recording()

        try:
            response = requests.post(
                f"{API_URL}/org/{self.org_id}/recording/cancel?recording_id={recording_id}",
                headers=self._auth.get_headers(),
            )
            response.raise_for_status()

            if response.json() == "WrongUser":
                raise RobotError("Cannot cancel recording initiated by another user")

            get_recording_state_manager().recording_stopped(
                robot_id=self.id, instance=self.instance, recording_id=recording_id
            )
        except requests.exceptions.ConnectionError:
            raise RobotError(
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            )
        except requests.exceptions.RequestException as e:
            raise RobotError(f"Failed to cancel recording: {str(e)}")


# Global robot registry
_robots: dict[RobotInstanceIdentifier, Robot] = {}
_robot_name_id_mapping: dict[str, str] = {}


def init(
    robot_name: str,
    instance: int,
    urdf_path: str | None = None,
    mjcf_path: str | None = None,
    overwrite: bool = False,
    shared: bool = False,
) -> Robot:
    """Initialize a robot and register it globally.

    Creates a new Robot instance, initializes it on the server, and registers
    it in the global robot registry for future access.

    Args:
        robot_name: Unique identifier for the robot type.
        instance: Instance number for multi-robot deployments.
        urdf_path: Path to URDF kinematic model file.
        mjcf_path: Path to MJCF kinematic model file.
        overwrite: Whether to overwrite existing robot configuration.
        shared: Whether the robot is shared/open-source.
            Note that setting shared=True is only available to specific
            members allocated by the Neuracore team.

    Returns:
        The initialized Robot instance.
    """
    if not robot_name:
        raise ValueError("Robot name cannot be empty")
    robot = Robot(robot_name, instance, urdf_path, mjcf_path, overwrite, shared)
    robot.init()
    if not robot.id:
        raise RobotError("Robot not initialized. Call init() first.")
    _robot_name_id_mapping[robot_name] = robot.id
    _robots[RobotInstanceIdentifier(robot_id=robot.id, robot_instance=instance)] = robot
    return robot


def get_robot(robot_name: str, instance: int) -> Robot:
    """Retrieve a registered robot instance by name and instance number.

    Args:
        robot_name: Name or ID of the robot to retrieve.
        instance: Instance number of the robot.

    Returns:
        The Robot instance if found.

    Raises:
        RobotError: If the robot is not found in the registry.
    """
    robot_id = _robot_name_id_mapping.get(robot_name, robot_name)
    key = RobotInstanceIdentifier(robot_id=robot_id, robot_instance=instance)
    if key not in _robots:
        raise RobotError(
            f"Robot {robot_name}:{instance} not initialized. Call init() first."
        )
    return _robots[key]


def _update_local_robot_name_cache(robot_id: str, new_robot_name: str) -> None:
    """Update in-memory robot name mappings for a renamed robot."""
    for name, rid in list(_robot_name_id_mapping.items()):
        if rid == robot_id and name != new_robot_name:
            _robot_name_id_mapping.pop(name, None)
    _robot_name_id_mapping[new_robot_name] = robot_id
    for robot in _robots.values():
        if robot.id == robot_id:
            robot.name = new_robot_name


def update_robot_name(
    robot_key: str,
    new_robot_name: str,
    instance: int = 0,
    shared: bool = False,
) -> str:
    """Update a robot's name on the server.

    Args:
        robot_key: Robot old name or ID to update.
        new_robot_name: New display name to set.
        instance: Robot instance to associate with the update.
        shared: Whether the robot is shared/open-source.

    Returns:
        The resolved robot ID.

    Raises:
        RobotError: If the update request fails.
        ValueError: If the robot key or new name is empty.
    """
    if not robot_key:
        raise ValueError("Robot old name or ID cannot be empty.")
    if not new_robot_name:
        raise ValueError("New robot name cannot be empty.")
    if not get_auth().is_authenticated:
        raise RobotError("Not authenticated. Please call nc.login() first.")

    resolved_org_id = get_current_org()
    if not resolved_org_id:
        raise RobotError("No organization selected. Please call nc.select_org() first.")
    robot_id = _robot_name_id_mapping.get(robot_key, robot_key)
    try:
        response = requests.post(
            f"{API_URL}/org/{resolved_org_id}/robots?is_shared={shared}&robot_id={robot_id}",
            json={"name": new_robot_name, "instance": instance},
            headers=get_auth().get_headers(),
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RobotError(
            "Failed to connect to neuracore server, "
            "please check your internet connection and try again."
        )
    except requests.exceptions.RequestException as e:
        raise RobotError(f"Failed to update robot name: {str(e)}")
    _update_local_robot_name_cache(robot_id, new_robot_name)
    RobotMapping.for_org(resolved_org_id, is_shared=shared).refresh()
    return robot_id


def list_organization_robots(
    org_id: str, is_shared: bool = False, mode: str = "current"
) -> list[dict]:
    """List all robots in an organization.

    Args:
        org_id: Organization ID
        is_shared: Whether to list shared robots
        mode: Robot list mode ("current", "archived", or "mixed")
    """
    if not get_auth().is_authenticated:
        raise RobotError("Not authenticated. Please call nc.login() first.")
    if not org_id:
        raise RobotError("No organization selected. Please call nc.select_org() first.")
    if mode not in ["current", "archived", "mixed"]:
        raise RobotError(
            "Invalid robot list mode. Please use 'current', 'archived', or 'mixed'."
        )
    try:
        response = requests.get(
            f"{API_URL}/org/{org_id}/robots?is_shared={is_shared}&mode={mode}",
            headers=get_auth().get_headers(),
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise RobotError(
            "Failed to connect to neuracore server, "
            "please check your internet connection and try again."
        )
    except requests.exceptions.RequestException as e:
        raise RobotError(f"Failed to list robots: {str(e)}")
