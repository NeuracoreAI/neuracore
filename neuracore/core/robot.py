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
from pathlib import Path
from threading import Lock
from typing import Optional
from warnings import warn

import requests
from neuracore_types import RobotInstanceIdentifier

from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.streaming.data_stream import DataStream
from neuracore.core.streaming.recording_state_manager import (
    RecordingStateManager,
    get_recording_state_manager,
)

from .auth import Auth, get_auth
from .const import API_URL, MAX_DATA_STREAMS
from .exceptions import RobotError, ValidationError

logger = logging.getLogger(__name__)


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
        urdf_path: Optional[str] = None,
        mjcf_path: Optional[str] = None,
        overwrite: bool = False,
        shared: bool = False,
        org_id: Optional[str] = None,
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
        self.id: Optional[str] = None
        self.archived: Optional[bool] = None
        self._auth: Auth = get_auth()
        self._temp_dir = None
        self._data_streams: dict[str, DataStream] = dict()

        self._recording_manager = get_recording_state_manager()
        self._recording_manager.add_listener(
            RecordingStateManager.RECORDING_STOPPED, self._recording_stopped
        )
        self._stop_streams_lock = Lock()

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
        if mjcf_path:
            if not os.path.isfile(mjcf_path):
                raise ValidationError(f"MJCF file not found: {mjcf_path}")
            if not mjcf_path.lower().endswith(".xml"):
                raise ValidationError("MJCF file must have .xml extension.")
            try:
                from .mjcf_to_urdf import convert
            except ImportError:
                raise ImportError("MJCF to URDF conversion requires mujoco")
            self._temp_dir = tempfile.TemporaryDirectory(prefix="neuracore")
            self.urdf_path = os.path.join(self._temp_dir.name, "model.urdf")
            convert(mjcf_path, Path(self.urdf_path), asset_file_prefix="meshes/")

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
                "Run `nc-select-org` and try again."
            )

        try:
            response = requests.post(
                f"{API_URL}/org/{self.org_id}/robots?is_shared={self.shared}",
                json={
                    "name": self.name,
                    "instance": self.instance,
                    "overwrite": self.overwrite,
                },  # TODO: Add camera support
                headers=self._auth.get_headers(),
            )
            response.raise_for_status()
            response_body = response.json()
            self.id = response_body["robot_id"]
            self.archived = response_body.get("archived")
            has_urdf = response_body["has_urdf"]
            # Upload URDF and meshes if provided
            if self.urdf_path and not has_urdf:
                self._upload_urdf_and_meshes()
                if self._temp_dir:
                    self._temp_dir.cleanup()
        except requests.exceptions.ConnectionError:
            raise RobotError((
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            ))
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                raise RobotError(
                    "Unauthorised: no organisation selected. "
                    "Run `nc-select-org` and try again."
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

    def get_data_stream(self, stream_id: str) -> Optional[DataStream]:
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

        try:

            response = requests.post(
                f"{API_URL}/org/{self.org_id}/recording/start",
                headers=self._auth.get_headers(),
                json={
                    "robot_id": self.id,
                    "instance": self.instance,
                    "dataset_id": dataset_id,
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
            raise RobotError((
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            ))
        except requests.exceptions.RequestException as e:
            raise RobotError(f"Failed to start recording: {str(e)}")

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
            raise RobotError((
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            ))
        except requests.exceptions.RequestException as e:
            raise RobotError(f"Failed to stop recording: {str(e)}")

    def _recording_stopped(
        self, robot_id: str, instance: int, recording_id: str
    ) -> None:
        """Handle recording stopped events from the recording state manager.

        Internal callback that stops data collection from all streams when
        a recording session ends.

        Args:
            robot_id: ID of the robot whose recording stopped.
            instance: Instance number of the robot.
            recording_id: ID of the recording that stopped.
        """
        if self.id != robot_id or self.instance != instance:
            return
        with self._stop_streams_lock:
            for data_stream in self._data_streams.values():
                if data_stream.is_recording():
                    data_stream.stop_recording()

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

    def get_current_recording_id(self) -> Optional[str]:
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
            raise RobotError((
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            ))
        except requests.exceptions.RequestException as e:
            raise RobotError(f"Failed to upload URDF package: {str(e)}")
        except Exception as e:
            raise RobotError(f"Error preparing URDF package: {str(e)}")


# Global robot registry
_robots: dict[RobotInstanceIdentifier, Robot] = {}
_robot_name_id_mapping: dict[str, str] = {}


def init(
    robot_name: str,
    instance: int,
    urdf_path: Optional[str] = None,
    mjcf_path: Optional[str] = None,
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
