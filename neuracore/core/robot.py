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
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING

import requests
from neuracore_types import DataType, RobotInstanceIdentifier

from neuracore.core.config.get_current_org import get_current_org, get_registry_org
from neuracore.core.daemon_rows import wait_by_default, wait_for_robot_id
from neuracore.core.streaming.data_stream import DataStream
from neuracore.core.utils.http_errors import extract_error_detail
from neuracore.core.utils.http_session import thread_local_session
from neuracore.core.utils.timestamp_utils import resolve_timestamp_us
from neuracore.data_daemon import bridge as recording_context
from neuracore.data_daemon.robot_packages import save_robot_package

from .auth import Auth, get_auth
from .const import API_URL, MAX_DATA_STREAMS
from .exceptions import AuthenticationError, RobotError, ValidationError

if TYPE_CHECKING:
    from neuracore.api.logging import JointStreamBinding, ResolvedJointGroup

logger = logging.getLogger(__name__)
DaemonRecordingContext = recording_context.RecordingContext

_ROBOTS_LIST_PAGE_SIZE = 30
_SUPPORTED_MESH_EXTENSIONS = (".stl", ".dae")


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

    deleted = False

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
                if files don't exist, if file extensions are incorrect,
                or if a URDF visual mesh has an unsupported format.
            ImportError: If MJCF conversion is requested but mujoco is not available.
        """
        self.name = robot_name
        self.instance = instance
        self.urdf_path = urdf_path
        self.mjcf_path = mjcf_path
        self.overwrite = overwrite
        self.shared = shared
        self._id: str | None = None
        self._robot_index: int | None = None
        self.archived: bool | None = None
        self._auth: Auth = get_auth()
        self._temp_dir = None
        self._data_streams: dict[str, DataStream] = dict()
        self._data_stream_counts: dict[DataType, int] = defaultdict(int)
        self._joint_stream_bindings: "dict[DataType, dict[str, JointStreamBinding]]" = (
            dict()
        )
        self._joint_group_cache: "dict[DataType, ResolvedJointGroup]" = dict()
        self._daemon_recording_context: DaemonRecordingContext | None = None

        self.org_id = org_id or get_registry_org()

        if urdf_path and mjcf_path:
            raise ValidationError(
                "Only one of urdf_path or mjcf_path should be provided."
            )
        if urdf_path:
            if not os.path.isfile(urdf_path):
                raise ValidationError(f"URDF file not found: {urdf_path}")
            if not urdf_path.lower().endswith(".urdf"):
                raise ValidationError("URDF file must have .urdf extension.")
            self._check_mesh_formats()
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

    def init(self, wait: bool | None = False) -> None:
        """Connect the robot through the data daemon.

        The daemon keeps a row for the robot instance and has the server
        register it, and upload its kinematic model if the server needs it, in
        the background, online or offline. Until then :attr:`id` is not known;
        reading it waits for the server.

        Args:
            wait: Wait for the server to register the robot before returning,
                so a refusal raises here. ``None`` waits unless the last login
                or the daemon is offline.

        Raises:
            RobotArchivedError: If ``wait`` and the robot is archived.
            RobotError: If ``wait`` and the server refused the robot, or the
                package could not be built.
            ServerUnreachableError: If ``wait`` and the last login could not
                reach the server.
        """
        package_path = None
        if self.urdf_path:
            try:
                package = self._package_urdf()["robot_package"][1]
            except RobotError as error:
                # The package is sent only if the server has no URDF for the
                # robot, which cannot be known yet; one asked to overwrite it
                # must be sent.
                if self.overwrite:
                    raise
                logger.warning(
                    "Robot '%s' is connected without its URDF package: %s. If "
                    "the server has no URDF for it yet, it is registered without "
                    "one.",
                    self.name,
                    error,
                )
            else:
                # An MJCF robot's converted URDF stays until close(): urdf_path
                # names it, and the importer loads it.
                package_path = str(save_robot_package(package))
        self._robot_index = recording_context.connect_robot_row(
            org_id=self.org_id,
            name=self.name,
            shared=self.shared,
            instance=self.instance,
            overwrite_urdf=self.overwrite,
            package_path=package_path,
        )
        robot_index = self._robot_index
        if wait_by_default(
            wait, lambda: recording_context.robot_row_state(robot_index)
        ):
            self.id  # noqa: B018 - reading it waits for the server

    @property
    def id(self) -> str:
        """The server's id for the robot.

        Known once the server registers the robot instance; reading it before
        then waits for that, as ``connect_robot(wait=True)`` does.

        Raises:
            RobotError: If the robot is not connected, the server refused it,
                or it was not registered in time.
            RobotArchivedError: If the robot is archived.
            ServerUnreachableError: If the last login could not reach the
                server and the id is not known yet.
        """
        if self._id is None and self._robot_index is not None:
            self._id = wait_for_robot_id(self._robot_index, self.name, self.instance)
            self.archived = False
            _register_robot_id(self)
        if self._id is None:
            raise RobotError("Robot not initialized. Call init() first.")
        return self._id

    @id.setter
    def id(self, value: str | None) -> None:
        self._id = value

    @property
    def is_registered(self) -> bool:
        """Whether the server has registered the robot, so :attr:`id` is known.

        Reading :attr:`id` returns at once when this is true; otherwise it
        waits for the server.
        """
        return self._id is not None

    @property
    def robot_index(self) -> int:
        """The robot's row in the data daemon, which labels its logged data.

        Raises:
            RobotError: If the robot is not connected.
        """
        if self._robot_index is None:
            raise RobotError("Robot not initialized. Call init() first.")
        return self._robot_index

    def add_data_stream(self, stream_id: str, stream: DataStream) -> None:
        """Add a data stream to the robot for sensor data collection.

        Args:
            stream_id: Unique identifier for the data stream.
            stream: DataStream instance for handling the sensor data.

        Raises:
            RuntimeError: If the maximum number of data streams is exceeded.
            ValueError: If a stream with the same ID already exists.
        """
        if self._data_stream_counts[stream.data_type] >= MAX_DATA_STREAMS:
            raise RuntimeError("Excessive number of data streams")
        if stream_id in self._data_streams:
            raise ValueError("Stream already exists")

        self._data_stream_counts[stream.data_type] += 1
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

    def start_recording(
        self, dataset_index: int, timestamp: float | None = None
    ) -> None:
        """Start recording data from all active streams to a dataset.

        Asks the daemon to open a recording window; it owns recording identity
        and mints the cloud id asynchronously, so there is nothing to return
        here. Ask :meth:`get_cloud_recording_id` once it exists.

        Args:
            dataset_index: The data daemon's row for the dataset to record into.
            timestamp: Optional capture time (Unix seconds) for the recording's
                start, matching the ``log_*`` methods.

        Raises:
            RobotError: If the robot is not initialized or if
                the recording fails to start.
            ValueError: If the timestamp is not finite or is outside the
                supported microsecond range.
        """
        timestamp_us = resolve_timestamp_us(timestamp)
        self._get_daemon_recording_context().start_recording(
            robot_id=str(self.robot_index),
            robot_instance=self.instance,
            robot_name=self.name,
            dataset_index=dataset_index,
            dataset_name=None,
            timestamp_us=timestamp_us,
        )

    def stop_recording(
        self,
        recording_id: str | None = None,
        timestamp: float | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Stop an active recording session.

        Ends the active recording session for this robot instance and stops
        data collection from all streams.

        Args:
            recording_id: Unused — the daemon stops the active recording for
                this source. Retained for call-site compatibility.
            timestamp: Optional capture time (Unix seconds) for the recording's
                stop, matching the ``log_*`` methods. It is the reported stop
                time, not the window's upper bound, which the daemon takes from
                the publish stamp at the send.
            tags: Normalized user tag names for the recording, which the daemon
                sends to the backend with the stop.

        Raises:
            RobotError: If the robot is not initialized.
            ValueError: If the timestamp is not finite or is outside the
                supported microsecond range.
        """
        timestamp_us = resolve_timestamp_us(timestamp)
        # The gate closes inside the notify, not here.
        self._notify_daemon_of_stop(recording_id, timestamp_us=timestamp_us, tags=tags)

    def _notify_daemon_of_stop(
        self,
        recording_id: str | None,
        timestamp_us: int,
        tags: list[str] | None = None,
    ) -> None:
        """Send the recording-stopped IPC message to the daemon."""
        try:
            context = self._get_daemon_recording_context()
            try:
                context.stop_recording(timestamp_us=timestamp_us, tags=tags)
            finally:
                # Runs even if the notify failed.
                context.flush_source()
        except Exception:
            logger.exception(
                "Failed to notify daemon of stop for recording_id=%s",
                recording_id,
            )

    def is_recording(self) -> bool:
        """Check if the robot is currently recording data.

        Reads the daemon-backed cache, which owns recording state. It therefore
        answers the same for a recording started here, in another process on
        this host, or from the web — a process that started nothing still gets a
        true answer.

        A recording this process bracketed is exact the moment the call returns.
        One started elsewhere is seen within a refresh interval (50 ms), which is
        far inside the propagation the backend itself takes.

        Returns:
            True if the robot is actively recording, False otherwise.

        Raises:
            RecordingStateUnavailableError: nothing has ever answered for this
                source. Silence is not "not recording" — what this process
                started says nothing about a recording opened elsewhere.
        """
        return (
            recording_context.query_recording_state(
                str(self.robot_index), self.instance
            )
            is not None
        )

    def _recording_epoch(self) -> int | None:
        """Which recording the daemon has open for this source, as a token.

        Opaque and only worth comparing: it changes when the source crosses a
        recording boundary, whoever opened it, and is ``None`` when none is
        open. The log path uses it to scope the monotonic-timestamp check to one
        recording, so unlike :meth:`is_recording` it must never block, not even
        on a source's first read.
        """
        if self._robot_index is None:
            return None
        return self._get_daemon_recording_context().recording_epoch()

    def get_cloud_recording_id(self, timeout_s: float = 30.0) -> str | None:
        """Resolve the daemon-owned cloud recording id for the open recording.

        The id is resolvable only while the recording is open: the daemon drops
        the source's entry when it stops, and this reads that entry.

        Args:
            timeout_s: Maximum time to wait for the daemon to mint the id. ``0``
                reads it once without waiting.

        Returns:
            The cloud recording id of the open recording, or ``None`` when the
            robot is not recording *or* the id is not minted within
            ``timeout_s`` — it is assigned asynchronously, and never at all for
            a recording made offline. Use :meth:`is_recording` to ask whether a
            recording is open.
        """
        recording = recording_context.query_recording_state(
            str(self.robot_index), self.instance, timeout_s=timeout_s
        )
        return recording.cloud_recording_id if recording is not None else None

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

    def _check_mesh_formats(self) -> None:
        """Check that every URDF visual mesh uses a supported file format.

        Raises:
            ValidationError: If a visual mesh filename has an unsupported extension.
        """
        unsupported_filenames: list[str] = []
        for mesh in self._load_urdf().findall(".//visual//mesh"):
            filename = mesh.get("filename")
            if not filename or filename in unsupported_filenames:
                continue
            extension = PureWindowsPath(filename).suffix.lower()
            if extension not in _SUPPORTED_MESH_EXTENSIONS:
                unsupported_filenames.append(filename)
        if unsupported_filenames:
            raise ValidationError(
                "URDF has unsupported visual mesh formats: "
                f"{', '.join(unsupported_filenames)}\n"
                "Convert these meshes to one of: "
                f"{', '.join(_SUPPORTED_MESH_EXTENSIONS)}"
            )

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

        try:
            with zipfile.ZipFile(io.BytesIO(self._fetch_urdf_package()), "r") as zf:
                if "robot.urdf" not in zf.namelist():
                    raise RobotError("URDF package does not contain robot.urdf")
                try:
                    return ET.fromstring(zf.read("robot.urdf"))
                except ET.ParseError as e:
                    raise RobotError(f"Failed to parse URDF: {str(e)}") from e
        except zipfile.BadZipFile as e:
            raise RobotError("Invalid URDF package received from server.") from e

    def _fetch_urdf_package(self) -> bytes:
        """Download the robot's URDF package ZIP from the server.

        Returns:
            Raw bytes of the ZIP package containing robot.urdf and meshes/.

        Raises:
            RobotError: If the robot is not initialized, no URDF has been
                uploaded, or the request fails.
        """
        if not self.id:
            raise RobotError("Robot not initialized. Call init() first.")
        if not self.org_id:
            raise RobotError(
                "Unauthorised: no organisation selected. "
                "Run `nc-select-org` and try again."
            )

        try:
            session = thread_local_session()
            response = session.get(
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
        return response.content

    def download_urdf(self, output_dir: str | Path) -> Path:
        """Download the robot's URDF and meshes from the server.

        The package is extracted into ``output_dir`` as ``robot.urdf`` alongside
        a ``meshes/`` directory referenced by the URDF.

        Args:
            output_dir: Directory to extract the URDF package into. Created if
                it does not exist.

        Returns:
            Path to the extracted ``robot.urdf`` file.

        Raises:
            RobotError: If the robot is not initialized, no URDF is available,
                the download fails, or the package is invalid.
        """
        package = self._fetch_urdf_package()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(io.BytesIO(package), "r") as zf:
                if "robot.urdf" not in zf.namelist():
                    raise RobotError("URDF package does not contain robot.urdf")
                zf.extractall(output_path)
        except zipfile.BadZipFile as e:
            raise RobotError("Invalid URDF package received from server.") from e
        return output_path / "robot.urdf"

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

    def cancel_recording(
        self, recording_id: str | None = None, timestamp: float | None = None
    ) -> None:
        """Cancel an active recording without saving any data.

        Args:
            recording_id: Unused — the daemon cancels the active recording for
                this source. Retained for call-site compatibility.
            timestamp: Optional capture time (Unix seconds) for the cancel,
                mirroring ``stop_recording``: the recording's captured stop
                time, the current time when omitted.

        Raises:
            RobotError: If the robot is not initialized.
            ValueError: If the timestamp is not finite or is outside the
                supported microsecond range.
        """
        timestamp_us = resolve_timestamp_us(timestamp)
        self._get_daemon_recording_context().cancel_recording(timestamp_us=timestamp_us)

    def _get_daemon_recording_context(self) -> DaemonRecordingContext:
        """Return a reusable daemon recording context, creating it lazily."""
        if self._daemon_recording_context is None:
            context = DaemonRecordingContext()
            context.bind_source(str(self.robot_index), self.instance)
            self._daemon_recording_context = context
        return self._daemon_recording_context

    def _cleanup_daemon_recording_context(self) -> None:
        """Release daemon recording context resources."""
        if self._daemon_recording_context is None:
            return
        try:
            # Seal this process's tail chunks on the way out: nothing else will
            # once the context is gone.
            self._daemon_recording_context.flush_source()
            self._daemon_recording_context.close()
        except Exception:
            logger.exception("Failed to cleanup daemon recording context")
        finally:
            self._daemon_recording_context = None

    def delete(self) -> None:
        """Delete this robot from Neuracore and invalidate this object.

        After a successful delete the robot is dropped from the global registry,
        local resources are released, and the object is turned into a tombstone,
        so any further attribute or method access raises RobotError.

        Raises:
            RobotError: If the robot is shared, has not been initialized, or the
                server rejects the delete.
        """
        if self.shared:
            raise RobotError("Cannot delete a shared robot.")
        robot_id = self.id
        if robot_id is None:
            raise RobotError("Robot not initialized. Call init() first.")
        try:
            session = thread_local_session()
            response = session.delete(
                f"{API_URL}/org/{self.org_id}/robots/{robot_id}",
                headers=self._auth.get_headers(),
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RobotError(
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            )
        except requests.exceptions.RequestException as e:
            detail = None
            if e.response is not None:
                detail = extract_error_detail(e.response)
            raise RobotError(f"Failed to delete robot: {detail or str(e)}")

        _robots.pop(
            RobotInstanceIdentifier(robot_id=robot_id, robot_instance=self.instance),
            None,
        )
        for robot_name, mapped_id in list(_robot_name_id_mapping.items()):
            if mapped_id == robot_id:
                del _robot_name_id_mapping[robot_name]
        for key, connected in list(_connected_robots.items()):
            if connected is self:
                del _connected_robots[key]

        self.close()
        self.__dict__.clear()
        self.__class__ = _DeletedRobot

    def set_archived(self, archived: bool = True) -> None:
        """Archive or unarchive this robot.

        Archiving hides the robot from the default robots list while keeping the
        robot record and URDF package so recordings can still be visualized.

        Unarchiving works before init(), because the server refuses to connect
        to an archived robot: the robot is then found by name among the
        archived robots.

        Args:
            archived: True to archive, False to unarchive.

        Raises:
            RobotError: If archiving a robot that has not been initialized, if
                no archived robot has this name, or if the server rejects the
                update.
        """
        if self.is_registered:
            robot_id = self.id
        elif not archived:
            robot_id = self._archived_robot_id()
        elif self._robot_index is not None:
            # Connected, but not registered yet: wait for the server's id.
            robot_id = self.id
        else:
            raise RobotError("Robot not initialized. Call init() first.")
        try:
            session = thread_local_session()
            response = session.put(
                f"{API_URL}/org/{self.org_id}/robots/{robot_id}/archive",
                json={"archived": archived},
                params={"is_shared": str(self.shared).lower()},
                headers=self._auth.get_headers(),
            )
            response.raise_for_status()
            self.archived = archived
        except requests.exceptions.ConnectionError:
            raise RobotError(
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            )
        except requests.exceptions.RequestException as e:
            detail = None
            if e.response is not None:
                detail = extract_error_detail(e.response)
            action = "archive" if archived else "unarchive"
            raise RobotError(f"Failed to {action} robot: {detail or str(e)}")

    def _archived_robot_id(self) -> str:
        """Find the ID of the archived robot with this robot's name.

        Raises:
            RobotError: If no archived robot has this name.
        """
        archived_robots = list_organization_robots(
            self.org_id, is_shared=self.shared, mode="archived"
        )
        for robot in archived_robots:
            if robot["name"] == self.name:
                return robot["id"]
        raise RobotError(f"No archived robot is named '{self.name}'.")

    def list_instance_ids(self) -> list[int]:
        """Return registered instance IDs for this robot.

        Raises:
            RobotError: If the robot has not been initialized or the server
                rejects the request.
        """
        if self.id is None:
            raise RobotError("Robot not initialized. Call init() first.")
        try:
            session = thread_local_session()
            response = session.get(
                f"{API_URL}/org/{self.org_id}/robots/{self.id}",
                params={"is_shared": str(self.shared).lower()},
                headers=self._auth.get_headers(),
            )
            response.raise_for_status()
            instances = response.json().get("instances") or []
            return [
                int(item["robot_instance"])
                for item in instances
                if item.get("robot_instance") is not None
            ]
        except requests.exceptions.ConnectionError:
            raise RobotError(
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            )
        except requests.exceptions.RequestException as e:
            detail = None
            if e.response is not None:
                detail = extract_error_detail(e.response)
            raise RobotError(f"Failed to list robot instances: {detail or str(e)}")
        except (KeyError, TypeError, ValueError) as e:
            raise RobotError(f"Failed to parse robot instances: {e}") from e

    def max_instance_id(self) -> int:
        """Return the highest registered instance ID, or 0 if none exist.

        Raises:
            RobotError: If the robot has not been initialized or the server
                rejects the request.
        """
        instance_ids = self.list_instance_ids()
        return max(instance_ids) if instance_ids else 0

    def remove_instance(self, instance: int) -> None:
        """Delete a registered robot instance without deleting the robot.

        Evicts the matching entry from the local robot registry if present.
        Does not tombstone this Robot object.

        Args:
            instance: Instance number to remove.

        Raises:
            RobotError: If the robot has not been initialized or the server
                rejects the delete.
        """
        if self.id is None:
            raise RobotError("Robot not initialized. Call init() first.")
        try:
            session = thread_local_session()
            response = session.delete(
                f"{API_URL}/org/{self.org_id}/robots/{self.id}/instances/{instance}",
                params={"is_shared": str(self.shared).lower()},
                headers=self._auth.get_headers(),
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RobotError(
                "Failed to connect to neuracore server, "
                "please check your internet connection and try again."
            )
        except requests.exceptions.RequestException as e:
            detail = None
            if e.response is not None:
                detail = extract_error_detail(e.response)
            raise RobotError(f"Failed to delete robot instance: {detail or str(e)}")

        _robots.pop(
            RobotInstanceIdentifier(robot_id=self.id, robot_instance=instance),
            None,
        )

    def close(self) -> None:
        """Release local resources owned by this Robot instance."""
        self._cleanup_daemon_recording_context()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __del__(self) -> None:
        """Best-effort cleanup for daemon recording resources."""
        self.close()


class _DeletedRobot(Robot):
    """Tombstone for a deleted Robot. Any attribute access raises."""

    def __getattribute__(self, name: str) -> object:
        """Raise on every access except the deleted flag."""
        if name == "deleted":
            return True
        raise RobotError("Robot has been deleted and can no longer be used.")

    def __del__(self) -> None:
        """Do nothing. Local resources were released during delete."""


# Global robot registry
_robots: dict[RobotInstanceIdentifier, Robot] = {}
_robot_name_id_mapping: dict[str, str] = {}
# Robots connected by this process, by name and instance, from the moment they
# are connected: their server id may not be known yet.
_connected_robots: dict[tuple[str, int], Robot] = {}


def _register_robot_id(robot: Robot) -> None:
    """Index a connected robot by its server id, once that is known."""
    if not robot.is_registered:
        return
    _robot_name_id_mapping[robot.name] = robot.id
    _robots[
        RobotInstanceIdentifier(robot_id=robot.id, robot_instance=robot.instance)
    ] = robot


def init(
    robot_name: str,
    instance: int,
    urdf_path: str | None = None,
    mjcf_path: str | None = None,
    overwrite: bool = False,
    shared: bool = False,
    wait: bool | None = False,
) -> Robot:
    """Initialize a robot and register it globally.

    Creates a new Robot instance, connects it through the data daemon, and
    registers it in the global robot registry for future access.

    Args:
        robot_name: Unique identifier for the robot type.
        instance: Instance number for multi-robot deployments.
        urdf_path: Path to URDF kinematic model file.
        mjcf_path: Path to MJCF kinematic model file.
        overwrite: Whether to overwrite existing robot configuration.
        shared: Whether the robot is shared/open-source.
            Note that setting shared=True is only available to specific
            members allocated by the Neuracore team.
        wait: Wait for the server to register the robot before returning.
            ``None`` waits unless the last login or the daemon is offline.

    Returns:
        The initialized Robot instance.
    """
    if not robot_name:
        raise ValueError("Robot name cannot be empty")
    robot = Robot(robot_name, instance, urdf_path, mjcf_path, overwrite, shared)
    robot.init(wait=wait)
    _connected_robots[(robot_name, instance)] = robot
    _register_robot_id(robot)
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
    connected = _connected_robots.get((robot_name, instance))
    if connected is not None:
        return connected
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
        if robot.is_registered and robot.id == robot_id:
            robot.name = new_robot_name
    for (name, instance), robot in list(_connected_robots.items()):
        if robot.is_registered and robot.id == robot_id and name != new_robot_name:
            _connected_robots[(new_robot_name, instance)] = _connected_robots.pop(
                (name, instance)
            )


def update_robot_name(
    robot_key: str,
    new_robot_name: str,
    shared: bool = False,
) -> str:
    """Update a robot's name on the server.

    Args:
        robot_key: Robot ID, or name of a robot connected in this session.
        new_robot_name: New display name to set.
        shared: Whether the robot is shared/open-source.

    Returns:
        The resolved robot ID.

    Raises:
        AuthenticationError: If the user is not authenticated.
        RobotError: If the update request fails.
        ValueError: If the robot key or new name is empty.
    """
    if not robot_key:
        raise ValueError("Robot old name or ID cannot be empty.")
    if not new_robot_name:
        raise ValueError("New robot name cannot be empty.")
    if not get_auth().is_authenticated:
        raise AuthenticationError("Not authenticated. Please call nc.login() first.")

    resolved_org_id = get_current_org()
    if not resolved_org_id:
        raise RobotError("No organization selected. Please call nc.select_org() first.")
    robot_id = _robot_name_id_mapping.get(robot_key)
    if robot_id is None:
        # A robot connected by this process whose id is not known yet: wait
        # for the server to register it.
        connected = next(
            (
                robot
                for (name, _), robot in _connected_robots.items()
                if name == robot_key
            ),
            None,
        )
        robot_id = connected.id if connected is not None else robot_key
    try:
        session = thread_local_session()
        response = session.put(
            f"{API_URL}/org/{resolved_org_id}/robots/{robot_id}/name",
            json={"name": new_robot_name},
            params={"is_shared": str(shared).lower()},
            headers=get_auth().get_headers(),
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RobotError(
            "Failed to connect to neuracore server, "
            "please check your internet connection and try again."
        )
    except requests.exceptions.RequestException as e:
        detail = None
        if e.response is not None:
            detail = extract_error_detail(e.response)
        raise RobotError(f"Failed to update robot name: {detail or str(e)}")
    _update_local_robot_name_cache(robot_id, new_robot_name)
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
    robots: list[dict] = []
    start_after: dict | None = None
    try:
        while True:
            session = thread_local_session()
            response = session.post(
                f"{API_URL}/org/{org_id}/robots/list",
                headers=get_auth().get_headers(),
                params={
                    "is_shared": str(is_shared).lower(),
                    "mode": mode,
                    "limit": str(_ROBOTS_LIST_PAGE_SIZE),
                },
                json=start_after,
            )
            response.raise_for_status()
            page = response.json()

            batch = page.get("data", [])
            robots.extend(batch)
            total = page.get("total", len(robots))
            if not batch or len(robots) >= total:
                break
            start_after = batch[-1]
        return robots
    except requests.exceptions.ConnectionError:
        raise RobotError(
            "Failed to connect to neuracore server, "
            "please check your internet connection and try again."
        )
    except requests.exceptions.RequestException as e:
        raise RobotError(f"Failed to list robots: {str(e)}")


def get_robot_id_from_name(robot_name: str, org_id: str | None = None) -> str:
    """Get the robot ID corresponding to a given robot name.

    This is for any robot in the org, not a specific robot instance.

    Will search through list of robots in given org and shared

    Args:
        robot_name: The name of the robot to look up.

    Returns:
        The unique ID of the robot.

    Raises:
        RobotError: If the robot name is not found in the registry.
    """
    if org_id is None:
        org_id = get_current_org()

    private_robots = list_organization_robots(org_id, is_shared=False)
    shared_robots = list_organization_robots(org_id, is_shared=True)
    robots = private_robots + shared_robots

    # There is a change the same name can exist in both private and shared,
    # we will return the private first in this case
    for robot in robots:
        if robot["name"] == robot_name:
            return robot["id"]
    raise RobotError(
        f"Robot with name '{robot_name}' not found in organization '{org_id}' "
        "or shared robots."
    )


def get_robot_name_by_id(robot_id: str, org_id: str | None = None) -> str:
    """Resolve a robot ID to its display name.

    Args:
        robot_id: Robot ID to look up.
        org_id: Organization ID. Uses the current org when omitted.

    Returns:
        The robot's display name.

    Raises:
        RobotError: If not authenticated, no org is selected, or the robot is unknown.
    """
    names = get_robot_names_by_ids([robot_id], org_id=org_id)
    name = names.get(robot_id)
    if name is None:
        raise RobotError(f"Robot with ID '{robot_id}' was not found.")
    return name


def get_robot_names_by_ids(
    robot_ids: list[str], org_id: str | None = None
) -> dict[str, str]:
    """Resolve robot IDs to display names via the Neuracore API.

    Args:
        robot_ids: Robot IDs to resolve. Duplicates are ignored.
        org_id: Organization ID. Uses the current org when omitted.

    Returns:
        Mapping of robot ID to display name for IDs that were found.
        Unknown IDs are omitted from the result.

    Raises:
        RobotError: If not authenticated, no org is selected, or the API call fails.
    """
    if not robot_ids:
        return {}
    if not get_auth().is_authenticated:
        raise RobotError("Not authenticated. Please call nc.login() first.")
    if org_id is None:
        org_id = get_current_org()
    if not org_id:
        raise RobotError("No organization selected. Please call nc.select_org() first.")

    unique_robot_ids = list(dict.fromkeys(robot_ids))
    try:
        session = thread_local_session()
        response = session.post(
            f"{API_URL}/org/{org_id}/robots/names",
            headers=get_auth().get_headers(),
            json={"robot_ids": unique_robot_ids},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return {robot["id"]: robot["name"] for robot in payload.get("robots", [])}
    except requests.exceptions.ConnectionError:
        raise RobotError(
            "Failed to connect to neuracore server, "
            "please check your internet connection and try again."
        )
    except requests.exceptions.RequestException as e:
        raise RobotError(f"Failed to resolve robot names: {str(e)}")
