import io
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from typing import Optional

import requests

from .auth import Auth, get_auth
from .const import API_URL
from .exceptions import RobotError, ValidationError

logger = logging.getLogger(__name__)


class Robot:
    def __init__(
        self,
        robot_name: str,
        instance: int = 0,
        urdf_path: Optional[str] = None,
        mjcf_path: Optional[str] = None,
        overwrite: bool = False,
        shared: bool = False,
    ):
        self.name = robot_name
        self.instance = instance
        self.urdf_path = urdf_path
        self.mjcf_path = mjcf_path
        self.overwrite = overwrite
        self.shared = shared
        self.id: str = None
        self.instanced_id: str = None
        self._auth: Auth = get_auth()
        self._temp_dir = None
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
            convert(mjcf_path, self.urdf_path, asset_file_prefix="meshes/")

    def init(self) -> None:
        """Initialize robot on the server."""
        if not self._auth.is_authenticated:
            raise RobotError("Not authenticated. Please call nc.login() first.")

        try:
            # First check if we already have a robot with the same name
            if not self.overwrite:
                response = requests.get(
                    f"{API_URL}/robots?is_shared={self.shared}",
                    headers=self._auth.get_headers(),
                )
                response.raise_for_status()
                robots = response.json()
                for robot in robots:
                    if robot["name"] == self.name:
                        self.id = robot["id"]
                        self.instanced_id = f"{self.id}_{self.instance}"
                        logger.info(f"Found existing robot: {self.name}")
                        return

            logger.info(f"Creating new robot: {self.name}")
            response = requests.post(
                f"{API_URL}/robots?is_shared={self.shared}",
                json={"name": self.name, "cameras": []},  # TODO: Add camera support
                headers=self._auth.get_headers(),
            )
            response.raise_for_status()
            self.id = response.json()
            # Allows multiple instances of the same robot to record at once
            self.instanced_id = f"{self.id}_{self.instance}"

            # Upload URDF and meshes if provided
            if self.urdf_path:
                self._upload_urdf_and_meshes()
                if self._temp_dir:
                    self._temp_dir.cleanup()

        except requests.exceptions.RequestException as e:
            raise RobotError(f"Failed to initialize robot: {str(e)}")

    def start_recording(self, dataset_id: str) -> str:
        """Start recording robot data."""
        if not self.instanced_id:
            raise RobotError("Robot not initialized. Call init() first.")

        try:
            response = requests.post(
                f"{API_URL}/recording/start",
                headers=self._auth.get_headers(),
                json={"robot_id": self.instanced_id, "dataset_id": dataset_id},
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise RobotError(f"Failed to start recording: {str(e)}")

    def stop_recording(self, recording_id: str) -> None:
        """Stop a recording.

        Args:
            recording_id: Identifier of the recording to stop.
        """
        if not self.id:
            raise RobotError("Robot not initialized. Call init() first.")

        try:
            response = requests.post(
                f"{API_URL}/recording/stop?recording_id={recording_id}",
                headers=self._auth.get_headers(),
            )
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise RobotError(f"Failed to stop recording: {str(e)}")

    def _package_urdf(self) -> dict:
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
        """Upload URDF and associated mesh files as a ZIP package."""

        try:
            # Create the files dict with the ZIP data
            files = self._package_urdf()

            # Upload the package
            response = requests.put(
                f"{API_URL}/robots/{self.id}/package?is_shared={self.shared}",
                headers=self._auth.get_headers(),
                files=files,
            )

            # Log response for debugging
            logger.info(f"Upload response status: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Upload error response: {response.text}")

            response.raise_for_status()

            logger.info(f"Successfully uploaded URDF package for robot {self.id}")

        except requests.exceptions.RequestException as e:
            raise RobotError(f"Failed to upload URDF package: {str(e)}")
        except Exception as e:
            raise RobotError(f"Error preparing URDF package: {str(e)}")


# Global robot registry
_robots = {}


def init(
    robot_name: str,
    instance: int = 0,
    urdf_path: Optional[str] = None,
    mjcf_path: Optional[str] = None,
    overwrite: bool = False,
    shared: bool = False,
) -> Robot:
    """Initialize a robot globally."""
    robot = Robot(robot_name, instance, urdf_path, mjcf_path, overwrite, shared)
    robot.init()
    _robots[(robot_name, instance)] = robot
    return robot


def get_robot(robot_name: str, instance: int = 0) -> Robot:
    """Get a registered robot instance."""
    key = (robot_name, instance)
    if key not in _robots:
        raise RobotError(
            f"Robot {robot_name}:{instance} not initialized. Call init() first."
        )
    return _robots[key]
