import io
import logging
import os
import xml.etree.ElementTree as ET
import zipfile

import requests

from neuracore.const import API_URL

from .auth import Auth, get_auth
from .exceptions import RobotError, ValidationError

logger = logging.getLogger(__name__)


class Robot:
    def __init__(
        self, robot_name: str, urdf_path: str | None = None, overwrite: bool = False
    ):
        self.name = robot_name
        self.urdf_path = urdf_path
        self.overwrite = overwrite
        self.id: str = None
        self._auth: Auth = get_auth()
        if self.urdf_path and not os.path.isfile(self.urdf_path):
            raise ValidationError(f"URDF file not found: {self.urdf_path}")

    def init(self) -> None:
        """Initialize robot on the server."""
        if not self._auth.is_authenticated:
            raise RobotError("Not authenticated. Please call nc.login() first.")

        try:
            # First check if we already have a robot with the same name
            if not self.overwrite:
                response = requests.get(
                    f"{API_URL}/robots",
                    headers=self._auth.get_headers(),
                )
                response.raise_for_status()
                robots = response.json()
                for robot in robots:
                    if robot["name"] == self.name:
                        self.id = robot["id"]
                        logger.info(f"Found existing robot: {self.name}")
                        return

            logger.info(f"Creating new robot: {self.name}")
            response = requests.post(
                f"{API_URL}/robots",
                json={"name": self.name, "cameras": []},  # TODO: Add camera support
                headers=self._auth.get_headers(),
            )
            response.raise_for_status()
            self.id = response.json()

            # Upload URDF and meshes if provided
            if self.urdf_path:
                self._upload_urdf_and_meshes()

        except requests.exceptions.RequestException as e:
            raise RobotError(f"Failed to initialize robot: {str(e)}")

    def start_recording(self, dataset_id: str) -> str:
        """Start recording robot data."""
        if not self.id:
            raise RobotError("Robot not initialized. Call init() first.")

        try:
            response = requests.post(
                f"{API_URL}/recording/start",
                headers=self._auth.get_headers(),
                json={"robot_id": self.id, "dataset_id": dataset_id},
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

    def _upload_urdf_and_meshes(self) -> None:
        """Upload URDF and associated mesh files as a ZIP package."""
        if not os.path.exists(self.urdf_path):
            raise ValidationError(f"URDF file not found: {self.urdf_path}")

        try:
            # Read and parse URDF to find all mesh files
            with open(self.urdf_path) as f:
                urdf_content = f.read()

            root = ET.fromstring(urdf_content)
            urdf_dir = os.path.dirname(os.path.abspath(self.urdf_path))
            mesh_files: list[str] = []

            # Collect all mesh files
            for mesh in root.findall(".//mesh"):
                filename = mesh.get("filename")
                if filename:
                    if filename.startswith("package://"):
                        filename = filename.replace("package://", "")

                    mesh_path = os.path.join(urdf_dir, filename)
                    if mesh_path not in mesh_files:
                        if os.path.exists(mesh_path):
                            mesh_files.append(mesh_path)
                        else:
                            raise RobotError(f"Mesh file not found: {mesh_path}")

            # Create ZIP file in memory using BytesIO
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                # Add URDF file as robot.urdf
                zf.writestr("robot.urdf", urdf_content)

                # Add mesh files maintaining relative paths
                for mesh_path in mesh_files:
                    rel_path = os.path.relpath(mesh_path, urdf_dir)
                    zf.write(mesh_path, rel_path)

            # Get the zip data
            zip_buffer.seek(0)
            zip_data = zip_buffer.getvalue()

            # Log the ZIP contents for debugging
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                logger.info("ZIP contents:")
                for info in zf.filelist:
                    logger.info(f"  {info.filename}: {info.file_size} bytes")

            # Create the files dict with the ZIP data
            files = {
                "robot_package": ("robot_package.zip", zip_data, "application/zip")
            }

            # Upload the package
            response = requests.put(
                f"{API_URL}/robots/{self.id}/package",
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
    robot_name: str, urdf_path: str | None = None, overwrite: bool = False
) -> Robot:
    """Initialize a robot globally."""
    robot = Robot(robot_name, urdf_path, overwrite)
    robot.init()
    _robots[robot_name] = robot
    return robot


def get_robot(robot_name: str) -> Robot:
    """Get a registered robot instance."""
    if robot_name not in _robots:
        raise RobotError(f"Robot {robot_name} not initialized. Call init() first.")
    return _robots[robot_name]
