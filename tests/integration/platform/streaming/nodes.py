"""Provider processes of the streaming integration tests."""

import logging
import multiprocessing
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.synchronize import Event

import numpy as np

# Keep neuracore out of module-level imports: a provider sets its URL first.

logger = logging.getLogger(__name__)


rgb_test_image = np.full((10, 10, 3), 100, dtype=np.uint8)
rgb_test_image[0, 0] = [255, 0, 0]  # Red pixel to identify the image

REMOTE_JOINT_NAME = "joint_from_remote"
REMOTE_JOINT_VALUE = 0.5
SIGNALLING_URL_VARIABLE = "NEURACORE_SIGNALLING_URL"
STOP_SECONDS = 15


@dataclass
class Provider:
    """A provider node process.

    Attributes:
        process: The node process.
        stop_event: Set to stop the node.
    """

    process: multiprocessing.Process
    stop_event: Event

    def stop(self) -> None:
        """Stop the node and wait for its process to end."""
        self.stop_event.set()
        self.process.join(timeout=STOP_SECONDS)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()


ProviderFactory = Callable[..., Provider]


def remote_node_logger(robot_name: str, instance: int, ready_event: Event):
    """Log data of a robot instance from a separate process."""
    # Use a separate import alias to avoid any potential global state issues

    import neuracore as nc_remote

    try:
        nc_remote.login()
        # Connect to the same robot instance. The main process has already created it.
        nc_remote.connect_robot(robot_name, instance=instance)

        # Log specific joint positions.
        joint_positions = {"joint_from_remote": 0.5}
        nc_remote.log_joint_positions(
            joint_positions, robot_name=robot_name, instance=instance
        )

        nc_remote.log_rgb(
            "cam_from_remote", rgb_test_image, robot_name=robot_name, instance=instance
        )
        nc_remote.log_custom_1d(
            "custom_data_from_remote", np.array([42.0], dtype=np.float32)
        )
        nc_remote.log_parallel_gripper_open_amounts(
            {"left_gripper": 0.5, "right_gripper": 0.5}
        )
        nc_remote.log_joint_target_positions({"joint_from_remote": 0.5})
        nc_remote.log_joint_velocities({"joint_from_remote": 0.5})
        nc_remote.log_joint_torques({"joint_from_remote": 0.5})
        # Signal that the remote node is ready and has logged data.
        ready_event.set()

        # Keep the process alive for a while so the main process can fetch data.
        while True:
            time.sleep(1)

    except Exception:
        logger.error("Remote node process failed.", exc_info=True)


def regional_node(
    robot_name: str,
    instance: int,
    signalling_url: str | None,
    ready_event: Event,
    stop_event: Event,
    joint_name: str = REMOTE_JOINT_NAME,
) -> None:
    """Run a provider node of a robot instance until the stop event is set.

    Args:
        robot_name: Name of the robot to connect to.
        instance: Robot instance of the node.
        signalling_url: Region URL to register in, or None for the shared endpoint.
        ready_event: Set once the node has logged its data.
        stop_event: The node stops when this is set.
        joint_name: Name of the joint position the node logs.
    """
    if signalling_url is None:
        os.environ.pop(SIGNALLING_URL_VARIABLE, None)
    else:
        os.environ[SIGNALLING_URL_VARIABLE] = signalling_url

    import neuracore as nc_remote

    try:
        nc_remote.login()
        nc_remote.connect_robot(robot_name, instance=instance)
        nc_remote.log_joint_positions(
            {joint_name: REMOTE_JOINT_VALUE}, robot_name=robot_name, instance=instance
        )
        ready_event.set()
        stop_event.wait()
    except Exception:
        logger.error("Regional node process failed.", exc_info=True)
        raise
