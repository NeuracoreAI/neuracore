"""Provider processes of the streaming integration tests."""

import logging
import time
from multiprocessing.synchronize import Event

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


rgb_test_image = np.full((10, 10, 3), 100, dtype=np.uint8)
rgb_test_image[0, 0] = [255, 0, 0]  # Red pixel to identify the image


def remote_node_logger(robot_name: str, instance: int, ready_event: Event):
    """
    Simulates a remote node that logs data for a given robot instance.
    This function is intended to be run in a separate process.
    """
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
        # Don't set the event, so the main test will time out and fail.
