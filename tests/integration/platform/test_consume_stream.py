"""
Integration test for verifying that get_latest_data aggregates data from multiple nodes.

"""

import logging
import multiprocessing
import time
from multiprocessing.synchronize import Event

import numpy as np
import pytest
from neuracore_types import DataType

import neuracore as nc

# Configure logging
logger = logging.getLogger(__name__)


rgb_test_image = np.full((10, 10, 3), 100, dtype=np.uint8)
rgb_test_image[0, 0] = [255, 0, 0]  # Red pixel to identify the image


MAXIMUM_WAITING_TIME_S = 60


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


# This test can be flaky, so run it multiple times to increase confidence in stability.s
@pytest.mark.parametrize("execution_number", range(10))
def test_get_latest_data_from_multiple_nodes(execution_number: int):
    """
    Tests that get_latest_data correctly aggregates data logged from multiple
    processes (nodes) for the same robot instance.
    """
    robot_name = "multinode-test-robot"
    instance = 0
    nc.login()
    # The main process creates/connects to the robot.
    nc.connect_robot(robot_name, instance=instance, overwrite=False)

    # 2. Launch remote node: Start a separate process to log data.
    ctx = multiprocessing.get_context("spawn")
    remote_ready_event = ctx.Event()

    remote_process = ctx.Process(
        target=remote_node_logger,
        args=(robot_name, instance, remote_ready_event),
    )
    remote_process.start()

    # 3. Log data from main process while waiting for the remote node.
    main_joint_velocities = {"j_vel_from_main": -0.5}
    nc.log_joint_velocities(
        main_joint_velocities, robot_name=robot_name, instance=instance
    )

    # Wait for the remote node to be up and to have logged its data.
    is_ready = remote_ready_event.wait(timeout=MAXIMUM_WAITING_TIME_S)
    assert is_ready, "Remote node process did not signal readiness in time."

    # 4. Fetch and verify data: Call get_latest_data and check the SynchronizedPoint.
    try:
        start_connection_time = time.time()
        while not nc.check_remote_nodes_connected(
            num_remote_nodes=1, robot_name=robot_name, instance=instance
        ):
            if time.time() - start_connection_time > MAXIMUM_WAITING_TIME_S:
                assert False, "Timed out waiting for remote nodes to fully connect."
            time.sleep(0.25)

        sync_point = nc.get_latest_sync_point(
            robot_name=robot_name, instance=instance, include_remote=True
        )

        assert nc.check_remote_nodes_connected(
            num_remote_nodes=1, robot_name=robot_name, instance=instance
        ), "Remote nodes should remain connected after fetching data."

        # 5. Assertions:

        assert sync_point is not None, "SynchronizedPoint not found"

        # --- Verify data from the main process ---
        assert (
            DataType.JOINT_VELOCITIES in sync_point.data
        ), "Joint velocities from main process not found"
        assert "j_vel_from_main" in sync_point[DataType.JOINT_VELOCITIES]
        assert sync_point[DataType.JOINT_VELOCITIES]["j_vel_from_main"].value == -0.5

        # --- Verify data from the remote process ---
        assert (
            DataType.JOINT_POSITIONS in sync_point.data
        ), "Joint positions from remote process not found"
        assert "joint_from_remote" in sync_point[DataType.JOINT_POSITIONS]
        assert sync_point[DataType.JOINT_POSITIONS]["joint_from_remote"].value == 0.5

        assert (
            DataType.RGB_IMAGES in sync_point.data
        ), "RGB image from remote process not found"

        assert "cam_from_remote" in sync_point[DataType.RGB_IMAGES]
        remote_image_data = sync_point[DataType.RGB_IMAGES]["cam_from_remote"]
        assert isinstance(remote_image_data.frame, np.ndarray)
        np.testing.assert_array_equal(remote_image_data.frame, rgb_test_image)

        assert (
            DataType.CUSTOM_1D in sync_point.data
        ), "Custom data from remote process not found"
        assert "custom_data_from_remote" in sync_point[DataType.CUSTOM_1D]
        np.testing.assert_array_equal(
            sync_point[DataType.CUSTOM_1D]["custom_data_from_remote"].data,
            np.array([42.0], dtype=np.float32),
        )

        assert (
            DataType.JOINT_TARGET_POSITIONS in sync_point.data
        ), "Joint target positions from remote process not found"
        assert "joint_from_remote" in sync_point[DataType.JOINT_TARGET_POSITIONS]
        assert (
            sync_point[DataType.JOINT_TARGET_POSITIONS]["joint_from_remote"].value
            == 0.5
        )

        assert (
            DataType.JOINT_TORQUES in sync_point.data
        ), "Joint torques from remote process not found"
        assert "joint_from_remote" in sync_point[DataType.JOINT_TORQUES]
        assert sync_point[DataType.JOINT_TORQUES]["joint_from_remote"].value == 0.5

    finally:
        # 6. Teardown: Clean up the remote process.
        remote_process.terminate()
        remote_process.join(timeout=5)
