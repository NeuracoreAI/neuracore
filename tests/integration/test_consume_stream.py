"""
Integration test for verifying that get_latest_data aggregates data from multiple nodes.

"""

import logging
import multiprocessing
import time
from multiprocessing.synchronize import Event

import numpy as np

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
        nc_remote.log_custom_data("custom_data_from_remote", {"key": "value"})
        nc_remote.log_gripper_data({"left_gripper": 0.5, "right_gripper": 0.5})
        nc_remote.log_joint_target_positions({"joint_from_remote": 0.5})
        nc_remote.log_joint_velocities({"joint_from_remote": 0.5})
        nc_remote.log_joint_torques({"joint_from_remote": 0.5})
        # Signal that the remote node is ready and has logged data.
        ready_event.set()

        # Keep the process alive for a while so the main process can fetch data.
        while True:
            time.sleep(1)

    except Exception as e:
        # Log any exceptions to help with debugging.
        logger.error(f"Remote node process failed: {e}", exc_info=True)
        # Don't set the event, so the main test will time out and fail.


EXPECTED_REMOTE_DATATYPES = (
    nc.DataType.JOINT_POSITIONS,
    nc.DataType.RGB_IMAGE,
    nc.DataType.CUSTOM,
    nc.DataType.END_EFFECTORS,
    nc.DataType.JOINT_TARGET_POSITIONS,
    nc.DataType.JOINT_VELOCITIES,
    nc.DataType.JOINT_TORQUES,
)


def test_get_latest_data_from_multiple_nodes():
    """
    Tests that get_latest_data correctly aggregates data logged from multiple
    processes (nodes) for the same robot instance.
    """
    robot_name = "multinode-test-robot"
    instance = 0
    nc.login()
    # The main process creates/connects to the robot.
    nc.connect_robot(robot_name, instance=instance, overwrite=True)

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

    # 4. Fetch and verify data: Call get_latest_data and check the SyncPoint.
    try:
        start_connection_time = time.time()
        while not nc.check_remote_nodes_connected(
            num_remote_nodes=1,
            data_types=EXPECTED_REMOTE_DATATYPES,
            robot_name=robot_name,
            instance=instance,
        ):
            if time.time() - start_connection_time > MAXIMUM_WAITING_TIME_S:
                assert False, "Timed out waiting for remote nodes to fully connect."
            time.sleep(0.25)

        sync_point = nc.get_latest_sync_point(
            robot_name=robot_name, instance=instance, include_remote=True
        )

        assert nc.check_remote_nodes_connected(
            num_remote_nodes=1,
            data_types=EXPECTED_REMOTE_DATATYPES,
            robot_name=robot_name,
            instance=instance,
        ), "Remote nodes should remain connected after fetching data."

        # 5. Assertions:

        assert sync_point is not None, "SyncPoint not found"

        # --- Verify data from the main process ---
        assert (
            sync_point.joint_velocities is not None
        ), "Joint velocities from main process not found"
        assert "j_vel_from_main" in sync_point.joint_velocities.values
        assert sync_point.joint_velocities.values["j_vel_from_main"] == -0.5

        # --- Verify data from the remote process ---
        assert (
            sync_point.joint_positions is not None
        ), "Joint positions from remote process not found"
        assert "joint_from_remote" in sync_point.joint_positions.values
        assert sync_point.joint_positions.values["joint_from_remote"] == 0.5

        assert (
            sync_point.rgb_images is not None
        ), "RGB image from remote process not found"

        assert "rgb_cam_from_remote" in sync_point.rgb_images
        remote_image_data = sync_point.rgb_images["rgb_cam_from_remote"]
        assert isinstance(remote_image_data.frame, np.ndarray)
        np.testing.assert_array_equal(remote_image_data.frame, rgb_test_image)

        assert (
            sync_point.custom_data is not None
        ), "Custom data from remote process not found"
        assert "custom_data_from_remote" in sync_point.custom_data
        assert (
            sync_point.custom_data["custom_data_from_remote"].data.get("key", None)
            == "value"
        )

        assert (
            sync_point.end_effectors is not None
        ), "Gripper data from remote process not found"
        assert "left_gripper" in sync_point.end_effectors.open_amounts
        assert sync_point.end_effectors.open_amounts["left_gripper"] == 0.5
        assert "right_gripper" in sync_point.end_effectors.open_amounts
        assert sync_point.end_effectors.open_amounts["right_gripper"] == 0.5

        assert (
            sync_point.joint_target_positions is not None
        ), "Joint target positions from remote process not found"
        assert "joint_from_remote" in sync_point.joint_target_positions.values
        assert sync_point.joint_target_positions.values["joint_from_remote"] == 0.5

        assert (
            sync_point.joint_torques is not None
        ), "Joint torques from remote process not found"
        assert "joint_from_remote" in sync_point.joint_torques.values
        assert sync_point.joint_torques.values["joint_from_remote"] == 0.5

    finally:
        # 6. Teardown: Clean up the remote process.
        remote_process.terminate()
        remote_process.join(timeout=5)


def relay_node_round_trip(robot_name: str, source_instance: int, dest_instance: int):
    """
    Process function: Connects to the source instance of a robot, waits for a SyncPoint,
    and re-logs it to the destination instance to simulate a relay round-trip.
    """
    import time

    import neuracore as nc_relay

    try:
        nc_relay.login()
        nc_relay.connect_robot(robot_name, instance=source_instance)
        nc_relay.connect_robot(robot_name, instance=dest_instance)

        # Wait for expected data on the source instance before pulling the SyncPoint.
        start = time.time()
        while not nc_relay.check_remote_nodes_connected(
            num_remote_nodes=0,
            data_types=EXPECTED_REMOTE_DATATYPES,
            robot_name=robot_name,
            instance=source_instance,
        ):
            if time.time() - start > MAXIMUM_WAITING_TIME_S:
                raise TimeoutError("Timeout waiting for data on source instance.")
            time.sleep(0.25)

        sync_point = nc_relay.get_latest_sync_point(
            robot_name, instance=source_instance
        )
        assert (
            sync_point is not None
        ), "Relay node: No SyncPoint found on source instance"

        if sync_point.joint_positions:
            nc_relay.log_joint_positions(
                positions=sync_point.joint_positions.values,
                additional_urdf_positions=sync_point.joint_positions.additional_values,
                robot_name=robot_name,
                instance=dest_instance,
                timestamp=sync_point.timestamp,
            )
        if sync_point.rgb_images:
            for cam_name, image in sync_point.rgb_images.items():
                nc_relay.log_rgb(
                    camera_id=cam_name,
                    image=image.frame,
                    extrinsics=image.extrinsics,
                    intrinsics=image.intrinsics,
                    robot_name=robot_name,
                    instance=dest_instance,
                    timestamp=sync_point.timestamp,
                )
        if sync_point.custom_data:
            for key, val in sync_point.custom_data.items():
                nc_relay.log_custom_data(
                    name=key,
                    data=val.data,
                    robot_name=robot_name,
                    instance=dest_instance,
                    timestamp=sync_point.timestamp,
                )
        if sync_point.end_effectors:
            nc_relay.log_gripper_data(
                open_amounts=sync_point.end_effectors.open_amounts,
                robot_name=robot_name,
                instance=dest_instance,
                timestamp=sync_point.timestamp,
            )
        if sync_point.joint_target_positions:
            nc_relay.log_joint_target_positions(
                target_positions=sync_point.joint_target_positions.values,
                additional_urdf_positions=sync_point.joint_target_positions.additional_values,
                robot_name=robot_name,
                instance=dest_instance,
                timestamp=sync_point.timestamp,
            )
        if sync_point.joint_velocities:
            nc_relay.log_joint_velocities(
                velocities=sync_point.joint_velocities.values,
                additional_urdf_velocities=sync_point.joint_velocities.additional_values,
                robot_name=robot_name,
                instance=dest_instance,
                timestamp=sync_point.timestamp,
            )
        if sync_point.joint_torques:
            nc_relay.log_joint_torques(
                torques=sync_point.joint_torques.values,
                additional_urdf_torques=sync_point.joint_torques.additional_values,
                robot_name=robot_name,
                instance=dest_instance,
                timestamp=sync_point.timestamp,
            )

        # Keep the process alive to simulate a real remote node
        while True:
            time.sleep(1)

    except Exception as e:
        logger.error(f"Relay node failed: {e}", exc_info=True)
        raise


def test_sync_point_round_trip_between_instances():
    """
    Tests that data can be logged to instance 0, relayed through a remote node,
    and re-logged to instance 1. Validates data round-trip integrity using SyncPoint.
    """
    robot_name = "roundtrip-test-robot"
    sender_instance = 0
    receiver_instance = 1

    nc.login()
    nc.connect_robot(robot_name=robot_name, instance=sender_instance, overwrite=True)
    nc.connect_robot(robot_name=robot_name, instance=receiver_instance, overwrite=False)

    # Log original data to instance 0
    nc.log_joint_positions(
        positions={"joint_roundtrip": 1.0},
        robot_name=robot_name,
        instance=sender_instance,
    )
    nc.log_rgb(
        "cam_roundtrip", rgb_test_image, robot_name=robot_name, instance=sender_instance
    )
    nc.log_custom_data(
        "custom_data_roundtrip",
        {"hello": "world"},
        robot_name=robot_name,
        instance=sender_instance,
    )
    nc.log_gripper_data(
        {"grip_left": 0.5}, robot_name=robot_name, instance=sender_instance
    )
    nc.log_joint_target_positions(
        {"joint_roundtrip": 1.0}, robot_name=robot_name, instance=sender_instance
    )
    nc.log_joint_velocities(
        {"joint_roundtrip": 1.0}, robot_name=robot_name, instance=sender_instance
    )
    nc.log_joint_torques(
        {"joint_roundtrip": 1.0}, robot_name=robot_name, instance=sender_instance
    )

    # Start relay node in separate process
    ctx = multiprocessing.get_context("spawn")
    relay_proc = ctx.Process(
        target=relay_node_round_trip,
        args=(robot_name, sender_instance, receiver_instance),
    )
    relay_proc.start()

    try:
        # Wait until the data from the relay is connected as a remote node on instance 1
        start = time.time()
        while not nc.check_remote_nodes_connected(
            num_remote_nodes=1,
            data_types=EXPECTED_REMOTE_DATATYPES,
            robot_name=robot_name,
            instance=receiver_instance,
        ):
            if time.time() - start > MAXIMUM_WAITING_TIME_S:
                assert False, "Timed out waiting for re-logged data on instance 1"
            time.sleep(0.25)

        # Retrieve SyncPoint on instance 1
        sync_point = nc.get_latest_sync_point(
            robot_name, instance=receiver_instance, include_remote=True
        )
        assert sync_point is not None, "No SyncPoint found on instance 1"

        # --- Verify data integrity ---
        assert "joint_roundtrip" in sync_point.joint_positions.values
        assert sync_point.joint_positions.values["joint_roundtrip"] == 1.0

        assert "rgb_cam_roundtrip" in sync_point.rgb_images
        np.testing.assert_array_equal(
            sync_point.rgb_images["rgb_cam_roundtrip"].frame, rgb_test_image
        )

        assert "custom_data_roundtrip" in sync_point.custom_data
        assert sync_point.custom_data["custom_data_roundtrip"].data["hello"] == "world"

        assert "grip_left" in sync_point.end_effectors.open_amounts
        assert sync_point.end_effectors.open_amounts["grip_left"] == 0.5

        assert sync_point.joint_target_positions.values["joint_roundtrip"] == 1.0
        assert sync_point.joint_velocities.values["joint_roundtrip"] == 1.0
        assert sync_point.joint_torques.values["joint_roundtrip"] == 1.0

    finally:
        relay_proc.terminate()
        relay_proc.join(timeout=5)
