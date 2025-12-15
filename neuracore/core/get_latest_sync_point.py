"""Synchronized data retrieval from robot sensors and remote sources.

This module provides functions to collect and synchronize the latest sensor
data from a robot, including camera feeds, joint states, and language inputs.
It supports merging data from local robot streams with data from remote
sources via the Neuracore platform's live data streaming capabilities.
"""

import time

from neuracore_types import DataType, JointData, SynchronizedPoint

from neuracore.api.globals import GlobalSingleton
from neuracore.core.exceptions import RobotError
from neuracore.core.robot import Robot
from neuracore.core.streaming.p2p.consumer.org_nodes_manager import (
    get_org_nodes_manager,
)
from neuracore.core.streaming.p2p.consumer.sync_point_parser import merge_sync_points
from neuracore.core.streaming.p2p.provider.global_live_data_enabled import (
    get_consume_live_data_enabled_manager,
)


def _maybe_add_existing_data(
    existing: JointData | None, to_add: JointData
) -> JointData:
    """Merge joint data from multiple streams into a single data structure.

    Combines joint data while preserving existing values and updating
    timestamps. Used to aggregate data from multiple joint streams.

    Args:
        existing: Existing joint data or None.
        to_add: New joint data to merge.

    Returns:
        Combined JointData with merged values.
    """
    # Check if the joint data already exists
    if existing is None:
        return to_add
    existing.timestamp = to_add.timestamp
    existing.values.update(to_add.values)
    if existing.additional_values and to_add.additional_values:
        existing.additional_values.update(to_add.additional_values)
    return existing


def check_remote_nodes_connected(robot: Robot, num_remote_nodes: int) -> bool:
    """Check if the specified number of remote nodes are connected for a robot.

    Always false if live data is disabled.


    Args:
        robot: The robot instance.
        num_remote_nodes: The number of remote nodes expected to be connected.

    Returns:
        True if the specified number of remote nodes are connected, False otherwise.

    Raises:
        RobotError: If the robot is not initialized.
    """
    if get_consume_live_data_enabled_manager().is_disabled():
        return False

    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")

    org_node_manager = get_org_nodes_manager(robot.org_id)

    consumer_manager = org_node_manager.get_robot_consumer(
        robot_id=robot.id, robot_instance=robot.instance
    )

    if consumer_manager.num_remote_nodes() < num_remote_nodes:
        return False

    return consumer_manager.all_remote_nodes_connected()


def get_latest_sync_point(
    robot: Robot | None = None, include_remote: bool = True
) -> SynchronizedPoint:
    """Create a synchronized data point from current robot sensor streams.

    Collects the latest data from all active robot streams including
    cameras, joint sensors, and language inputs. Organizes the data
    into a synchronized structure with consistent timestamps.

    Returns:
        SynchronizedPoint containing all current sensor data.

    Raises:
        NotImplementedError: If an unsupported stream type is encountered.
    """
    robot = GlobalSingleton()._active_robot
    if robot is None:
        raise ValueError("No active robot found. Please initialize a robot instance.")
    sync_point = SynchronizedPoint(timestamp=time.time())
    for stream_name, stream in robot.list_all_streams().items():
        stream_data = stream.get_latest_data()
        assert stream_data is not None
        dt_name, name_of_sensor_data = stream_name.split(":", 1)
        dt = DataType[dt_name]
        if dt not in sync_point.data:
            sync_point[dt] = {}
        sync_point[dt][name_of_sensor_data] = stream_data

    if not include_remote or get_consume_live_data_enabled_manager().is_disabled():
        return sync_point

    if robot.id is None:
        raise RobotError("Robot not initialized. Call init() first.")

    org_node_manager = get_org_nodes_manager(robot.org_id)

    consumer_manager = org_node_manager.get_robot_consumer(
        robot_id=robot.id, robot_instance=robot.instance
    )

    return merge_sync_points(sync_point, consumer_manager.get_latest_data())
