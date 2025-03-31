#!/usr/bin/env python3
import argparse
import sys
from typing import List

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray

import neuracore as nc

from .const import QOS_BEST_EFFORT

# ruff: noqa: E402
sys.path.append("/ros2_ws/src/neuracore/examples")
from common.constants import BIMANUAL_VIPERX_URDF_PATH

CAMERA_NAMES = ["top", "angle", "vis"]


class LeftArmLoggerNode(Node):
    """Node dedicated to logging left arm joint states."""

    def __init__(self):
        super().__init__("left_arm_logger_node")

        # Connect to our robot in this node
        nc.connect_robot("Mujoco VX300s")

        # Subscribe to left arm joint states
        self.left_arm_sub = self.create_subscription(
            JointState,
            "/left_arm/joint_states",
            self.left_arm_callback,
            QOS_BEST_EFFORT,
        )

        self.get_logger().info("Left arm logger node initialized and running")

    def left_arm_callback(self, msg):
        """Process left arm joint states and log to neuracore."""
        # Convert ROS JointState to format expected by neuracore
        joint_positions = {}
        for i, name in enumerate(msg.name):
            joint_positions[name] = msg.position[i]

        self.get_logger().info(f"Received left arm joint states: {joint_positions}")

        # Log only left arm data
        nc.log_joint_positions(joint_positions)


class RightArmLoggerNode(Node):
    """Node dedicated to logging right arm joint states."""

    def __init__(self):
        super().__init__("right_arm_logger_node")

        # Subscribe to right arm joint states
        self.right_arm_sub = self.create_subscription(
            JointState,
            "/right_arm/joint_states",
            self.right_arm_callback,
            QOS_BEST_EFFORT,
        )

        self.get_logger().info("Right arm logger node initialized and running")

    def right_arm_callback(self, msg):
        """Process right arm joint states and log to neuracore."""
        # Convert ROS JointState to format expected by neuracore
        joint_positions = {}
        for i, name in enumerate(msg.name):
            joint_positions[name] = msg.position[i]

        self.get_logger().info(f"Received right arm joint states: {joint_positions}")

        # Log only right arm data
        nc.log_joint_positions(joint_positions, "right_arm")


class CameraLoggerNode(Node):
    """Node dedicated to logging camera images."""

    def __init__(self, camera_names: List[str]):
        super().__init__("camera_logger_node")

        # Initialize CV bridge for image conversion
        self.cv_bridge = CvBridge()

        # Subscribe to camera images
        self.camera_subs = {}
        for cam_name in camera_names:
            self.camera_subs[cam_name] = self.create_subscription(
                Image,
                f"/camera/{cam_name}/image_raw",
                lambda msg, cam=cam_name: self.camera_callback(msg, cam),
                QOS_BEST_EFFORT,
            )

        self.get_logger().info("Camera logger node initialized and running")

    def camera_callback(self, msg, cam_name):
        """Process camera images and log to neuracore."""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.get_logger().info(
                f"Received {cam_name} camera image with shape: {cv_image.shape}"
            )

            # Log image to neuracore
            nc.log_rgb(cam_name, cv_image)
        except Exception as e:
            self.get_logger().error(f"Error processing {cam_name} camera: {e}")


class ActionLoggerNode(Node):
    """Node dedicated to logging robot actions."""

    def __init__(self):
        super().__init__("action_logger_node")

        # Subscribe to actions
        self.action_sub = self.create_subscription(
            Float32MultiArray, "/robot/actions", self.action_callback, QOS_BEST_EFFORT
        )

        self.get_logger().info("Action logger node initialized and running")

    def action_callback(self, msg):
        """Process action commands and log to neuracore."""
        # Convert ROS Float32MultiArray to format expected by neuracore
        action_array = np.array(msg.data)

        # Format the action as a dictionary
        left_arm_action = action_array[:6]
        right_arm_action = action_array[7:13]
        left_gripper_action = action_array[6]
        right_gripper_action = action_array[13]

        action = {
            "left_arm": left_arm_action.tolist(),
            "left_gripper": float(left_gripper_action),
            "right_arm": right_arm_action.tolist(),
            "right_gripper": float(right_gripper_action),
        }
        self.get_logger().info(f"Received action: {action}")

        # Log action to neuracore
        nc.log_action(action)


class NeuracoreManagerNode(Node):
    """Node to handle Neuracore initialization and recording management."""

    def __init__(self, record: bool = False, dataset_name: str = "ROS2 Dataset"):
        super().__init__("neuracore_manager_node")

        self.record = record

        # Initialize neuracore
        self.get_logger().info("Initializing neuracore...")
        nc.login()
        nc.connect_robot(
            robot_name="Mujoco VX300s",
            urdf_path=BIMANUAL_VIPERX_URDF_PATH,
            overwrite=False,
        )

        # Setup recording
        if self.record:
            nc.create_dataset(
                name=dataset_name, description="ROS2 distributed data collection"
            )
            self.get_logger().info(f"Created dataset: {dataset_name}")

            # Start recording
            nc.start_recording()
            self.get_logger().info("Started recording")

    def shutdown(self):
        """Clean shutdown of neuracore."""
        if self.record:
            self.get_logger().info("Stopping recording...")
            # nc.stop_recording()
            self.get_logger().info("Recording stopped")


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        action="store_true",
        help="Whether to record with neuracore",
        default=False,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset to create",
        default="ROS2 Dataset",
    )
    parsed_args, remaining = parser.parse_known_args()

    # Create all nodes
    neuracore_manager = NeuracoreManagerNode(
        record=parsed_args.record, dataset_name=parsed_args.dataset_name
    )
    left_arm_logger = LeftArmLoggerNode()
    right_arm_logger = RightArmLoggerNode()
    camera_logger = CameraLoggerNode(camera_names=CAMERA_NAMES)
    action_logger = ActionLoggerNode()

    # Create a multithreaded executor to spin all nodes concurrently
    executor = MultiThreadedExecutor()

    # Add nodes to the executor
    executor.add_node(neuracore_manager)
    executor.add_node(left_arm_logger)
    executor.add_node(right_arm_logger)
    executor.add_node(camera_logger)
    executor.add_node(action_logger)

    try:
        # Spin all nodes
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        neuracore_manager.shutdown()

        # Destroy all nodes
        neuracore_manager.destroy_node()
        left_arm_logger.destroy_node()
        right_arm_logger.destroy_node()
        camera_logger.destroy_node()
        action_logger.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
