#!/usr/bin/env python3
import sys
import threading

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray

from .const import QOS_BEST_EFFORT

# ruff: noqa: E402
sys.path.append("/ros2_ws/src/neuracore/examples")
from common.constants import (  # CAMERA_NAMES
    LEFT_ARM_JOINT_NAMES,
    LEFT_GRIPPER_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_GRIPPER_JOINT_NAMES,
)
from common.sim_env import make_sim_env

CAMERA_NAMES = ["top", "angle", "vis"]


class SimulationNode(Node):
    def __init__(self):
        super().__init__("simulation_node")

        # Create publishers for different data streams
        # Joint states at 100Hz
        self.left_arm_pub = self.create_publisher(
            JointState, "/left_arm/joint_states", QOS_BEST_EFFORT
        )

        self.right_arm_pub = self.create_publisher(
            JointState, "/right_arm/joint_states", QOS_BEST_EFFORT
        )

        # Camera images at 30Hz
        self.camera_pubs = {}
        for cam_name in CAMERA_NAMES:
            self.camera_pubs[cam_name] = self.create_publisher(
                Image, f"/camera/{cam_name}/image_raw", QOS_BEST_EFFORT
            )

        # Subscribe to action commands
        self.action_sub = self.create_subscription(
            Float32MultiArray, "/robot/actions", self.action_callback, QOS_BEST_EFFORT
        )

        # Initialize simulation
        self.get_logger().info("Initializing simulation environment...")
        self.env = make_sim_env()
        self.cv_bridge = CvBridge()
        self.ts = self.env.reset()
        self.action = None
        self.action_lock = threading.Lock()

        # Create timers for different frequency publishers
        self.create_timer(1.0 / 40.0, self.publish_joint_states)  # 40Hz
        self.create_timer(1.0 / 10.0, self.publish_camera_images)  # 10Hz

        # Create timer for simulation step
        self.create_timer(1.0 / 50.0, self.simulation_step)  # 50Hz

        self.get_logger().info("Simulation node initialized and running")

    def action_callback(self, msg):
        with self.action_lock:
            # Convert ROS message to action format expected by simulation
            action_array = np.array(msg.data)

            # Process the action into the format expected by the simulation
            # This depends on the exact format of your action space
            # For example, if using a dictionary format:
            left_arm_action = action_array[:6]
            right_arm_action = action_array[7:13]
            left_gripper_action = action_array[6]
            right_gripper_action = action_array[13]

            self.action = {
                "left_arm": left_arm_action,
                "left_gripper": left_gripper_action,
                "right_arm": right_arm_action,
                "right_gripper": right_gripper_action,
            }

    def simulation_step(self):
        with self.action_lock:
            if self.action is not None:
                # Convert action dictionary to the format expected by env.step
                # This depends on the exact format of your environment
                action_list = np.concatenate([
                    self.action["left_arm"],
                    [self.action["left_gripper"]],
                    self.action["right_arm"],
                    [self.action["right_gripper"]],
                ])

                # Step the simulation
                self.ts = self.env.step(action_list)
                self.action = None
            else:
                # If no action received, keep the simulation running
                # This could be a no-op action or just a physics step
                # For a real robot, you might want to implement a safety controller here
                pass

    def publish_joint_states(self):
        if not hasattr(self, "ts"):
            return

        # Extract joint positions from the simulation state
        qpos = self.ts.observation["qpos"]

        # Create JointState messages
        left_js = JointState()
        right_js = JointState()

        # Set header
        left_js.header.stamp = self.get_clock().now().to_msg()
        right_js.header.stamp = self.get_clock().now().to_msg()

        left_js.name = LEFT_ARM_JOINT_NAMES + LEFT_GRIPPER_JOINT_NAMES
        right_js.name = RIGHT_ARM_JOINT_NAMES + RIGHT_GRIPPER_JOINT_NAMES

        # Set joint positions
        # This assumes qpos is a dictionary with joint names as keys
        left_js.position = [qpos[joint] for joint in left_js.name]
        right_js.position = [qpos[joint] for joint in right_js.name]

        # Publish
        self.left_arm_pub.publish(left_js)
        self.right_arm_pub.publish(right_js)

    def publish_camera_images(self):
        if not hasattr(self, "ts"):
            return

        # Extract images from the simulation state
        images = self.ts.observation["images"]

        # Publish each camera image
        for cam_name, img in images.items():
            # Convert to ROS Image message
            img_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="rgb8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = f"camera_{cam_name}_frame"

            # Publish
            self.camera_pubs[cam_name].publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimulationNode()

    # Use a MultiThreadedExecutor to handle multiple timers concurrently
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        node.get_logger().info("Starting simulation node...")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received")
    finally:
        node.get_logger().info("Shutting down simulation node")
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
