import sys
from typing import List

import rclpy
from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

import neuracore as nc

from .const import QOS_BEST_EFFORT

# ruff: noqa: E402
sys.path.append("/ros2_ws/src/neuracore/examples")

CAMERA_NAMES = ["top", "angle", "vis"]


class LeftArmLoggerNode(Node):
    """Node dedicated to logging left arm joint states."""

    def __init__(self):
        super().__init__("left_arm_logger_node")

        # Connect to our robot in this node
        nc.login()
        nc.connect_robot("Mujoco VX300s")

        # Subscribe to left arm joint states
        self.left_arm_sub = self.create_subscription(
            JointState,
            "/left_arm/joint_states",
            self.left_arm_callback,
            QOS_BEST_EFFORT,
        )

    def left_arm_callback(self, msg):
        """Process left arm joint states and log to neuracore."""
        # Convert ROS JointState to format expected by neuracore
        joint_positions = {}
        for i, name in enumerate(msg.name):
            joint_positions[name] = float(msg.position[i])

        # Log only left arm data
        nc.log_joint_positions(joint_positions)


class RightArmLoggerNode(Node):
    """Node dedicated to logging right arm joint states."""

    def __init__(self):
        super().__init__("right_arm_logger_node")

        # Connect to our robot in this node
        nc.login()
        nc.connect_robot("Mujoco VX300s")

        # Subscribe to right arm joint states
        self.right_arm_sub = self.create_subscription(
            JointState,
            "/right_arm/joint_states",
            self.right_arm_callback,
            QOS_BEST_EFFORT,
        )

    def right_arm_callback(self, msg):
        """Process right arm joint states and log to neuracore."""
        # Convert ROS JointState to format expected by neuracore
        joint_positions = {}
        for i, name in enumerate(msg.name):
            joint_positions[name] = float(msg.position[i])

        # Log only right arm data
        nc.log_joint_positions(joint_positions)


class CameraLoggerNode(Node):
    """Node dedicated to logging camera images."""

    def __init__(self, camera_names: List[str]):
        super().__init__("camera_logger_node")

        # Connect to our robot in this node
        nc.login()
        nc.connect_robot("Mujoco VX300s")

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

    def camera_callback(self, msg, cam_name):
        """Process camera images and log to neuracore."""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            # Log image to neuracore
            nc.log_rgb(cam_name, cv_image)
        except Exception as e:
            self.get_logger().error(f"Error processing {cam_name} camera: {e}")


def main(args=None):
    rclpy.init(args=args)

    # Create all nodes
    left_arm_logger = LeftArmLoggerNode()
    right_arm_logger = RightArmLoggerNode()
    camera_logger = CameraLoggerNode(camera_names=CAMERA_NAMES)

    # Create a multithreaded executor to spin all nodes concurrently
    executor = SingleThreadedExecutor()

    # Add nodes to the executor
    executor.add_node(left_arm_logger)
    executor.add_node(right_arm_logger)
    executor.add_node(camera_logger)

    try:
        # Spin all nodes
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        left_arm_logger.destroy_node()
        right_arm_logger.destroy_node()
        camera_logger.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
