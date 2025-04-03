#!/usr/bin/env python3
import sys
import threading

import rclpy
from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from .const import QOS_BEST_EFFORT

# ruff: noqa: E402
sys.path.append("/ros2_ws/src/neuracore/examples")
from common.constants import (  # CAMERA_NAMES
    BIMANUAL_VIPERX_URDF_PATH,
    LEFT_ARM_JOINT_NAMES,
    LEFT_GRIPPER_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_GRIPPER_JOINT_NAMES,
)

import neuracore as nc

CAMERA_NAMES = ["top", "angle", "vis"]


class SimulationNode(Node):
    def __init__(self):
        super().__init__("simulation_node")

        nc.login()
        nc.connect_robot(
            robot_name="Mujoco VX300s",
            urdf_path=BIMANUAL_VIPERX_URDF_PATH,
            overwrite=False,
        )

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

        # Initialize simulation
        self.get_logger().info("Initializing simulation environment...")
        self.cv_bridge = CvBridge()
        self.action = None
        self.action_lock = threading.Lock()

        self.env = None
        self.ts = None
        self.action_traj = []
        self.record = True

        # Initialize environment first, before setting up timers
        self.initialize_environment()

        if self.record:
            dataset_name = "ROS2_BimanualVX300s_Dataset"
            nc.create_dataset(
                name=dataset_name, description="ROS2 distributed data collection"
            )
            self.get_logger().info(f"Created dataset: {dataset_name}")

            # Start recording
            nc.start_recording()
            self.get_logger().info("Started recording")

        # Create all timers in a specific order, with simulation_step last
        self.joint_states_timer = self.create_timer(
            1.0 / 40.0, self.publish_joint_states
        )  # 40Hz
        self.camera_timer = self.create_timer(
            1.0 / 10.0, self.publish_camera_images
        )  # 10Hz

        # Create timer for simulation step - must be last to ensure environment is ready
        self.sim_step_timer = self.create_timer(
            1.0 / 50.0, self.simulation_step
        )  # 50Hz

    def initialize_environment(self):
        """Initialize the environment on the main thread before setting up timers"""
        try:
            from common.rollout_utils import rollout_policy
            from common.sim_env import BOX_POSE, make_sim_env

            self.get_logger().info("Generating a demo action trajectory...")
            self.action_traj, subtask_info, max_reward = rollout_policy()

            BOX_POSE[0] = subtask_info
            self.env = make_sim_env()
            self.ts = self.env.reset()

            self.get_logger().info("Environment initialized successfully")
        except Exception as e:
            self.get_logger().error(f"Error initializing environment: {e}")
            raise

    def simulation_step(self):
        """Execute one simulation step, keeping GL context on the same thread"""
        try:
            if len(self.action_traj) > 0:
                action = self.action_traj.pop(0)
                nc.log_action(action)
                self.ts = self.env.step(list(action.values()))
            else:
                self.sim_step_timer.cancel()
                self.get_logger().info("No more actions in the trajectory")
                if self.record:
                    self.get_logger().info("Stopping recording...")
                    nc.stop_recording()
                    self.get_logger().info("Recording stopped")
                self.sim_step_timer.cancel()
        except Exception as e:
            self.get_logger().error(f"Error in simulation step: {e}")

    def publish_joint_states(self):
        """Publish joint states without modifying the environment"""
        try:
            if not self.ts:
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
            left_js.position = [qpos[joint] for joint in left_js.name]
            right_js.position = [qpos[joint] for joint in right_js.name]

            # Publish
            self.left_arm_pub.publish(left_js)
            self.right_arm_pub.publish(right_js)
        except Exception as e:
            self.get_logger().error(f"Error publishing joint states: {e}")

    def publish_camera_images(self):
        """Publish camera images without modifying the environment"""
        try:
            if not self.ts:
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
        except Exception as e:
            self.get_logger().error(f"Error publishing camera images: {e}")


def main(args=None):
    rclpy.init(args=args)

    # Create the node
    node = SimulationNode()

    # Use a SingleThreadedExecutor to ensure the GL context is on the same thread
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        node.get_logger().info(
            "Starting simulation node with single-threaded executor..."
        )
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received")
    except Exception as e:
        node.get_logger().error(f"Error during execution: {e}")
    finally:
        node.get_logger().info("Shutting down simulation node")
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
