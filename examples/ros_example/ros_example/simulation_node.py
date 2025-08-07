#!/usr/bin/env python3
import sys
import threading

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from .const import QOS_BEST_EFFORT

# ruff: noqa: E402
sys.path.append("/ros2_ws/src/neuracore/examples")
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH

import neuracore as nc

CAMERA_NAMES = ["angle"]


class SimulationNode(Node):
    def __init__(self):
        super().__init__("simulation_node")

        # Declare parameters
        self.declare_parameter("max_episodes", 10)
        self.max_episodes = self.get_parameter("max_episodes").value
        self.current_episode = 0
        self.is_complete = False

        nc.login()
        nc.connect_robot(
            robot_name="Mujoco VX300s",
            urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
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
        self.cv_bridge = CvBridge()
        self.action = None
        self.action_lock = threading.Lock()

        self.env = None
        self.obs = None
        self.action_traj = []
        self.current_action_traj = []
        self.record = True

        # Initialize environment first, before setting up timers
        self.initialize_environment()

        if self.record:
            dataset_name = "ROS2_BimanualVX300s_Dataset"
            nc.create_dataset(
                name=dataset_name, description="ROS2 distributed data collection"
            )
            self.get_logger().info(f"Created dataset: {dataset_name}")

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

        self.get_logger().info(f"Simulation will run for {self.max_episodes} episodes")

    def initialize_environment(self):
        """Initialize the environment on the main thread before setting up timers"""
        try:
            from common.rollout_utils import rollout_policy
            from common.transfer_cube import make_sim_env

            self.get_logger().info(
                "Generating a demo action trajectory. This will take a few moments..."
            )
            self.action_traj = rollout_policy()
            self.current_action_traj = self.action_traj.copy()
            self.get_logger().info(
                "Demo action trajectory generated successfully! "
                "This will now be replayed as if a human is controlling the robot."
            )

            self.env = make_sim_env()
            self.obs = self.env.reset()

        except Exception as e:
            self.get_logger().error(f"Error initializing environment: {e}")
            raise

    def simulation_step(self):
        """Execute one simulation step, keeping GL context on the same thread"""
        try:
            if self.is_complete:
                return

            if len(self.current_action_traj) > 0:
                action = self.current_action_traj.pop(0)
                nc.log_joint_target_positions(action)
                self.obs, _, _ = self.env.step(np.array(list(action.values())))
            else:
                if self.record:
                    nc.stop_recording()
                    self.get_logger().info("Recording stopped")

                # Increment episode counter
                self.current_episode += 1
                self.get_logger().info(
                    f"Completed episode {self.current_episode} of {self.max_episodes}"
                )

                # Check if we've reached the maximum number of episodes
                if self.current_episode >= self.max_episodes:
                    self.is_complete = True
                    self.get_logger().info("All episodes completed. CTRL-C to stop.")
                    return

                # Reset the action trajectory
                self.obs = self.env.reset()
                self.current_action_traj = self.action_traj.copy()

                if self.record:
                    nc.start_recording()
                    self.get_logger().info("Recording Started")

        except Exception as e:
            self.get_logger().error(f"Error in simulation step: {e}")

    def publish_joint_states(self):
        """Publish joint states without modifying the environment"""
        try:
            if not self.obs or self.is_complete:
                return

            # Extract joint positions from the simulation state
            qpos = self.obs.qpos

            # Create JointState messages
            left_js = JointState()
            right_js = JointState()

            # Set header
            left_js.header.stamp = self.get_clock().now().to_msg()
            right_js.header.stamp = self.get_clock().now().to_msg()

            left_js.name = (
                self.env.LEFT_ARM_JOINT_NAMES + self.env.LEFT_GRIPPER_JOINT_NAMES
            )
            right_js.name = (
                self.env.RIGHT_ARM_JOINT_NAMES + self.env.RIGHT_GRIPPER_JOINT_NAMES
            )

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
            if not self.obs or self.is_complete:
                return

            # Publish each camera image
            for cam_name, cam_data in self.obs.cameras.items():
                # Convert to ROS Image message
                img_msg = self.cv_bridge.cv2_to_imgmsg(cam_data.rgb, encoding="rgb8")
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
