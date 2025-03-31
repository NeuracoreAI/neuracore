#!/usr/bin/env python3
import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from .const import QOS_BEST_EFFORT

# ruff: noqa: E402
sys.path.append("/ros2_ws/src/neuracore/examples")
from common.rollout_utils import rollout_policy


class ActionGeneratorNode(Node):
    """Client node that sends action commands to the simulation."""

    def __init__(self, num_episodes=1, wait_for_sim=True):
        super().__init__("action_client")

        self.num_episodes = num_episodes
        self.current_episode = 0
        self.episode_step = 0
        self.current_action_traj = None
        self.subtask_info = None
        self.max_reward = None

        # Publishers
        self.action_pub = self.create_publisher(
            Float32MultiArray, "/robot/actions", QOS_BEST_EFFORT
        )

        # Subscribers for monitoring the simulation state
        self.reward_sub = self.create_subscription(
            Float32MultiArray, "/reward", self.reward_callback, QOS_BEST_EFFORT
        )

        self.left_arm_sub = self.create_subscription(
            JointState,
            "/left_arm/joint_states",
            self.joint_state_callback,
            QOS_BEST_EFFORT,
        )

        # State tracking
        self.sim_ready = False
        self.episode_success = False
        self.episode_reward = 0.0

        # Wait for simulation to be ready
        if wait_for_sim:
            self.get_logger().info("Waiting for simulation to be ready...")
            while not self.sim_ready and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)
                time.sleep(0.1)

        # Start the first episode after a delay
        self.create_timer(3.0, self.start_episode)

        self.get_logger().info("Action client initialized and running")

    def joint_state_callback(self, msg):
        """Use joint state messages to detect if simulation is running."""
        self.sim_ready = True

    def reward_callback(self, msg):
        """Process reward updates from the simulation."""
        if len(msg.data) >= 2:
            reward = msg.data[0]
            success = bool(msg.data[1])

            self.episode_reward = reward
            if success and not self.episode_success:
                self.episode_success = True
                self.get_logger().info(f"Episode successful! Reward: {reward}")

    def start_episode(self):
        """Start a new episode."""
        if self.current_episode >= self.num_episodes:
            self.get_logger().info("All episodes completed!")
            return

        self.get_logger().info(
            f"Starting episode {self.current_episode+1}/{self.num_episodes}"
        )

        # Reset episode state
        self.episode_step = 0
        self.episode_success = False
        self.episode_reward = 0.0

        # Get action trajectory from policy rollout
        action_traj, subtask_info, max_reward = rollout_policy()
        self.current_action_traj = action_traj
        self.subtask_info = subtask_info
        self.max_reward = max_reward

        # Start sending actions at regular intervals
        self.action_timer = self.create_timer(
            0.05, self.send_action
        )  # 20Hz to match simulation

    def send_action(self):
        """Send the next action in the trajectory."""
        if self.episode_step >= len(self.current_action_traj):
            # End of trajectory
            self.action_timer.cancel()
            self.get_logger().info(f"Episode {self.current_episode+1} completed")

            # Start next episode after a delay
            self.current_episode += 1
            if self.current_episode < self.num_episodes:
                self.create_timer(3.0, self.start_episode)
            return

        # Get current action
        action = self.current_action_traj[self.episode_step]

        # Format action as expected by the simulation node
        left_arm = action.get("left_arm", np.zeros(6))
        right_arm = action.get("right_arm", np.zeros(6))
        left_gripper = action.get("left_gripper", 0.0)
        right_gripper = action.get("right_gripper", 0.0)

        # Create action message
        msg = Float32MultiArray()
        msg.data = np.concatenate(
            [left_arm, [left_gripper], right_arm, [right_gripper]]
        ).tolist()

        # Publish action
        self.action_pub.publish(msg)
        self.episode_step += 1


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_episodes",
        type=int,
        help="Number of episodes to run",
        default=5,
    )
    parser.add_argument(
        "--wait_for_sim",
        action="store_true",
        help="Wait for simulation to be ready before starting",
        default=True,
    )
    parsed_args, remaining = parser.parse_known_args()

    node = ActionGeneratorNode(
        num_episodes=parsed_args.num_episodes, wait_for_sim=parsed_args.wait_for_sim
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
