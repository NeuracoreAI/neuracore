"""Scripted policies for bimanual manipulation tasks."""

from typing import Dict, List, NamedTuple, Protocol

import numpy as np
from pyquaternion import Quaternion


class Waypoint(NamedTuple):
    """Single waypoint in a trajectory."""

    t: int
    xyz: np.ndarray
    quat: np.ndarray
    gripper: float


class TimeStep(Protocol):
    """Protocol for timestep objects."""

    observation: Dict


class BasePolicy:
    """Base class for scripted policies with trajectory interpolation."""

    def __init__(self, inject_noise: bool = False) -> None:
        """Initialize base policy.

        Args:
            inject_noise: Whether to inject noise into actions.
        """
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory: List[Waypoint] = []
        self.right_trajectory: List[Waypoint] = []
        self.curr_left_waypoint: Waypoint = None
        self.curr_right_waypoint: Waypoint = None

    def generate_trajectory(self, ts_first: TimeStep) -> None:
        """Generate trajectory waypoints for both arms.

        Args:
            ts_first: Initial timestep with observation.
        """
        raise NotImplementedError

    @staticmethod
    def interpolate(
        curr_waypoint: Waypoint, next_waypoint: Waypoint, t: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Interpolate between two waypoints.

        Args:
            curr_waypoint: Current waypoint.
            next_waypoint: Next waypoint.
            t: Current timestep.

        Returns:
            Interpolated (xyz, quat, gripper) tuple.
        """
        t_frac = (t - curr_waypoint.t) / (next_waypoint.t - curr_waypoint.t)

        xyz = curr_waypoint.xyz + (next_waypoint.xyz - curr_waypoint.xyz) * t_frac
        quat = curr_waypoint.quat + (next_waypoint.quat - curr_waypoint.quat) * t_frac
        gripper = (
            curr_waypoint.gripper
            + (next_waypoint.gripper - curr_waypoint.gripper) * t_frac
        )

        return xyz, quat, gripper

    def __call__(self, ts: TimeStep) -> np.ndarray:
        """Execute policy to get action.

        Args:
            ts: Current timestep.

        Returns:
            Action array for both arms.
        """
        # Generate trajectory at first timestep
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # Update current waypoints if needed
        if self.left_trajectory[0].t == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0].t == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # Interpolate between waypoints
        left_xyz, left_quat, left_gripper = self.interpolate(
            self.curr_left_waypoint, next_left_waypoint, self.step_count
        )
        right_xyz, right_quat, right_gripper = self.interpolate(
            self.curr_right_waypoint, next_right_waypoint, self.step_count
        )

        # Inject noise if enabled
        if self.inject_noise:
            noise_scale = 0.01
            left_xyz = left_xyz + self.rng.uniform(
                -noise_scale, noise_scale, left_xyz.shape
            )
            right_xyz = right_xyz + self.rng.uniform(
                -noise_scale, noise_scale, right_xyz.shape
            )

        # Combine actions
        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])

    @property
    def rng(self) -> np.random.Generator:
        """Get random number generator."""
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng()
        return self._rng


class PickAndTransferPolicy(BasePolicy):
    """Policy for picking up cube with right arm and transferring to left arm."""

    def generate_trajectory(self, ts_first: TimeStep) -> None:
        """Generate pick and transfer trajectory.

        Args:
            ts_first: Initial timestep with observation data.
        """
        obs = ts_first.observation

        # Get initial poses
        init_mocap_pose_right = obs["mocap_pose_right"]
        init_mocap_pose_left = obs["mocap_pose_left"]

        # Get box position
        box_info = np.array(obs["env_state"])
        box_xyz = box_info[:3]

        # Define key orientations
        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(
            axis=[0.0, 1.0, 0.0], degrees=-60
        )
        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)
        meet_xyz = np.array([0, 0.5, 0.25])

        # Left arm trajectory (receiver)
        self.left_trajectory = [
            Waypoint(
                t=0,
                xyz=init_mocap_pose_left[:3],
                quat=init_mocap_pose_left[3:],
                gripper=0,
            ),  # Sleep
            Waypoint(
                t=100,
                xyz=meet_xyz + np.array([-0.1, 0, -0.02]),
                quat=meet_left_quat.elements,
                gripper=1,
            ),  # Approach meet position
            Waypoint(
                t=260,
                xyz=meet_xyz + np.array([0.02, 0, -0.02]),
                quat=meet_left_quat.elements,
                gripper=1,
            ),  # Move to meet position
            Waypoint(
                t=310,
                xyz=meet_xyz + np.array([0.02, 0, -0.02]),
                quat=meet_left_quat.elements,
                gripper=0,
            ),  # Close gripper
            Waypoint(
                t=360,
                xyz=meet_xyz + np.array([-0.1, 0, -0.02]),
                quat=np.array([1, 0, 0, 0]),
                gripper=0,
            ),  # Move left
            Waypoint(
                t=400,
                xyz=meet_xyz + np.array([-0.1, 0, -0.02]),
                quat=np.array([1, 0, 0, 0]),
                gripper=0,
            ),  # Stay
        ]

        # Right arm trajectory (picker)
        self.right_trajectory = [
            Waypoint(
                t=0,
                xyz=init_mocap_pose_right[:3],
                quat=init_mocap_pose_right[3:],
                gripper=0,
            ),  # Sleep
            Waypoint(
                t=90,
                xyz=box_xyz + np.array([0, 0, 0.08]),
                quat=gripper_pick_quat.elements,
                gripper=1,
            ),  # Approach the cube
            Waypoint(
                t=130,
                xyz=box_xyz + np.array([0, 0, -0.015]),
                quat=gripper_pick_quat.elements,
                gripper=1,
            ),  # Go down
            Waypoint(
                t=170,
                xyz=box_xyz + np.array([0, 0, -0.015]),
                quat=gripper_pick_quat.elements,
                gripper=0,
            ),  # Close gripper
            Waypoint(
                t=200,
                xyz=meet_xyz + np.array([0.05, 0, 0]),
                quat=gripper_pick_quat.elements,
                gripper=0,
            ),  # Approach meet position
            Waypoint(
                t=220, xyz=meet_xyz, quat=gripper_pick_quat.elements, gripper=0
            ),  # Move to meet position
            Waypoint(
                t=310, xyz=meet_xyz, quat=gripper_pick_quat.elements, gripper=1
            ),  # Open gripper
            Waypoint(
                t=360,
                xyz=meet_xyz + np.array([0.1, 0, 0]),
                quat=gripper_pick_quat.elements,
                gripper=1,
            ),  # Move to right
            Waypoint(
                t=400,
                xyz=meet_xyz + np.array([0.1, 0, 0]),
                quat=gripper_pick_quat.elements,
                gripper=1,
            ),  # Stay
        ]
